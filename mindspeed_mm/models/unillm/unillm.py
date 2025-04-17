# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union

import torch
import torch_npu
from torch import nn
from einops import rearrange

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.activations import ACT2FN


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim,
        rope_theta,
        max_position_embeddings,
        rope_type="default",
        partial_rotary_factor=None,
        factor=None,
        device=None
    ):
        super().__init__()
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.rope_type = rope_type
        self.partial_rotary_factor = partial_rotary_factor

        if self.rope_type == "default":
            inv_freq, self.attention_scaling = self._compute_default_rope_parameters(
                rope_theta=self.rope_theta,
                head_dim=self.head_dim,
                partial_rotary_factor=self.partial_rotary_factor,
                device=device)
        elif self.rope_type == "dynamic":
            inv_freq, self.attention_scaling = self._compute_default_rope_parameters(
                rope_theta=self.rope_theta,
                head_dim=self.head_dim,
                partial_rotary_factor=self.partial_rotary_factor,
                factor=factor,
                device=device
            )
        else:
            raise NotImplementedError(f"The rope_type: {self.rope_type} is not support Now, Only support 'default' and 'dynamic'")
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                rope_theta=self.rope_theta,
                head_dim=self.head_dim,
                partial_rotary_factor=self.partial_rotary_factor, 
                device=device
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    def _compute_default_rope_parameters(
        self,
        rope_theta,
        head_dim,
        partial_rotary_factor,
        factor=None,
        device=None
    ):
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = rope_theta
        partial_rotary_factor = partial_rotary_factor or 1.0
        dim = int(head_dim * partial_rotary_factor)
        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        return inv_freq, attention_factor

    def _compute_dynamic_ntk_parameters(
        self,
        rope_theta,
        head_dim,
        seq_len,
        partial_rotary_factor,
        factor,
        max_position_embeddings,
    ):
        if factor is None:
            raise ValueError(f"The param factor can not be None")
        partial_rotary_factor = partial_rotary_factor or 1.0
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE
        # seq_len: default to max_position_embeddings, e.g. at init time
        seq_len = seq_len if seq_len is not None and seq_len > max_position_embeddings else max_position_embeddings

        # Compute the inverse frequencies
        base = rope_theta * ((factor * seq_len / max_position_embeddings) - (factor - 1)) ** (dim / (dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        return inv_freq, attention_factor


class LlamaMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, mlp_bias, hidden_act):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=mlp_bias)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        layer_idx,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        attention_dropout: 0.0,
        attention_bias: False
        ):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = hidden_size // num_attention_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = attention_dropout
        self.is_causal = True

        # text
        self.q_proj_text = nn.Linear(
            hidden_size, num_attention_heads * self.head_dim, bias=attention_bias
        )
        self.k_proj_text = nn.Linear(
            hidden_size, num_key_value_heads * self.head_dim, bias=attention_bias
        )
        self.v_proj_text = nn.Linear(
            hidden_size, num_key_value_heads * self.head_dim, bias=attention_bias
        )
        self.o_proj_text = nn.Linear(
            num_attention_heads * self.head_dim, hidden_size, bias=attention_bias
        )

        # vision
        self.q_proj_vision = nn.Linear(
            hidden_size, num_attention_heads * self.head_dim, bias=attention_bias
        )
        self.k_proj_vision = nn.Linear(
            hidden_size, num_key_value_heads * self.head_dim, bias=attention_bias
        )
        self.v_proj_vision = nn.Linear(
            hidden_size, num_key_value_heads * self.head_dim, bias=attention_bias
        )
        self.o_proj_vision = nn.Linear(
            num_attention_heads * self.head_dim, hidden_size, bias=attention_bias
        )
 
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # ADD
        image_token_nums: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if image_token_nums > 0:
            input_shape_vision = hidden_states[:, -image_token_nums:].shape[:-1]
            hidden_shape_vision = (*input_shape_vision, -1, self.head_dim) 

            if hidden_states.shape[-2] - image_token_nums > 0:
                input_shape_text = hidden_states[:, :-image_token_nums].shape[:-1]
                hidden_shape_text = (*input_shape_text, -1, self.head_dim)
        else:
            input_shape_text = hidden_states.shape[:-1]
            hidden_shape_text = (*input_shape_text, -1, self.head_dim)

        # Vision
        if image_token_nums > 0:
            query_states_vision = self.q_proj_vision(hidden_states[:, -image_token_nums:]).view(hidden_shape_vision).transpose(1, 2)
            key_states_vision = self.k_proj_vision(hidden_states[:, -image_token_nums:]).view(hidden_shape_vision).transpose(1, 2)
            value_states_vision = self.v_proj_vision(hidden_states[:, -image_token_nums:]).view(hidden_shape_vision).transpose(1, 2)

            # 推理视觉token
            if hidden_states.shape[-2] == 1:
                # inference
                query_states = query_states_vision
                key_states = key_states_vision
                value_states = value_states_vision
            # 视觉文本训练
            else:
                query_states_text = self.q_proj_text(hidden_states[:, :-image_token_nums]).view(hidden_shape_text).transpose(1, 2)
                key_states_text = self.k_proj_text(hidden_states[:, :-image_token_nums]).view(hidden_shape_text).transpose(1, 2)
                value_states_text = self.v_proj_text(hidden_states[:, :-image_token_nums]).view(hidden_shape_text).transpose(1, 2)

                # cat text and vision
                query_states = torch.cat([query_states_text, query_states_vision], dim=-2)
                key_states = torch.cat([key_states_text, key_states_vision], dim=-2)
                value_states = torch.cat([value_states_text, value_states_vision], dim=-2)
        else:
            # 训练文本token和推理文本token都可以
            query_states = self.q_proj_text(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj_text(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj_text(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # ===   NPU training acceleration ===
        head_dim = self.head_dim
        FA_head_num = query_states.shape[1]
        input_layout = "BSH"
        enable_FA = True
        query_states = rearrange(query_states, 'b h s d -> b s (h d)', h=FA_head_num)
        key_states = rearrange(key_states, 'b h s d -> b s  (h d)', h=FA_head_num)
        value_states = rearrange(value_states, 'b h s d -> b s (h d)', h=FA_head_num)
        if enable_FA:
            hidden_states = torch_npu.npu_fusion_attention(query_states, key_states, value_states,
                                                        atten_mask=attention_mask.to(torch.bool), 
                                                        input_layout=input_layout,
                                                        scale=self.scaling,
                                                        head_num=FA_head_num)[0]
            attn_output = hidden_states

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()

        if image_token_nums > 0:
            if hidden_states.shape[-2] == 1:
                attn_output_vision = self.o_proj_vision(attn_output[:, -image_token_nums:])
                attn_output = attn_output_vision
            else:
                attn_output_text = self.o_proj_text(attn_output[:, :-image_token_nums])
                attn_output_vision = self.o_proj_vision(attn_output[:, -image_token_nums:])
                attn_output = torch.cat([attn_output_text, attn_output_vision], dim=-2) 
        else:
            attn_output = self.o_proj_text(attn_output)

        return attn_output, None


class LlamaDecoderLayer(nn.Module):
    def __init__(
        self,
        layer_idx,
        hidden_size,
        num_attention_heads,
        num_key_value_heads,
        attention_dropout,
        attention_bias,
        intermediate_size,
        mlp_bias,
        hidden_act,
        rms_norm_eps
    ):
        super().__init__()
        self.self_attn = LlamaAttention(
            layer_idx=layer_idx,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias
        )

        self.mlp_text = LlamaMLP(hidden_size, intermediate_size, mlp_bias, hidden_act)
        self.mlp_vision = LlamaMLP(hidden_size, intermediate_size, mlp_bias, hidden_act)

        self.input_layernorm_text = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm_text = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)

        self.input_layernorm_vision = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm_vision = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC

        # ADD
        image_token_nums: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states

        if image_token_nums > 0 and hidden_states.shape[-2] == 1:
            hidden_states = self.input_layernorm_vision(hidden_states)
        elif image_token_nums == 0:
            hidden_states = self.input_layernorm_text(hidden_states)
        else:
            hidden_states_text = self.input_layernorm_text(hidden_states[:, :-image_token_nums])
            hidden_states_vision = self.input_layernorm_vision(hidden_states[:, -image_token_nums:])
            hidden_states = torch.cat([hidden_states_text, hidden_states_vision], dim=-2)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            image_token_nums=image_token_nums,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states

        moe_losses = []
        if image_token_nums > 0 and hidden_states.shape[-2] == 1:
            hidden_states = self.post_attention_layernorm_vision(hidden_states)
            hidden_states = self.mlp_vision(hidden_states)
            if isinstance(hidden_states, tuple):
                moe_losses.append(hidden_states[1])
                hidden_states = hidden_states[0]
        elif image_token_nums == 0:
            hidden_states = self.post_attention_layernorm_text(hidden_states)
            hidden_states = self.mlp_text(hidden_states)
            if isinstance(hidden_states, tuple):
                moe_losses.append(hidden_states[1])
                hidden_states = hidden_states[0]
        else:
            hidden_states_text = self.post_attention_layernorm_text(hidden_states[:, :-image_token_nums])
            hidden_states_vision = self.post_attention_layernorm_vision(hidden_states[:, -image_token_nums:])
            hidden_states_text = self.mlp_text(hidden_states_text)
            if isinstance(hidden_states_text, tuple):
                moe_losses.append(hidden_states_text[1])
                hidden_states_text = hidden_states_text[0]
            hidden_states_vision = self.mlp_vision(hidden_states_vision)
            if isinstance(hidden_states_vision, tuple):
                moe_losses.append(hidden_states_vision[1])
                hidden_states_vision = hidden_states_vision[0]
            hidden_states = torch.cat([hidden_states_text, hidden_states_vision], dim=-2)

        hidden_states = residual + hidden_states

        outputs = (hidden_states, )
        if output_attentions:
            outputs += (self_attn_weights,)
        outputs += (moe_losses,)
        return outputs


class LlamaModel(nn.Module):
    """
    Transformer decoder consisting of *num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    """

    def __init__(
        self,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        attention_dropout,
        attention_bias,
        intermediate_size,
        mlp_bias,
        hidden_act,
        rope_theta,
        max_position_embeddings,
        vocab_size,
        rms_norm_eps,
        padding_idx
    ):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(
                layer_idx=layer_idx,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                attention_dropout=attention_dropout,
                attention_bias=attention_bias,
                intermediate_size=intermediate_size,
                mlp_bias=mlp_bias,
                hidden_act=hidden_act,
                rms_norm_eps=rms_norm_eps
            ) 
            for layer_idx in range(num_hidden_layers)]
        )

        self.norm_text = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)
        self.norm_vision = LlamaRMSNorm(hidden_size, eps=rms_norm_eps)

        self.rotary_emb = LlamaRotaryEmbedding(
            head_dim=hidden_size // num_attention_heads,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings
        )

        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        image_token_nums: Union[int, torch.Tensor] = 0,
        **flash_attn_kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            print("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_moe_loss = []
        
        for decoder_layer in self.layers[: self.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
            # if self.gradient_checkpointing:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,

                    # ADD
                    image_token_nums,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,

                    # ADD
                    image_token_nums=image_token_nums,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            # ADD
            all_moe_loss.extend(layer_outputs[-1])

        if image_token_nums > 0 and hidden_states.shape[-2] == 1:
            hidden_states = self.norm_vision(hidden_states)
        elif image_token_nums == 0:
            hidden_states = self.norm_text(hidden_states)
        else:
            hidden_states_text = self.norm_text(hidden_states[:, :-image_token_nums])
            hidden_states_vision = self.norm_vision(hidden_states[:, -image_token_nums:])
            hidden_states = torch.cat([hidden_states_text, hidden_states_vision], dim=1) 

        
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        output.all_moe_loss = all_moe_loss
        return output if return_dict else output.to_tuple()

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        _attn_implementation = "sdpa"
        if _attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if _attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            _attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to place the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class LlamaForCausalLM(nn.Module):
    def __init__(
        self,
        hidden_size,
        intermediate_size,
        max_position_embeddings,
        num_attention_heads,
        num_hidden_layers,
        num_key_value_heads,
        attention_dropout,
        attention_bias,
        mlp_bias,
        hidden_act,
        rope_theta,
        rms_norm_eps,
        padding_idx,
        torch_dtype,
        vocab_size,
        image_token_size,
        image_token_embed,
        n_embed,
        **kwargs
    ):
        super().__init__()
        self.model = LlamaModel(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attention_dropout=attention_dropout,
            attention_bias=attention_bias,
            intermediate_size=intermediate_size,
            mlp_bias=mlp_bias,
            hidden_act=hidden_act,
            rope_theta=rope_theta,
            max_position_embeddings=max_position_embeddings,
            vocab_size=vocab_size,
            rms_norm_eps=rms_norm_eps,
            padding_idx=padding_idx
            )
        self.vocab_size = vocab_size
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.vision_head = nn.Sequential(
            nn.Linear(n_embed, image_token_embed),  # n_embed, image_token_embed
            nn.GELU(),
            nn.Linear(image_token_embed, image_token_size),  # image_token_embed, image_token_size
        )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        image_gen: Optional[bool] = None,
        vocab_size: Union[int, torch.Tensor] = 0,
        image_token_nums: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            
            #ADD
            image_token_nums=image_token_nums,
            **kwargs,
        )

        hidden_states = outputs[0]

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        # ADD
        if image_gen:
            # 图像生成没有进行slice切分，是通过ar_token_nums来分离ar and non-ar。
            logits = self.vision_head(hidden_states[:, slice_indices, :])
        else:
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.vocab_size if not vocab_size else vocab_size, **kwargs)
        
        if len(outputs.all_moe_loss):
            loss += 0.01 * sum(outputs.all_moe_loss)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def loss_function(self, logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
        def fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
            reduction = "sum" if num_items_in_batch is not None else "mean"
            loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
            if reduction == "sum":
                loss = loss / num_items_in_batch
            return loss
        # Upcast to float if we need to compute the loss to avoid potential precision issues
        logits = logits.float()
        labels = labels.to(logits.device)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tokens
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = fixed_cross_entropy(shift_logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
        return loss
