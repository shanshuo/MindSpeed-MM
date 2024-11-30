from curses import KEY_A1
from typing import Optional, Tuple, Dict

from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from megatron.core import mpu, tensor_parallel
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args

from mindspeed_mm.models.common.ffn import FeedForward as TensorParallelFeedForward
from mindspeed_mm.models.common.communications import split_forward_gather_backward, gather_forward_split_backward
from mindspeed_mm.models.common.embeddings.pos_embeddings import Rotary3DPositionEmbedding
from mindspeed_mm.models.common.embeddings.time_embeddings import TimeStepEmbedding
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.embeddings.patch_embeddings import VideoPatchEmbed2D, VideoPatch2D
from mindspeed_mm.models.common.attention import SelfAttentionBSH, ParallelSelfAttentionSBH


class SatDiT(MultiModalModule):
    """
    A video dit model for video generation. can process both standard continuous images of shape
    (batch_size, num_channels, width, height) as well as quantized image embeddings of shape
    (batch_size, num_image_vectors). Define whether input is continuous or discrete depending on config.

    Args:
        num_layers: The number of layers for VideoDiTBlock.
        num_heads: The number of heads to use for multi-head attention.
        head_dim: The number of channels in each head.
        in_channels: The number of channels in the input (specify if the input is continuous).
        out_channels: The number of channels in the output.
        dropout: The dropout probability to use.
        cross_attention_dim: The number of prompt dimensions to use.
        attention_bias: Whether to use bias in VideoDiTBlock's attention.
        input_size: The shape of the latents (specify if the input is discrete).
        patch_size: The shape of the patchs.
        activation_fn: The name of activation function use in VideoDiTBlock.
        norm_type: can be 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'.
        num_embeds_ada_norm: The number of diffusion steps used during training. Pass if at least one of the norm_layers is
                             `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings
                             that are added to the hidden states.
        norm_elementswise_affine: Whether to use learnable elementwise affine parameters for normalization.
        norm_eps: The eps of he normalization.
        use_rope: Whether to use rope in attention block.
        interpolation_scale: The scale for interpolation.
        qk_ln: Whether to use layer norm in qk.
        elementwise_affine: Whether to use learnable elementwise affine parameters for qk normalization.
    """

    def __init__(
        self,
        num_layers: int = 1,
        num_heads: int = 16,
        head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        input_size: Tuple[int] = None,
        patch_size: Tuple[int] = None,
        activation_fn: str = "geglu",
        norm_type: str = "layer_norm",
        num_embeds_ada_norm: Optional[int] = None,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        caption_channels: int = None,
        use_rope: bool = False,
        interpolation_scale: Tuple[float] = None,
        elementwise_affine: bool = True,
        text_length=None,
        text_hidden_size=None,
        time_embed_dim=None,
        concat_text_embed=None,
        learnable_pos_embed=False,
        **kwargs
    ):
        super().__init__(config=None)
        # Validate inputs and init args.
        if patch_size is not None:
            if norm_type not in ["ada_norm", "ada_norm_zero", "ada_norm_single", "qk_ln"]:
                raise NotImplementedError(
                    f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
                )
            elif norm_type in ["ada_norm", "ada_norm_zero"] and num_embeds_ada_norm is None:
                raise ValueError(
                    f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
                )
        self.patch_size_t, self.patch_size_h, self.patch_size_w = patch_size
        self.norm_type = norm_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.concat_text_embed = concat_text_embed
        t, h, w = input_size
        seq_len = text_length + t // self.patch_size_t * h // self.patch_size_h * w // self.patch_size_w
        seq_begin = (seq_len // mpu.get_context_parallel_world_size()) * mpu.get_context_parallel_rank()
        seq_end = (seq_len // mpu.get_context_parallel_world_size()) * (mpu.get_context_parallel_rank() + 1)
        if seq_end < text_length:
            self.text_length = seq_len // mpu.get_context_parallel_world_size()
        elif seq_begin > text_length:
            self.text_length = 0
        else:
            self.text_length = text_length - seq_begin
        self.text_hidden_size = text_hidden_size
        self.elementwise_affine = elementwise_affine
        inner_dim = num_heads * head_dim
        self.time_embed_dim = time_embed_dim if time_embed_dim is not None else inner_dim

        args = get_args()
        self.recompute_granularity = args.recompute_granularity
        self.distribute_saved_activations = args.distribute_saved_activations
        self.recompute_method = args.recompute_method
        self.recompute_num_layers = args.recompute_num_layers
        if self.recompute_granularity == "selective":
            raise ValueError("recompute_granularity does not support selective mode in VideoDiT")
        if self.distribute_saved_activations:
            raise NotImplementedError("distribute_saved_activations is currently not supported")

        if mpu.get_context_parallel_world_size() > 1:
            self.enable_sequence_parallelism = True
        else:
            self.enable_sequence_parallelism = False

        # Initialize blocks
        # Init PatchEmbed
        self.time_embed = TimeStepEmbedding(inner_dim, self.time_embed_dim)
        self.patch_embed = VideoPatch2D(in_channels, inner_dim, self.patch_size_h)
        self.pos_embed = Rotary3DPositionEmbedding(
            hidden_size_head=head_dim,
            text_length=text_length,
            height=input_size[1] // self.patch_size_h,
            width=input_size[2] // self.patch_size_w,
            compressed_num_frames=(input_size[0] - 1) // interpolation_scale[0] + 1,
            hidden_size=inner_dim,
            learnable_pos_embed=learnable_pos_embed
        )
        # Init VideoDiTBlock
        self.videodit_blocks = nn.ModuleList(
            [
                VideoDiTBlock(
                    dim=inner_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    use_rope=use_rope,
                    pos_embed=self.pos_embed,
                    interpolation_scale=interpolation_scale,
                    enable_sequence_parallelism=self.enable_sequence_parallelism,
                    time_embed_dim=self.time_embed_dim,
                    text_length=self.text_length,
                )
                for i in range(num_layers)
            ]
        )
        # Init Norm
        self.norm_final = nn.LayerNorm(inner_dim, elementwise_affine=elementwise_affine, eps=1e-5)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.time_embed_dim, 2 * inner_dim, bias=True))
        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=elementwise_affine, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim,
                                  self.patch_size_t * self.patch_size_h * self.patch_size_w * self.out_channels)
        # Init Projection
        self.caption_projection = None
        if text_hidden_size is not None:
            self.caption_projection = nn.Linear(self.text_hidden_size, inner_dim)
        print(self)

    def forward(
        self,
        latents: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        prompt: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        use_image_num: Optional[int] = 0,
        **kwargs
    ) -> torch.Tensor:
        """
        Args:
            latents: Shape (batch size, num latent pixels) if discrete, shape (batch size, channel, height, width) if continuous.
            timestep: Used to indicate denoising step. Optional timestep to be applied as an embedding in AdaLayerNorm.
            prompt: Conditional embeddings for cross attention layer.
            video_mask: An attention mask of shape (batch, key_tokens) is applied to latents.
            prompt_mask: Cross-attention mask applied to prompt.
            added_cond_kwargs: resolution or aspect_ratio.
            class_labels: Used to indicate class labels conditioning.
            use_image_num: The number of images use for trainning.
        """
        b, _, t, _, _ = latents.shape
        frames = t - use_image_num
        vid_mask, img_mask = None, None
        prompt_vid_mask, prompt_img_mask = None, None

        # 1. Input
        frames = ((frames - 1) // self.patch_size_t + 1) if frames % 2 == 1 else frames // self.patch_size_t  # patchfy
        height, width = latents.shape[-2] // self.patch_size_h, latents.shape[-1] // self.patch_size_w

        if "masked_video" in kwargs.keys() and kwargs["masked_video"] is not None:
            latents = torch.cat([latents, kwargs["masked_video"]], dim=1)
        
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        latents_vid, latents_img, prompt_vid, prompt_img, timestep_vid, timestep_img, \
            embedded_timestep_vid, embedded_timestep_img = self._operate_on_patched_inputs(
            latents, prompt, timestep, added_cond_kwargs, b, frames, use_image_num
        )
        if self.concat_text_embed:
            latents_vid = torch.cat((prompt_vid, latents_vid), dim=1)

        if self.enable_sequence_parallelism and latents_vid is not None and prompt_vid is not None:
            latents_vid = rearrange(latents_vid, 'b s h -> s b h', b=b).contiguous()
            latents_vid = split_forward_gather_backward(latents_vid, mpu.get_context_parallel_group(), dim=0,
                                                        grad_scale='down')

        frames = torch.tensor(frames)
        height = torch.tensor(height)
        width = torch.tensor(width)
        if self.recompute_granularity == "full":
            if latents_vid is not None:
                latents_vid = self._checkpointed_forward(
                    latents_vid,
                    video_mask=vid_mask,
                    prompt=prompt_vid,
                    prompt_mask=prompt_vid_mask,
                    timestep=timestep_vid,
                    class_labels=class_labels,
                    frames=frames,
                    height=height,
                    width=width
                )
        else:
            for block in self.videodit_blocks:
                if latents_vid is not None:
                    latents_vid = block(
                        latents_vid,
                        video_mask=vid_mask,
                        prompt=prompt_vid,
                        prompt_mask=prompt_vid_mask,
                        timestep=timestep_vid,
                        class_labels=class_labels,
                        frames=frames,
                        height=height,
                        width=width
                    )

        if self.enable_sequence_parallelism and latents_vid is not None:
            latents_vid = rearrange(latents_vid, 's b h -> b s h', b=b).contiguous()
            latents_vid = gather_forward_split_backward(latents_vid, mpu.get_context_parallel_group(), dim=1,
                                                        grad_scale='up')

        # 3. Output
        output_vid, output_img = None, None
        if latents_vid is not None:
            output_vid = self._get_output_for_patched_inputs(
                latents=latents_vid,
                timestep=timestep_vid,
                class_labels=class_labels,
                embedded_timestep=embedded_timestep_vid,
                num_frames=frames,
                height=height,
                width=width,
            )  # [b, c, t, h, w]

        if output_vid is not None and output_img is not None:
            output = torch.cat([output_vid, output_img], dim=2)
        elif output_vid is not None:
            output = output_vid
        elif output_img is not None:
            output = output_img
        return output

    def _get_block(self, layer_number):
        return self.videodit_blocks[layer_number]

    def _checkpointed_forward(
        self,
        latents,
        video_mask,
        prompt,
        prompt_mask,
        timestep,
        class_labels,
        frames,
        height,
        width):
        """Forward method with activation checkpointing."""

        def custom(start, end):
            def custom_forward(*args, **kwargs):
                x_, *args = args
                for index in range(start, end):
                    layer = self._get_block(index)
                    x_ = layer(x_, *args, **kwargs)
                return x_
            return custom_forward

        if self.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and
            # checkpoint the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            layer_num = 0
            while layer_num < self.num_layers:
                latents = tensor_parallel.checkpoint(
                    custom(layer_num, layer_num + self.recompute_num_layers),
                    self.distribute_saved_activations,
                    latents,
                    prompt,
                    video_mask,
                    prompt_mask,
                    timestep,
                    class_labels,
                    frames,
                    height,
                    width
                )
                layer_num += self.recompute_num_layers
        elif self.recompute_method == "block":
            for layer_num in range(self.num_layers):
                if layer_num < self.recompute_num_layers:
                    latents = tensor_parallel.checkpoint(
                        custom(layer_num, layer_num + 1),
                        self.distribute_saved_activations,
                        latents,
                        prompt,
                        video_mask,
                        prompt_mask,
                        timestep,
                        class_labels,
                        frames,
                        height,
                        width
                    )
                else:
                    block = self._get_block(layer_num)
                    latents = block(
                        latents,
                        video_mask=video_mask,
                        prompt=prompt,
                        prompt_mask=prompt_mask,
                        timestep=timestep,
                        class_labels=class_labels,
                        frames=frames,
                        height=height,
                        width=width
                    )
        else:
            raise ValueError("Invalid activation recompute method.")
        return latents

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the module (assuming that all the module parameters have the same dtype)."""
        params = tuple(self.parameters())
        if len(params) > 0:
            return params[0].dtype
        else:
            buffers = tuple(self.buffers())
            return buffers[0].dtype

    def _operate_on_patched_inputs(self, latents, prompt, timestep, added_cond_kwargs, batch_size, frames,
                                   use_image_num):
        if self.pos_embed is not None:
            latents_vid, latents_img = self.patch_embed(latents.to(self.dtype), prompt)
            _, seq_len, _ = latents_vid.shape
            pos_emb = self.pos_embed.position_embedding_forward(latents.to(self.dtype),
                                                                seq_length=seq_len - self.text_length)
            if pos_emb is not None:
                latents_vid = latents_vid + pos_emb
        else:
            latents_vid, latents_img = self.patch_embed(latents.to(self.dtype), frames)
        timestep_vid, timestep_img = None, None
        embedded_timestep_vid, embedded_timestep_img = None, None
        prompt_vid, prompt_img = None, None

        if self.time_embed is not None:
            timestep_vid = self.time_embed(timestep)
        if self.caption_projection is not None:
            prompt = self.caption_projection(prompt)
            if latents_vid is None:
                prompt_img = rearrange(prompt, 'b 1 l d -> (b 1) l d')
            else:
                prompt_vid = rearrange(prompt[:, :1], 'b 1 l d -> (b 1) l d')
                if latents_img is not None:
                    prompt_img = rearrange(prompt[:, 1:], 'b i l d -> (b i) l d')

        return latents_vid, latents_img, prompt_vid, prompt_img, timestep_vid, timestep_img, embedded_timestep_vid, embedded_timestep_img

    def _get_output_for_patched_inputs(self, latents, timestep, class_labels, embedded_timestep, num_frames,
                                       height=None, width=None):
        x = self.norm_final(latents)
        x = x[:, self.pos_embed.text_length:, :]
        shift, scale = self.adaLN_modulation(timestep).chunk(2, dim=1)
        x = self.norm_out(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.proj_out(x)
        latents = x

        # unpatchify
        output = rearrange(latents, "b (t h w) (c o p q) -> b (t o) c (h p) (w q)",
                           b=latents.shape[0], h=height, w=width,
                           o=self.patch_size_t, p=self.patch_size_h, q=self.patch_size_w,
                           c=self.out_channels).transpose(1, 2)
        return output


class VideoDiTBlock(nn.Module):
    """
    A basic dit block for video generation.

    Args:
        dim: The number out channels in the input and output.
        num_heads: The number of heads to use for multi-head attention.
        head_dim: The number of channels in each head.
        in_channels: The number of channels in the input (specify if the input is continuous).
        out_channels: The number of channels in the output.
        dropout: The dropout probability to use.
        cross_attention_dim: The number of prompt dimensions to use.
        attention_bias: Whether to use bias in VideoDiTBlock's attention.
        activation_fn: The name of activation function use in VideoDiTBlock.
        norm_type: can be 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'.
        num_embeds_ada_norm: The number of diffusion steps used during training. Pass if at least one of the norm_layers is
                             `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings
                             that are added to the hidden states.
        norm_elementswise_affine: Whether to use learnable elementwise affine parameters for normalization.
        norm_eps: The eps of he normalization.
        interpolation_scale: The scale for interpolation.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        attention_out_bias: bool = True,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        use_rope: bool = False,
        interpolation_scale: Tuple[float] = None,
        enable_sequence_parallelism: bool = False,
        time_embed_dim=None,
        text_length=None,
        pos_embed=None,
    ):
        super().__init__()
        self.time_embed_dim = time_embed_dim if time_embed_dim is not None else dim
        self.cross_attention_dim = cross_attention_dim
        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )
        self.norm_type = norm_type
        self.text_length = text_length
        self.positional_embeddings = positional_embeddings

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError("If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined.")
        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = pos_embed

        # Define three blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.enable_sequence_parallelism = enable_sequence_parallelism
        if self.enable_sequence_parallelism:
            attention = ParallelSelfAttentionSBH
        else:
            attention = SelfAttentionBSH

        self.self_atten = attention(
            query_dim=dim,
            key_dim=None,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            proj_qkv_bias=attention_bias,
            proj_out_bias=attention_out_bias,
            qk_ln=(norm_type == "qk_ln"),
            use_rope=use_rope,
            rope=pos_embed,
            interpolation_scale=interpolation_scale
        )


        # 2. Feed-forward
        self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.ff = TensorParallelFeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias
        )

        # 3. Scale-shift.
        args = get_args()
        config = core_transformer_config_from_args(args)
        self.scale_shift_table = nn.Sequential(
            nn.SiLU(),
            tensor_parallel.ColumnParallelLinear(
                self.time_embed_dim,
                12 * dim,
                config=config,
                init_method=config.init_method,
                gather_output=True
            )
        )

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self,
        latents: torch.Tensor,
        prompt: Optional[torch.Tensor] = None,
        video_mask: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        timestep: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.Tensor] = None,
        frames: torch.int64 = None,
        height: torch.int64 = None,
        width: torch.int64 = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.FloatTensor:
        # 1. Self-Attention
        frames = frames.item()
        height = height.item()
        width = width.item()
        if self.enable_sequence_parallelism:
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                text_shift_msa,
                text_scale_msa,
                text_gate_msa,
                text_shift_mlp,
                text_scale_mlp,
                text_gate_mlp,
            ) = self.scale_shift_table(timestep)[0].unsqueeze(0).chunk(12, dim=2)
            latents_text = latents[:self.text_length]
            latents_vid = latents[self.text_length:]
        else:
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
                text_shift_msa,
                text_scale_msa,
                text_gate_msa,
                text_shift_mlp,
                text_scale_mlp,
                text_gate_mlp,
            ) = self.scale_shift_table(timestep)[0].unsqueeze(1).chunk(12, dim=2)
            latents_text = latents[:, :self.text_length]
            latents_vid = latents[:, self.text_length:]
        latents_vid = self.norm1(latents_vid)
        latents_text = self.norm1(latents_text)
        latents_vid = latents_vid * (1 + scale_msa) + shift_msa
        latents_text = latents_text * (1 + text_scale_msa) + text_shift_msa
        if self.enable_sequence_parallelism:
            norm_latents = torch.cat((latents_text, latents_vid), dim=0)  # (s_t + t * h/2 * w/2, b, n * d)
        else:
            norm_latents = torch.cat((latents_text, latents_vid), dim=1)  # (b, s_t + t * h/2 * w/2, n * d)

        if self.pos_embed is not None and self.positional_embeddings is not None:
            norm_latents = self.pos_embed(norm_latents)

        attn_output = self.self_atten(
            query=norm_latents,
            key=None,
            mask=None,
            frames=frames,
            height=height,
            width=width,
        )
        if self.enable_sequence_parallelism:
            attn_vid_output = gate_msa * attn_output[self.text_length:]
            attn_text_output = text_gate_msa * attn_output[:self.text_length]
            attn_output = torch.cat((attn_text_output, attn_vid_output), dim=0)
        else:
            attn_vid_output = gate_msa * attn_output[:, self.text_length:]
            attn_text_output = text_gate_msa * attn_output[:, :self.text_length]
            attn_output = torch.cat((attn_text_output, attn_vid_output), dim=1)

        latents = attn_output + latents

        # 2. Feed-forward
        if self.enable_sequence_parallelism:
            latents_text = latents[:self.text_length]
            latents_vid = latents[self.text_length:]
            latents_text = self.norm2(latents_text)
            latents_vid = self.norm2(latents_vid)
            latents_vid = latents_vid * (1 + scale_mlp) + shift_mlp
            latents_text = latents_text * (1 + text_scale_mlp) + text_shift_mlp
            norm_latents = torch.cat((latents_text, latents_vid), dim=0)
        else:
            latents_text = latents[:, :self.text_length]
            latents_vid = latents[:, self.text_length:]
            latents_text = self.norm2(latents_text)
            latents_vid = self.norm2(latents_vid)
            latents_vid = latents_vid * (1 + scale_mlp) + shift_mlp
            latents_text = latents_text * (1 + text_scale_mlp) + text_shift_mlp
            norm_latents = torch.cat((latents_text, latents_vid), dim=1)

        ff_output = self.ff(norm_latents)

        if self.enable_sequence_parallelism:
            ff_vid_output = gate_mlp * ff_output[self.text_length:]
            ff_text_output = text_gate_mlp * ff_output[:self.text_length]
            ff_output = torch.cat((ff_text_output, ff_vid_output), dim=0)
        else:
            ff_vid_output = gate_mlp * ff_output[:, self.text_length:]
            ff_text_output = text_gate_mlp * ff_output[:, :self.text_length]
            ff_output = torch.cat((ff_text_output, ff_vid_output), dim=1)

        latents = ff_output + latents
        return latents

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        self._chunk_size = chunk_size
        self._chunk_dim = dim
