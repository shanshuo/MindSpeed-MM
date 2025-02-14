import math
from typing import List, Optional, Union, Tuple
from contextlib import nullcontext
from einops import rearrange
import torch
from torch import nn
import torch_npu

from megatron.core import mpu, tensor_parallel
from megatron.training import get_args
from megatron.training.arguments import core_transformer_config_from_args
from mindspeed.ops.npu_rotary_position_embedding import npu_rotary_position_embedding

from mindspeed_mm.models.common import MultiModalModule
from mindspeed_mm.models.common.embeddings import (
    PatchEmbed3D,
    TimeStepEmbedding,
    TextProjection,
    SingleTokenRefiner,
    get_nd_rotary_pos_embed
)
from mindspeed_mm.models.common.blocks import MLP, ModulateDiT
from mindspeed_mm.models.common.activations import get_activation_layer
from mindspeed_mm.models.common.normalize import normalize


def _get_cu_seqlens(text_mask, img_len):
    """Calculate cu_seqlens_q, cu_seqlens_kv using text_mask and img_len

    Args:
        text_mask (torch.Tensor): the mask of text
        img_len (int): the length of image

    Returns:
        torch.Tensor: the calculated cu_seqlens for flash attention
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="npu")

    for i in range(batch_size):
        seq_len = text_len[i] + img_len
        seq_len1 = i * max_len + seq_len
        seq_len2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = seq_len1
        cu_seqlens[2 * i + 2] = seq_len2

    return cu_seqlens


class HunyuanVideoDiT(MultiModalModule):
    def __init__(
            self,
            patch_size: Tuple[int] = (1, 2, 2),
            in_channels: int = 4, # should be VAE.latent_channels,
            out_channels: Optional[int] = None,
            num_heads: int = 24,
            head_dim: int = 128,
            mlp_width_ratio: int = 4,
            mlp_act_type: str = "gelu_tanh",
            mm_double_blocks_depth: int = 20,
            mm_single_blocks_depth: int = 40,
            double_stream_recompute_layers: Optional[int] = None,
            single_stream_recompute_layers: Optional[int] = None,
            rope_dim_list: Tuple[int] = (16, 56, 56),
            qkv_bias: bool = True,
            qk_norm: bool = True,
            qk_norm_type: str = "rmsnorm",
            guidance_embed: bool = False, 
            text_projection: str = "single_refiner",
            text_states_dim: Tuple[int] = (4096, 768),
            timestep_hidden_dim: int = 256,
            use_attention_mask: bool = True,
            rope_theta: int = 256,
            embeded_guidance_scale: float = 6.016,
            **kwargs
        ):
        super().__init__(config=None)

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        self.text_states_dim = text_states_dim

        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection

        self.hidden_size = num_heads * head_dim
        self.heads_num = num_heads
        self.embeded_guidance_scale = embeded_guidance_scale

        args = get_args()
        config = core_transformer_config_from_args(args)
        self.recompute_granularity = args.recompute_granularity
        self.distribute_saved_activations = args.distribute_saved_activations
        self.recompute_method = args.recompute_method
        self.double_stream_recompute_layers = double_stream_recompute_layers if double_stream_recompute_layers else mm_double_blocks_depth
        self.single_stream_recompute_layers = single_stream_recompute_layers if single_stream_recompute_layers else mm_single_blocks_depth
        if self.recompute_granularity == "selective":
            raise ValueError("recompute_granularity does not support selective mode in HunyuanvideoDiT")
        if self.distribute_saved_activations:
            raise NotImplementedError("distribute_saved_activations is currently not supported")
        self.enable_tensor_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        self.sequence_parallel = args.sequence_parallel and self.enable_tensor_parallel
        
        # context parallel setting
        self.context_parallel_algo = args.context_parallel_algo if mpu.get_context_parallel_world_size() > 1 else None
        if self.context_parallel_algo is not None and self.context_parallel_algo not in ["ulysses_cp_algo"]:
            raise NotImplementedError(f"Context_parallel_algo {self.context_parallel_algo} is not implemented")
        
        if sum(rope_dim_list) != head_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {head_dim}"
            )

        # time modulation
        self.time_in = TimeStepEmbedding(timestep_hidden_dim, time_embed_dim=self.hidden_size)

        # txt modulation
        self.vector_in = MLP(
            in_channels=self.text_states_dim[1],
            hidden_channels=self.hidden_size,
            out_features=self.hidden_size,
            act_layer=get_activation_layer("silu"),
            enable_tensor_parallel=self.enable_tensor_parallel,
            enable_tp_sp=False
        )
        
        self.img_in = PatchEmbed3D(
            self.patch_size, self.in_channels, self.hidden_size
        )

        # guidance modulation
        self.guidance_in = (
            TimeStepEmbedding(timestep_hidden_dim, self.hidden_size)
            if guidance_embed
            else None
        )

        # text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                self.text_states_dim[0],
                self.hidden_size,
                nn.SiLU,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                in_channels=self.text_states_dim[0],
                hidden_size=self.hidden_size,
                time_embed_dim=timestep_hidden_dim,
                heads_num=num_heads,
                depth=2
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        # double blocks
        self.double_blocks = nn.ModuleList([
            MMDoubleStreamBlock(
                self.hidden_size,
                self.heads_num,
                mlp_width_ratio=mlp_width_ratio,
                mlp_act_type=mlp_act_type,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
                qkv_bias=qkv_bias
            )
            for _ in range(mm_double_blocks_depth)
        ])

        # single blocks
        self.single_blocks = nn.ModuleList([
            MMSingleStreamBlock(
                self.hidden_size,
                self.heads_num,
                mlp_width_ratio=mlp_width_ratio,
                mlp_act_type=mlp_act_type,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type
            )
            for _ in range(mm_single_blocks_depth)
        ])

        if self.enable_tensor_parallel:
            config.sequence_parallel = False
            self.adaLN_modulation = nn.Sequential(
                get_activation_layer("silu")(),
                tensor_parallel.ColumnParallelLinear(
                    self.hidden_size,
                    2 * self.hidden_size,
                    bias=True,
                    config=config,
                    init_method=config.init_method,
                    gather_output=False
                )
            )
            config.sequence_parallel = self.sequence_parallel
        else:
            self.adaLN_modulation = nn.Sequential(
            get_activation_layer("silu")(),
            nn.Linear(self.hidden_size, 2 * self.hidden_size, bias=True)
        )

        self.norm_final = nn.LayerNorm(self.hidden_size, elementwise_affine=False, eps=1e-6)
        for param in self.norm_final.parameters():
            setattr(param, "sequence_parallel", self.sequence_parallel)
        
        if self.enable_tensor_parallel:
            self.proj_out = tensor_parallel.ColumnParallelLinear(
                self.hidden_size,
                math.prod(patch_size) * out_channels,
                bias=True,
                config=config,
                init_method=config.init_method,
                gather_output=False
            )
        else:
            self.proj_out = nn.Linear(self.hidden_size, math.prod(patch_size) * out_channels, bias=True)

        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
        
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the module (assuming that all the module parameters have the same dtype)."""
        params = tuple(self.parameters())
        if len(params) > 0:
            return params[0].dtype
        else:
            buffers = tuple(self.buffers())
            return buffers[0].dtype

    def unpatchify(self, x, t, h, w) -> torch.Tensor:
        """
        input:
            x: (B, t*h*w, d)
        return:
            x: (B, c, T, H, W)
        """
        b = x.shape[0]
        c = self.unpatchify_channels
        patch_t, patch_h, patch_w = self.patch_size
        x = x.reshape(b, t, h, w, c, patch_t, patch_h, patch_w)
        x = torch.einsum("nthwcopq->nctohpwq", x)
        output = x.reshape(b, c, t * patch_t, h * patch_h, w * patch_w)
        return output
    
    def _get_block(
            self,
            dit_type: str,
            layer_number: int
    ):
        if dit_type == "double_stream":
            return self.double_blocks[layer_number]
        elif dit_type == "single_stream":
            return self.single_blocks[layer_number]
        else:
            raise NotImplementedError(f"dit type: {dit_type} is not implemented! ")
    
    def _checkpointed_forward(
            self,
            dit_type: str,
            img_and_txt: tuple,
            *args
    ):
        "Forward method with activation checkpointing."
        if dit_type == "double_stream":
            num_layers = len(self.double_blocks)
            recompute_layers = self.double_stream_recompute_layers
        elif dit_type == "single_stream":
            num_layers = len(self.single_blocks)
            recompute_layers = self.single_stream_recompute_layers
        else:
            raise NotImplementedError(f"dit type: {dit_type} is not implemented! ")
        
        def custom(start, end):
            def custom_forward(*args):
                for index in range(start, end):
                    layer = self._get_block(dit_type, index)
                    x_ = layer(*args)
                return x_
            return custom_forward
        
        if self.recompute_method == "uniform":
            # Uniformly divide the total number of Transformer layers and
            # checkpoint the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            _layer_num = 0
            while _layer_num < num_layers:
                img_and_txt = tensor_parallel.checkpoint(
                    custom(_layer_num, _layer_num + recompute_layers),
                    self.distribute_saved_activations,
                    *img_and_txt,
                    *args
                )
                _layer_num += recompute_layers

        elif self.recompute_method == "block":
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            for _layer_num in range(num_layers):
                if _layer_num < recompute_layers:
                    img_and_txt = tensor_parallel.checkpoint(
                        custom(_layer_num, _layer_num + 1),
                        self.distribute_saved_activations,
                        *img_and_txt,
                        *args
                    )
                else:
                    block = self._get_block(dit_type, _layer_num)
                    img_and_txt = block(*img_and_txt, *args)
        else:
            raise ValueError(f"Invalid activation recompute method {self.recompute_method}.")
        
        return img_and_txt
    
    def forward(
            self, 
            x: torch.Tensor, 
            timestep: torch.Tensor,
            prompt: List[torch.Tensor],
            prompt_mask: Union[torch.Tensor, List[torch.Tensor]] = None,
            guidance: torch.Tensor = None,
            **kwargs
    ):
        bs, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )
        x = x.to(self.dtype)

        if isinstance(prompt_mask, list):
            prompt_mask = prompt_mask[0]
        prompt_mask = prompt_mask.to(x.device)
        prompt[0] = prompt[0].view(-1, prompt[0].shape[-2], prompt[0].shape[-1]).to(self.dtype) # B*N, seq_len, Dim
        prompt[1] = prompt[1].to(self.dtype) # B, N, Dim
        prompt_mask = prompt_mask.view(-1, prompt_mask.shape[-1]) # B*N, seqlen

        # Prepare modulation vectors
        vec = self.time_in(timestep)

        # text modulation
        vec = vec + self.vector_in(prompt[1])

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                guidance = torch.tensor(
                    [self.embeded_guidance_scale] * x.shape[0],
                    dtype=torch.float32,
                ).to(vec.device).to(vec.dtype) * 1000.0
            vec = vec + self.guidance_in(guidance)

        img = self.img_in(x)
        
        if self.text_projection == "linear":
            txt = self.txt_in(prompt[0])
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(prompt[0], timestep, prompt_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # compute cu_squlens and max_seqlen for flash attention 
        cu_seqlens_q = _get_cu_seqlens(prompt_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        rope_sizes = list(x.shape)[-3:]
        rope_sizes = [rope_sizes[i] // self.patch_size[i] for i in range(3)]
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            self.rope_dim_list,
            rope_sizes,
            theta=self.rope_theta, 
            theta_rescale_factor=1
        )
        freqs_cos = freqs_cos.unsqueeze(0).unsqueeze(2).to(device=vec.device, dtype=vec.dtype) # b s n d
        freqs_sin = freqs_sin.unsqueeze(0).unsqueeze(2).to(device=vec.device, dtype=vec.dtype) # b s n d

        if bs == 1:
            txt = txt[:, :prompt_mask.sum()]
            txt_seq_len = txt.shape[1]

        # RNG context
        if self.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        # cp split
        if self.context_parallel_algo is not None:
            from mindspeed.core.context_parallel.unaligned_cp.mapping import (
                split_forward_gather_backward,
                gather_forward_split_backward,
            )
            img = split_forward_gather_backward(
                img, 
                mpu.get_context_parallel_group(),
                dim=1,
                grad_scale="down"
            ) # b s n d
            freqs_cos = split_forward_gather_backward(
                freqs_cos,
                mpu.get_context_parallel_group(),
                dim=1,
                grad_scale="down"
            )
            freqs_sin = split_forward_gather_backward(
                freqs_sin,
                mpu.get_context_parallel_group(),
                dim=1,
                grad_scale="down"
            )
            img_seq_len = img_seq_len // mpu.get_context_parallel_world_size()
        
        img_seq_len = torch.tensor(img_seq_len)
        max_seqlen_q = torch.tensor(max_seqlen_q)
        max_seqlen_kv = torch.tensor(max_seqlen_kv)

        if self.sequence_parallel:
            # b s h -> s b h
            img = img.transpose(0, 1).contiguous()
            # split img_seq for tp-sp
            img = tensor_parallel.scatter_to_sequence_parallel_region(img)

        # --------------------------- Pass through DiT blocks --------------------------
        with rng_context:
            if self.recompute_granularity == "full":
                # double_stream
                img, txt = self._checkpointed_forward(
                    "double_stream",
                    (img, txt),
                    vec,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    freqs_cos,
                    freqs_sin
                )

                if self.sequence_parallel:
                    txt = txt.transpose(0, 1).contiguous()
                    x = torch.cat([img, txt], dim=0)
                else:
                    x = torch.cat([img, txt], dim=1)

                # single_stream
                x = self._checkpointed_forward(
                    "single_stream",
                    (x, ),
                    vec,
                    img_seq_len,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    freqs_cos,
                    freqs_sin
                )[0]
            
            else:
                for _, block in enumerate(self.double_blocks):
                    img, txt = block(
                        img=img, 
                        txt=txt,
                        vec=vec,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_kv=max_seqlen_kv,
                        freqs_cos=freqs_cos,
                        freqs_sin=freqs_sin
                    )
                if self.sequence_parallel:
                    txt = txt.transpose(0, 1).contiguous()
                    x = torch.cat([img, txt], dim=0)
                else:
                    x = torch.cat([img, txt], dim=1)

                for _, block in enumerate(self.single_blocks):
                    x = block(
                        x=x,
                        vec=vec,
                        img_len=img_seq_len,
                        cu_seqlens_q=cu_seqlens_q,
                        cu_seqlens_kv=cu_seqlens_kv,
                        max_seqlen_q=max_seqlen_q,
                        max_seqlen_kv=max_seqlen_kv,
                        freqs_cos=freqs_cos,
                        freqs_sin=freqs_sin
                    )[0]
        
        if self.sequence_parallel:
            img = x[:img_seq_len // mpu.get_tensor_model_parallel_world_size()]
        else:
            img = x[:, :img_seq_len]

        # --------------------- Final layer ------------
        if self.enable_tensor_parallel:
            shift_scale = self.adaLN_modulation(vec)[0]
            if self.sequence_parallel:
                shift_scale = tensor_parallel.mappings.all_gather_last_dim_from_tensor_parallel_region(
                    shift_scale
                )
            else:
                shift_scale = tensor_parallel.mappings.gather_from_tensor_model_parallel_region(
                    shift_scale
                )
            shift, scale = shift_scale.chunk(2, dim=-1)
        else:
            shift, scale = self.adaLN_modulation(vec).chunk(2, dim=-1)

        x = self.norm_final(img) * (1 + scale) + shift

        if self.enable_tensor_parallel:
            x = self.proj_out(x)[0]
            if self.sequence_parallel:
                x = tensor_parallel.mappings.all_gather_last_dim_from_tensor_parallel_region(x)
            else:
                x = tensor_parallel.mappings.gather_from_tensor_model_parallel_region(x)
        else:
            x = self.proj_out(x)

        if self.sequence_parallel:
            x.transpose(0, 1).contiguous() # s b h -> b s h

        if self.context_parallel_algo is not None:
            x = gather_forward_split_backward(
                x, 
                mpu.get_context_parallel_group(),
                dim=1,
                grad_scale="up"
            )
        output = self.unpatchify(x, tt, th, tw)
        return output
    

class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal dit block with seperate modulation for text and image/video
    """
    def __init__(
            self, 
            hidden_size: int,
            heads_num: int,
            mlp_width_ratio: float, 
            mlp_act_type: str = "gelu_tanh",
            qk_norm: bool = True,
            qk_norm_type: str = "rmsnorm",
            qkv_bias: bool = False
    ):
        super().__init__()

        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        self.head_dim = head_dim
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        # context parallel setting
        args = get_args()
        config = core_transformer_config_from_args(args)
        self.context_parallel_algo = args.context_parallel_algo if mpu.get_context_parallel_world_size() > 1 else None
        if self.context_parallel_algo is not None and self.context_parallel_algo not in ["ulysses_cp_algo"]:
            raise NotImplementedError(f"Context_parallel_algo {self.context_parallel_algo} is not implemented")
        self.enable_tensor_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        self.sequence_parallel = args.sequence_parallel and self.enable_tensor_parallel
        self.num_attention_heads_per_partition = self.heads_num // mpu.get_tensor_model_parallel_world_size()

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            enable_tensor_parallel=self.enable_tensor_parallel,
            gather_tensor_parallel_output=True
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            enable_tensor_parallel=self.enable_tensor_parallel,
            gather_tensor_parallel_output=True
        )

        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        for param in self.img_norm1.parameters():
            setattr(param, "sequence_parallel", self.sequence_parallel)

        if self.enable_tensor_parallel:
            self.img_attn_qkv = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                hidden_size * 3,
                bias=qkv_bias,
                config=config,
                init_method=config.init_method,
                gather_output=False
            )
        else:
            self.img_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        self.img_attn_q_norm = (
            normalize(head_dim, affine=True, eps=1e-6, norm_type=qk_norm_type)
            if qk_norm
            else nn.Identity()
        )

        self.img_attn_k_norm = (
            normalize(head_dim, affine=True, eps=1e-6, norm_type=qk_norm_type)
            if qk_norm
            else nn.Identity()
        )

        if qk_norm:
            for param in self.img_attn_q_norm.parameters():
                setattr(param, "sequence_parallel", self.sequence_parallel)
            for param in self.img_attn_k_norm.parameters():
                setattr(param, "sequence_parallel", self.sequence_parallel)
        
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        for param in self.txt_norm1.parameters():
            setattr(param, "sequence_parallel", self.sequence_parallel)

        if self.enable_tensor_parallel:
            config.sequence_parallel = False # disable txt_seq sp
            self.txt_attn_qkv = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                hidden_size * 3,
                bias=qkv_bias,
                config=config,
                init_method=config.init_method,
                gather_output=False
            )
            config.sequence_parallel = self.sequence_parallel
        else:
            self.txt_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias)

        self.txt_attn_q_norm = (
            normalize(head_dim, affine=True, eps=1e-6, norm_type=qk_norm_type)
            if qk_norm
            else nn.Identity()
        )

        self.txt_attn_k_norm = (
            normalize(head_dim, affine=True, eps=1e-6, norm_type=qk_norm_type)
            if qk_norm
            else nn.Identity()
        )

        # disable txt_seq sp
        for param in self.txt_attn_q_norm.parameters():
            setattr(param, "sequence_parallel", False)
        for param in self.txt_attn_k_norm.parameters():
            setattr(param, "sequence_parallel", False)
        
        if self.enable_tensor_parallel:
            self.img_attn_proj = tensor_parallel.RowParallelLinear(
                hidden_size,
                hidden_size,
                bias=qkv_bias,
                config=config,
                init_method=config.init_method,
                input_is_parallel=True,
                skip_bias_add=False
            )
        else:
            self.img_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        for param in self.img_norm2.parameters():
            setattr(param, "sequence_parallel", True)

        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            enable_tensor_parallel=self.enable_tensor_parallel,
            enable_tp_sp=True
        )

        if self.enable_tensor_parallel:
            config.sequence_parallel = False # disable txt_seq sp
            self.txt_attn_proj = tensor_parallel.RowParallelLinear(
                hidden_size,
                hidden_size,
                bias=qkv_bias,
                config=config,
                init_method=config.init_method,
                input_is_parallel=True,
                skip_bias_add=False
            )
            config.sequence_parallel = self.sequence_parallel
        else:
            self.txt_attn_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)        
        
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        for param in self.txt_norm2.parameters():
            setattr(param, "sequence_parallel", False)

        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            enable_tensor_parallel=self.enable_tensor_parallel,
            enable_tp_sp=False
        )

    def forward(
            self, 
            img: torch.Tensor,
            txt: torch.Tensor,
            vec: torch.Tensor,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            max_seqlen_q: Optional[torch.Tensor] = None,
            max_seqlen_kv: Optional[torch.Tensor] = None,
            freqs_cos: Optional[torch.Tensor] = None,
            freqs_sin: Optional[torch.Tensor] = None
    ):
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate
        ) = self.img_mod(vec).chunk(6, dim=-1)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        # Prepare image for attention,
        img_modulated = self.img_norm1(img)
        img_modulated = img_modulated * (1 + img_mod1_scale) + img_mod1_shift

        if self.enable_tensor_parallel:
            img_qkv = self.img_attn_qkv(img_modulated)[0]
            if self.sequence_parallel:
                img_qkv = img_qkv.transpose(0, 1).contiguous() # s b h -> b s h
        else:
            img_qkv = self.img_attn_qkv(img_modulated)

        # b s h
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_attention_heads_per_partition)
        img_seq_len = img_q.shape[1]

        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q)
        img_k = self.img_attn_k_norm(img_k)

        # Apply RoPE if needed
        if freqs_cos is not None and freqs_sin is not None:
            img_q = npu_rotary_position_embedding(img_q, freqs_cos, freqs_sin, mode=1)
            img_k = npu_rotary_position_embedding(img_k, freqs_cos, freqs_sin, mode=1)

        # Prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = txt_modulated * (1 + txt_mod1_scale) + txt_mod1_shift

        if self.enable_tensor_parallel:
            txt_qkv = self.txt_attn_qkv(txt_modulated)[0]
        else:
            txt_qkv = self.txt_attn_qkv(txt_modulated)

        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        txt_q = self.txt_attn_q_norm(txt_q)
        txt_k = self.txt_attn_k_norm(txt_k)

        # Run actual attention
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            heads_num=self.num_attention_heads_per_partition,
            head_dim=self.head_dim,
            attn_mask=None,
            actual_seq_qlen=cu_seqlens_q,
            actual_seq_kvlen=cu_seqlens_kv,
            context_parallel_algo=self.context_parallel_algo,
        )

        img_attn, txt_attn = attn[:, :img_seq_len], attn[:, img_seq_len:]

        # Calculate the img blocks
        if self.enable_tensor_parallel:
            if self.sequence_parallel:
                img_attn = img_attn.transpose(0, 1).contiguous() # b s h -> s b h
            img = img + self.img_attn_proj(img_attn)[0] * img_mod1_gate
        else:
            img = img + self.img_attn_proj(img_attn) * img_mod1_gate

        img = img + \
            self.img_mlp(
                self.img_norm2(img) * (1 + img_mod2_scale) + img_mod2_shift
            ) * img_mod2_gate
        
        # Calculate the txt blocks

        if self.enable_tensor_parallel:
            txt = txt + self.txt_attn_proj(txt_attn)[0] * txt_mod1_gate
        else:
            txt = txt + self.txt_attn_proj(txt_attn) * txt_mod1_gate
        txt = txt + \
            self.txt_mlp(
                self.txt_norm2(txt) * (1 + txt_mod2_scale) + txt_mod2_shift
            ) * txt_mod2_gate
        
        return img, txt
    

class MMSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers.
    """
    def __init__(
            self,
            hidden_size: int,
            heads_num: int,
            mlp_width_ratio: int = 4,
            mlp_act_type: str = "gelu_tanh",
            qk_norm: bool = True,
            qk_norm_type: str = "rmsnorm",
            qk_scale: float = None
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        self.head_dim = head_dim
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        # context parallel setting
        args = get_args()
        config = core_transformer_config_from_args(args)
        self.context_parallel_algo = args.context_parallel_algo if mpu.get_context_parallel_world_size() > 1 else None
        if self.context_parallel_algo is not None and self.context_parallel_algo not in ["ulysses_cp_algo"]:
            raise NotImplementedError(f"Context_parallel_algo {self.context_parallel_algo} is not implemented")
        self.enable_tensor_parallel = mpu.get_tensor_model_parallel_world_size() > 1
        self.tp_size = mpu.get_tensor_model_parallel_world_size()
        self.sequence_parallel = args.sequence_parallel and self.enable_tensor_parallel
        self.num_attention_heads_per_partition = self.heads_num // mpu.get_tensor_model_parallel_world_size()

        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            enable_tensor_parallel=self.enable_tensor_parallel,
            gather_tensor_parallel_output=True
        )

        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        if self.enable_tensor_parallel:
            self.linear1 = tensor_parallel.ColumnParallelLinear(
                hidden_size,
                hidden_size * 3 + mlp_hidden_dim,
                config=config,
                init_method=config.init_method,
                gather_output=False,
                bias=True
            )
        else:
            self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim)
        
        self.q_norm = (
            normalize(head_dim, affine=True, eps=1e-6, norm_type=qk_norm_type)
            if qk_norm
            else nn.Identity()
        )

        self.k_norm = (
            normalize(head_dim, affine=True, eps=1e-6, norm_type=qk_norm_type)
            if qk_norm
            else nn.Identity()
        )

        if self.enable_tensor_parallel:
            self.linear2 = tensor_parallel.RowParallelLinear(
                hidden_size + mlp_hidden_dim,
                hidden_size,
                config=config,
                init_method=config.init_method,
                input_is_parallel=True,
                skip_bias_add=False,
                bias=True
            )
        else:
            self.linear2 = nn.Linear(hidden_size + mlp_hidden_dim, hidden_size)

        for param in self.q_norm.parameters():
            setattr(param, "sequence_parallel", self.sequence_parallel)
        for param in self.k_norm.parameters():
            setattr(param, "sequence_parallel", self.sequence_parallel)
        for param in self.pre_norm.parameters():
            setattr(param, "sequence_parallel", self.sequence_parallel)

        self.mlp_act = get_activation_layer(mlp_act_type)()

    def forward(
            self,
            x: torch.Tensor,
            vec: torch.Tensor,
            img_len: int,
            cu_seqlens_q: Optional[torch.Tensor] = None,
            cu_seqlens_kv: Optional[torch.Tensor] = None,
            max_seqlen_q: Optional[torch.Tensor] = None,
            max_seqlen_kv: Optional[torch.Tensor] = None,
            freqs_cos: Optional[torch.Tensor] = None,
            freqs_sin: Optional[torch.Tensor] = None
    ):
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = self.pre_norm(x) * (1 + mod_scale) + mod_shift

        if self.enable_tensor_parallel:
            if self.sequence_parallel:
                img_qkv, img_mlp = torch.split(
                    self.linear1(x_mod[:img_len // self.tp_size])[0],
                    [3 * self.hidden_size // self.tp_size, self.mlp_hidden_dim // self.tp_size],
                    dim=-1
                )
                # disable txt_seq sp
                self.linear1.sequence_parallel = False
                txt_qkv, txt_mlp = torch.split(
                    self.linear1(x_mod[img_len // self.tp_size:])[0],
                    [3 * self.hidden_size // self.tp_size, self.mlp_hidden_dim // self.tp_size],
                    dim=-1
                )
                self.linear1.sequence_parallel = self.sequence_parallel
                qkv = torch.cat([img_qkv, txt_qkv], dim=0)
                qkv = qkv.transpose(0, 1) # s b h -> b s h
            else:
                qkv, mlp = torch.split(
                    self.linear1(x_mod)[0],
                    [3 * self.hidden_size // self.tp_size, self.mlp_hidden_dim // self.tp_size],
                    dim=-1
                )
        else:
            qkv, mlp = torch.split(
                self.linear1(x_mod), 
                [3 * self.hidden_size, self.mlp_hidden_dim], 
                dim=-1
            )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_attention_heads_per_partition)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        # Apply QK-Norm if needed
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE if needed
        if freqs_cos is not None and freqs_sin is not None:
            img_q, txt_q = q[:, :img_len, :, :], q[:, img_len:, :, :] # b s n d
            img_k, txt_k = k[:, :img_len, :, :], k[:, img_len:, :, :]

            img_q = npu_rotary_position_embedding(img_q, freqs_cos, freqs_sin, mode=1)
            img_k = npu_rotary_position_embedding(img_k, freqs_cos, freqs_sin, mode=1)
        
        img_v = v[:, :img_len, :, :]
        txt_v = v[:, img_len:, :, :]
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            heads_num=self.num_attention_heads_per_partition,
            head_dim=self.head_dim,
            attn_mask=None,
            actual_seq_qlen=cu_seqlens_q,
            actual_seq_kvlen=cu_seqlens_kv,
            context_parallel_algo=self.context_parallel_algo
        )

        if self.enable_tensor_parallel:
            if self.sequence_parallel:
                attn = attn.transpose(0, 1).contiguous() # b s h -> s b h
                img_attn = attn[:img_len]
                txt_attn = attn[img_len:]
                img_output = self.linear2(torch.cat((img_attn, self.mlp_act(img_mlp)), 2))[0]

                # disable txt_seq sequence_parallel
                self.linear2.sequence_parallel = False
                txt_output = self.linear2(torch.cat((txt_attn, self.mlp_act(txt_mlp)), 2))[0]
                self.linear2.sequence_parallel = True

                output = torch.cat([img_output, txt_output], 0)
            else:
                output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))[0]
        else:
            output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return (x + output * mod_gate, )
    

def parallel_attention(
    q: Tuple[torch.Tensor, torch.Tensor],
    k: Tuple[torch.Tensor, torch.Tensor],
    v: Tuple[torch.Tensor, torch.Tensor],
    heads_num: int, 
    head_dim: int,
    attn_mask: Optional[torch.Tensor] = None,
    actual_seq_qlen: Optional[torch.Tensor] = None,
    actual_seq_kvlen: Optional[torch.Tensor] = None,
    context_parallel_algo: str = None,
):
    img_q, txt_q = q
    img_k, txt_k = k
    img_v, txt_v = v
    bs, img_seq_len, _, _ = img_q.shape
    # input: b s n d, output: b s h

    if context_parallel_algo == "ulysses_cp_algo":
        from mindspeed.core.context_parallel.unaligned_cp.mapping import (
            split_forward_gather_backward,
            gather_forward_split_backward,
            all_to_all
        )
        img_q = all_to_all(img_q, mpu.get_context_parallel_group(), scatter_dim=2, gather_dim=1)
        img_k = all_to_all(img_k, mpu.get_context_parallel_group(), scatter_dim=2, gather_dim=1)
        img_v = all_to_all(img_v, mpu.get_context_parallel_group(), scatter_dim=2, gather_dim=1)
        
        def shrink_head(txt, dim=2):
            txt = split_forward_gather_backward(txt, mpu.get_context_parallel_group(), dim=dim)
            return txt
        
        if bs == 1:
            attn = torch_npu.npu_fusion_attention(
                torch.cat([img_q, shrink_head(txt_q, 2)], dim=1),
                torch.cat([img_k, shrink_head(txt_k, 2)], dim=1),
                torch.cat([img_v, shrink_head(txt_v, 2)], dim=1),
                head_num=heads_num // mpu.get_context_parallel_world_size(),
                atten_mask=attn_mask,
                input_layout="BSND",
                scale=1 / math.sqrt(head_dim)
            )[0]
            img_attn = attn[:, :img_seq_len * mpu.get_context_parallel_world_size()] # b img_seq sub_n d
            txt_attn = attn[:, img_seq_len * mpu.get_context_parallel_world_size():] # b txt_seq sub_n d
        else:
            raise NotImplementedError("not implemented ulysses_cp_algo for batch_size > 1")
        
        img_attn = all_to_all(img_attn, mpu.get_context_parallel_group(), scatter_dim=1, gather_dim=2) # b sub_img_seq n d
        txt_attn = gather_forward_split_backward(txt_attn, mpu.get_context_parallel_group(), dim=2) # b txt_seq n d
        attn = torch.cat([img_attn, txt_attn], dim=1)
        attn = attn.view(bs, -1, heads_num * head_dim)
    else:
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        if bs == 1:
            attn = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                head_num=heads_num,
                atten_mask=attn_mask,
                input_layout="BSND",
                scale=1 / math.sqrt(head_dim),
            )[0]
            attn = attn.view(bs, -1, heads_num * head_dim)
        else:
            q = q.view(-1, heads_num * head_dim)
            k = k.view(-1, heads_num * head_dim)
            v = v.view(-1, heads_num * head_dim)
            attn = torch_npu.npu_fusion_attention(
                q,
                k,
                v,
                head_num=heads_num,
                atten_mask=attn_mask,
                input_layout="TND",
                scale=1 / math.sqrt(head_dim),
                actual_seq_qlen=actual_seq_qlen.tolist(),
                actual_seq_kvlen=actual_seq_kvlen.tolist()
            )[0]
            attn = attn.view(bs, -1, heads_num * head_dim)

    return attn