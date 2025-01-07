import math
from math import pi
import functools
from typing import Optional, List
from beartype import beartype
from beartype.typing import Literal, Union, Optional
from einops import rearrange, repeat
import numpy as np
import torch
from torch import nn, einsum, broadcast_tensors, Tensor
from torch.cuda.amp import autocast

from mindspeed.ops.npu_rotary_position_embedding import npu_rotary_position_embedding


def get_3d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    interpolation_scale=(1.0, 1.0, 1.0),
    base_size=None,
) -> np.array:
    """
    embed_dim: output dimension for each position
    grid_size: int of the grid height and width
    return:
        pos_embed: [grid_size*grid_size, embed_dim] or
        [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """

    grid_t = np.arange(grid_size[0], dtype=np.float32) / interpolation_scale[0]
    grid_h = np.arange(grid_size[1], dtype=np.float32) / interpolation_scale[1]
    grid_w = np.arange(grid_size[2], dtype=np.float32) / interpolation_scale[2]
    if base_size is not None:
        grid_t *= base_size[0] / grid_size[0]
        grid_h *= base_size[1] / grid_size[1]
        grid_w *= base_size[2] / grid_size[2]
    grid = np.meshgrid(grid_w, grid_h, grid_t)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size[2], grid_size[1], grid_size[0]])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid) -> np.array:
    """
    embed_dim: output dimension for each position
    grid: list of grid size
    """
    if embed_dim % 3 != 0:
        raise ValueError("embed_dim must be divisible by 3")

    # use 1/3 of dimensions to encode grid_t/h/w
    emb_t = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (T*H*W, D/3)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (T*H*W, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (T*H*W, D/3)

    emb = np.concatenate([emb_t, emb_h, emb_w], axis=1)  # (T*H*W, D)
    return emb


def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    interpolation_scale=(1.0, 1.0),
    base_size=None,
) -> np.array:
    """
    embed_dim: output dimension for each position
    grid_size: int of the grid height and width
    return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32) / interpolation_scale[0]
    grid_w = np.arange(grid_size[1], dtype=np.float32) / interpolation_scale[1]
    if base_size is not None:
        grid_h *= base_size[0] / grid_size[0]
        grid_w *= base_size[1] / grid_size[1]

    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid) -> np.array:
    """
    embed_dim: output dimension for each position
    grid: list of grid size
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos) -> np.array:
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    interpolation_scale=1.0,
    base_size=None,
) -> np.array:
    """
    embed_dim: output dimension for each position
    grid_size: int of the grid
    return:
        pos_embed: [grid_size, embed_dim] or
        [1+grid_size, embed_dim] (w/ or w/o cls_token)
    """

    grid = np.arange(grid_size, dtype=np.float32) / interpolation_scale
    if base_size is not None:
        grid *= base_size / grid_size

    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid)  # (H*W, D/2)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_meshgrid_nd(rope_sizes, dim=2, dtype=torch.float32):
    """
    Get n-D meshgrid
    """
    axis_grid = [torch.linspace(0, rope_sizes[i], rope_sizes[i] + 1, dtype=dtype)[:rope_sizes[i]] for i in range(dim)]
    grid = torch.meshgrid(*axis_grid, indexing="ij") # dim x [W, H, D]
    grid = torch.stack(grid, dim=0) # [dim, W, H, D]

    return grid


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[torch.FloatTensor, int],
    theta: float = 10000.0,
    theta_rescale_factor: float = 1.0,
    interpolation_factor: float = 1.0,
):
    """
    Precompute the frequency tensor for complex exponential (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponential using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.

    Args:
        dim (int): Dimension of the frequency tensor.
        pos (int or torch.FloatTensor): Position indices for the frequency tensor. [S] or scalar
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.
        theta_rescale_factor (float, optional): Rescale factor for theta. Defaults to 1.0.

    Returns:
        freqs_cos, freqs_sin: Precomputed frequency tensor with real and imaginary parts separately. [S, D]
    """
    if isinstance(pos, int):
        pos = torch.arange(pos).float()

    if not math.isclose(theta_rescale_factor, 1.0, rel_tol=1e-9):
        theta *= theta_rescale_factor ** (dim / (dim - 2))

    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )  # [D/2]

    freqs = torch.outer(pos * interpolation_factor, freqs)  # [S, D/2]

    freqs_cos = freqs.cos().repeat_interleave(2, dim=1)  # [S, D]
    freqs_sin = freqs.sin().repeat_interleave(2, dim=1)  # [S, D]
    return freqs_cos, freqs_sin


def get_nd_rotary_pos_embed(
    rope_dim_list,
    rope_sizes,
    theta=10000.0,
    theta_rescale_factor: Union[float, List[float]] = 1.0,
    interpolation_factor: Union[float, List[float]] = 1.0,
):
    """
    This is a n-d version of precompute_freqs_cis, which is a RoPE for tokens with n-d structure.

    Args:
        rope_dim_list (list of int): Dimension of each rope. len(rope_dim_list) should equal to n.
            sum(rope_dim_list) should equal to head_dim of attention layer.
        rope_sizes (int | tuple of int | list of int): rotary embed sizes for each dim
        theta (float): Scaling factor for frequency computation. Defaults to 10000.0.
        theta_rescale_factor (float): Rescale factor for theta. Defaults to 1.0.

    Returns:
        pos_embed (torch.Tensor): [HW, D/2]
    """

    grid = get_meshgrid_nd(
        rope_sizes, dim=len(rope_dim_list)
    )  # [3, W, H, D] / [2, W, H]

    if isinstance(theta_rescale_factor, int) or isinstance(theta_rescale_factor, float):
        theta_rescale_factor = [theta_rescale_factor] * len(rope_dim_list)
    elif isinstance(theta_rescale_factor, list) and len(theta_rescale_factor) == 1:
        theta_rescale_factor = [theta_rescale_factor[0]] * len(rope_dim_list)

    if len(theta_rescale_factor) != len(rope_dim_list):
        raise ValueError(f"len(theta_rescale_factor): {len(theta_rescale_factor)} should equal to len(rope_dim_list): {len(rope_dim_list)}")

    if isinstance(interpolation_factor, int) or isinstance(interpolation_factor, float):
        interpolation_factor = [interpolation_factor] * len(rope_dim_list)
    elif isinstance(interpolation_factor, list) and len(interpolation_factor) == 1:
        interpolation_factor = [interpolation_factor[0]] * len(rope_dim_list)

    if len(interpolation_factor) != len(rope_dim_list):
        raise ValueError(f"len(interpolation_factor): {len(interpolation_factor)} should equal to len(rope_dim_list): {len(rope_dim_list)}")

    # use 1/ndim of dimensions to encode grid_axis
    embs = []
    for i, rope_dim in enumerate(rope_dim_list):
        emb = get_1d_rotary_pos_embed(
            rope_dim,
            grid[i].reshape(-1),
            theta,
            theta_rescale_factor=theta_rescale_factor[i],
            interpolation_factor=interpolation_factor[i],
        )  # 2 x [WHD, rope_dim_list[i]]
        embs.append(emb)

    cos = torch.cat([emb[0] for emb in embs], dim=1)  # (WHD, D/2)
    sin = torch.cat([emb[1] for emb in embs], dim=1)  # (WHD, D/2)
    return cos, sin


class PositionEmbedding2D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        if dim % 4 != 0:
            raise Exception("dim must be divisible by 4")
        
        half_dim = dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _get_sin_cos_emb(self, t: torch.Tensor):
        out = torch.einsum("i,d->id", t, self.inv_freq)
        emb_cos = torch.cos(out)
        emb_sin = torch.sin(out)
        return torch.cat((emb_sin, emb_cos), dim=-1)

    @functools.lru_cache(maxsize=512)
    def _get_cached_emb(
        self,
        device: torch.device,
        dtype: torch.dtype,
        h: int,
        w: int,
        scale=1.0,
        base_size=None,
    ):
        grid_h = torch.arange(h, device=device) / scale
        grid_w = torch.arange(w, device=device) / scale
        if base_size is not None:
            grid_h *= base_size / h
            grid_w *= base_size / w
        grid_h, grid_w = torch.meshgrid(
            grid_w,
            grid_h,
            indexing="ij",
        )  # here w goes first
        grid_h = grid_h.t().reshape(-1)
        grid_w = grid_w.t().reshape(-1)
        emb_h = self._get_sin_cos_emb(grid_h)
        emb_w = self._get_sin_cos_emb(grid_w)
        return torch.concat([emb_h, emb_w], dim=-1).unsqueeze(0).to(dtype)

    def forward(
        self,
        x: torch.Tensor,
        h: int,
        w: int,
        scale: Optional[float] = 1.0,
        base_size: Optional[int] = None,
    ) -> torch.Tensor:
        return self._get_cached_emb(x.device, x.dtype, h, w, scale, base_size)
    
    
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2).contiguous()
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, '... d r -> ... (d r)')


@autocast(enabled=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1., seq_dim=-2):
    dtype = t.dtype
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:]

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    if rot_dim > t.shape[-1]:
        raise Exception(f"feature dimension {t.shape[-1]} is not \
                        of sufficient size to rotate in all the positions {rot_dim}")

    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    out = torch.cat((t_left, t, t_right), dim=-1)

    return out.type(dtype)


class NpuRotaryEmbedding(nn.Module):
    @beartype
    def __init__(
        self,
        dim,
        custom_freqs: Optional[Tensor] = None,
        freqs_for: Union[
            Literal['lang'],
            Literal['pixel'],
            Literal['constant']
        ] = 'lang',
        theta=10000,
        max_freq=10,
        num_freqs=1,
        learned_freq=False,
        use_xpos=False,
        xpos_scale_base=512,
        interpolate_factor=1.,
        theta_rescale_factor=1.,
        seq_before_head_dim=False,
        cache_if_possible=True
    ):
        super().__init__()
        # proposed by reddit user bloc97, to rescale rotary embeddings to longer sequence length without fine-tuning
        # has some connection to NTK literature

        theta *= theta_rescale_factor ** (dim / (dim - 2))

        self.freqs_for = freqs_for

        if exists(custom_freqs):
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()

        self.cache_if_possible = cache_if_possible

        self.tmp_store('cached_freqs', None)
        self.tmp_store('cached_scales', None)

        self.freqs = nn.Parameter(freqs, requires_grad=learned_freq)

        self.learned_freq = learned_freq

        # dummy for device
        self.tmp_store('dummy', torch.tensor(0))

        # default sequence dimension
        self.seq_before_head_dim = seq_before_head_dim
        self.default_seq_dim = -3 if seq_before_head_dim else -2

        # interpolation factors
        if interpolate_factor < 1.:
            raise Exception("interpolate_factor must less than 1.")
        self.interpolate_factor = interpolate_factor

        # xpos
        self.use_xpos = use_xpos
        if not use_xpos:
            self.tmp_store('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = xpos_scale_base
        self.tmp_store('scale', scale)

        # add apply_rotary_emb as static method
        self.apply_rotary_emb = staticmethod(apply_rotary_emb)

    @property
    def device(self):
        return self.dummy.device

    def tmp_store(self, key, value):
        self.register_buffer(key, value, persistent=False)

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return (torch.arange(seq_len, device=device, dtype=dtype) + offset) / self.interpolate_factor

    def rotate_queries_or_keys(self, t, seq_dim=None, offset=0):
        seq_dim = default(seq_dim, self.default_seq_dim)

        if self.use_xpos:
            raise Exception("you must use `.rotate_queries_and_keys` method \
                             instead and pass in both queries and keys, \
                             for length extrapolatable rotary embeddings")

        device, dtype, seq_len = t.device, t.dtype, t.shape[seq_dim]

        freqs = self.forward(self.get_seq_pos(seq_len, device=device, dtype=dtype, offset=offset), seq_len=seq_len, offset=offset)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')

        return apply_rotary_emb(freqs, t, seq_dim=seq_dim)

    def rotate_queries_with_cached_keys(self, q, k, seq_dim=None, offset=0):
        seq_dim = default(seq_dim, self.default_seq_dim)

        q_len, k_len = q.shape[seq_dim], k.shape[seq_dim]
        if q_len > k_len:
            raise Exception("q_len must ")

        rotated_q = self.rotate_queries_or_keys(q, seq_dim=seq_dim, offset=k_len - q_len + offset)
        rotated_k = self.rotate_queries_or_keys(k, seq_dim=seq_dim, offset=offset)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    def rotate_queries_and_keys(self, q, k, seq_dim=None):
        seq_dim = default(seq_dim, self.default_seq_dim)
        
        if not self.use_xpos:
            raise Exception("use_xpos must be true when we use rotate_queries_and_keys")

        device, dtype, seq_len = q.device, q.dtype, q.shape[seq_dim]

        seq = self.get_seq_pos(seq_len, dtype=dtype, device=device)

        freqs = self.forward(seq, seq_len=seq_len)
        scale = self.get_scale(seq, seq_len=seq_len).to(dtype)

        if seq_dim == -3:
            freqs = rearrange(freqs, 'n d -> n 1 d')
            scale = rearrange(scale, 'n d -> n 1 d')

        rotated_q = apply_rotary_emb(freqs, q, scale=scale, seq_dim=seq_dim)
        rotated_k = apply_rotary_emb(freqs, k, scale=scale ** -1, seq_dim=seq_dim)

        rotated_q = rotated_q.type(q.dtype)
        rotated_k = rotated_k.type(k.dtype)

        return rotated_q, rotated_k

    @beartype
    def get_scale(
        self,
        t: Tensor,
        seq_len: Optional[int] = None,
        offset=0
    ):
        if not self.use_xpos:
            raise Exception("use_xpos must be true when we use get_scale method")

        should_cache = (
            self.cache_if_possible and
            exists(seq_len)
        )

        if (
            should_cache and \
            exists(self.cached_scales) and \
            (seq_len + offset) <= self.cached_scales.shape[0]
        ):
            return self.cached_scales[offset:(offset + seq_len)]

        scale = 1.
        if self.use_xpos:
            power = (t - len(t) // 2) / self.scale_base
            scale = self.scale ** rearrange(power, 'n -> n 1')
            scale = torch.cat((scale, scale), dim=-1)

        if should_cache:
            self.tmp_store('cached_scales', scale)

        return scale

    def get_axial_freqs(self, *dims):
        Colon = slice(None)
        all_freqs = []

        for ind, dim in enumerate(dims):
            if self.freqs_for == 'pixel':
                pos = torch.linspace(-1, 1, steps=dim, device=self.device)
            else:
                pos = torch.arange(dim, device=self.device)

            freqs = self.forward(pos, seq_len=dim)

            all_axis = [None] * len(dims)
            all_axis[ind] = Colon

            new_axis_slice = (Ellipsis, *all_axis, Colon)
            all_freqs.append(freqs[new_axis_slice])

        all_freqs = broadcast_tensors(*all_freqs)
        return torch.cat(all_freqs, dim=-1)

    @autocast(enabled=False)
    def forward(
        self,
        t: Tensor,
        seq_len=None,
        offset=0
    ):
        should_cache = (
            self.cache_if_possible and \
            not self.learned_freq and \
            exists(seq_len) and \
            self.freqs_for != 'pixel'
        )

        if (
            should_cache and \
            exists(self.cached_freqs) and \
            (offset + seq_len) <= self.cached_freqs.shape[0]
        ):
            return self.cached_freqs[offset:(offset + seq_len)].detach()

        freqs = self.freqs

        freqs = einsum('..., f -> ... f', t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, '... n -> ... (n r)', r=2)

        if should_cache:
            self.tmp_store('cached_freqs', freqs.detach())

        return freqs


def broad_cat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


class Rotary3DPositionEmbedding(nn.Module):
    def __init__(
        self,
        height,
        width,
        compressed_num_frames,
        hidden_size,
        hidden_size_head,
        text_length,
        theta=10000,
        rot_v=False,
        learnable_pos_embed=False,
    ):
        super().__init__()
        self.rot_v = rot_v

        dim_t = hidden_size_head // 4
        dim_h = hidden_size_head // 8 * 3
        dim_w = hidden_size_head // 8 * 3

        freqs_t = 1.0 / (theta ** (torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t))
        freqs_h = 1.0 / (theta ** (torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h))
        freqs_w = 1.0 / (theta ** (torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w))

        grid_t = torch.arange(compressed_num_frames, dtype=torch.float32)
        grid_h = torch.arange(height, dtype=torch.float32)
        grid_w = torch.arange(width, dtype=torch.float32)

        freqs_t = torch.einsum("..., f -> ... f", grid_t, freqs_t)
        freqs_h = torch.einsum("..., f -> ... f", grid_h, freqs_h)
        freqs_w = torch.einsum("..., f -> ... f", grid_w, freqs_w)

        freqs_t = repeat(freqs_t, "... n -> ... (n r)", r=2)
        freqs_h = repeat(freqs_h, "... n -> ... (n r)", r=2)
        freqs_w = repeat(freqs_w, "... n -> ... (n r)", r=2)

        freqs = broad_cat((freqs_t[:, None, None, :], freqs_h[None, :, None, :], freqs_w[None, None, :, :]), dim=-1)

        freqs = freqs.contiguous()
        self.freqs_sin = freqs.sin().npu()
        self.freqs_cos = freqs.cos().npu()

        self.text_length = text_length
        if learnable_pos_embed:
            num_patches = int(height * width * compressed_num_frames + text_length)
            self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, int(hidden_size)), requires_grad=True)
        else:
            self.pos_embedding = None

    def rotary(self, t, **kwargs):
        # input shape: bnsd
        def reshape_freq(freqs):
            freqs = freqs[: kwargs["rope_T"], : kwargs["rope_H"], : kwargs["rope_W"]].contiguous()
            freqs = rearrange(freqs, "t h w d -> (t h w) d")
            freqs = freqs.unsqueeze(0).unsqueeze(0)
            return freqs

        freqs_cos = reshape_freq(self.freqs_cos).to(t.dtype)
        freqs_sin = reshape_freq(self.freqs_sin).to(t.dtype)

        return npu_rotary_position_embedding(t, freqs_cos, freqs_sin, mode=1)

    def position_embedding_forward(self, position_ids, **kwargs):
        if self.pos_embedding is not None:
            return self.pos_embedding[:, :self.text_length + kwargs.get("seq_length", 0)]
        else:
            return None

    def forward(self, x, **kwargs):
        # input shape: bnsd
        x[:, :, self.text_length:] = self.rotary(x[:, :, self.text_length:], **kwargs)
        return x
