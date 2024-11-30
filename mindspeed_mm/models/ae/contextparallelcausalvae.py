from typing import Tuple

import os
import torch
from torch import nn
from einops import rearrange
import numpy as np

from megatron.core import mpu
from megatron.training import print_rank_0
from mindspeed_mm.models.common.activations import Sigmoid
from mindspeed_mm.models.common.module import MultiModalModule
from mindspeed_mm.models.common.conv import Conv2d, CausalConv3d, ContextParallelCausalConv3d
from mindspeed_mm.models.common.normalize import normalize, Normalize3D
from mindspeed_mm.models.common.attention import CausalConv3dAttnBlock
from mindspeed_mm.models.common.resnet_block import ResnetBlock2D, ResnetBlock3D, ContextParallelResnetBlock3D
from mindspeed_mm.models.common.updownsample import (SpatialDownsample2x, TimeDownsample2x, SpatialUpsample2x,
                                                     TimeUpsample2x,
                                                     TimeUpsampleRes2x, Upsample3D, Downsample, DownSample3D,
                                                     Spatial2xTime2x3DDownsample, Spatial2xTime2x3DUpsample)
from mindspeed_mm.models.common.regularizer import DiagonalGaussianDistribution
from mindspeed_mm.models.common.communications import _conv_split, _conv_gather
from mindspeed_mm.utils.utils import (
    is_context_parallel_initialized,
    initialize_context_parallel,
    get_context_parallel_group,
    get_context_parallel_group_rank,
    get_context_parallel_world_size,
    get_context_parallel_rank
)

CASUALVAE_MODULE_MAPPINGS = {
    "Conv2d": Conv2d,
    "ResnetBlock2D": ResnetBlock2D,
    "CausalConv3d": CausalConv3d,
    "AttnBlock3D": CausalConv3dAttnBlock,
    "ResnetBlock3D": ResnetBlock3D,
    "Downsample": Downsample,
    "SpatialDownsample2x": SpatialDownsample2x,
    "TimeDownsample2x": TimeDownsample2x,
    "SpatialUpsample2x": SpatialUpsample2x,
    "TimeUpsample2x": TimeUpsample2x,
    "TimeUpsampleRes2x": TimeUpsampleRes2x,
    "Spatial2xTime2x3DDownsample": Spatial2xTime2x3DDownsample,
    "Spatial2xTime2x3DUpsample": Spatial2xTime2x3DUpsample,
    "SiLU": nn.SiLU,
    "swish": Sigmoid,
    "ContextParallelResnetBlock3D": ContextParallelResnetBlock3D,
    "ContextParallelCausalConv3d": ContextParallelCausalConv3d,
    "DownSample3D": DownSample3D,
    "Upsample3D": Upsample3D
}


def model_name_to_cls(model_name):
    if model_name in CASUALVAE_MODULE_MAPPINGS:
        return CASUALVAE_MODULE_MAPPINGS[model_name]
    else:
        raise ValueError(f"Model name {model_name} not supported")


class ContextParallelCasualVAE(MultiModalModule):
    def __init__(
        self,
        from_pretrained: str = None,
        cp_size: str = 0,
        hidden_size: int = 128,
        z_channels: int = 4,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = (),
        dropout: float = 0.0,
        resolution: int = 256,
        double_z: bool = True,
        embed_dim: int = 4,
        num_res_blocks: int = 2,
        q_conv: str = "CausalConv3d",
        conv_padding: int = 1,
        encoder_conv_in: str = "CausalConv3d",
        encoder_conv_out: str = "CausalConv3d",
        encoder_attention: str = "AttnBlock3D",
        encoder_nonlinearity: str = "SiLU",
        encoder_resnet_blocks: Tuple[str] = (
                "ResnetBlock3D",
                "ResnetBlock3D",
                "ResnetBlock3D",
                "ResnetBlock3D",
        ),
        encoder_spatial_downsample: Tuple[str] = (
                "SpatialDownsample2x",
                "SpatialDownsample2x",
                "SpatialDownsample2x",
                "",
        ),
        encoder_temporal_downsample: Tuple[str] = (
                "",
                "TimeDownsample2x",
                "TimeDownsample2x",
                "",
        ),
        encoder_mid_resnet: str = "ResnetBlock3D",
        decoder_conv_in: str = "CausalConv3d",
        decoder_conv_out: str = "CausalConv3d",
        decoder_attention: str = "AttnBlock3D",
        decoder_nonlinearity: str = "SiLU",
        decoder_resnet_blocks: Tuple[str] = (
                "ResnetBlock3D",
                "ResnetBlock3D",
                "ResnetBlock3D",
                "ResnetBlock3D",
        ),
        decoder_spatial_upsample: Tuple[str] = (
                "",
                "SpatialUpsample2x",
                "SpatialUpsample2x",
                "SpatialUpsample2x",
        ),
        decoder_temporal_upsample: Tuple[str] = ("", "", "TimeUpsample2x", "TimeUpsample2x"),
        decoder_mid_resnet: str = "ResnetBlock3D",
        tile_sample_min_size: int = 256,
        tile_sample_min_size_t: int = 33,
        tile_latent_min_size_t: int = 16,
        tile_overlap_factor: int = 0.125,
        vae_scale_factor: list = None,
        use_tiling: bool = False,
        use_quant_layer: bool = True,
        encoder_gather_norm: bool = False,
        decoder_gather_norm: bool = False,
        **kwargs
    ) -> None:
        super().__init__(config=None)
        self.cp_size = cp_size
        self.tile_sample_min_size = tile_sample_min_size
        self.tile_sample_min_size_t = tile_sample_min_size_t
        self.tile_latent_min_size = int(self.tile_sample_min_size / (2 ** (len(hidden_size_mult) - 1)))

        self.tile_latent_min_size_t = tile_latent_min_size_t
        self.tile_overlap_factor = tile_overlap_factor
        self.vae_scale_factor = vae_scale_factor
        self.use_tiling = use_tiling
        self.use_quant_layer = use_quant_layer

        self.encoder = Encoder(
            z_channels=z_channels,
            hidden_size=hidden_size,
            hidden_size_mult=hidden_size_mult,
            attn_resolutions=attn_resolutions,
            conv_in=encoder_conv_in,
            conv_out=encoder_conv_out,
            conv_padding=conv_padding,
            nonlinearity=encoder_nonlinearity,
            attention=encoder_attention,
            resnet_blocks=encoder_resnet_blocks,
            spatial_downsample=encoder_spatial_downsample,
            temporal_downsample=encoder_temporal_downsample,
            mid_resnet=encoder_mid_resnet,
            dropout=dropout,
            resolution=resolution,
            num_res_blocks=num_res_blocks,
            double_z=double_z,
            gather_norm=encoder_gather_norm
        )

        self.decoder = Decoder(
            z_channels=z_channels,
            hidden_size=hidden_size,
            hidden_size_mult=hidden_size_mult,
            attn_resolutions=attn_resolutions,
            conv_in=decoder_conv_in,
            conv_out=decoder_conv_out,
            conv_padding=conv_padding,
            nonlinearity=decoder_nonlinearity,
            attention=decoder_attention,
            resnet_blocks=decoder_resnet_blocks,
            spatial_upsample=decoder_spatial_upsample,
            temporal_upsample=decoder_temporal_upsample,
            mid_resnet=decoder_mid_resnet,
            dropout=dropout,
            resolution=resolution,
            num_res_blocks=num_res_blocks,
            gather_norm=decoder_gather_norm
        )
        if self.use_quant_layer:
            quant_conv_cls = model_name_to_cls(q_conv)
            self.quant_conv = quant_conv_cls(2 * z_channels, 2 * embed_dim, 1)
            self.post_quant_conv = quant_conv_cls(embed_dim, z_channels, 1)

        if from_pretrained is not None:
            self.load_checkpoint(from_pretrained)

        self.dp_group_nums = torch.distributed.get_world_size() // mpu.get_data_parallel_world_size()

        if self.cp_size > 0:
            if not is_context_parallel_initialized():
                initialize_context_parallel(self.cp_size)

    def get_encoder(self):
        if self.use_quant_layer:
            return [self.quant_conv, self.encoder]
        return [self.encoder]

    def get_decoder(self):
        if self.use_quant_layer:
            return [self.post_quant_conv, self.decoder]
        return [self.decoder]

    def _bs_split_and_pad(self, x, split_size):
        bs = x.shape[0]
        remain = bs % split_size
        if remain == 0:
            return torch.tensor_split(x, split_size, dim=0)
        else:
            print_rank_0(f"[WARNING]: data batch size {bs} is not divisible by split size {split_size}, which may cause waste!")
            x = torch.cat([x, x[-1:].repeat_interleave(split_size - remain, dim=0)], dim=0)
            return torch.tensor_split(x, split_size, dim=0)

    def encode(self, x, enable_cp=True):
        if not enable_cp:
            return self._encode(x, enable_cp=False)

        if self.cp_size % self.dp_group_nums == 0 and self.cp_size > self.dp_group_nums:
            # loop cp
            data_list = [torch.empty_like(x) for _ in range(self.cp_size)]
            data_list[get_context_parallel_rank()] = x
            torch.distributed.all_gather(data_list, x, group=get_context_parallel_group())
            data_list = data_list[::self.dp_group_nums]
            latents = []
            for data in data_list:
                latents.append(self._encode(data))
            return latents[get_context_parallel_group_rank() % self.dp_group_nums]

        elif self.dp_group_nums % self.cp_size == 0 and self.cp_size < self.dp_group_nums:
            # split
            bs = x.shape[0]
            data_list = self._bs_split_and_pad(x, self.dp_group_nums // self.cp_size)
            data = data_list[get_context_parallel_rank() % (self.dp_group_nums // self.cp_size)]

            _latent = self._encode(data)

            if mpu.get_tensor_model_parallel_world_size() > 1:
                latents_tp = [torch.empty_like(_latent) for _ in range(mpu.get_tensor_model_parallel_world_size())]
                torch.distributed.all_gather(latents_tp, _latent, group=mpu.get_tensor_model_parallel_group())
                latents_tp = torch.cat(latents_tp, dim=0)
            else:
                latents_tp = _latent

            if mpu.get_context_parallel_world_size() > 1:
                latents_cp = [torch.empty_like(latents_tp) for _ in range(mpu.get_context_parallel_world_size())]
                torch.distributed.all_gather(latents_cp, latents_tp, group=mpu.get_context_parallel_group())
                latents = torch.cat(latents_cp, dim=0)
            else:
                latents = latents_tp

            latents = latents[::self.cp_size]
            return latents[:bs]

        elif self.cp_size == self.dp_group_nums:
            return self._encode(x)
        else:
            raise NotImplementedError(f"Not supported megatron data parallel group nums {self.dp_group_nums} and VAE cp_size {self.cp_size}!")

    def _encode(self, x, enable_cp=True):
        x = x.permute(0, 2, 1, 3, 4).contiguous()

        if self.cp_size > 0 and enable_cp:
            global_src_rank = get_context_parallel_group_rank() * self.cp_size
            torch.distributed.broadcast(x, src=global_src_rank, group=get_context_parallel_group())
            x = _conv_split(x, dim=2, kernel_size=1)

        h = self.encoder(x, enable_cp=enable_cp)
        if self.use_quant_layer:
            h = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(h)

        res = posterior.sample()

        if self.cp_size > 0 and enable_cp:
            res = _conv_gather(res, dim=2, kernel_size=1)

        res = 0.7 * res

        return res

    def decode(self, z, **kwargs):
        if self.cp_size > 0:            
            global_src_rank = get_context_parallel_group_rank() * self.cp_size
            torch.distributed.broadcast(z, src=global_src_rank, group=get_context_parallel_group())

            z = _conv_split(z, dim=2, kernel_size=1)

        if self.use_tiling:
            if (z.shape[-1] > self.tile_latent_min_size
                    or z.shape[-2] > self.tile_latent_min_size
                    or z.shape[-3] > self.tile_latent_min_size_t):
                dec = self.tiled_decode(z)
        else:
            if self.use_quant_layer:
                z = self.post_quant_conv(z)
            dec = self.decoder(z)

        if self.cp_size > 0:
            dec = _conv_gather(dec, dim=2, kernel_size=1)

        return dec

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + \
                               b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + \
                               b[:, :, :, :, x] * (x / blend_extent)
        return b

    def tiled_encode(self, x):
        t = x.shape[2]
        t_chunk_idx = [i for i in range(0, t, self.tile_sample_min_size_t - 1)]
        if len(t_chunk_idx) == 1 and t_chunk_idx[0] == 0:
            t_chunk_start_end = [[0, t]]
        else:
            t_chunk_start_end = [[t_chunk_idx[i], t_chunk_idx[i + 1] + 1] for i in range(len(t_chunk_idx) - 1)]
            if t_chunk_start_end[-1][-1] > t:
                t_chunk_start_end[-1][-1] = t
            elif t_chunk_start_end[-1][-1] < t:
                last_start_end = [t_chunk_idx[-1], t]
                t_chunk_start_end.append(last_start_end)
        moments = []
        for idx, (start, end) in enumerate(t_chunk_start_end):
            chunk_x = x[:, :, start: end]
            if idx != 0:
                moment = self.tiled_encode2d(chunk_x, return_moments=True)[:, :, 1:]
            else:
                moment = self.tiled_encode2d(chunk_x, return_moments=True)
            moments.append(moment)
        moments = torch.cat(moments, dim=2)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def tiled_decode(self, x):
        t = x.shape[2]
        t_chunk_idx = [i for i in range(0, t, self.tile_latent_min_size_t - 1)]
        if len(t_chunk_idx) == 1 and t_chunk_idx[0] == 0:
            t_chunk_start_end = [[0, t]]
        else:
            t_chunk_start_end = [[t_chunk_idx[i], t_chunk_idx[i + 1] + 1] for i in range(len(t_chunk_idx) - 1)]
            if t_chunk_start_end[-1][-1] > t:
                t_chunk_start_end[-1][-1] = t
            elif t_chunk_start_end[-1][-1] < t:
                last_start_end = [t_chunk_idx[-1], t]
                t_chunk_start_end.append(last_start_end)
        dec_ = []
        for idx, (start, end) in enumerate(t_chunk_start_end):
            chunk_x = x[:, :, start: end]
            if idx != 0:
                dec = self.tiled_decode2d(chunk_x)[:, :, 1:]
            else:
                dec = self.tiled_decode2d(chunk_x)
            dec_.append(dec)
        dec_ = torch.cat(dec_, dim=2)
        return dec_

    def tiled_encode2d(self, x, return_moments=False):
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[3], overlap_size):
            row = []
            for j in range(0, x.shape[4], overlap_size):
                tile = x[:, :, :,
                       i: i + self.tile_sample_min_size,
                       j: j + self.tile_sample_min_size,
                       ]
                tile = self.encoder(tile)
                if self.use_quant_layer:
                    tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))

        moments = torch.cat(result_rows, dim=3)
        posterior = DiagonalGaussianDistribution(moments)
        if return_moments:
            return moments
        return posterior

    def tiled_decode2d(self, z):
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[3], overlap_size):
            row = []
            for j in range(0, z.shape[4], overlap_size):
                tile = z[:, :, :,
                       i: i + self.tile_latent_min_size,
                       j: j + self.tile_latent_min_size,
                       ]
                if self.use_quant_layer:
                    tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)
        return dec

    def enable_tiling(self, use_tiling: bool = True):
        self.use_tiling = use_tiling

    def disable_tiling(self):
        self.enable_tiling(False)

    def load_checkpoint(self, ckpt_path):
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Could not find checkpoint at {ckpt_path}")

        if ckpt_path.endswith("pt") or ckpt_path.endswith("pth"):
            ckpt_dict = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        elif ckpt_path.endswith(".safetensors"):
            ckpt_dict = safetensors.torch.load_file(ckpt_path)
        else:
            raise ValueError(f"Invalid checkpoint path: {ckpt_path}")

        if "state_dict" in ckpt_dict.keys():
            ckpt_dict = ckpt_dict["state_dict"]

        missing_keys, unexpected_keys = self.load_state_dict(ckpt_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")


class Encoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        hidden_size: int,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = (16,),
        conv_in: str = "Conv2d",
        conv_out: str = "CasualConv3d",
        conv_padding: int = 1,
        attention: str = "AttnBlock2D",
        resnet_blocks: Tuple[str] = (
                "ResnetBlock2D",
                "ResnetBlock2D",
                "ResnetBlock2D",
                "ResnetBlock3D",
        ),
        spatial_downsample: Tuple[str] = (
                "Downsample",
                "Downsample",
                "Downsample",
                "",
        ),
        temporal_downsample: Tuple[str] = ("", "", "TimeDownsampleRes2x", ""),
        mid_resnet: str = "ResnetBlock3D",
        nonlinearity: str = "SiLU",
        dropout: float = 0.0,
        resolution: int = 256,
        num_res_blocks: int = 2,
        double_z: bool = True,
        temporal_compress_times: int = 4,
        gather_norm: bool = False,
    ) -> None:
        super().__init__()
        if len(resnet_blocks) != len(hidden_size_mult):
            raise AssertionError(f"the length of resnet_blocks and hidden_size_mult must be equal")
        # ---- Config ----
        self.temb_ch = 0
        self.num_resolutions = len(hidden_size_mult)
        self.resolution = resolution
        self.num_res_blocks = num_res_blocks
        self.enable_nonlinearity = nonlinearity
        self.enbale_attn1 = attention
        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        # ---- Nonlinearity ----
        if self.enable_nonlinearity:
            self.nonlinearity = model_name_to_cls(nonlinearity)()

        # ---- In ----
        self.conv_in = model_name_to_cls(conv_in)(
            3, hidden_size, kernel_size=3, stride=1, padding=conv_padding
        )

        # ---- Downsample ----
        curr_res = resolution
        in_ch_mult = (1,) + tuple(hidden_size_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = hidden_size * in_ch_mult[i_level]  # [1,1,2,2,4]
            block_out = hidden_size * hidden_size_mult[i_level]  # [1,2,2,4]
            for i_block in range(self.num_res_blocks):
                block.append(
                    model_name_to_cls(resnet_blocks[i_level])(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        gather_norm=gather_norm,
                        temb_channels=self.temb_ch,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(model_name_to_cls(attention)(
                        in_channels=block_in,
                        out_channels=block_in
                    )
                    )
            down = nn.Module()
            down.block = block
            down.attn = attn

            if i_level != self.num_resolutions - 1:  # 3,最后一个downsample不压缩
                if i_level < self.temporal_compress_level:
                    down.downsample = DownSample3D(block_in, compress_time=True)
                else:
                    down.downsample = DownSample3D(block_in, compress_time=False)
                curr_res = curr_res // 2

            if temporal_downsample[i_level]:
                down.time_downsample = model_name_to_cls(temporal_downsample[i_level])(
                    block_in, block_in
                )
            self.down.append(down)

        # ---- Mid ----
        self.mid = nn.Module()

        self.mid.block_1 = model_name_to_cls(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            gather_norm=gather_norm,
            temb_channels=self.temb_ch,
        )

        if self.enbale_attn1:
            self.mid.attn_1 = model_name_to_cls(attention)(
                in_channels=block_in,
                out_channels=block_in
            )

        self.mid.block_2 = model_name_to_cls(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            dropout=dropout,
            gather_norm=gather_norm,
            temb_channels=self.temb_ch,
        )
        # ---- Out ----
        self.norm_out = normalize(block_in, gather=gather_norm)

        self.conv_out = model_name_to_cls(conv_out)(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=conv_padding
        )

    def forward(self, x, enable_cp=True):
        h = self.conv_in(x, enable_cp=enable_cp)

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, enable_cp=enable_cp)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        h = self.mid.block_1(h, enable_cp=enable_cp)
        if self.enbale_attn1:
            h = self.mid.attn_1(h, enable_cp=enable_cp)
        h = self.mid.block_2(h, enable_cp=enable_cp)
        h = self.norm_out(h, enable_cp=enable_cp)
        if self.enable_nonlinearity:
            h = self.nonlinearity(h)
        h = self.conv_out(h, enable_cp=enable_cp)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels: int,
        hidden_size: int,
        hidden_size_mult: Tuple[int] = (1, 2, 4, 4),
        attn_resolutions: Tuple[int] = (16,),
        conv_in: str = "Conv2d",
        conv_out: str = "CasualConv3d",
        conv_padding: int = 1,
        attention: str = "AttnBlock2D",
        resnet_blocks: Tuple[str] = (
                "ResnetBlock3D",
                "ResnetBlock3D",
                "ResnetBlock3D",
                "ResnetBlock3D",
        ),
        spatial_upsample: Tuple[str] = (
                "",
                "SpatialUpsample2x",
                "SpatialUpsample2x",
                "SpatialUpsample2x",
        ),
        temporal_upsample: Tuple[str] = ("", "", "", "TimeUpsampleRes2x"),
        mid_resnet: str = "ResnetBlock3D",
        nonlinearity: str = "SiLU",
        dropout: float = 0.0,
        resolution: int = 256,
        num_res_blocks: int = 2,
        temporal_compress_times: int = 4,
        gather_norm: bool = False,
    ):
        super().__init__()
        # ---- Config ----
        self.temb_ch = 0
        self.num_resolutions = len(hidden_size_mult)
        self.resolution = resolution
        self.num_res_blocks = num_res_blocks
        self.enable_attention = attention
        self.enable_nonlinearity = nonlinearity

        # ---- Nonlinearity ----
        if self.enable_nonlinearity:
            self.nonlinearity = model_name_to_cls(nonlinearity)()

        # log2 of temporal compress_times
        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        # ---- In ----
        block_in = hidden_size * hidden_size_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.conv_in = model_name_to_cls(conv_in)(
            z_channels, block_in, kernel_size=3, padding=conv_padding
        )

        # ---- Mid ----
        self.mid = nn.Module()
        self.mid.block_1 = model_name_to_cls(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            zq_ch=z_channels,
            dropout=dropout,
            gather_norm=gather_norm,
            temb_channels=self.temb_ch,
            normalization=Normalize3D
        )
        if self.enable_attention:
            self.mid.attn_1 = model_name_to_cls(attention)(
                in_channels=block_in,
                out_channels=block_in
            )
        self.mid.block_2 = model_name_to_cls(mid_resnet)(
            in_channels=block_in,
            out_channels=block_in,
            zq_ch=z_channels,
            dropout=dropout,
            gather_norm=gather_norm,
            temb_channels=self.temb_ch,
            normalization=Normalize3D
        )

        # ---- Upsample ----
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = hidden_size * hidden_size_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    model_name_to_cls(resnet_blocks[i_level])(
                        in_channels=block_in,
                        out_channels=block_out,
                        zq_ch=z_channels,
                        dropout=dropout,
                        gather_norm=gather_norm,
                        temb_channels=self.temb_ch,
                        normalization=Normalize3D
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(model_name_to_cls(attention)(
                        in_channels=block_in,
                        out_channels=block_in
                    )
                    )
            up = nn.Module()
            up.block = block
            up.attn = attn
            if spatial_upsample[i_level]:
                compress_time = i_level >= self.num_resolutions - self.temporal_compress_level
                up.upsample = model_name_to_cls(spatial_upsample[i_level])(
                    block_in, block_in, compress_time=compress_time
                )
                curr_res = curr_res * 2
            if temporal_upsample[i_level]:
                up.time_upsample = model_name_to_cls(temporal_upsample[i_level])(
                    block_in, block_in
                )
            self.up.insert(0, up)

        # ---- Out ----
        self.norm_out = Normalize3D(block_in, z_channels, gather=gather_norm)
        self.conv_out = model_name_to_cls(conv_out)(
            block_in, 3, kernel_size=3, padding=conv_padding
        )

    def forward(self, z, **kwargs):
        zq = z

        h = self.conv_in(z)
        h = self.mid.block_1(h, zq=zq)
        if self.enable_attention:
            h = self.mid.attn_1(h)
        h = self.mid.block_2(h, zq=zq)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, zq=zq)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq=zq)
            if hasattr(self.up[i_level], "upsample"):
                h = self.up[i_level].upsample(h)
            if hasattr(self.up[i_level], "time_upsample"):
                h = self.up[i_level].time_upsample(h)

        h = self.norm_out(h, zq=zq)
        if self.enable_nonlinearity:
            h = self.nonlinearity(h)
        h = self.conv_out(h)
        return h
