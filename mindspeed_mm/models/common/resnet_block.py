import torch
from torch import nn

from mindspeed_mm.utils.utils import video_to_image
from mindspeed_mm.models.common.conv import CausalConv3d, SafeConv3d, ContextParallelCausalConv3d, WfCausalConv3d
from mindspeed_mm.models.common.normalize import normalize
from mindspeed_mm.models.common.activations import Sigmoid


class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        dropout,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        conv_shortcut=False,
        num_groups=32,
        eps=1e-6,
        affine=True,
        norm_type="groupnorm",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalize(in_channels, num_groups, eps, affine, norm_type=norm_type)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.norm2 = normalize(out_channels, num_groups, eps, affine, norm_type=norm_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.activation = nn.SiLU()
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    @video_to_image
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.activation(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        x = x + h
        return x


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        num_groups=32,
        eps=1e-6,
        affine=True,
        conv_shortcut=False,
        dropout=0,
        norm_type="groupnorm",
        conv_type="CausalConv3d"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = normalize(in_channels, num_groups, eps, affine, norm_type=norm_type)
        if conv_type == "WfCausalConv3d":
            ConvLayer = WfCausalConv3d
        elif conv_type == "CausalConv3d":
            ConvLayer = CausalConv3d
        else:
            raise ValueError(f"Unsupported convolution type: {conv_type}")
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm2 = normalize(out_channels, num_groups, eps, affine, norm_type=norm_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, padding=padding)
        self.activation = nn.SiLU()

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = ConvLayer(in_channels, out_channels, kernel_size, padding=padding)
            else:
                self.nin_shortcut = ConvLayer(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.activation(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class ContextParallelResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        zq_ch=None,
        add_conv=False,
        gather_norm=False,
        normalization=normalize,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalization(
            in_channels,
            zq_ch=zq_ch,
            add_conv=add_conv,
            gather=gather_norm,
        )

        self.conv1 = ContextParallelCausalConv3d(
            chan_in=in_channels,
            chan_out=out_channels,
            kernel_size=3,
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = normalization(
            out_channels,
            zq_ch=zq_ch,
            add_conv=add_conv,
            gather=gather_norm,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = ContextParallelCausalConv3d(
            chan_in=out_channels,
            chan_out=out_channels,
            kernel_size=3,
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = ContextParallelCausalConv3d(
                    chan_in=in_channels,
                    chan_out=out_channels,
                    kernel_size=3,
                )
            else:
                self.nin_shortcut = SafeConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
        self.act = Sigmoid()

    def forward(self, x, temb=None, zq=None, clear_fake_cp_cache=True):
        h = x

        if zq is not None:
            h = self.norm1(h, zq, clear_fake_cp_cache=clear_fake_cp_cache)
        else:
            h = self.norm1(h)

        h = self.act(h)
        h = self.conv1(h, clear_cache=clear_fake_cp_cache)

        if temb is not None:
            h = h + self.temb_proj(self.act(temb))[:, :, None, None, None]

        if zq is not None:
            h = self.norm2(h, zq, clear_fake_cp_cache=clear_fake_cp_cache)
        else:
            h = self.norm2(h)

        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h, clear_cache=clear_fake_cp_cache)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x, clear_cache=clear_fake_cp_cache)
            else:
                x = self.nin_shortcut(x)

        return x + h