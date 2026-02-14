# -----------------------------------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# -----------------------------------------------------------------------------------
# VMamba: Visual State Space Model
# Copyright (c) 2024 MzeroMiko
# -----------------------------------------------------------------------------------
# Spatial-Mamba: Effective Visual State Space Models via Structure-Aware State Fusion
# Modified by Chaodong Xiao
# -----------------------------------------------------------------------------------

import copy
import math
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from fvcore.nn import flop_count, parameter_count
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from .attention_utils import SEAttention
from .wtconv import WTConv2d

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

try:
    from kernels.dwconv2d.Dwconv.dwconv_layer import DepthwiseFunction
except:
    DepthwiseFunction = None
import selective_scan_cuda_oflex_rh


def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_C = True, with_D=True, with_Z=False, with_complex=False,):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32
    
    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu] 
    """
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    if with_C:
        flops = 9 * B * L * D * N
    else:
        flops = 7 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops


def selective_scan_flop_jit(inputs, outputs, flops_fn=flops_selective_scan_fn):
    # print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_fn(B=B, L=L, D=D, N=N, with_C=True, with_D=True, with_Z=False)
    return flops


def selective_scan_state_flop_jit(inputs, outputs, flops_fn=flops_selective_scan_fn):
    # print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    assert N == 1
    flops = flops_fn(B=B, L=L, D=D, N=N, with_C=False, with_D=False, with_Z=False)
    return flops


class SelectiveScanStateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, B, D=None, z=None, delta_bias=None, delta_softplus=False,
                return_last_state=False, lag=0):
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if z is not None and z.stride(-1) != 1:
            z = z.contiguous()
        if B.dim() == 3:
            B = rearrange(B, "b dstate l -> b 1 dstate l")
            ctx.squeeze_B = True

        out, x, *rest = selective_scan_cuda_oflex_rh.fwd(u, delta, A, B, D, delta_bias, delta_softplus, 1, True)
        ctx.delta_softplus = delta_softplus
        ctx.has_z = z is not None
        last_state = x[:, :, -1, 1::2]  # (batch, dim, dstate)
        if not ctx.has_z:
            ctx.save_for_backward(u, delta, A, B, D, delta_bias, x)
            return out if not return_last_state else (out, last_state)
        else:
            ctx.save_for_backward(u, delta, A, B, D, z, delta_bias, x, out)
            out_z = rest[0]
            return out_z if not return_last_state else (out_z, last_state)

    @staticmethod
    def backward(ctx, dout, *args):
        if not ctx.has_z:
            u, delta, A, B, D, delta_bias, x = ctx.saved_tensors
            z = None
            out = None
        else:
            u, delta, A, B, D, z, delta_bias, x, out = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        # The kernel supports passing in a pre-allocated dz (e.g., in case we want to fuse the
        # backward of selective_scan_cuda with the backward of chunk).
        # Here we just pass in None and dz will be allocated in the C++ code.
        du, ddelta, dA, dB, dD, ddelta_bias, *rest = selective_scan_cuda_oflex_rh.bwd(
            u, delta, A, B, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        dz = rest[0] if ctx.has_z else None
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        return (du, ddelta, dA, dB,
                dD if D is not None else None,
                dz,
                ddelta_bias if delta_bias is not None else None,
                None,
                None,
                None)


def selective_scan_fn(u, delta, A, B, D=None, z=None, delta_bias=None, delta_softplus=False,
                     return_last_state=False):
    """if return_last_state is True, returns (out, last_state)
    last_state has shape (batch, dim, dstate). Note that the gradient of the last state is
    not considered in the backward pass.
    """

    return SelectiveScanStateFn.apply(u, delta, A, B, D, z, delta_bias, delta_softplus, return_last_state)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dilation=1, groups=1,
                 bias=True, dropout=0, norm=nn.BatchNorm2d, act_func=nn.ReLU):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class DownSampling(nn.Module):
    r""" DownSampling Layer.

    Args:
        dim (int): Number of input channels.
    """

    def __init__(self, dim, ratio=4.0):
        super().__init__()
        self.dim = dim
        in_channels = dim
        out_channels = dim * 2
        self.conv = nn.Sequential(ConvLayer(in_channels, out_channels, kernel_size=3, stride=2, padding=1, act_func=None))
        # self.conv = nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

    def forward(self, x):
        x = self.conv(rearrange(x, 'b h w d -> b d h w').contiguous())
        x = rearrange(x, 'b d h w -> b h w d')
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StateFusion(nn.Module):
    """multi-scale local context extraction module based on depth-wise convolution.
    re-parameterization techniques are used to accelerate inference speed.

    """
    def __init__(self, dim):
        super(StateFusion, self).__init__()
        self.dim = dim
        self.kernel_3   = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.kernel_3_1 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.kernel_3_2 = nn.Parameter(torch.ones(dim, 1, 3, 3))
        self.alpha = nn.Parameter(torch.ones(3), requires_grad=True)

    @staticmethod
    def padding(input_tensor, padding):
        return torch.nn.functional.pad(input_tensor, padding, mode='replicate')

    def forward(self, h):
        if self.training:
            h1 = F.conv2d(self.padding(h, (1,1,1,1)), self.kernel_3,   padding=0, dilation=1, groups=self.dim)
            h2 = F.conv2d(self.padding(h, (3,3,3,3)), self.kernel_3_1, padding=0, dilation=3, groups=self.dim)
            h3 = F.conv2d(self.padding(h, (5,5,5,5)), self.kernel_3_2, padding=0, dilation=5, groups=self.dim)
            out = self.alpha[0]*h1 + self.alpha[1]*h2 + self.alpha[2]*h3
            return out
        else:
            if not hasattr(self, "_merge_weight"):
                self._merge_weight = torch.zeros((self.dim, 1, 11, 11), device=h.device)
                self._merge_weight[:, :, 4:7, 4:7] = self.alpha[0]*self.kernel_3

                self._merge_weight[:, :, 2:3, 2:3] = self.alpha[1]*self.kernel_3_1[:,:,0:1,0:1]
                self._merge_weight[:, :, 2:3, 5:6] = self.alpha[1]*self.kernel_3_1[:,:,0:1,1:2]
                self._merge_weight[:, :, 2:3, 8:9] = self.alpha[1]*self.kernel_3_1[:,:,0:1,2:3]
                self._merge_weight[:, :, 5:6, 2:3] = self.alpha[1]*self.kernel_3_1[:,:,1:2,0:1]
                self._merge_weight[:, :, 5:6, 5:6] += self.alpha[1]*self.kernel_3_1[:,:,1:2,1:2]
                self._merge_weight[:, :, 5:6, 8:9] = self.alpha[1]*self.kernel_3_1[:,:,1:2,2:3]
                self._merge_weight[:, :, 8:9, 2:3] = self.alpha[1]*self.kernel_3_1[:,:,2:3,0:1]
                self._merge_weight[:, :, 8:9, 5:6] = self.alpha[1]*self.kernel_3_1[:,:,2:3,1:2]
                self._merge_weight[:, :, 8:9, 8:9] = self.alpha[1]*self.kernel_3_1[:,:,2:3,2:3]

                self._merge_weight[:, :, 0:1, 0:1] = self.alpha[2]*self.kernel_3_2[:,:,0:1,0:1]
                self._merge_weight[:, :, 0:1, 5:6] = self.alpha[2]*self.kernel_3_2[:,:,0:1,1:2]
                self._merge_weight[:, :, 0:1, 10:11] = self.alpha[2]*self.kernel_3_2[:,:,0:1,2:3]
                self._merge_weight[:, :, 5:6, 0:1] = self.alpha[2]*self.kernel_3_2[:,:,1:2,0:1]
                self._merge_weight[:, :, 5:6, 5:6] += self.alpha[2]*self.kernel_3_2[:,:,1:2,1:2]
                self._merge_weight[:, :, 5:6, 10:11] = self.alpha[2]*self.kernel_3_2[:,:,1:2,2:3]
                self._merge_weight[:, :, 10:11, 0:1] = self.alpha[2]*self.kernel_3_2[:,:,2:3,0:1]
                self._merge_weight[:, :, 10:11, 5:6] = self.alpha[2]*self.kernel_3_2[:,:,2:3,1:2]
                self._merge_weight[:, :, 10:11, 10:11] = self.alpha[2]*self.kernel_3_2[:,:,2:3,2:3]

            out = DepthwiseFunction.apply(h, self._merge_weight, None, 11//2, 11//2, False)
            return out


class SpectralSASF(nn.Module):
    """
    Spectral Structure-Aware State Fusion
    Input:  (B, H, W, C)
    Output: (B, H, W, C)
    """

    def __init__(
        self,
        channels,
        kernel_size=3,
        dilations=(1, 2, 4),
        residual=True,
        use_gate=True,
    ):
        super().__init__()

        self.residual = residual
        self.use_gate = use_gate

        self.branches = nn.ModuleList()

        for d in dilations:
            padding = d * (kernel_size // 2)

            conv = nn.Conv1d(
                in_channels=1,
                out_channels=1,
                kernel_size=kernel_size,
                dilation=d,
                padding=padding,
                groups=1, # for each spectral channel, we apply a separate convolution, so groups=1
                bias=False,
            )
            self.branches.append(conv)

        if use_gate:
            self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        x: (B, H, W, C)
        """

        B, H, W, C = x.shape

        # reshape as (B*H*W, 1, C)
        x_reshaped = x.view(B * H * W, 1, C)

        out = 0
        for conv in self.branches:
            out = out + conv(x_reshaped)

        if self.residual:
            if self.use_gate:
                out = x_reshaped + torch.sigmoid(self.gate) * out
            else:
                out = x_reshaped + out

        # reshape to original shape
        out = out.view(B, H, W, C)

        return out


class FastSpectralSASF(nn.Module):
    def __init__(self, C, kernel_size=3, dilations=(1,2,4)):
        super().__init__()
        self.branches = nn.ModuleList()

        for d in dilations:
            conv = nn.Conv3d(
                in_channels=1,
                out_channels=1,
                kernel_size=(kernel_size, 1, 1),
                dilation=(d, 1, 1),
                padding=(d, 0, 0),
                groups=1,
                bias=False
            )
            self.branches.append(conv)

    def forward(self, x):
        # x: (B, H, W, C)
        # reshape to (B, 1, C, H, W)
        x = x.permute(0, 3, 1, 2).unsqueeze(1)

        out = 0
        for conv in self.branches:
            out = out + conv(x)

        # reshape back to (B, H, W, C)
        out = out.squeeze(1).permute(0, 2, 3, 1)

        return out


class StateFusionSS(nn.Module):
    def __init__(self, in_channels, kernel_size: int = 3, stride: int = 1, pad: int = 2):
        super(StateFusionSS, self).__init__()
        self.ss_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=(stride, stride, stride),
            padding=(pad, pad, pad),
            # groups=in_channels, # poor performance.
            dilation=(pad, pad, pad)
        )
        self.pointwise_conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        x1 = x.unsqueeze(2)
        x1 = self.ss_conv(x1)
        x1 = x1.squeeze(2)
        return x1


# WTConv from Wavelet Convolutions for Large Receptive Fields
class StateFusionWTConv(nn.Module):
    def __init__(self, in_channels, kernel_size: int = 5, stride: int = 1, wt_levels: int = 3):
        super(StateFusionWTConv, self).__init__()
        self.wt_conv = WTConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=True,
            wt_levels=wt_levels,
        )

    def forward(self, x):
        x = self.wt_conv(x)
        return x


# BottConv from SCSegamba lightweight convolution
class StateFusionBottConv(nn.Module):
    def __init__(self, in_channels, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super(StateFusionBottConv, self).__init__()
        mid_channels = in_channels // 8
        out_channels = in_channels
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=True)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x


class StateFusionWTConv_BottConv(nn.Module):
    def __init__(self, in_channels, kernel_size: int = 3, stride: int = 1, padding: int = 1, wt_levels: int = 3):
        super(StateFusionWTConv_BottConv, self).__init__()
        self.freq_branch = StateFusionWTConv(in_channels, kernel_size=kernel_size, stride=stride, wt_levels=wt_levels)
        self.spa_branch = StateFusionBottConv(in_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.weights = nn.Parameter(torch.ones(2) / 2)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        x_feq = self.freq_branch(x)
        x_spa = self.spa_branch(x)
        weights = self.softmax(self.weights)
        fusion_x = x_feq * weights[0] + x_spa * weights[1]
        return fusion_x


# Channel Attention (SE Block)
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


# Spatial Attention
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)
        return x * y


# Adaptive Fusion Spectral-Spatial Attention Module
# The performance is not good enough, compare to the other fusion methods.
class StateFusionASSF(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=3):
        super(StateFusionASSF, self).__init__()
        self.channel_attention = SEBlock(channels, reduction)
        # self.spatial_attention = SpatialAttention(spatial_kernel)
        self.spatial_attention = StateFusionBottConv(channels)

        # Adaptive fusion parameter: a learnable scalar
        self.alpha_param = nn.Parameter(torch.tensor(0.5))  # Initial value = 0.5

    def forward(self, x):
        ca = self.channel_attention(x)  # Channel attention
        sa = self.spatial_attention(x)  # Spatial attention

        # Sigmoid to restrict it to [0, 1]
        alpha = torch.sigmoid(self.alpha_param)

        # Adaptive fusion
        out = alpha * ca + (1 - alpha) * sa
        return out


class StructureAwareSSM(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        # project the input dim in 4 times of d_model
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        # extract the local patch context (depth-wise Conv) and keep the spatial resolution unchanged
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        # the learnable parameters of mamba
        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state*2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight)
        del self.x_proj

        self.dt_projs = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_projs.weight)
        self.dt_projs_bias = nn.Parameter(self.dt_projs.bias)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, dt_init)
        self.Ds = self.D_init(self.d_inner, dt_init)

        self.selective_scan = selective_scan_fn
        # the main contribution of SpatialMamba
        self.state_fusion = StateFusion(self.d_inner)
        # Ours
        # self.state_fusion = StateFusionWTConv(self.d_inner)
        # self.state_fusion = StateFusionBottConv(self.d_inner)
        # self.state_fusion = StateFusionWTConv_BottConv(self.d_inner)
        # self.state_fusion = StateFusionASSF(self.d_inner)
        # self.state_fusion = StateFusionSS(self.d_inner)
        # self.state_fusion = SpectralSASF(self.d_inner)
        # self.state_fusion = FastSpectralSASF(self.d_inner)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, bias=True,**factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=bias, **factory_kwargs)

        if bias:
            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            dt = torch.exp(
                torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
                + math.log(dt_min)
            ).clamp(min=dt_init_floor)
            # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
            inv_dt = dt + torch.log(-torch.expm1(-dt))

            with torch.no_grad():
                dt_proj.bias.copy_(inv_dt)
            # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
            dt_proj.bias._no_reinit = True

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        elif dt_init == "simple":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.randn((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.randn((d_inner)))
                dt_proj.bias._no_reinit = True
        elif dt_init == "zero":
            with torch.no_grad():
                dt_proj.weight.copy_(0.1 * torch.rand((d_inner, dt_rank)))
                dt_proj.bias.copy_(0.1 * torch.rand((d_inner)))
                dt_proj.bias._no_reinit = True
        else:
            raise NotImplementedError

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, init, device=None):
        if init=="random" or "constant":
            # S4D real initialization
            A = repeat(
                torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=d_inner,
            ).contiguous()
            A_log = torch.log(A)
            A_log = nn.Parameter(A_log)
            A_log._no_weight_decay = True
        elif init=="simple":
            A_log = nn.Parameter(torch.randn((d_inner, d_state)))
        elif init=="zero":
            A_log = nn.Parameter(torch.zeros((d_inner, d_state)))
        else:
            raise NotImplementedError
        return A_log

    @staticmethod
    def D_init(d_inner, init="random", device=None):
        if init=="random" or "constant":
            # D "skip" parameter
            D = torch.ones(d_inner, device=device)
            D = nn.Parameter(D) 
            D._no_weight_decay = True
        elif init == "simple" or "zero":
            D = nn.Parameter(torch.ones(d_inner))
        else:
            raise NotImplementedError
        return D

    def ssm(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W

        xs = x.view(B, -1, L)
        
        x_dbl = torch.matmul(self.x_proj_weight.view(1, -1, C), xs)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=1)
        dts = torch.matmul(self.dt_projs_weight.view(1, C, -1), dts)
        
        As = -torch.exp(self.A_logs)
        Ds = self.Ds
        dts = dts.contiguous()
        dt_projs_bias = self.dt_projs_bias

        h = self.selective_scan(
            xs, dts, 
            As, Bs, None,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        )

        h = rearrange(h, "b d 1 (h w) -> b (d 1) h w", h=H, w=W)
        h = self.state_fusion(h)
        h = rearrange(h, "b d h w -> b d (h w)")
        
        y = h * Cs
        y = y + xs * Ds.view(-1, 1)

        return y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # split the xz into 2 parts along the last dimension

        x = rearrange(x, 'b h w d -> b d h w').contiguous()
        x = self.act(self.conv2d(x))
        y = self.ssm(x)
        y = rearrange(y, 'b d (h w)-> b h w d', h=H, w=W)

        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return y


class SpatialMambaBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        dt_init: str = "random",
        num_heads: int = 8,
        mlp_ratio = 4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        **kwargs,
    ):
        super().__init__()
        # depth-wise convolution
        self.cpe1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.ln_1 = norm_layer(hidden_dim)
        self.SASSM = StructureAwareSSM(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, dt_init=dt_init, **kwargs)
        self.drop_path = DropPath(drop_path)
        # depth-wise convolution
        self.cpe2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, groups=hidden_dim)
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLP(in_features=hidden_dim, hidden_features=int(hidden_dim*mlp_ratio), act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=False)

    def forward(self, x: torch.Tensor):
        x = x + self.cpe1(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)
        x = x + self.drop_path(self.SASSM(self.ln_1(x)))
        x = x + self.cpe2(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1)
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class SpatialMambaLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks for one stage.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
    def __init__(
        self, 
        dim, 
        depth, 
        attn_drop=0.,
        drop_path=0., 
        norm_layer=nn.LayerNorm, 
        downsample=None, 
        use_checkpoint=False, 
        d_state=16,
        dt_init="random",
        mlp_ratio=4.0,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SpatialMambaBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
                dt_init=dt_init,
                mlp_ratio=mlp_ratio,
            )
            for i in range(depth)])

        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim)
        else:
            self.downsample = None

    def forward(self, x):
        # Input BHWD
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ChannelGateWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelGateWithAttention, self).__init__()

        # Step 1: Depthwise Convolution (each channel processed independently)
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)

        # Step 2: Learnable Scaling Factor for Gate Mechanism (soft attention)
        self.channel_gate = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation to generate gating coefficients between 0 and 1
        self.scale_factor = nn.Parameter(torch.ones(1))  # Learnable scaling factor to adjust the gate's impact

        # Step 3: Pointwise Convolution (reduce channel dimensions after gating)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # Apply depthwise convolution to extract features for each channel independently
        x = self.depthwise_conv(x)

        # Apply channel gating mechanism to learn channel-wise attention
        gate = self.channel_gate(x)  # Generate gating coefficients for each channel
        gate = self.sigmoid(gate)  # Apply sigmoid to ensure values are between 0 and 1
        x = x * gate  # Element-wise multiplication (apply gate to the features)

        # Apply a learnable scaling factor to control the impact of the gate
        x = x * self.scale_factor  # Scale the gated features

        # Apply pointwise convolution to reduce the channel dimension
        x = self.pointwise_conv(x)
        return x


class SpatialMambaHSI(nn.Module):
    def __init__(self, 
                 in_channels: int = 128,
                 num_classes: int = 16,
                 depth: int = 2,
                 hidden_dim: int = 96,
                 d_state = 1, 
                 dt_init = "random",
                 mlp_ratio = 4.0,
                 drop_rate = 0.1, 
                 attn_drop_rate = 0., 
                 drop_path_rate = 0.1,
                 norm_layer = nn.LayerNorm, 
                 use_checkpoint = False, 
                 group_num: int = 4,
                 **kwargs):
        super(SpatialMambaHSI, self).__init__()

        # channel gate, aiming to reduce the input channel dimension, but the performance gets poor
        # self.gate = ChannelGateWithAttention(in_channels=in_channels, out_channels=64)
        # just a normal conv+gn+silu cnn stem
        self.patch_embedding = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
                                             nn.GroupNorm(group_num, hidden_dim),
                                             nn.SiLU())

        self.depth = depth
        # print(self.depth)
        self.feat_dim = [int(hidden_dim * 2 ** i_layer) for i_layer in range(self.depth)]
        # print("self.feat_dim: ", self.feat_dim)
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        # print(self.dpr)
        # mamba blocks
        self.mamba_blocks = nn.ModuleList([
                SpatialMambaLayer(
                    dim=self.feat_dim[i],
                    depth=1, # inside the SpatialMambaLayer, the depth is 1 for one stage
                    d_state=d_state, # d_state==1 is mandatory here, issue at https://github.com/EdwardChasel/Spatial-Mamba/issues/7#issuecomment-2493858325
                    dt_init=dt_init,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=self.dpr[i],
                    norm_layer=nn.LayerNorm,
                    downsample=DownSampling if (i < self.depth - 1) else None,
                    use_checkpoint=use_checkpoint,
            )
            for i in range(self.depth)])

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.norm = norm_layer(self.feat_dim[-1])
        # aiming to boost the performance, but the performance gets poor
        # self.ratio = 2
        # self.channel_attn = SEAttention(hidden_dim * self.ratio)
        self.cls_head = nn.Sequential(nn.Conv2d(in_channels=self.feat_dim[-1], out_channels=128, kernel_size=1, stride=1, padding=0),
                                      nn.GroupNorm(group_num, 128),
                                      nn.SiLU(),
                                      nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        for name, p in m.named_parameters():
            if name in ["out_proj.weight"]:
                p = p.clone().detach_()
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))

    def forward(self, x):
        # x = self.gate(x)
        x = self.patch_embedding(x) # shape: BCHW -> B hidden_dim HW
        x = rearrange(x, "b c h w -> b h w c")
        x = self.pos_drop(x)
        for i, mamba in enumerate(self.mamba_blocks):
            x = mamba(x)
        x = self.norm(x)
        x = rearrange(x, "b h w c -> b c h w")
        # x = self.channel_attn(x)
        logits = self.cls_head(x)
        return logits


if __name__=='__main__':
    # batch, length, dim = 2, 512*512, 256
    # x = torch.randn(batch, length, dim).to("cuda")
    batch, C, H, W = 1, 128, 512, 512
    x = torch.randn(batch, H, W, C).to("cuda")
    model = SpatialMambaLayer(
                dim=128,
                depth=4,
                d_state=1,
                dt_init="random",
                mlp_ratio=4.0,
                drop=0.0, 
                attn_drop=0.0,
                drop_path=0.0,
                norm_layer=nn.LayerNorm,
                downsample=DownSampling,
                use_checkpoint=False,
            ).to("cuda")
    y = model(x)
    print(x.shape)
    print(y.shape)