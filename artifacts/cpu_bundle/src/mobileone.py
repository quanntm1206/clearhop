"""
MobileOne building blocks with train-time multi-branch structure and deploy-time
single-branch reparameterization.

This implementation follows the MobileOne paper's core idea:
- During training: multiple Conv+BN branches (+ optional identity/scale branches)
- During deployment: a single fused Conv (with bias), obtained by analytically
  fusing Conv+BN and identity BN into one convolution kernel and bias.

References:
- "MobileOne: An Improved One millisecond Mobile Backbone"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv2d(nn.Conv2d):
    """Conv2d with symmetric frequency padding and left-only time padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding_freq: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.padding_freq = int(padding_freq)
        self.causal_left = int((kernel_size - 1) * dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(
            x,
            (self.causal_left, 0, self.padding_freq, self.padding_freq),
        )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            (0, 0),
            self.dilation,
            self.groups,
        )


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fuse a Conv2d + BatchNorm2d into an equivalent Conv2d kernel and bias.

    Args:
        conv: Convolution layer (must have weight; bias may be None).
        bn: BatchNorm2d layer.

    Returns:
        (kernel, bias) tensors for an equivalent convolution with bias.
    """
    if conv.bias is None:
        conv_bias = torch.zeros(conv.weight.size(0), device=conv.weight.device, dtype=conv.weight.dtype)
    else:
        conv_bias = conv.bias

    w = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps

    std = torch.sqrt(running_var + eps)
    scale = (gamma / std).reshape(-1, 1, 1, 1)
    fused_w = w * scale
    fused_b = beta + (conv_bias - running_mean) * (gamma / std)
    return fused_w, fused_b


def _fuse_identity_bn(
    bn: nn.BatchNorm2d,
    num_channels: int,
    device: torch.device,
    dtype: torch.dtype,
    causal_time: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create an equivalent 3x3 conv kernel + bias for an identity path with BN.

    The identity is represented as a Dirac delta kernel. For causal kernels,
    the temporal impulse belongs at the right edge (current frame), not the
    geometric center (which would add one frame of delay after fusion).
    """
    # Identity kernel: shape (C_out, C_in, 3, 3) for groups=1 and C_in==C_out==C.
    kernel = torch.zeros((num_channels, num_channels, 3, 3), device=device, dtype=dtype)
    idx = torch.arange(num_channels, device=device)
    time_index = 2 if causal_time else 1
    kernel[idx, idx, 1, time_index] = 1.0

    running_mean = bn.running_mean.to(device=device, dtype=dtype)
    running_var = bn.running_var.to(device=device, dtype=dtype)
    gamma = bn.weight.to(device=device, dtype=dtype)
    beta = bn.bias.to(device=device, dtype=dtype)
    eps = bn.eps

    std = torch.sqrt(running_var + eps)
    scale = (gamma / std).reshape(-1, 1, 1, 1)
    fused_w = kernel * scale
    fused_b = beta - running_mean * (gamma / std)
    return fused_w, fused_b


def _pad_1x1_to_3x3(kernel_1x1: torch.Tensor, *, causal_time: bool = False) -> torch.Tensor:
    """Pad a 1x1 kernel to 3x3, aligned to the current causal frame."""
    if kernel_1x1.size(-1) != 1 or kernel_1x1.size(-2) != 1:
        raise ValueError("Expected a 1x1 kernel.")
    if not causal_time:
        return F.pad(kernel_1x1, [1, 1, 1, 1])
    padded = torch.zeros(
        (*kernel_1x1.shape[:-2], 3, 3),
        device=kernel_1x1.device,
        dtype=kernel_1x1.dtype,
    )
    padded[..., 1, 2] = kernel_1x1[..., 0, 0]
    return padded


@dataclass
class MobileOneConfig:
    """Configuration for a MobileOne block."""

    in_channels: int
    out_channels: int
    stride: int = 1
    padding: int = 1
    dilation: int = 1
    groups: int = 1
    num_conv_branches: int = 1
    use_scale_branch: bool = True  # optional 1x1 conv+bn branch
    use_identity_branch: bool = True  # identity+bn branch when shape allows
    activation: str = "silu"  # "silu" or "relu" or "none"
    causal_time: bool = False


class MobileOneBlock(nn.Module):
    """
    MobileOne block with train-time multi-branch structure and deploy-time fusion.

    Train-time branches:
    - Multiple 3x3 conv+bn branches (num_conv_branches)
    - Optional 1x1 conv+bn scale branch
    - Optional identity+bn branch (only if stride=1 and in_channels==out_channels)

    Deploy-time:
    - Single 3x3 conv with bias (equivalent to the sum of train-time branches)
    """

    def __init__(self, cfg: MobileOneConfig, inference_mode: bool = False) -> None:
        super().__init__()
        self.cfg = cfg
        self.inference_mode = inference_mode

        if cfg.groups != 1:
            # The classic MobileOne paper includes group conv variants; for this project,
            # we keep groups=1 to simplify identity fusion and keep numerics predictable.
            raise ValueError("This implementation currently supports groups=1 only.")

        self.act = self._make_activation(cfg.activation)

        if self.inference_mode:
            self.rbr_reparam = self._make_reparam_conv()
            self.rbr_conv = None
            self.rbr_scale = None
            self.rbr_identity = None
        else:
            self.rbr_conv = nn.ModuleList()
            for _ in range(cfg.num_conv_branches):
                self.rbr_conv.append(self._conv_bn(kernel_size=3))

            self.rbr_scale = self._conv_bn(kernel_size=1) if cfg.use_scale_branch else None

            can_use_identity = (
                cfg.use_identity_branch
                and cfg.stride == 1
                and cfg.in_channels == cfg.out_channels
            )
            self.rbr_identity = nn.BatchNorm2d(cfg.in_channels) if can_use_identity else None
            self.rbr_reparam = None

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "silu":
            return nn.SiLU(inplace=True)
        if name == "relu":
            return nn.ReLU(inplace=True)
        if name in ("none", "identity"):
            return nn.Identity()
        raise ValueError(f"Unknown activation: {name}")

    def _conv_bn(self, kernel_size: int) -> nn.Sequential:
        padding = self.cfg.padding if kernel_size == 3 else 0
        if kernel_size == 3 and self.cfg.causal_time:
            conv = CausalConv2d(
                self.cfg.in_channels,
                self.cfg.out_channels,
                kernel_size=kernel_size,
                stride=self.cfg.stride,
                padding_freq=padding,
                dilation=self.cfg.dilation,
                groups=self.cfg.groups,
                bias=False,
            )
        else:
            conv = nn.Conv2d(
                self.cfg.in_channels,
                self.cfg.out_channels,
                kernel_size=kernel_size,
                stride=self.cfg.stride,
                padding=padding,
                dilation=self.cfg.dilation,
                groups=self.cfg.groups,
                bias=False,
            )
        bn = nn.BatchNorm2d(self.cfg.out_channels)
        return nn.Sequential(conv, bn)

    def _make_reparam_conv(self) -> nn.Conv2d:
        if self.cfg.causal_time:
            return CausalConv2d(
                self.cfg.in_channels,
                self.cfg.out_channels,
                kernel_size=3,
                stride=self.cfg.stride,
                padding_freq=self.cfg.padding,
                dilation=self.cfg.dilation,
                groups=self.cfg.groups,
                bias=True,
            )
        return nn.Conv2d(
            self.cfg.in_channels,
            self.cfg.out_channels,
            kernel_size=3,
            stride=self.cfg.stride,
            padding=self.cfg.padding,
            dilation=self.cfg.dilation,
            groups=self.cfg.groups,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inference_mode:
            out = self.rbr_reparam(x)
            return self.act(out)

        assert self.rbr_conv is not None
        out = 0.0
        for branch in self.rbr_conv:
            out = out + branch(x)

        if self.rbr_scale is not None:
            out = out + self.rbr_scale(x)

        if self.rbr_identity is not None:
            out = out + self.rbr_identity(x)

        return self.act(out)

    @torch.no_grad()
    def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the fused 3x3 kernel and bias for the sum of all branches.
        """
        if self.inference_mode:
            w = self.rbr_reparam.weight
            b = self.rbr_reparam.bias
            return w, b

        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        kernel_sum = torch.zeros(
            (self.cfg.out_channels, self.cfg.in_channels, 3, 3),
            device=device,
            dtype=dtype,
        )
        bias_sum = torch.zeros((self.cfg.out_channels,), device=device, dtype=dtype)

        assert self.rbr_conv is not None
        for branch in self.rbr_conv:
            conv: nn.Conv2d = branch[0]
            bn: nn.BatchNorm2d = branch[1]
            k, b = fuse_conv_bn(conv, bn)
            kernel_sum = kernel_sum + k
            bias_sum = bias_sum + b

        if self.rbr_scale is not None:
            conv_1x1: nn.Conv2d = self.rbr_scale[0]
            bn_1x1: nn.BatchNorm2d = self.rbr_scale[1]
            k1, b1 = fuse_conv_bn(conv_1x1, bn_1x1)
            kernel_sum = kernel_sum + _pad_1x1_to_3x3(k1, causal_time=self.cfg.causal_time)
            bias_sum = bias_sum + b1

        if self.rbr_identity is not None:
            kid, bid = _fuse_identity_bn(
                self.rbr_identity,
                num_channels=self.cfg.in_channels,
                device=device,
                dtype=dtype,
                causal_time=self.cfg.causal_time,
            )
            kernel_sum = kernel_sum + kid
            bias_sum = bias_sum + bid

        return kernel_sum, bias_sum

    @torch.no_grad()
    def reparameterize(self) -> "MobileOneBlock":
        """
        Convert this block to deploy-time inference mode by fusing all branches
        into a single Conv2d.

        Returns:
            self (now in inference_mode with a single fused conv).
        """
        if self.inference_mode:
            return self

        kernel, bias = self.get_equivalent_kernel_bias()

        self.rbr_reparam = self._make_reparam_conv()
        self.rbr_reparam.weight.data.copy_(kernel)
        self.rbr_reparam.bias.data.copy_(bias)

        # Remove training-time branches to avoid accidental usage.
        self.rbr_conv = None
        self.rbr_scale = None
        self.rbr_identity = None

        self.inference_mode = True
        return self
