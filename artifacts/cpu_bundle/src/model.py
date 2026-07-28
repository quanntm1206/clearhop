"""
MobileDeepFilterNet: a compact streaming noise reduction network.

Pipeline-aligned components:
- Encoder: MobileOne-style blocks (2D conv over (freq, time))
- Temporal module: causal GRU with stateful streaming support
- Decoder heads:
    - spectral mask (sigmoid)
    - complex deep filter taps (real + imag)

This is a reference implementation tailored for:
    sample_rate=16 kHz, n_fft=320, freq_bins=161, hop=160
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .mobileone import CausalConv2d, MobileOneBlock, MobileOneConfig


@dataclass
class MobileDeepFilterNetConfig:
    freq_bins: int = 161
    enc_channels: int = 32
    num_encoder_blocks: int = 3
    gru_hidden: int = 64
    gru_layers: int = 1
    k_tap: int = 3
    activation: str = "silu"
    causal_time: bool = True


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable conv (depthwise 3x3 + pointwise 1x1) for 2D features."""

    def __init__(self, channels: int, activation: str = "silu", causal_time: bool = False) -> None:
        super().__init__()
        self.dw = (
            CausalConv2d(
                channels,
                channels,
                kernel_size=3,
                padding_freq=1,
                groups=channels,
                bias=False,
            )
            if causal_time
            else nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        )
        self.dw_bn = nn.BatchNorm2d(channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True) if activation == "silu" else nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.dw_bn(self.dw(x)))
        x = self.act(self.pw_bn(self.pw(x)))
        return x


class MobileDeepFilterNet(nn.Module):
    """
    Streaming-capable noise reduction model.

    Input features are expected as log-power magnitude:
        feats: (B, 1, F=161, T)

    Outputs:
        mask: (B, F, T) in [0,1]
        w_taps: (B, F, K, T, 2) complex deep filter taps
    """

    def __init__(self, cfg: MobileDeepFilterNetConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or MobileDeepFilterNetConfig()

        if self.cfg.freq_bins != 161:
            raise ValueError("This project expects freq_bins=161.")

        # Encoder: initial projection + MobileOne blocks + separable conv refinement.
        in_conv = (
            CausalConv2d(1, self.cfg.enc_channels, kernel_size=3, padding_freq=1, bias=False)
            if self.cfg.causal_time
            else nn.Conv2d(1, self.cfg.enc_channels, kernel_size=3, padding=1, bias=False)
        )
        self.in_proj = nn.Sequential(
            in_conv,
            nn.BatchNorm2d(self.cfg.enc_channels),
            nn.SiLU(inplace=True) if self.cfg.activation == "silu" else nn.ReLU(inplace=True),
        )

        blocks = []
        for _ in range(self.cfg.num_encoder_blocks):
            blocks.append(
                MobileOneBlock(
                    MobileOneConfig(
                        in_channels=self.cfg.enc_channels,
                        out_channels=self.cfg.enc_channels,
                        stride=1,
                        padding=1,
                        num_conv_branches=2,
                        use_scale_branch=True,
                        use_identity_branch=True,
                        activation=self.cfg.activation,
                        causal_time=self.cfg.causal_time,
                    ),
                    inference_mode=False,
                )
            )
            blocks.append(
                DepthwiseSeparableConv(
                    self.cfg.enc_channels,
                    activation=self.cfg.activation,
                    causal_time=self.cfg.causal_time,
                )
            )
        self.encoder = nn.Sequential(*blocks)

        # Temporal module: causal GRU over time. We pool over frequency to keep it compact.
        self.temporal_in = nn.Linear(self.cfg.enc_channels, self.cfg.gru_hidden)
        self.gru = nn.GRU(
            input_size=self.cfg.gru_hidden,
            hidden_size=self.cfg.gru_hidden,
            num_layers=self.cfg.gru_layers,
            batch_first=True,
        )

        # Merge temporal context back into per-bin representation.
        self.temporal_out = nn.Linear(self.cfg.gru_hidden, self.cfg.enc_channels)

        # Mask head: per-frequency sigmoid mask.
        self.mask_head = nn.Sequential(
            nn.Conv2d(self.cfg.enc_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # Deep filter head: complex taps (real+imag) per frequency.
        self.df_head = nn.Conv2d(self.cfg.enc_channels, 2 * self.cfg.k_tap, kernel_size=1)

    def forward(
        self,
        feats_logp: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass on a chunk.

        Args:
            feats_logp: (B, 1, F, T)
            h: optional GRU hidden state (num_layers, B, hidden)

        Returns:
            mask: (B, F, T)
            w_taps: (B, F, K, T, 2)
            h_new: updated GRU hidden state
        """
        if feats_logp.ndim != 4 or feats_logp.size(1) != 1 or feats_logp.size(2) != self.cfg.freq_bins:
            raise ValueError("feats_logp must have shape (B, 1, 161, T).")

        x = self.in_proj(feats_logp)  # (B, C, F, T)
        x = self.encoder(x)  # (B, C, F, T)

        # Temporal pooling over frequency -> (B, C, T) -> (B, T, C)
        x_pool = x.mean(dim=2).transpose(1, 2)
        x_in = self.temporal_in(x_pool)
        y_seq, h_new = self.gru(x_in, h)
        y_ctx = self.temporal_out(y_seq)  # (B, T, C)

        # Broadcast temporal context over frequency.
        y_ctx_2d = y_ctx.transpose(1, 2).unsqueeze(2)  # (B, C, 1, T)
        x_ctx = x + y_ctx_2d

        mask = self.mask_head(x_ctx).squeeze(1)  # (B, F, T)
        df = self.df_head(x_ctx)  # (B, 2K, F, T)
        b, _, f, t = df.shape
        df = df.view(b, 2, self.cfg.k_tap, f, t).permute(0, 3, 2, 4, 1).contiguous()
        # df: (B, F, K, T, 2)
        return mask, df, h_new

    def forward_streaming(
        self,
        feats_logp_ctx: torch.Tensor,
        h: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Streaming-friendly wrapper.

        For real-time usage, you typically pass a small context window and take
        only the last time step from the outputs.

        Args:
            feats_logp_ctx: (B, 1, F, T_ctx)
            h: GRU hidden state

        Returns:
            mask_last: (B, F) for the newest frame
            w_last: (B, F, K, 2) taps for the newest frame
            h_new: updated GRU hidden
        """
        if feats_logp_ctx.ndim != 4 or feats_logp_ctx.size(1) != 1 or feats_logp_ctx.size(2) != self.cfg.freq_bins:
            raise ValueError("feats_logp_ctx must have shape (B, 1, 161, T_ctx).")
        x = self.encoder(self.in_proj(feats_logp_ctx))[..., -1:]
        x_pool = x.mean(dim=2).transpose(1, 2)
        x_in = self.temporal_in(x_pool)
        y_seq, h_new = self.gru(x_in, h)
        y_ctx = self.temporal_out(y_seq).transpose(1, 2).unsqueeze(2)
        x_ctx = x + y_ctx
        mask_last = self.mask_head(x_ctx).squeeze(1).squeeze(-1)
        df = self.df_head(x_ctx)
        b, _, f, _ = df.shape
        w_last = df.view(b, 2, self.cfg.k_tap, f).permute(0, 3, 2, 1).contiguous()
        return mask_last, w_last, h_new

    @property
    def temporal_receptive_frames(self) -> int:
        """Encoder frames required to reproduce the newest offline output."""
        return 3 + 4 * self.cfg.num_encoder_blocks

    @torch.no_grad()
    def reparameterize_encoder(self) -> None:
        """
        Fuse all MobileOne blocks in the encoder into deploy-time convs.
        """
        for m in self.modules():
            if isinstance(m, MobileOneBlock):
                m.reparameterize()
