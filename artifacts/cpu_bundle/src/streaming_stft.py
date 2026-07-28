"""
Streaming STFT utilities for real-time (hop-by-hop) processing.

Target configuration (must match project spec):
- sample_rate: 16000
- n_fft: 320  (20 ms)
- hop: 160    (10 ms)
- freq_bins: 161

Design goals:
- Push hop-sized chunks and compute one STFT frame per hop (center=False).
- Maintain ring buffers for time-context features (log power magnitude).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class StreamingSTFTConfig:
    sample_rate: int = 16000
    n_fft: int = 320
    hop: int = 160
    freq_bins: int = 161


class StreamingSTFT:
    """
    Hop-synchronous streaming STFT with ring buffers.

    Typical usage:
        stft = StreamingSTFT()
        for each hop of audio (160 samples):
            x_frame_ri, logp = stft.push_frame(hop_samples)
            ctx = stft.get_context(K_ctx)
    """

    def __init__(
        self,
        cfg: StreamingSTFTConfig | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        window: str = "hann",
        feature_ring_size: int = 512,
    ) -> None:
        self.cfg = cfg or StreamingSTFTConfig()
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype

        if self.cfg.n_fft != 320 or self.cfg.hop != 160 or self.cfg.freq_bins != 161:
            raise ValueError("This project expects n_fft=320, hop=160, freq_bins=161.")

        if window.lower() != "hann":
            raise ValueError("Only Hann window is supported in this reference implementation.")

        self.window = torch.hann_window(self.cfg.n_fft, periodic=True, device=self.device, dtype=self.dtype)

        # Sample ring buffer (most recent n_fft samples).
        self._sample_ring = torch.zeros((self.cfg.n_fft,), device=self.device, dtype=self.dtype)

        # Feature ring buffers (time frames).
        self._feature_ring_size = int(feature_ring_size)
        self._stft_ring = torch.zeros(
            (self._feature_ring_size, self.cfg.freq_bins, 2),
            device=self.device,
            dtype=self.dtype,
        )
        self._logp_ring = torch.zeros(
            (self._feature_ring_size, self.cfg.freq_bins),
            device=self.device,
            dtype=self.dtype,
        )
        self._ring_write_idx = 0
        self._num_frames = 0

    @property
    def num_frames(self) -> int:
        return self._num_frames

    def reset(self) -> None:
        """Reset internal state for a new stream."""
        self._sample_ring.zero_()
        self._stft_ring.zero_()
        self._logp_ring.zero_()
        self._ring_write_idx = 0
        self._num_frames = 0

    def push_frame(self, hop_samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Push one hop (160 samples) and compute the corresponding STFT frame.

        Args:
            hop_samples: (hop,) tensor, float32/float16.

        Returns:
            x_frame_ri: (F, 2) real/imag
            logp: (F,) log power magnitude feature
        """
        if hop_samples.ndim != 1 or hop_samples.numel() != self.cfg.hop:
            raise ValueError(f"hop_samples must have shape ({self.cfg.hop},).")
        if not torch.is_floating_point(hop_samples) or not torch.isfinite(hop_samples).all():
            raise ValueError("hop_samples must be finite floating-point samples.")

        hop_samples = hop_samples.to(device=self.device, dtype=self.dtype)

        # Shift ring left by hop and append new samples.
        self._sample_ring = torch.roll(self._sample_ring, shifts=-self.cfg.hop, dims=0)
        self._sample_ring[-self.cfg.hop :] = hop_samples

        # Windowed frame for STFT.
        frame = self._sample_ring * self.window
        x = torch.fft.rfft(frame, n=self.cfg.n_fft)  # (F,) complex
        if x.numel() != self.cfg.freq_bins:
            raise RuntimeError("Unexpected rFFT size; check n_fft/freq_bins.")

        x_ri = torch.stack([x.real, x.imag], dim=-1)  # (F,2)
        mag2 = x.real * x.real + x.imag * x.imag
        logp = torch.log(mag2 + 1e-12)

        # Write to rings.
        self._stft_ring[self._ring_write_idx] = x_ri
        self._logp_ring[self._ring_write_idx] = logp
        self._ring_write_idx = (self._ring_write_idx + 1) % self._feature_ring_size
        self._num_frames += 1

        return x_ri, logp

    def _gather_ring(self, ring: torch.Tensor, k: int) -> torch.Tensor:
        """Gather last k entries from a ring buffer into (k, ...)."""
        if k <= 0:
            raise ValueError("k must be > 0.")
        k = min(k, min(self._num_frames, self._feature_ring_size))
        # Compute indices of last k frames in chronological order.
        end = self._ring_write_idx
        start = (end - k) % self._feature_ring_size
        if start < end:
            return ring[start:end]
        return torch.cat([ring[start:], ring[:end]], dim=0)

    def get_context(self, k_ctx: int) -> torch.Tensor:
        """
        Return the most recent STFT frames.

        Returns:
            x_ctx_ri: (k_ctx, F, 2) with chronological order (oldest -> newest).
        """
        return self._gather_ring(self._stft_ring, k_ctx)

    def get_logp_context(self, k_ctx: int) -> torch.Tensor:
        """
        Return the most recent log-power magnitude frames.

        Returns:
            logp_ctx: (k_ctx, F) chronological order.
        """
        return self._gather_ring(self._logp_ring, k_ctx)
