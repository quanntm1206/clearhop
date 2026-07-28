"""Shared offline and hop-by-hop inference contracts."""

from __future__ import annotations

from typing import Optional

import torch

from .filtering import causal_deep_filter, stft_analysis_ri, stft_synthesis
from .model import MobileDeepFilterNet
from .streaming_stft import StreamingSTFT, StreamingSTFTConfig


@torch.no_grad()
def enhance_offline(
    model: MobileDeepFilterNet,
    noisy: torch.Tensor,
    *,
    stft_cfg: StreamingSTFTConfig | None = None,
) -> torch.Tensor:
    """Enhance a batch of waveforms using the same causal spectral contract."""
    if noisy.ndim != 2:
        raise ValueError("noisy must have shape (B, samples).")
    cfg = stft_cfg or StreamingSTFTConfig()
    window = torch.hann_window(cfg.n_fft, device=noisy.device, dtype=noisy.dtype)
    spec_ri = stft_analysis_ri(noisy, cfg.n_fft, cfg.hop, window)
    feats = torch.log(spec_ri[..., 0].square() + spec_ri[..., 1].square() + 1e-12).unsqueeze(1)
    mask, taps, _ = model(feats, None)
    enhanced_ri = causal_deep_filter(spec_ri, mask, taps)
    return stft_synthesis(
        enhanced_ri,
        n_fft=cfg.n_fft,
        hop=cfg.hop,
        window=window,
        length=noisy.size(-1),
    )


class StreamingEnhancer:
    """Stateful one-hop enhancer with explicit model, analysis, and OLA state."""

    def __init__(
        self,
        model: MobileDeepFilterNet,
        stft: StreamingSTFT | None = None,
    ) -> None:
        self.model = model
        self.model.eval()
        self.stft = stft or StreamingSTFT(device=next(model.parameters()).device)
        self._hidden: Optional[torch.Tensor] = None
        self._history: list[torch.Tensor] = []
        cfg = self.stft.cfg
        self._window = self.stft.window
        self._ola = torch.zeros(cfg.n_fft, device=self.stft.device, dtype=self.stft.dtype)
        self._norm = torch.zeros_like(self._ola)
        self._tap_count = int(model.cfg.k_tap)

    @property
    def num_frames(self) -> int:
        return self.stft.num_frames

    @property
    def history_length(self) -> int:
        return len(self._history)

    def reset(self) -> None:
        self.stft.reset()
        self._hidden = None
        self._history.clear()
        self._ola.zero_()
        self._norm.zero_()

    @torch.no_grad()
    def push(self, hop_samples: torch.Tensor) -> torch.Tensor:
        cfg = self.stft.cfg
        if hop_samples.ndim != 1 or hop_samples.numel() != cfg.hop:
            raise ValueError(f"hop_samples must have shape ({cfg.hop},).")
        if not torch.is_floating_point(hop_samples) or not torch.isfinite(hop_samples).all():
            raise ValueError("hop_samples must be finite floating-point samples.")
        x_frame, _ = self.stft.push_frame(hop_samples)
        logp_ctx = self.stft.get_logp_context(self.model.temporal_receptive_frames)
        feats = logp_ctx.transpose(0, 1).unsqueeze(0).unsqueeze(0)
        mask, taps, self._hidden = self.model.forward_streaming(feats, self._hidden)

        self._history.append(x_frame)
        self._history = self._history[-self._tap_count :]
        y_frame = torch.zeros_like(x_frame)
        for tap in range(self._tap_count):
            history_index = len(self._history) - 1 - tap
            if history_index < 0:
                continue
            y_frame = y_frame + _complex_mul(taps[0, :, tap], self._history[history_index])
        y_frame = y_frame * mask[0].unsqueeze(-1)

        time_frame = torch.fft.irfft(
            torch.complex(y_frame[:, 0], y_frame[:, 1]), n=cfg.n_fft
        )
        self._ola += time_frame * self._window
        self._norm += self._window.square()
        output = self._ola[: cfg.hop] / self._norm[: cfg.hop].clamp_min(1e-8)
        self._ola = torch.cat([self._ola[cfg.hop :], torch.zeros(cfg.hop, device=self._ola.device, dtype=self._ola.dtype)])
        self._norm = torch.cat([self._norm[cfg.hop :], torch.zeros(cfg.hop, device=self._norm.device, dtype=self._norm.dtype)])
        return torch.nan_to_num(output)

    @torch.no_grad()
    def flush(self) -> torch.Tensor:
        """Emit the final synthesis hop by advancing one explicit zero frame."""
        return self.push(
            torch.zeros(
                self.stft.cfg.hop,
                device=self.stft.device,
                dtype=self.stft.dtype,
            )
        )


def _complex_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1], a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]],
        dim=-1,
    )
