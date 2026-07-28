"""
Causal complex deep filtering for streaming noise reduction.

Implements:

    Y(f,t) = mask(f,t) * sum_k W(f,k,t) * X(f,t-k)

where X and W are complex-valued. We represent complex values using a real/imag
channel of size 2 for portability across runtimes (TorchScript / ONNX).
"""

from __future__ import annotations

from typing import Tuple

import torch


def stft_analysis_ri(
    waveform: torch.Tensor,
    n_fft: int,
    hop: int,
    window: torch.Tensor,
) -> torch.Tensor:
    """Causal STFT with explicit left context and right-hop completion."""
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape (B, samples).")
    if n_fft <= hop or waveform.size(-1) <= 0:
        raise ValueError("Expected n_fft > hop and a non-empty waveform.")
    left = n_fft - hop
    frames = (waveform.size(-1) + hop - 1) // hop
    right = left + frames * hop - waveform.size(-1)
    padded = torch.nn.functional.pad(waveform, (left, right))
    spectrum = torch.stft(
        padded,
        n_fft=n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=False,
        return_complex=True,
    )
    return complex_to_stft_ri(spectrum)


def stft_synthesis(
    frames_ri: torch.Tensor,
    n_fft: int,
    hop: int,
    window: torch.Tensor,
    length: int,
) -> torch.Tensor:
    """Synthesize a causal STFT and remove its explicit analysis latency."""
    if length <= 0:
        raise ValueError("length must be positive.")
    padded = istft_overlap_add(frames_ri, n_fft, hop, window)
    left = n_fft - hop
    end = left + length
    if padded.size(-1) < end:
        padded = torch.nn.functional.pad(padded, (0, end - padded.size(-1)))
    return padded[..., left:end]


def complex_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Complex multiplication using real/imag channel.

    Args:
        a: (..., 2) real/imag
        b: (..., 2) real/imag

    Returns:
        (..., 2) real/imag
    """
    ar, ai = a[..., 0], a[..., 1]
    br, bi = b[..., 0], b[..., 1]
    real = ar * br - ai * bi
    imag = ar * bi + ai * br
    return torch.stack([real, imag], dim=-1)


def causal_deep_filter(
    x_stft: torch.Tensor,
    mask: torch.Tensor,
    w_taps: torch.Tensor,
) -> torch.Tensor:
    """
    Apply causal complex deep filtering to a streaming STFT sequence.

    Shapes:
        x_stft: (B, F, T, 2)
        mask:   (B, F, T)           in [0,1]
        w_taps: (B, F, K, T, 2)     complex taps for each time

    Returns:
        y_stft: (B, F, T, 2)
    """
    if x_stft.ndim != 4 or x_stft.size(-1) != 2:
        raise ValueError("x_stft must have shape (B, F, T, 2).")
    if w_taps.ndim != 5 or w_taps.size(-1) != 2:
        raise ValueError("w_taps must have shape (B, F, K, T, 2).")
    if mask.ndim != 3:
        raise ValueError("mask must have shape (B, F, T).")

    b, f, t, _ = x_stft.shape
    _, _, k, t2, _ = w_taps.shape
    if t2 != t:
        raise ValueError("w_taps time dimension must match x_stft.")

    # Compute filtered STFT:
    # y(f,t) = sum_{k=0..K-1} w(f,k,t) * x(f,t-k)   with x(f, t-k)=0 for t-k < 0.
    y = torch.zeros_like(x_stft)
    for tap in range(k):
        # shift right by tap along time axis: x[..., :t-tap] aligns to y[..., tap:].
        if tap == 0:
            x_shift = x_stft
        else:
            x_shift = torch.zeros_like(x_stft)
            x_shift[:, :, tap:, :] = x_stft[:, :, : t - tap, :]

        y = y + complex_mul(w_taps[:, :, tap, :, :], x_shift)

    y = y * mask.unsqueeze(-1)
    return y


def stft_ri_to_complex(x_ri: torch.Tensor) -> torch.Tensor:
    """Convert (..., 2) real/imag to torch.complex."""
    return torch.complex(x_ri[..., 0], x_ri[..., 1])


def complex_to_stft_ri(x_c: torch.Tensor) -> torch.Tensor:
    """Convert complex tensor to (..., 2) real/imag."""
    return torch.stack([x_c.real, x_c.imag], dim=-1)


def istft_overlap_add(
    y_frames_ri: torch.Tensor,
    n_fft: int,
    hop: int,
    window: torch.Tensor,
    length: int | None = None,
) -> torch.Tensor:
    """
    Convenience helper for inverse STFT (batched) from stacked frames.

    Args:
        y_frames_ri: (B, F, T, 2)
        n_fft: FFT size (320).
        hop: Hop size (160).
        window: (n_fft,) window tensor on same device/dtype as y_frames_ri.
        length: Optional output length.

    Returns:
        waveform: (B, N)
    """
    if y_frames_ri.ndim != 4 or y_frames_ri.size(-1) != 2:
        raise ValueError("y_frames_ri must have shape (B, F, T, 2).")
    if window.numel() != n_fft:
        raise ValueError("window length must equal n_fft.")

    y_complex = stft_ri_to_complex(y_frames_ri)  # (B, F, T)
    b, _, frames = y_complex.shape
    time_frames = torch.fft.irfft(y_complex, n=n_fft, dim=1)
    output_len = n_fft + max(0, frames - 1) * hop
    output = torch.zeros((b, output_len), device=y_frames_ri.device, dtype=y_frames_ri.dtype)
    norm = torch.zeros_like(output)
    window = window.to(device=y_frames_ri.device, dtype=y_frames_ri.dtype)
    window_sq = window * window
    for frame_index in range(frames):
        start = frame_index * hop
        frame = time_frames[:, :, frame_index] * window
        output[:, start : start + n_fft] += frame
        norm[:, start : start + n_fft] += window_sq
    output = torch.where(
        norm > 0,
        output / norm.clamp_min(torch.finfo(output.dtype).tiny),
        torch.zeros_like(output),
    )
    if length is None:
        return output
    if length < output.size(-1):
        return output[..., :length]
    if length > output.size(-1):
        return torch.nn.functional.pad(output, (0, length - output.size(-1)))
    return output
