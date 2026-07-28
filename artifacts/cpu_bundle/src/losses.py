"""Pure, differentiable calibration losses for spectral enhancement."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _validate_eps(eps: float) -> None:
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")


def _validate_complex_pair(est: torch.Tensor, ref: torch.Tensor) -> None:
    if est.ndim != 4 or ref.ndim != 4 or est.shape != ref.shape or est.shape[-1] != 2:
        raise ValueError("complex tensors must share shape (B, F, T, 2)")


def _validate_waveform_pair(est: torch.Tensor, ref: torch.Tensor) -> None:
    if est.ndim != 2 or ref.ndim != 2 or est.shape != ref.shape:
        raise ValueError("waveform tensors must share shape (B, T)")


def _require_finite(loss: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(loss).all():
        raise FloatingPointError("loss is non-finite")
    return loss


def si_sdr_loss(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Negative scale-invariant SDR, averaged over batch examples."""
    _validate_waveform_pair(est, ref)
    _validate_eps(eps)
    ref_zm = ref - ref.mean(dim=-1, keepdim=True)
    est_zm = est - est.mean(dim=-1, keepdim=True)
    target = (est_zm.mul(ref_zm).sum(dim=-1, keepdim=True) /
              ref_zm.square().sum(dim=-1, keepdim=True).add(eps)) * ref_zm
    noise = est_zm - target
    ratio = target.square().sum(dim=-1).add(eps) / noise.square().sum(dim=-1).add(eps)
    return _require_finite(-(10.0 * torch.log10(ratio)).mean())


def complex_mse(est: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Mean squared error over real/imaginary spectral channels."""
    _validate_complex_pair(est, ref)
    return _require_finite(F.mse_loss(est, ref))


def complex_nmse(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-example complex normalized MSE, then batch mean."""
    _validate_complex_pair(est, ref)
    _validate_eps(eps)
    reduce_dims = tuple(range(1, est.ndim))
    error = (est - ref).square().sum(dim=reduce_dims)
    target = ref.square().sum(dim=reduce_dims).clamp_min(eps)
    return _require_finite((error / target).mean())


def compressed_complex_loss(
    est: torch.Tensor,
    ref: torch.Tensor,
    exponent: float,
    complex_weight: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Energy-normalized magnitude and complex loss in compressed space."""
    _validate_complex_pair(est, ref)
    _validate_eps(eps)
    if not math.isfinite(exponent) or not 0.0 < exponent <= 1.0:
        raise ValueError("exponent must be finite and in (0, 1]")
    if not math.isfinite(complex_weight) or not 0.0 <= complex_weight <= 1.0:
        raise ValueError("complex_weight must be finite and in [0, 1]")

    est_magnitude = torch.sqrt(est.square().sum(dim=-1) + eps)
    ref_magnitude = torch.sqrt(ref.square().sum(dim=-1) + eps)
    est_compressed = est * est_magnitude.unsqueeze(-1).pow(exponent - 1.0)
    ref_compressed = ref * ref_magnitude.unsqueeze(-1).pow(exponent - 1.0)
    est_magnitude = est_magnitude.pow(exponent)
    ref_magnitude = ref_magnitude.pow(exponent)

    magnitude_error = (est_magnitude - ref_magnitude).square().sum(dim=(1, 2))
    magnitude_energy = ref_magnitude.square().sum(dim=(1, 2)).clamp_min(eps)
    complex_error = (est_compressed - ref_compressed).square().sum(dim=(1, 2, 3))
    complex_energy = ref_compressed.square().sum(dim=(1, 2, 3)).clamp_min(eps)
    magnitude_loss = (magnitude_error / magnitude_energy).mean()
    complex_loss = (complex_error / complex_energy).mean()
    return _require_finite((1.0 - complex_weight) * magnitude_loss + complex_weight * complex_loss)


def scheduled_weight(step: int, target: float, start: int, end: int) -> float:
    """Linearly ramp a loss weight from zero over the inclusive interval."""
    if end <= start:
        raise ValueError("end must be greater than start")
    if not math.isfinite(target):
        raise ValueError("target must be finite")
    if step <= start:
        return 0.0
    if step >= end:
        return float(target)
    return float(target) * (step - start) / (end - start)
