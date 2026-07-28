"""Metric-complete evaluation for held-out paired audio batches."""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import numpy as np
import torch

from .filtering import causal_deep_filter, stft_analysis_ri, stft_synthesis
from .model import MobileDeepFilterNet
from .utils import snr_db


_EVALUATION_PROFILES = {
    "full": (("noisy", "mask_only", "enhanced"), True, True),
    "screen": (("noisy", "enhanced"), True, False),
}


def evaluation_profile_options(profile: str) -> tuple[tuple[str, ...], bool, bool]:
    """Return outputs, STOI, and PESQ switches for a locked evaluation profile."""
    try:
        return _EVALUATION_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"Unknown evaluation profile: {profile}") from exc


def si_sdr(reference: np.ndarray, estimate: np.ndarray, eps: float = 1e-8) -> float:
    ref = reference.astype(np.float64) - float(np.mean(reference))
    est = estimate.astype(np.float64) - float(np.mean(estimate))
    target = np.dot(est, ref) / (np.dot(ref, ref) + eps) * ref
    error = est - target
    return float(10.0 * math.log10((np.dot(target, target) + eps) / (np.dot(error, error) + eps)))


def _optional_stoi(clean: np.ndarray, estimate: np.ndarray, sr: int) -> Optional[float]:
    try:
        from pystoi.stoi import stoi  # type: ignore

        return float(stoi(clean, estimate, sr, extended=False))
    except Exception:
        return None


def _optional_pesq(clean: np.ndarray, estimate: np.ndarray, sr: int) -> Optional[float]:
    """Return PESQ when the optional package accepts the audio contract."""
    try:
        from pesq import pesq  # type: ignore

        mode = "wb" if sr == 16000 else "nb" if sr == 8000 else None
        if mode is None:
            return None
        return float(pesq(sr, clean, estimate, mode))
    except Exception:
        return None


def calibration_metrics(clean: np.ndarray, output: np.ndarray, eps: float = 1e-12) -> dict[str, float]:
    """Measure output gain and polarity relative to the clean reference."""
    gain = float(np.dot(output, clean) / max(float(np.dot(clean, clean)), eps))
    return {
        "projection_gain": gain,
        "gain_error_db": float(20.0 * np.log10(max(abs(gain), eps))),
        "polarity_failure": float(gain <= 0.0),
    }


def _aggregate(rows: list[dict[str, float]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0}
    keys = sorted(rows[0])
    result: dict[str, Any] = {"n": len(rows)}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


@torch.no_grad()
def evaluate_model(
    model: MobileDeepFilterNet,
    loader: Iterable[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    sr: int = 16000,
    max_items: int | None = None,
    profile: str = "full",
    return_items: bool = False,
) -> dict[str, Any]:
    """Evaluate the outputs and perceptual metrics locked by ``profile``."""
    output_names, compute_stoi, compute_pesq = evaluation_profile_options(profile)
    model.eval()
    window = torch.hann_window(320, device=device)
    rows: dict[str, list[dict[str, float]]] = {name: [] for name in output_names}
    item_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in output_names}
    seen = 0
    for batch in loader:
        clean = batch["clean"].to(device)
        noisy = batch["noisy"].to(device)
        spec_ri = stft_analysis_ri(noisy, 320, 160, window)
        feats = torch.log(spec_ri[..., 0].square() + spec_ri[..., 1].square() + 1e-12).unsqueeze(1)
        mask, taps, _ = model(feats, None)
        enhanced_ri = causal_deep_filter(spec_ri, mask, taps)
        enhanced = stft_synthesis(enhanced_ri, 320, 160, window, length=clean.size(-1))
        mask_wave = None
        if "mask_only" in output_names:
            mask_ri = spec_ri * mask.unsqueeze(-1)
            mask_wave = stft_synthesis(mask_ri, 320, 160, window, length=clean.size(-1))

        for index in range(clean.size(0)):
            item_index = seen
            batch_meta = batch.get("meta", [])
            item_meta = batch_meta[index] if isinstance(batch_meta, list) and index < len(batch_meta) and isinstance(batch_meta[index], dict) else {}
            snr_value = item_meta.get("snr_db")
            snr_band = None
            if isinstance(snr_value, (int, float)):
                snr_band = "low" if float(snr_value) < 0 else "mid" if float(snr_value) < 10 else "high"
            noise_paths = item_meta.get("noise_paths") if isinstance(item_meta.get("noise_paths"), list) else item_meta.get("noise_files", [])
            first_noise = str(noise_paths[0]) if noise_paths else "unknown"
            noise_name = first_noise.replace("\\", "/").rsplit("/", 1)[-1]
            noise_family = "esc50" if noise_name.lower().startswith("esc50_") else "other"
            clean_name = str(item_meta.get("clean_path", "")).replace("\\", "/").rsplit("/", 1)[-1]
            speaker = clean_name.split("-", 1)[0] if "-" in clean_name else "unknown"
            clean_np = clean[index].detach().cpu().numpy()
            noisy_np = noisy[index].detach().cpu().numpy()
            available_outputs = {"noisy": noisy_np, "enhanced": enhanced[index].detach().cpu().numpy()}
            if mask_wave is not None:
                available_outputs["mask_only"] = mask_wave[index].detach().cpu().numpy()
            outputs = {name: available_outputs[name] for name in output_names}
            for name, output in outputs.items():
                row = {
                    "si_sdr": si_sdr(clean_np, output),
                    "snr": snr_db(clean_np, output),
                    **calibration_metrics(clean_np, output),
                }
                if compute_stoi:
                    stoi_value = _optional_stoi(clean_np, output, sr)
                    if stoi_value is not None:
                        row["stoi"] = stoi_value
                if compute_pesq:
                    pesq_value = _optional_pesq(clean_np, output, sr)
                    if pesq_value is not None:
                        row["pesq"] = pesq_value
                rows[name].append(row)
                if return_items:
                    item_rows[name].append({
                        "index": int(item_meta.get("manifest_id", item_index)),
                        "metadata": {"snr_db": snr_value, "snr_band": snr_band, "noise_family": noise_family, "speaker": speaker},
                        **row,
                    })
            seen += 1
            if max_items is not None and seen >= max_items:
                break
        if max_items is not None and seen >= max_items:
            break

    result = {name: _aggregate(values) for name, values in rows.items()}
    for name in ("mask_only", "enhanced"):
        if name not in result:
            continue
        if result[name].get("n", 0) and result["noisy"].get("n", 0):
            result[name]["si_sdr_improvement_mean"] = result[name]["si_sdr"]["mean"] - result["noisy"]["si_sdr"]["mean"]
            result[name]["snr_improvement_mean"] = result[name]["snr"]["mean"] - result["noisy"]["snr"]["mean"]
    if return_items:
        result["items"] = item_rows
    return result
