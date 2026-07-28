"""
Realistic noise mixing + augmentation helpers for noise suppression training.

Implements the exact mixing strategy requested:
- Segment length default 4.0s at 16 kHz
- SNR triangular sampling biased to low SNRs: triangular(-10, 0, 20)
- Optional time-varying SNR (p_vary=0.4) by splitting into 2..6 subsegments
  and sampling snr_db from triangular(-10, 0, 10) per subsegment
- Multi-noise mixing: 1..3 noise files
- Optional RIR on clean before mixing (p_rir=0.35)
- Device effects: bandlimiting, hum, clipping/saturation, small time/pitch perturbation

Notes:
- This module is pure NumPy/SciPy-oriented for speed in workers.
- Randomness is controlled externally via a passed-in RNG seed in metadata.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .utils import compute_rms, crop_or_repeat, load_audio_mono, scale_noise_to_snr, soft_clip


def _butter_bandpass(
    x: np.ndarray,
    sr: int,
    low_hz: Optional[float],
    high_hz: Optional[float],
) -> np.ndarray:
    """Simple SOS bandpass/bandlimit filter."""
    from scipy.signal import butter, sosfilt

    nyq = 0.5 * sr
    lo = None if low_hz is None else max(5.0, float(low_hz) / nyq)
    hi = None if high_hz is None else min(0.999, float(high_hz) / nyq)

    if lo is None and hi is None:
        return x
    if lo is None:
        sos = butter(4, hi, btype="lowpass", output="sos")
    elif hi is None:
        sos = butter(4, lo, btype="highpass", output="sos")
    else:
        if lo >= hi:
            return x
        sos = butter(4, [lo, hi], btype="bandpass", output="sos")
    return sosfilt(sos, x).astype(np.float32)


def _add_hum(x: np.ndarray, sr: int, rng: np.random.RandomState) -> Tuple[np.ndarray, Dict]:
    """Add a weak 50/60 Hz hum with random amplitude."""
    if rng.rand() > 0.25:
        return x, {"hum": None}
    freq = 50.0 if rng.rand() < 0.5 else 60.0
    amp = float(rng.uniform(1e-4, 5e-3))
    t = np.arange(x.size, dtype=np.float32) / float(sr)
    hum = amp * np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    return (x + hum).astype(np.float32), {"hum": {"freq": freq, "amp": amp}}


def _maybe_time_or_pitch_perturb(noise: np.ndarray, sr: int, rng: np.random.RandomState) -> Tuple[np.ndarray, Dict]:
    """
    Apply small random time-stretch or pitch shift to noise.

    - time-stretch rate in [0.95, 1.05]
    - pitch shift +/- 1 semitone (very small)
    """
    meta: Dict = {"time_stretch": None, "pitch_shift": None}
    if rng.rand() > 0.35:
        return noise, meta

    # Prefer time-stretch (robust). If librosa unavailable, skip.
    try:
        import librosa

        if rng.rand() < 0.7:
            rate = float(rng.uniform(0.95, 1.05))
            y = librosa.effects.time_stretch(noise.astype(np.float32), rate=rate)
            meta["time_stretch"] = rate
        else:
            n_steps = float(rng.uniform(-1.0, 1.0))
            y = librosa.effects.pitch_shift(noise.astype(np.float32), sr=sr, n_steps=n_steps)
            meta["pitch_shift"] = n_steps
        y = y.astype(np.float32)
        return y, meta
    except Exception:
        return noise, meta


def _convolve_rir(clean: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """FFT convolve and normalize to avoid energy blow-up."""
    from scipy.signal import fftconvolve

    y = fftconvolve(clean.astype(np.float32), rir.astype(np.float32), mode="full")[: clean.size]
    # Normalize by energy ratio to keep scale stable.
    rms_in = compute_rms(clean)
    rms_out = compute_rms(y)
    if rms_out > 0:
        y = y * (rms_in / (rms_out + 1e-12))
    return y.astype(np.float32)


def _generate_synthetic_rir(sr: int, rng: np.random.RandomState) -> np.ndarray:
    """
    Fallback synthetic RIR generator: exponentially decaying noise burst.
    This is used only when pyroomacoustics and local RIR dataset are absent.
    """
    t60 = float(rng.uniform(0.08, 0.8))
    length = int(min(sr * 1.0, max(sr * 0.2, sr * (t60 * 1.2))))
    t = np.arange(length, dtype=np.float32) / float(sr)
    decay = np.exp(-6.91 * t / max(t60, 1e-3)).astype(np.float32)
    rir = (rng.randn(length).astype(np.float32) * decay).astype(np.float32)
    rir[0] += 1.0  # direct path
    # Normalize peak
    rir = rir / (np.max(np.abs(rir)) + 1e-12)
    return rir.astype(np.float32)


def load_or_generate_rir(project_root: Path, sr: int, rng: np.random.RandomState) -> Tuple[np.ndarray, Dict]:
    """
    Prefer pyroomacoustics synthesis if available; else load from data/rirs; else synthetic.
    """
    # Try pyroomacoustics shoebox synthesis.
    try:
        import pyroomacoustics as pra  # type: ignore

        room_dim = rng.uniform([3.0, 3.0, 2.3], [9.0, 7.0, 3.2]).tolist()
        t60 = float(rng.uniform(0.08, 0.8))
        abs_coeff, max_order = pra.inverse_sabine(t60, room_dim)
        room = pra.ShoeBox(
            room_dim,
            fs=sr,
            materials=pra.Material(abs_coeff),
            max_order=max_order,
        )
        mic = rng.uniform([0.5, 0.5, 1.0], [room_dim[0] - 0.5, room_dim[1] - 0.5, 1.6]).tolist()
        src = rng.uniform([0.5, 0.5, 1.0], [room_dim[0] - 0.5, room_dim[1] - 0.5, 1.8]).tolist()
        room.add_microphone_array(np.array(mic, dtype=np.float32).reshape(3, 1))
        room.add_source(np.array(src, dtype=np.float32))
        room.compute_rir()
        rir = np.asarray(room.rir[0][0], dtype=np.float32)
        rir = rir / (np.max(np.abs(rir)) + 1e-12)
        return rir, {"rir_source": "pyroomacoustics", "t60": t60}
    except Exception:
        pass

    rir_root = project_root / "data" / "rirs"
    if rir_root.exists():
        files = [p for p in rir_root.rglob("*") if p.suffix.lower() in {".wav", ".flac"}]
        if files:
            pick = files[int(rng.randint(0, len(files)))]
            rir = load_audio_mono(str(pick), sr=sr)
            rir = rir.astype(np.float32)
            # Keep only early part
            rir = rir[: int(sr * 1.0)]
            rir = rir / (np.max(np.abs(rir)) + 1e-12)
            return rir, {"rir_source": "dataset", "path": str(pick)}

    rir = _generate_synthetic_rir(sr, rng)
    return rir, {"rir_source": "synthetic"}


def _apply_device_effects(x: np.ndarray, sr: int, rng: np.random.RandomState) -> Tuple[np.ndarray, Dict]:
    meta: Dict = {"bandlimit": None, "clip": None, "soft_clip": None}

    # Random bandlimiting / device bandwidth simulation.
    if rng.rand() < 0.35:
        low = None
        high = float(rng.uniform(3000.0, 7800.0))
        if rng.rand() < 0.2:
            low = float(rng.uniform(80.0, 200.0))
        x = _butter_bandpass(x, sr=sr, low_hz=low, high_hz=high)
        meta["bandlimit"] = {"low_hz": low, "high_hz": high}

    # Optional hard clipping then light lowpass (anti-aliasing feel).
    if rng.rand() < 0.12:
        thr = float(rng.uniform(0.85, 0.95))
        x = np.clip(x, -thr, thr).astype(np.float32)
        x = _butter_bandpass(x, sr=sr, low_hz=None, high_hz=7800.0)
        meta["clip"] = {"threshold": thr}

    # Optional soft saturation.
    if rng.rand() < 0.25:
        thr = float(rng.uniform(0.90, 0.99))
        x = soft_clip(x, threshold=thr).astype(np.float32)
        meta["soft_clip"] = {"threshold": thr}

    return x.astype(np.float32), meta


def _sample_snr_db(rng: np.random.RandomState, snr_config: Dict, time_varying: bool) -> float:
    # Allow tests to override fixed SNR deterministically.
    if "snr_db" in snr_config and snr_config["snr_db"] is not None and not time_varying:
        return float(snr_config["snr_db"])
    configured = snr_config.get("snr_range", (-10.0, 20.0))
    low, high = (float(configured[0]), float(configured[1]))
    if low > high:
        raise ValueError("snr_range minimum must not exceed maximum.")
    if low == high:
        return low
    mode = float((low + high) / 2.0)
    if time_varying:
        mode = float((low + mode) / 2.0)
    return float(rng.triangular(low, mode, high))


def mix_clean_with_noises(
    clean: np.ndarray,
    noise_paths: List[str],
    sr: int,
    segment_len: float,
    snr_config: Dict,
    rir: Optional[np.ndarray] = None,
    time_varying: bool = False,
    multi_noise: Tuple[int, int] = (1, 3),
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Mix clean speech with 1..3 noises using RMS-based SNR scaling.

    Args:
        clean: clean waveform (float32), arbitrary length
        noise_paths: list of noise file paths to sample from
        sr: sampling rate (16 kHz)
        segment_len: seconds (default 4.0)
        snr_config: dict controlling SNR sampling and overrides
        rir: optional RIR to convolve clean with before mixing
        time_varying: if True, split into subsegments and vary SNR per segment
        multi_noise: (min,max) noises to mix (uniform)

    Returns:
        clean_target: processed clean waveform used to construct the mixture
        noisy: mixed waveform (float32, length=segment_len*sr)
        meta: metadata with SNRs, noise files, augmentation decisions
    """
    project_root = Path(snr_config.get("project_root", Path.cwd()))
    rng_seed = int(snr_config.get("seed", 0))
    rng = np.random.RandomState(rng_seed)

    length = int(round(float(segment_len) * float(sr)))
    clean_seg = crop_or_repeat(clean.astype(np.float32), length, rng)

    meta: Dict = {
        "seed": rng_seed,
        "noise_files": [],
        "snr_db": None,
        "snr_db_segments": None,
        "rir": None,
        "aug": {},
    }

    # Optional RIR on clean BEFORE mixing.
    if rir is not None:
        clean_proc = _convolve_rir(clean_seg, rir)
        meta["rir"] = {"provided": True, "len": int(rir.size)}
    else:
        clean_proc = clean_seg

    # Choose how many noises to mix.
    n_min, n_max = int(multi_noise[0]), int(multi_noise[1])
    n_pick = int(rng.randint(n_min, n_max + 1))

    chosen = []
    for _ in range(n_pick):
        chosen.append(noise_paths[int(rng.randint(0, len(noise_paths)))])
    meta["noise_files"] = list(chosen)

    # Load and preprocess noises, sum them.
    noise_sum = np.zeros_like(clean_proc, dtype=np.float32)
    per_noise_meta: List[Dict] = []
    for p in chosen:
        n = load_audio_mono(p, sr=sr)
        n, perturb_meta = _maybe_time_or_pitch_perturb(n, sr=sr, rng=rng)
        n = crop_or_repeat(n.astype(np.float32), length, rng)
        # Optional per-noise bandpass to simulate device capture differences.
        if rng.rand() < 0.35:
            low = None
            high = float(rng.uniform(2500.0, 7800.0))
            if rng.rand() < 0.15:
                low = float(rng.uniform(60.0, 200.0))
            n = _butter_bandpass(n, sr=sr, low_hz=low, high_hz=high)
            perturb_meta["bandlimit"] = {"low_hz": low, "high_hz": high}
        # Per-file amplitude scaling.
        gain_db = float(rng.uniform(-6.0, 6.0))
        n = n * (10.0 ** (gain_db / 20.0))
        perturb_meta["gain_db"] = gain_db
        per_noise_meta.append(perturb_meta)
        noise_sum = (noise_sum + n.astype(np.float32)).astype(np.float32)

    meta["aug"]["per_noise"] = per_noise_meta

    # Time-varying SNR scaling by subsegments.
    if time_varying:
        # Tests may optionally provide an explicit pattern.
        forced_list = snr_config.get("vary_snr_db_list")
        forced_bounds = snr_config.get("vary_boundaries")

        if forced_list is not None and forced_bounds is not None:
            snr_list = [float(v) for v in forced_list]
            bounds = [int(b) for b in forced_bounds]
        else:
            n_seg = int(rng.randint(2, 7))  # 2..6
            # Random boundaries with at least ~0.3s segments.
            min_len = int(0.30 * sr)
            cut_points = sorted(rng.randint(min_len, length - min_len, size=n_seg - 1).tolist())
            bounds = [0] + cut_points + [length]
            snr_list = [float(_sample_snr_db(rng, snr_config, time_varying=True)) for _ in range(n_seg)]

        noise_scaled = np.zeros_like(noise_sum, dtype=np.float32)
        for i in range(len(snr_list)):
            a, b = bounds[i], bounds[i + 1]
            s_sub = clean_proc[a:b]
            n_sub = noise_sum[a:b]
            n_sub_scaled = scale_noise_to_snr(s_sub, n_sub, snr_list[i]).astype(np.float32)
            noise_scaled[a:b] = n_sub_scaled

        meta["snr_db_segments"] = {"bounds": bounds, "snr_db": snr_list}
        # Overall SNR is not a single number; store mean for convenience.
        meta["snr_db"] = float(np.mean(snr_list))
        noisy = (clean_proc + noise_scaled).astype(np.float32)
    else:
        snr_db = float(_sample_snr_db(rng, snr_config, time_varying=False))
        meta["snr_db"] = snr_db
        noise_scaled = scale_noise_to_snr(clean_proc, noise_sum, snr_db).astype(np.float32)
        noisy = (clean_proc + noise_scaled).astype(np.float32)

    # Additional augmentations on the mixed waveform.
    noisy, hum_meta = _add_hum(noisy, sr=sr, rng=rng)
    meta["aug"]["hum"] = hum_meta["hum"]

    noisy, dev_meta = _apply_device_effects(noisy, sr=sr, rng=rng)
    meta["aug"]["device"] = dev_meta

    # Final gain +/- 3 dB and peak normalize to avoid clipping.
    final_gain_db = float(rng.uniform(-3.0, 3.0))
    noisy = (noisy * (10.0 ** (final_gain_db / 20.0))).astype(np.float32)
    clean_proc = (clean_proc * (10.0 ** (final_gain_db / 20.0))).astype(np.float32)
    meta["aug"]["final_gain_db"] = final_gain_db

    # Apply one shared peak scale so the supervised pair and requested SNR stay aligned.
    if rng.rand() < 0.9:
        pk = float(rng.uniform(0.90, 0.99))
        current_peak = float(np.max(np.abs(noisy)) + 1e-12)
        pair_scale = pk / current_peak
        noisy = (noisy * pair_scale).astype(np.float32)
        clean_proc = (clean_proc * pair_scale).astype(np.float32)
        meta["aug"]["peak_norm"] = pk

    if rng.rand() < 0.25:
        thr = float(rng.uniform(0.90, 0.98))
        noisy = soft_clip(noisy, threshold=thr).astype(np.float32)
        meta["aug"]["soft_clip_post"] = thr

    meta["clean_rms"] = compute_rms(clean_proc)
    meta["noisy_rms"] = compute_rms(noisy)
    return clean_proc.astype(np.float32), noisy.astype(np.float32), meta
