"""
Utilities for audio I/O, reproducibility, metrics, and deterministic manifests.

Key requirements covered:
- Load audio as mono, resample to 16 kHz
- RMS/SNR helpers (RMS-based, numerically stable)
- Soft clipping
- Seed control for deterministic validation mixing
- Manifest generation for reproducible validation mixtures
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _debug_log(message: str, data: Dict[str, Any], *, hypothesis_id: str, run_id: str) -> None:
    """Append one NDJSON debug line for this debug session (no secrets)."""
    payload = {
        "sessionId": "6913e0",
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": "src/utils.py",
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open("debug-6913e0.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Never break training/data loading due to debug logging.
        pass


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and (if available) PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        # Torch not installed in some environments; keep this utility import-safe.
        pass


def derive_item_seed(global_seed: int, epoch: int, item_index: int, worker_id: int) -> int:
    """Derive a stable per-item seed while allowing mixtures to vary by epoch."""
    sequence = np.random.SeedSequence(
        [int(global_seed), int(epoch), int(item_index), int(worker_id)]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0] % (2**31 - 1))


def compute_rms(x: np.ndarray) -> float:
    """Compute RMS with float64 accumulation for stability."""
    x64 = x.astype(np.float64)
    return float(np.sqrt(np.mean(x64 * x64) + 1e-12))


def scale_noise_to_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Scale noise to reach a target SNR relative to clean (RMS-based).

    SNR(dB) = 20*log10(rms(clean)/rms(noise_scaled))
    """
    rms_s = compute_rms(clean)
    rms_n_raw = compute_rms(noise) + 1e-12
    rms_n_target = rms_s / (10.0 ** (snr_db / 20.0))
    alpha = rms_n_target / rms_n_raw
    return noise * alpha


def soft_clip(x: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """
    Soft-clip using tanh to simulate analog saturation.

    Args:
        x: waveform in [-1,1] roughly
        threshold: larger means less saturation
    """
    threshold = float(threshold)
    return np.tanh(x / threshold) * threshold


def snr_db(clean: np.ndarray, test: np.ndarray) -> float:
    """
    Compute SNR(clean vs error) in dB:
        10*log10( P_clean / P_error )
    """
    clean64 = clean.astype(np.float64)
    test64 = test.astype(np.float64)
    err = test64 - clean64
    p_clean = float(np.mean(clean64 * clean64) + 1e-12)
    p_err = float(np.mean(err * err) + 1e-12)
    return 10.0 * math.log10(p_clean / p_err)


def list_audio_files(root: Path) -> List[str]:
    """Recursively list common audio files."""
    exts = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
    # #region agent log
    _debug_log(
        "list_audio_files: start",
        {"root": str(root), "exists": bool(root.exists()), "exts": sorted(exts)},
        hypothesis_id="H1",
        run_id="pre-fix",
    )
    # #endregion

    paths = [p for p in root.rglob("*") if p.is_file()]
    picked = [p for p in paths if p.suffix.lower() in exts]

    # Summarize extensions present to debug mismatches (e.g., dataset contains .txt/.json or nested archives).
    ext_counts: Dict[str, int] = {}
    for p in paths[:20000]:  # cap to keep it fast
        ext = p.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    # #region agent log
    _debug_log(
        "list_audio_files: result",
        {
            "total_files_scanned_cap": min(len(paths), 20000),
            "total_files_found": len(paths),
            "audio_files_found": len(picked),
            "top_exts": sorted(ext_counts.items(), key=lambda kv: kv[1], reverse=True)[:12],
            "sample_audio_paths": [str(p) for p in picked[:5]],
            "sample_any_paths": [str(p) for p in paths[:5]],
        },
        hypothesis_id="H1",
        run_id="pre-fix",
    )
    # #endregion

    return [str(p) for p in picked]


def load_audio_mono(path: str, sr: int) -> np.ndarray:
    """
    Load audio as mono and resample to target sr.

    Uses soundfile if available; falls back to librosa.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    y: np.ndarray
    orig_sr: int

    try:
        import soundfile as sf

        y, orig_sr = sf.read(str(p), dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = np.mean(y, axis=1).astype(np.float32)
    except Exception:
        import librosa

        y, orig_sr = librosa.load(str(p), sr=None, mono=True)
        y = y.astype(np.float32)

    if orig_sr != sr:
        try:
            import librosa

            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr).astype(np.float32)
        except Exception:
            # scipy fallback
            from scipy.signal import resample_poly

            g = math.gcd(int(orig_sr), int(sr))
            up = int(sr // g)
            down = int(orig_sr // g)
            y = resample_poly(y, up, down).astype(np.float32)

    return y


def crop_or_repeat(x: np.ndarray, length: int, rng: np.random.RandomState) -> np.ndarray:
    """Crop a random segment or repeat to reach exact length."""
    if x.size == length:
        return x
    if x.size > length:
        start = int(rng.randint(0, x.size - length + 1))
        return x[start : start + length]
    # Repeat wrap
    reps = int(np.ceil(length / x.size))
    y = np.tile(x, reps)[:length]
    return y


def peak_normalize(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Scale to a target peak if current peak is above 0."""
    m = float(np.max(np.abs(x)) + 1e-12)
    return x * (float(peak) / m)


@dataclass
class ValManifestEntry:
    clean_path: str
    noise_paths: List[str]
    seed: int
    segment_len: float


def save_manifest(entries: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def generate_val_manifest(
    clean_list: List[str],
    noise_list: List[str],
    out_path: Path,
    n_val: int = 500,
    seed: int = 42,
    segment_len: float = 4.0,
) -> Path:
    """
    Create a deterministic validation manifest that stores only file choices and a seed.

    Mixing itself remains on-the-fly but fully reproducible because __getitem__()
    will use the stored per-item seed.
    """
    rng = np.random.RandomState(seed)
    entries: List[Dict[str, Any]] = []
    for i in range(int(n_val)):
        clean_path = clean_list[int(rng.randint(0, len(clean_list)))]
        n_noise = int(rng.randint(1, 4))  # 1..3
        noise_paths = [noise_list[int(rng.randint(0, len(noise_list)))] for _ in range(n_noise)]
        item_seed = int(rng.randint(0, 2**31 - 1))
        entries.append(
            {
                "id": i,
                "clean_path": clean_path,
                "noise_paths": noise_paths,
                "seed": item_seed,
                "segment_len": float(segment_len),
            }
        )
    save_manifest(entries, out_path)
    return out_path
