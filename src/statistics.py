"""Deterministic paired statistics for research receipts."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _paired(candidate: Sequence[float], baseline: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(candidate, dtype=np.float64)
    b = np.asarray(baseline, dtype=np.float64)
    if c.ndim != 1 or b.ndim != 1 or c.size == 0 or c.size != b.size:
        raise ValueError("paired samples must be non-empty one-dimensional arrays of equal length")
    if not np.isfinite(c).all() or not np.isfinite(b).all():
        raise ValueError("paired samples must be finite")
    return c, b


def paired_bootstrap_delta(
    candidate: Sequence[float],
    baseline: Sequence[float],
    *,
    seed: int = 0,
    resamples: int = 10000,
) -> dict[str, float | int]:
    c, b = _paired(candidate, baseline)
    if resamples < 100:
        raise ValueError("resamples must be at least 100")
    delta = c - b
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=np.float64)
    chunk = 1000
    for start in range(0, resamples, chunk):
        count = min(chunk, resamples - start)
        indices = rng.integers(0, delta.size, size=(count, delta.size))
        means[start : start + count] = delta[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "n": int(delta.size),
        "seed": int(seed),
        "resamples": int(resamples),
        "mean_delta": float(delta.mean()),
        "median_delta": float(np.median(delta)),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def paired_significance(candidate: Sequence[float], baseline: Sequence[float]) -> dict[str, float | str | int]:
    c, b = _paired(candidate, baseline)
    delta = c - b
    std = float(delta.std(ddof=1)) if delta.size > 1 else 0.0
    effect = float(delta.mean() / std) if std > 0 else 0.0
    if np.all(delta == 0):
        return {"test": "wilcoxon_signed_rank", "n": int(delta.size), "statistic": 0.0, "p_value": 1.0, "cohen_dz": effect}
    try:
        from scipy.stats import wilcoxon

        result = wilcoxon(c, b, alternative="two-sided", zero_method="wilcox", method="auto")
    except ImportError as exc:
        raise RuntimeError("scipy is required for the predeclared Wilcoxon test") from exc
    return {
        "test": "wilcoxon_signed_rank",
        "n": int(delta.size),
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "cohen_dz": effect,
    }
