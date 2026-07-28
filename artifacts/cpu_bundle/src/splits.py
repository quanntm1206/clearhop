"""Deterministic speaker-aware dataset splitting and portable manifests."""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class FoldSplit:
    train: tuple[Path, ...]
    val: tuple[Path, ...]
    test: tuple[Path, ...]


def infer_speaker_id(path: Path) -> str:
    """Infer a stable corpus-qualified speaker ID from supported file names."""
    stem = Path(path).stem
    match = re.match(r"^(\d+)-\d+-\d+$", stem)
    if match:
        return f"libri:{match.group(1)}"
    match = re.match(r"^(VIVOSSPK\d+)_", stem, flags=re.IGNORECASE)
    if match:
        return f"vivos:{match.group(1).upper()}"
    match = re.match(r"^vctk_(p\d+)_", stem, flags=re.IGNORECASE)
    if match:
        return f"vctk:{match.group(1).lower()}"
    match = re.match(r"^(p\d+)_", stem, flags=re.IGNORECASE)
    if match:
        return f"vctk:{match.group(1).lower()}"
    parent = Path(path).parent.name or "root"
    prefix = re.split(r"[_-]", stem, maxsplit=1)[0]
    return f"unknown:{parent}:{prefix}"


def build_grouped_splits(
    paths: Sequence[Path],
    n_folds: int = 5,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> list[FoldSplit]:
    """Build folds with mutually exclusive speaker groups."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    grouped: dict[str, list[Path]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        grouped.setdefault(infer_speaker_id(path), []).append(path)
    groups = sorted(grouped)
    if len(groups) < n_folds + 1:
        raise ValueError(
            f"Need at least {n_folds + 1} speaker groups, found {len(groups)}."
        )

    random.Random(seed).shuffle(groups)
    folds: list[FoldSplit] = []
    for fold_index in range(n_folds):
        test_groups = set(groups[fold_index::n_folds])
        remaining = [group for group in groups if group not in test_groups]
        random.Random(seed + 1009 * (fold_index + 1)).shuffle(remaining)
        val_count = max(1, int(round(len(remaining) * val_fraction)))
        val_count = min(val_count, len(remaining) - 1)
        val_groups = set(remaining[:val_count])
        train_groups = set(remaining[val_count:])

        def collect(selected: set[str]) -> tuple[Path, ...]:
            return tuple(
                sorted(
                    (path for group in selected for path in grouped[group]),
                    key=lambda item: item.as_posix(),
                )
            )

        folds.append(
            FoldSplit(
                train=collect(train_groups),
                val=collect(val_groups),
                test=collect(test_groups),
            )
        )
    return folds


def build_manifest_entries(
    clean_paths: Sequence[Path],
    noise_paths: Sequence[Path],
    seed: int,
    segment_len: float,
) -> list[dict[str, Any]]:
    """Assign deterministic validation/test noises to clean utterances."""
    if not noise_paths:
        raise ValueError("noise_paths is empty.")
    noises = [Path(path) for path in noise_paths]
    rng = random.Random(seed)
    entries: list[dict[str, Any]] = []
    for index, raw_clean_path in enumerate(clean_paths):
        count = rng.randint(1, min(3, len(noises)))
        entries.append(
            {
                "id": index,
                "clean_path": Path(raw_clean_path),
                "noise_paths": rng.sample(noises, count),
                "seed": rng.randrange(0, 2**31 - 1),
                "segment_len": float(segment_len),
            }
        )
    return entries


def _portable_path(path: Path, project_root: Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(Path(project_root).resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Manifest path is outside project root: {resolved}") from exc


def write_manifest(
    entries: Iterable[Mapping[str, Any]],
    path: Path,
    project_root: Path,
) -> str:
    """Write canonical relative-path JSONL and return its SHA-256 fingerprint."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = Path(project_root).resolve()
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for entry in entries:
            row = dict(entry)
            row["clean_path"] = _portable_path(Path(row["clean_path"]), root)
            if "noise_paths" in row:
                row["noise_paths"] = [
                    _portable_path(Path(noise_path), root)
                    for noise_path in row["noise_paths"]
                ]
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
    return manifest_fingerprint(output_path)


def manifest_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slice_fingerprint(path: Path, offset: int, count: int) -> str:
    """Hash one non-empty, ordered manifest slice without truncating it."""
    if offset < 0:
        raise ValueError("offset must be non-negative.")
    if count <= 0:
        raise ValueError("count must be positive.")

    with Path(path).open("r", encoding="utf-8") as handle:
        rows = [line for line in handle if line.strip()]
    end = offset + count
    if end > len(rows):
        raise ValueError(f"Slice [{offset}, {end}) exceeds manifest length {len(rows)}.")
    return hashlib.sha256("".join(rows[offset:end]).encode("utf-8")).hexdigest()


def manifest_slice_fingerprint(path: Path, *, offset: int, count: int) -> str:
    """Backward-compatible named wrapper for :func:`slice_fingerprint`."""
    return slice_fingerprint(path, offset, count)
