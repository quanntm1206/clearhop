"""Checkpoint schema and metadata validation for reproducible runs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from .splits import manifest_fingerprint


SCHEMA_VERSION = 2


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a file without loading it all at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_fingerprints(paths: Mapping[str, Path]) -> dict[str, str]:
    """Return deterministic SHA-256 fingerprints for existing manifests."""
    result: dict[str, str] = {}
    for name, path in paths.items():
        resolved = Path(path)
        if resolved.exists():
            result[str(name)] = manifest_fingerprint(resolved)
    return result


def validate_checkpoint_metadata(state: Mapping[str, Any]) -> None:
    """Fail fast on checkpoints incompatible with the production contract."""
    version = int(state.get("schema_version", 0))
    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint schema_version={version}; expected {SCHEMA_VERSION}.")
    for key in ("model", "model_cfg", "audio_cfg", "config"):
        if key not in state:
            raise ValueError(f"Checkpoint missing required key: {key}")
    audio = dict(state["audio_cfg"])
    contract = {"sr": 16000, "n_fft": 320, "hop": 160, "freq_bins": 161}
    mismatch = {key: audio.get(key) for key, expected in contract.items() if audio.get(key) != expected}
    if mismatch:
        raise ValueError(f"Checkpoint audio contract mismatch: {mismatch}")
