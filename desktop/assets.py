"""Pinned desktop asset manifest and SHA-256 verification helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


class AssetHashMismatchError(ValueError):
    """Raised when an asset differs from its pinned manifest digest."""


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset(path: str | Path, expected_sha256: str) -> str:
    """Verify and return a file digest. Never silently activate a bad asset."""
    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("expected_sha256 must be a 64-character hexadecimal digest")
    actual = sha256_file(path)
    if actual != expected:
        raise AssetHashMismatchError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def load_asset_manifest(path: str | Path) -> dict[str, Any]:
    """Load a desktop asset manifest and validate its security-critical shape."""
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid asset manifest: {manifest_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("asset manifest must be a JSON object")
    assets = payload.get("assets", [])
    if isinstance(assets, dict):
        assets = [dict(value, name=name) if isinstance(value, dict) else value for name, value in assets.items()]
        payload["assets"] = assets
    if not isinstance(assets, list):
        raise ValueError("asset manifest assets must be a list or object")
    names: set[str] = set()
    for asset in assets:
        if not isinstance(asset, dict) or not asset.get("name"):
            raise ValueError("each asset requires a name")
        name = str(asset["name"])
        if name in names:
            raise ValueError(f"duplicate asset name: {name}")
        names.add(name)
        digest = str(asset.get("sha256", "")).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"asset {name} requires a valid sha256")
        if not (asset.get("url") or asset.get("path") or asset.get("filename")):
            raise ValueError(f"asset {name} requires url, path, or filename")
    payload.setdefault("schema_version", 1)
    return payload


def manifest_asset(manifest: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Return one normalized asset entry by name."""
    for asset in manifest.get("assets", []):
        if isinstance(asset, Mapping) and str(asset.get("name")) == name:
            return dict(asset)
    raise KeyError(name)


def verify_manifest_asset(manifest: Mapping[str, Any], name: str, path: str | Path) -> str:
    asset = manifest_asset(manifest, name)
    return verify_asset(path, str(asset["sha256"]))
