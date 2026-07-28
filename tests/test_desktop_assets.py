from pathlib import Path
import json
import hashlib

import pytest

from desktop.assets import AssetHashMismatchError, load_asset_manifest, verify_asset


def test_manifest_load_and_hash_validation(tmp_path: Path):
    payload = b"model"
    asset = tmp_path / "model.bin"
    asset.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    manifest = tmp_path / "assets.json"
    manifest.write_text(json.dumps({"schema_version": 1, "version": "1.0", "assets": [{"name": "model", "path": "model.bin", "sha256": digest}]}))

    loaded = load_asset_manifest(manifest)
    assert loaded["version"] == "1.0"
    assert verify_asset(asset, digest) == digest

    with pytest.raises(AssetHashMismatchError):
        verify_asset(asset, "0" * 64)


def test_manifest_rejects_missing_or_invalid_hash(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"assets": [{"name": "x", "sha256": "not-a-hash"}]}))
    with pytest.raises(ValueError):
        load_asset_manifest(p)
