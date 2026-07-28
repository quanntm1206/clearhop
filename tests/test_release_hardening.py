from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.validate_release import validate_release


def _project(root: Path, *, version: str = "1.2.3") -> None:
    (root / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n', encoding="utf-8")
    (root / "CITATION.cff").write_text(f'version: "{version}"\n', encoding="utf-8")
    assets = root / "assets"
    assets.mkdir()
    import hashlib

    rows = []
    for name in ("checkpoint.pth", "model.onnx"):
        payload = assets / name
        payload.write_bytes(name.encode("ascii"))
        digest = hashlib.sha256(payload.read_bytes()).hexdigest()
        rows.append({"name": name, "offline_path": f"assets/{name}", "sha256": digest, "url": f"https://github.com/quanntm1206/clearhop/releases/download/v{version}/{name}"})
    manifest = {"schema_version": 1, "release_version": version, "assets": rows}
    config = root / "configs"
    config.mkdir()
    (config / "desktop_assets.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_release_metadata_and_assets_are_bound(tmp_path: Path) -> None:
    _project(tmp_path)
    result = validate_release(tmp_path, "v1.2.3")
    assert result["status"] == "pass"
    assert all(result["checks"].values())


@pytest.mark.parametrize("tag", ["v1.2.4", "main", ""])
def test_release_rejects_tag_or_version_drift(tmp_path: Path, tag: str) -> None:
    _project(tmp_path)
    assert validate_release(tmp_path, tag)["status"] == "fail"


def test_release_rejects_asset_hash_mismatch(tmp_path: Path) -> None:
    _project(tmp_path)
    (tmp_path / "assets/model.onnx").write_bytes(b"tampered")
    result = validate_release(tmp_path, "v1.2.3")
    assert result["checks"]["asset_hashes"] is False


def test_release_rejects_incomplete_duplicate_or_unofficial_assets(tmp_path: Path) -> None:
    _project(tmp_path)
    path = tmp_path / "configs/desktop_assets.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["assets"] = [manifest["assets"][0], dict(manifest["assets"][0])]
    manifest["assets"][0]["url"] = "https://attacker.invalid/releases/download/v1.2.3/checkpoint.pth"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_release(tmp_path, "v1.2.3")

    assert result["checks"]["asset_inventory"] is False
    assert result["checks"]["asset_urls"] is False


def test_packaged_smoke_is_real_denoise_not_launch_only() -> None:
    script = (Path(__file__).resolve().parents[1] / "scripts/smoke-packaged-desktop.ps1").read_text(encoding="utf-8")
    for token in ("--smoke-input", "--smoke-output", "--smoke-receipt", "output_sha256", "ExpectedCheckpointSha256", "input_samples", "WaitForExit"):
        assert token in script
    assert "Start-Sleep" not in script


def test_release_clean_wheel_smoke_and_publish_commit_binding() -> None:
    workflow = (Path(__file__).resolve().parents[1] / ".github/workflows/release.yml").read_text(encoding="utf-8")
    assert "--system-site-packages" not in workflow
    assert "--no-deps" not in workflow
    assert "Scripts/noise-reduce-desktop.exe\") --help" not in workflow
    for token in ("--smoke-input", "--smoke-output", "--smoke-receipt", "verified_commit", "needs.verify-build.outputs.verified_commit", "git fetch --force origin"):
        assert token in workflow
