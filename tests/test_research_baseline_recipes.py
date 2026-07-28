import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_rnnoise_blocker_has_runnable_pinned_docker_recipe() -> None:
    registry = json.loads((ROOT / "configs/research_baselines.json").read_text(encoding="utf-8"))
    rnnoise = next(row for row in registry["baselines"] if row["name"] == "RNNoise")
    reproduction = rnnoise["reproduction"]
    assert set(reproduction) == {
        "setup_script", "setup_script_sha256", "dockerfile", "dockerfile_sha256",
        "toolchain_manifest", "toolchain_manifest_sha256", "command", "pinned_commit",
        "status", "blocker",
    }
    assert reproduction["status"] == "blocked"
    assert reproduction["pinned_commit"] == "70f1d256acd4b34a572f999a05c87bf00b67730d"
    assert reproduction["command"] == "pwsh -File scripts/setup-rnnoise-baseline.ps1 -OutputDir <external-cache>"
    script = ROOT / reproduction["setup_script"]
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert hashlib.sha256(script.read_bytes()).hexdigest() == reproduction["setup_script_sha256"]
    dockerfile = ROOT / reproduction["dockerfile"]
    manifest_path = ROOT / reproduction["toolchain_manifest"]
    assert hashlib.sha256(dockerfile.read_bytes()).hexdigest() == reproduction["dockerfile_sha256"]
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == reproduction["toolchain_manifest_sha256"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source_commit"] == reproduction["pinned_commit"]
    assert manifest["base_image"].startswith("ubuntu@sha256:")
    assert len(manifest["base_image"].split("sha256:", 1)[1]) == 64
    assert len(manifest["source_archive_sha256"]) == 64
    assert len(manifest["model_archive_sha256"]) == 64
    assert manifest["packages"]
    assert manifest["apt_snapshot"].startswith("https://snapshot.ubuntu.com/ubuntu/")
    docker_text = dockerfile.read_text(encoding="utf-8")
    assert manifest["base_image"] in docker_text
    assert '"APT_SNAPSHOT_URL=$($manifest.apt_snapshot)"' in text
    for token in ("docker", "[switch]$DryRun", reproduction["toolchain_manifest"]):
        assert token in text
    assert "cp -a /out/. /export/" in text
    assert "docker create" not in text
    assert "toolchain package mismatch" in text
    for token in ("ARG BASE_IMAGE", "apt-cache policy", "sha256sum -c", "RNNOISE_ARCHIVE_URL"):
        assert token in docker_text


def test_deepfilternet_registry_binds_reproduced_receipt_and_assets() -> None:
    registry = json.loads((ROOT / "configs/research_baselines.json").read_text(encoding="utf-8"))
    baseline = next(row for row in registry["baselines"] if row["name"] == "DeepFilterNet3")
    assert baseline["evidence_class"] == "reproduced_local"
    assert baseline["weights_bundled"] is False
    assert baseline["provenance"]["weight_sha256"] == "23b92884f63ccf54bb026014604625ab231657b6480df65db4095c4c171e6003"
    assert baseline["provenance"]["model_archive_sha256"] == "49c52edc8947ae1f9bf50d81530beaf3a2c3245aeaf34b6f31ff535cd22284d2"
    assert baseline["reproduction"]["receipt"] == "reports/public/deepfilternet3_reproduction.json"
