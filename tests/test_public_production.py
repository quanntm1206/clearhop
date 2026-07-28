"""Fail-closed checks for tracked production evidence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

from scripts.verify_public_production import verify_public_production


def _evidence_root(tmp_path: Path) -> Path:
    source = Path(__file__).resolve().parents[1]
    root = tmp_path / "repo"
    (root / "artifacts").mkdir(parents=True)
    (root / "reports/public").mkdir(parents=True)
    shutil.copytree(source / "artifacts/cpu_bundle", root / "artifacts/cpu_bundle")
    shutil.copy2(
        source / "reports/public/production_readiness_verify.json",
        root / "reports/public/production_readiness_verify.json",
    )
    return root


def _mutate_receipt(root: Path, mutate) -> None:
    path = root / "reports/public/production_readiness_verify.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_tracked_production_evidence_passes(tmp_path: Path) -> None:
    result = verify_public_production(_evidence_root(tmp_path))
    assert result["status"] == "pass"
    assert result["production_eligible"] is True
    assert all(result["checks"].values())


def test_bundle_tamper_fails_closed(tmp_path: Path) -> None:
    root = _evidence_root(tmp_path)
    with (root / "artifacts/cpu_bundle/model.onnx").open("ab") as stream:
        stream.write(b"tampered")
    result = verify_public_production(root)
    assert result["status"] == "fail"
    assert result["checks"]["bundle_integrity"] is False


def test_benchmark_threshold_and_binding_fail_closed(tmp_path: Path) -> None:
    root = _evidence_root(tmp_path)
    _mutate_receipt(root, lambda value: value["benchmark"].update(iterations=4999))
    assert verify_public_production(root)["checks"]["cpu_benchmark"] is False

    root = _evidence_root(tmp_path / "binding")
    _mutate_receipt(root, lambda value: value["benchmark"].update(onnx_sha256="0" * 64))
    assert verify_public_production(root)["checks"]["checkpoint_onnx_binding"] is False


def test_soak_and_original_verifier_fail_closed(tmp_path: Path) -> None:
    root = _evidence_root(tmp_path)
    _mutate_receipt(root, lambda value: value["soak"].update(elapsed_seconds=7198.9))
    assert verify_public_production(root)["checks"]["cpu_soak"] is False

    root = _evidence_root(tmp_path / "verifier")
    _mutate_receipt(root, lambda value: value["verifier"]["checks"].update(runtime_smoke=False))
    assert verify_public_production(root)["checks"]["original_verifier"] is False


def test_export_runtime_load_or_parity_failure_fails_closed(tmp_path: Path) -> None:
    root = _evidence_root(tmp_path)
    with patch("torch.jit.load", side_effect=RuntimeError("invalid export")):
        result = verify_public_production(root)
    assert result["status"] == "fail"
    assert result["checks"]["export_runtime_parity"] is False


def test_embedded_evidence_hash_drift_fails_closed(tmp_path: Path) -> None:
    root = _evidence_root(tmp_path)
    _mutate_receipt(root, lambda value: value["benchmark"]["environment"].update(processor="tampered"))
    result = verify_public_production(root)
    assert result["status"] == "fail"
    assert result["checks"]["embedded_evidence_hashes"] is False
