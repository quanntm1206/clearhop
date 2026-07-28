import copy
import json
from pathlib import Path

from scripts import verify_rnnoise_build as rnnoise_build
from scripts.verify_rnnoise_build import audit_rnnoise_build, canonical_receipt_sha256


ROOT = Path(__file__).resolve().parents[1]


def test_public_rnnoise_build_receipt_and_ci_contract_pass() -> None:
    result = audit_rnnoise_build(ROOT)
    assert result["status"] == "pass", result
    assert all(result["checks"].values()), result


def test_rnnoise_build_receipt_rejects_hash_drift() -> None:
    receipt = json.loads((ROOT / "reports/public/rnnoise_build.json").read_text(encoding="utf-8"))
    receipt["outputs"]["files"]["rnnoise_demo"] = "0" * 64
    result = audit_rnnoise_build(ROOT, receipt=receipt)
    assert result["checks"]["receipt_hash"] is False


def test_rnnoise_build_receipt_rejects_stale_static_inputs() -> None:
    receipt = json.loads((ROOT / "reports/public/rnnoise_build.json").read_text(encoding="utf-8"))
    receipt = copy.deepcopy(receipt)
    receipt["inputs"]["dockerfile"]["sha256"] = "0" * 64
    receipt["receipt_sha256"] = canonical_receipt_sha256(receipt)
    result = audit_rnnoise_build(ROOT, receipt=receipt)
    assert result["checks"]["receipt_inputs"] is False


def test_rnnoise_build_requires_linux_ci_rebuild_command() -> None:
    result = audit_rnnoise_build(ROOT, workflow_text="name: CI\n")
    assert result["checks"]["workflow_contract"] is False


def test_linux_docker_export_uses_host_owner(monkeypatch, tmp_path: Path) -> None:
    commands: list[list[str]] = []
    monkeypatch.setattr(rnnoise_build.subprocess, "run", lambda command, **_: commands.append(command))
    monkeypatch.setattr(rnnoise_build.os, "name", "posix")
    monkeypatch.setattr(rnnoise_build.os, "getuid", lambda: 1234, raising=False)
    monkeypatch.setattr(rnnoise_build.os, "getgid", lambda: 5678, raising=False)

    rnnoise_build._docker_build(ROOT, tmp_path / "out")

    export_command = commands[-1]
    owner_index = export_command.index("--user")
    assert export_command[owner_index + 1] == "1234:5678"
