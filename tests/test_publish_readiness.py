from __future__ import annotations

import json
from pathlib import Path

from scripts.verify import publish_readiness


ROOT = Path(__file__).resolve().parents[1]


def test_publish_readiness_returns_three_pillar_scores() -> None:
    result = publish_readiness(ROOT)
    assert result["schema_version"] == 1
    assert set(result["scores"]) == {"github", "production", "research"}
    assert all(0 <= score <= 10 for score in result["scores"].values())
    assert isinstance(result["checks"], dict)
    assert result["score_scope"].startswith("repository and evidence readiness")
    assert set(result["research_coverage"]) == {"external_total", "external_reproduced", "external_blocked", "comparison_status"}


def test_publish_readiness_rejects_readme_generated_evidence(tmp_path: Path) -> None:
    for path in ("README.md", "LICENSE", "CITATION.cff", "SECURITY.md", "CONTRIBUTING.md", "CHANGELOG.md", "MODEL_CARD.md", "docs/research-comparison.md"):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("placeholder", encoding="utf-8")
    (tmp_path / "README.md").write_text("reports/generated/cpu_benchmark.json", encoding="utf-8")
    result = publish_readiness(tmp_path)
    assert result["checks"]["readme_public_evidence"] is False


def test_publish_readiness_rejects_placeholder_and_large_inventory(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("https://example.com/your-org/TODO", encoding="utf-8")
    (tmp_path / "raw.bin").write_bytes(b"x" * (51 * 1024 * 1024))
    result = publish_readiness(tmp_path)
    assert result["checks"]["no_placeholders"] is False
    assert result["checks"]["inventory"] is False


def test_publish_readiness_requires_version_consistency(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nversion = "1.0.0"\n', encoding="utf-8")
    (tmp_path / "CITATION.cff").write_text('version: "0.1.0"\n', encoding="utf-8")
    result = publish_readiness(tmp_path)
    assert result["checks"]["version_consistency"] is False


def test_publish_readiness_ignores_local_artifact_environments(tmp_path: Path) -> None:
    artifact = tmp_path / ".artifacts" / "venv" / "certifi" / "cacert.pem"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"local-only")
    result = publish_readiness(tmp_path)
    assert result["checks"]["inventory"] is True


def test_publish_readiness_rejects_local_paths_in_public_receipts(tmp_path: Path) -> None:
    receipt = tmp_path / "reports" / "public" / "model_comparison.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps({"source": r"C:\Users\name\.venv\report.json"}), encoding="utf-8")
    result = publish_readiness(tmp_path)
    assert result["checks"]["public_receipt_hygiene"] is False
