from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts.verify import publish_readiness


ROOT = Path(__file__).resolve().parents[1]


def test_publish_readiness_returns_three_pillar_scores() -> None:
    result = publish_readiness(ROOT)
    assert result["schema_version"] == 1
    assert set(result["scores"]) == {"github", "production", "research"}
    assert all(0 <= score <= 10 for score in result["scores"].values())
    assert isinstance(result["checks"], dict)
    assert result["score_scope"].startswith("repository and evidence readiness")
    assert set(result["research_coverage"]) == {
        "external_total", "external_reproduced", "external_blocked", "comparison_status",
        "verified_blocker_recipes", "eligible", "tier",
    }
    assert result["research_coverage"]["tier"] == "one_plus_recipe"
    assert result["scores"]["research"] == 9.0


def test_publish_readiness_rejects_readme_generated_evidence(tmp_path: Path) -> None:
    for path in ("README.md", "LICENSE", "CITATION.cff", "SECURITY.md", "CONTRIBUTING.md", "CHANGELOG.md", "MODEL_CARD.md", "docs/research-comparison.md"):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("placeholder", encoding="utf-8")
    (tmp_path / "README.md").write_text("reports/generated/cpu_benchmark.json", encoding="utf-8")
    (tmp_path / "SECURITY.md").write_text("", encoding="utf-8")
    result = publish_readiness(tmp_path)
    assert result["checks"]["readme_public_evidence"] is False
    assert result["checks"]["required_documents"] is False


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


def test_publish_readiness_uses_fail_closed_public_evidence_audits(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.verify_public_production.verify_public_production",
        lambda root: {"status": "fail", "checks": {"bundle_integrity": False}},
    )
    monkeypatch.setattr(
        "scripts.verify_public_research.audit_public_research",
        lambda root: {"status": "fail", "checks": {"source_hashes": False}},
    )

    result = publish_readiness(ROOT)

    assert result["checks"]["production_receipt"] is False
    assert result["checks"]["research_receipt"] is False
    assert result["scores"]["production"] < 10.0
    assert result["scores"]["research"] < 10.0


def test_publish_readiness_rejects_tracked_private_payloads(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.verify.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="raw_data.zip\n", stderr=""),
    )
    result = publish_readiness(ROOT)
    assert result["checks"]["excluded_payloads_untracked"] is False


def test_publish_readiness_requires_deepfilter_public_receipt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.verify_public_production.verify_public_production",
        lambda root: {"status": "pass", "checks": {}},
    )
    monkeypatch.setattr(
        "scripts.verify_public_research.audit_public_research",
        lambda root: {"status": "pass", "checks": {}},
    )
    public = tmp_path / "reports/public"
    public.mkdir(parents=True)
    for name in ("model_comparison.json", "production_readiness_verify.json", "research_readiness.json", "rnnoise_build.json"):
        (public / name).write_text("{}", encoding="utf-8")
    assert publish_readiness(tmp_path)["checks"]["public_receipts"] is False
    (public / "deepfilternet3_reproduction.json").write_text("{}", encoding="utf-8")
    assert publish_readiness(tmp_path)["checks"]["public_receipts"] is True


def test_public_docs_report_external_evidence_without_overclaiming() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    comparison = (ROOT / "docs/research-comparison.md").read_text(encoding="utf-8")
    model_card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    combined = "\n".join((readme, comparison, model_card, changelog))
    assert "reports/public/deepfilternet3_reproduction.json" in readme
    assert "one_plus_recipe" in combined
    assert "6.390235" in comparison
    assert "2.018783" in comparison
    assert "RNNoise" in comparison and "pinned" in comparison
    assert "DeepFilterNet3, RNNoise, DTLN, and WebRTC NS are currently `blocked`" not in combined
    assert "achieves `10/10`" not in combined


def test_research_score_cap_recomputes_coverage_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.verify_public_production.verify_public_production",
        lambda root: {"status": "pass", "checks": {}},
    )
    monkeypatch.setattr(
        "scripts.verify_public_research.audit_public_research",
        lambda root: {
            "status": "pass",
            "checks": {},
            "coverage": {
                "external_reproduced": 2,
                "verified_blocker_recipes": 0,
                "eligible": True,
                "tier": "two_reproduced",
            },
        },
    )
    result = publish_readiness(ROOT)
    assert result["research_coverage"]["external_reproduced"] == 1
    assert result["research_coverage"]["tier"] == "insufficient"
    assert result["scores"]["research"] < 9.0
