"""Create small, path-safe public evidence receipts from local generated reports."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]|^[/\\]{2}|^/")
_DROP_KEYS = {"command", "stdout", "stderr", "python"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _receipt_sha256(value: dict[str, Any]) -> str:
    return _canonical_sha256({key: item for key, item in value.items() if key != "receipt_sha256"})


def _safe(value: Any, root: Path) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe(item, root) for key, item in value.items() if str(key) not in _DROP_KEYS}
    if isinstance(value, list):
        return [_safe(item, root) for item in value]
    if isinstance(value, str):
        value = value.replace(str(root), ".").replace(str(root).replace("\\", "/"), ".")
        if ".venv" in value.replace("\\", "/"):
            return "<local-runtime-redacted>"
        if not _ABSOLUTE.match(value):
            return value
        candidate = Path(value)
        try:
            return candidate.resolve().relative_to(root).as_posix()
        except (OSError, ValueError):
            return "<local-path-redacted>"
    return value


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def curate(root: Path) -> None:
    generated = root / "reports" / "generated"
    public = root / "reports" / "public"
    public.mkdir(parents=True, exist_ok=True)

    verify = _load(generated / "production_readiness_verify.json")
    benchmark = _load(generated / "cpu_benchmark.json")
    soak = _load(generated / "cpu_soak.json")
    bundle = _load(root / "artifacts" / "cpu_bundle" / "bundle.json")
    verifier_public = _safe(verify["production_readiness"], root)
    benchmark_public = _safe(benchmark, root)
    soak_public = _safe(soak, root)
    bundle_public = _safe(bundle, root)
    production = {
        "schema_version": 1,
        "receipt_type": "production_readiness",
        "status": verify["production_readiness"]["status"],
        "timestamp_utc": verify.get("timestamp_utc"),
        "verifier": verifier_public,
        "benchmark": benchmark_public,
        "soak": soak_public,
        "bundle": bundle_public,
        "embedded_sha256": {
            "verifier": _canonical_sha256(verifier_public),
            "benchmark": _canonical_sha256(benchmark_public),
            "soak": _canonical_sha256(soak_public),
            "bundle": _canonical_sha256(bundle_public),
        },
        "source_sha256": {
            "cpu_benchmark.json": _sha256(generated / "cpu_benchmark.json"),
            "cpu_soak.json": _sha256(generated / "cpu_soak.json"),
            "bundle.json": _sha256(root / "artifacts" / "cpu_bundle" / "bundle.json"),
        },
    }
    (public / "production_readiness_verify.json").write_text(json.dumps(production, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verify = _load(generated / "research_readiness.json")
    names = ("research_training", "research_evaluations", "research_selection", "significance", "robustness_matrix", "failure_analysis", "failure_audio")
    evidence = {name: _safe(_load(generated / f"{name}.json"), root) for name in names}
    verifier_public = _safe(verify["research_readiness"], root)
    manifest_public = _safe(verify.get("manifest_audit", {}), root)
    research = {
        "schema_version": 1,
        "receipt_type": "research_readiness",
        "status": verify["research_readiness"]["status"],
        "timestamp_utc": verify.get("timestamp_utc"),
        "verifier": verifier_public,
        "manifest": manifest_public,
        "evidence": evidence,
        "embedded_sha256": {
            "verifier": _canonical_sha256(verifier_public),
            "manifest": _canonical_sha256(manifest_public),
            **{name: _canonical_sha256(value) for name, value in evidence.items()},
        },
        "source_sha256": {f"{name}.json": _sha256(generated / f"{name}.json") for name in names},
    }
    research["receipt_sha256"] = _receipt_sha256(research)
    (public / "research_readiness.json").write_text(json.dumps(research, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    curate(Path(__file__).resolve().parents[1])
