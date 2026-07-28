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
    production = {
        "schema_version": 1,
        "receipt_type": "production_readiness",
        "status": verify["production_readiness"]["status"],
        "timestamp_utc": verify.get("timestamp_utc"),
        "verifier": _safe(verify["production_readiness"], root),
        "benchmark": _safe(benchmark, root),
        "soak": _safe(soak, root),
        "bundle": _safe(bundle, root),
        "source_sha256": {
            "cpu_benchmark.json": _sha256(generated / "cpu_benchmark.json"),
            "cpu_soak.json": _sha256(generated / "cpu_soak.json"),
            "bundle.json": _sha256(root / "artifacts" / "cpu_bundle" / "bundle.json"),
        },
    }
    (public / "production_readiness_verify.json").write_text(json.dumps(production, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verify = _load(generated / "research_readiness.json")
    names = ("research_training", "research_evaluations", "research_selection", "significance", "robustness_matrix", "failure_analysis", "failure_audio")
    research = {
        "schema_version": 1,
        "receipt_type": "research_readiness",
        "status": verify["research_readiness"]["status"],
        "timestamp_utc": verify.get("timestamp_utc"),
        "verifier": _safe(verify["research_readiness"], root),
        "manifest": _safe(verify.get("manifest_audit", {}), root),
        "evidence": {name: _safe(_load(generated / f"{name}.json"), root) for name in names},
        "source_sha256": {f"{name}.json": _sha256(generated / f"{name}.json") for name in names},
    }
    (public / "research_readiness.json").write_text(json.dumps(research, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    curate(Path(__file__).resolve().parents[1])
