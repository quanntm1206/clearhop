"""Verify tracked, hash-bound production evidence without private artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any


REQUIRED_VERIFIER_CHECKS = {
    "required_artifacts",
    "cpu_benchmark",
    "cpu_soak",
    "runtime_smoke",
    "export_hashes",
    "export_structure",
    "bundle_integrity",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _finite_number(value: object, *, minimum: float = 0.0) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= minimum
    )


def _safe_manifest_name(name: object) -> bool:
    if not isinstance(name, str) or not name or "\\" in name:
        return False
    path = PurePosixPath(name)
    return not path.is_absolute() and all(part not in ("", ".", "..") for part in path.parts)


def _bundle_integrity(bundle_root: Path, manifest: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    files = manifest.get("files")
    if manifest.get("schema_version") != 1 or not isinstance(files, dict) or not files:
        return False, ["bundle manifest schema/files invalid"]
    expected_names = {"bundle.json"}
    for name, entry in files.items():
        if not _safe_manifest_name(name) or not isinstance(entry, dict):
            errors.append(f"unsafe or invalid manifest entry: {name!r}")
            continue
        expected_names.add(name)
        candidate = bundle_root.joinpath(*PurePosixPath(name).parts)
        if candidate.is_symlink() or not candidate.is_file():
            errors.append(f"missing regular bundle file: {name}")
            continue
        if entry.get("bytes") != candidate.stat().st_size:
            errors.append(f"size mismatch: {name}")
        if entry.get("sha256") != _sha256(candidate):
            errors.append(f"SHA-256 mismatch: {name}")
    actual_names = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if (path.is_file() or path.is_symlink())
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    if actual_names != expected_names:
        errors.append("bundle inventory differs from manifest")
    return not errors, errors


def _benchmark_valid(benchmark: object) -> bool:
    if not isinstance(benchmark, dict):
        return False
    if not (
        benchmark.get("schema_version") == 1
        and benchmark.get("status") == "pass"
        and benchmark.get("device") == "cpu"
        and isinstance(benchmark.get("iterations"), int)
        and not isinstance(benchmark.get("iterations"), bool)
        and int(benchmark["iterations"]) >= 5000
    ):
        return False
    for name in ("streaming_end_to_end", "onnx_core"):
        section = benchmark.get(name)
        if not isinstance(section, dict) or int(section.get("n", 0)) < 5000:
            return False
        for metric in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms", "realtime_factor"):
            if not _finite_number(section.get(metric)):
                return False
        if not 0.0 < float(section["p95_ms"]) < 10.0:
            return False
    threshold = benchmark.get("thresholds")
    return isinstance(threshold, dict) and float(threshold.get("streaming_p95_ms_lt", 0.0)) == 10.0


def _soak_valid(soak: object) -> bool:
    if not isinstance(soak, dict):
        return False
    faults = soak.get("faults")
    reset = soak.get("reset")
    memory = soak.get("memory")
    return (
        soak.get("schema_version") == 1
        and soak.get("status") == "pass"
        and soak.get("device") == "cpu"
        and _finite_number(soak.get("requested_seconds"), minimum=7200.0)
        and _finite_number(soak.get("elapsed_seconds"), minimum=7199.0)
        and isinstance(soak.get("iterations"), int)
        and not isinstance(soak.get("iterations"), bool)
        and int(soak["iterations"]) > 0
        and soak.get("failures") == []
        and isinstance(faults, dict)
        and all(faults.get(name) is True for name in ("empty", "nan", "wrong_shape"))
        and isinstance(reset, dict)
        and reset.get("idempotent") is True
        and reset.get("frames_after") == 0
        and isinstance(memory, dict)
        and memory.get("bounded") is True
    )


def _export_runtime_parity(bundle_root: Path) -> None:
    import numpy as np
    import onnxruntime as ort
    import torch

    from src.checkpoint import validate_checkpoint_metadata
    from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig

    state = torch.load(bundle_root / "checkpoint.pth", map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(state)
    model = MobileDeepFilterNet(MobileDeepFilterNetConfig(**state["model_cfg"]))
    model.load_state_dict(state["model"], strict=True)
    scripted = torch.jit.load(str(bundle_root / "model.ts"), map_location="cpu").eval()
    session = ort.InferenceSession(str(bundle_root / "model.onnx"), providers=["CPUExecutionProvider"])
    if [item.name for item in session.get_inputs()] != ["feats_logp", "hidden"]:
        raise ValueError("unexpected ONNX input contract")

    rng = np.random.default_rng(0)
    hidden_ts = torch.zeros(1, 1, model.cfg.gru_hidden, dtype=torch.float32)
    hidden_onnx = hidden_ts.numpy()
    for _ in range(2):
        features = rng.standard_normal((1, 1, model.cfg.freq_bins, 1), dtype=np.float32)
        with torch.no_grad():
            outputs_ts = scripted(torch.from_numpy(features), hidden_ts)
        outputs_onnx = session.run(None, {"feats_logp": features, "hidden": hidden_onnx})
        if len(outputs_ts) != len(outputs_onnx) or any(
            not np.allclose(torch.as_tensor(actual).detach().cpu().numpy(), expected, rtol=1e-4, atol=1e-5)
            for actual, expected in zip(outputs_ts, outputs_onnx)
        ):
            raise ValueError("TorchScript/ONNX recurrent parity mismatch")
        hidden_ts = outputs_ts[-1]
        hidden_onnx = outputs_onnx[-1]


def verify_public_production(root: str | Path) -> dict[str, Any]:
    """Return a fail-closed audit of public receipt and every bundled file."""
    root = Path(root).resolve()
    receipt_path = root / "reports/public/production_readiness_verify.json"
    manifest_path = root / "artifacts/cpu_bundle/bundle.json"
    checks = {
        "required_artifacts": receipt_path.is_file() and manifest_path.is_file(),
        "receipt_schema": False,
        "embedded_evidence_hashes": False,
        "bundle_integrity": False,
        "checkpoint_onnx_binding": False,
        "export_runtime_parity": False,
        "cpu_benchmark": False,
        "cpu_soak": False,
        "original_verifier": False,
    }
    errors: list[str] = []
    if not checks["required_artifacts"]:
        return {"schema_version": 1, "status": "fail", "production_eligible": False, "checks": checks, "errors": ["required public receipt or bundle manifest missing"]}
    try:
        receipt = _load_object(receipt_path)
        manifest = _load_object(manifest_path)
        embedded_manifest = receipt.get("bundle")
        sources = receipt.get("source_sha256")
        checks["receipt_schema"] = (
            receipt.get("schema_version") == 1
            and receipt.get("receipt_type") == "production_readiness"
            and receipt.get("status") == "pass"
            and isinstance(sources, dict)
            and sources.get("bundle.json") == _sha256(manifest_path)
            and embedded_manifest == manifest
        )
        embedded_hashes = receipt.get("embedded_sha256")
        checks["embedded_evidence_hashes"] = (
            isinstance(embedded_hashes, dict)
            and set(embedded_hashes) == {"verifier", "benchmark", "soak", "bundle"}
            and all(
                embedded_hashes.get(name) == _canonical_sha256(receipt.get(name))
                for name in ("verifier", "benchmark", "soak", "bundle")
            )
        )
        checks["bundle_integrity"], bundle_errors = _bundle_integrity(manifest_path.parent, manifest)
        errors.extend(bundle_errors)

        files = manifest.get("files", {})
        checkpoint_entry = files.get("checkpoint.pth") if isinstance(files, dict) else None
        onnx_entry = files.get("model.onnx") if isinstance(files, dict) else None
        benchmark = receipt.get("benchmark")
        soak = receipt.get("soak")
        checkpoint_hash = _sha256(manifest_path.parent / "checkpoint.pth")
        onnx_hash = _sha256(manifest_path.parent / "model.onnx")
        checks["checkpoint_onnx_binding"] = (
            isinstance(checkpoint_entry, dict)
            and isinstance(onnx_entry, dict)
            and isinstance(benchmark, dict)
            and isinstance(soak, dict)
            and manifest.get("checkpoint_sha256") == checkpoint_hash
            and checkpoint_entry.get("sha256") == checkpoint_hash
            and onnx_entry.get("sha256") == onnx_hash
            and benchmark.get("checkpoint_sha256") == checkpoint_hash
            and benchmark.get("onnx_sha256") == onnx_hash
            and soak.get("checkpoint_sha256") == checkpoint_hash
        )
        try:
            _export_runtime_parity(manifest_path.parent)
            checks["export_runtime_parity"] = True
        except Exception as exc:
            errors.append(f"export runtime/parity: {type(exc).__name__}: {exc}")
        checks["cpu_benchmark"] = _benchmark_valid(benchmark)
        checks["cpu_soak"] = _soak_valid(soak)

        verifier = receipt.get("verifier")
        verifier_checks = verifier.get("checks") if isinstance(verifier, dict) else None
        checks["original_verifier"] = (
            isinstance(verifier, dict)
            and verifier.get("schema_version") == 1
            and verifier.get("status") == "pass"
            and verifier.get("production_eligible") is True
            and isinstance(verifier_checks, dict)
            and REQUIRED_VERIFIER_CHECKS.issubset(verifier_checks)
            and all(value is True for value in verifier_checks.values())
        )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    status = "pass" if all(checks.values()) else "fail"
    return {
        "schema_version": 1,
        "status": status,
        "production_eligible": status == "pass",
        "checks": checks,
        **({"errors": errors} if errors else {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = verify_public_production(args.root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
