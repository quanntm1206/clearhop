"""Run the repository verification gate and write a machine-readable report."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(command: list[str], root: Path) -> dict[str, object]:
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-12000:],
        "stderr": completed.stderr[-12000:],
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_or_inf(value: float) -> float:
    return value if math.isfinite(value) else float("inf")


def manifest_audit(root: Path) -> dict[str, object]:
    summary_path = root / "manifests" / "v2" / "fold_0_summary.json"
    if not summary_path.exists():
        return {"status": "missing", "path": str(summary_path)}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    overlap = summary.get("speaker_overlap", {})
    fingerprints = summary.get("fingerprints", {})
    status = "pass" if overlap and all(int(value) == 0 for value in overlap.values()) and len(fingerprints) == 3 else "fail"
    return {"status": status, "summary": summary}


def artifact_audit(root: Path, *, require_full: bool = False) -> dict[str, object]:
    if require_full:
        paths = {
            "full_best_checkpoint": root / "checkpoints" / "best.pth",
            "full_last_checkpoint": root / "checkpoints" / "step_30000.pth",
            "full_evaluation": root / "reports" / "generated" / "full_best_evaluation.json",
            "full_benchmark": root / "reports" / "generated" / "full_best_benchmark.json",
            "full_export_parity": root / "reports" / "generated" / "full_export_parity.json",
            "full_torchscript": root / "checkpoints" / "full_best_export.ts",
            "full_onnx": root / "checkpoints" / "full_best_export.onnx",
            "full_export_metadata": root / "checkpoints" / "full_best_export.json",
            "full_training_report": root / "reports" / "full_training_report.md",
        }
    else:
        paths = {
            "production_checkpoint": root / "checkpoints" / "production_smoke_best.pth",
            "production_evaluation": root / "reports" / "generated" / "production_smoke_evaluation.json",
            "production_benchmark": root / "reports" / "generated" / "production_smoke_benchmark.json",
            "production_torchscript": root / "checkpoints" / "production_smoke_export.ts",
            "production_onnx": root / "checkpoints" / "production_smoke_export.onnx",
        }
    artifacts = {
        name: {"path": str(path), "exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}
        for name, path in paths.items()
    }
    files_ok = all(item["exists"] and int(item["bytes"]) > 0 for item in artifacts.values())
    checks: dict[str, bool] = {}
    if require_full and files_ok:
        try:
            checkpoint_hash = _file_sha256(paths["full_best_checkpoint"])
            torchscript_hash = _file_sha256(paths["full_torchscript"])
            onnx_hash = _file_sha256(paths["full_onnx"])
            evaluation = json.loads(paths["full_evaluation"].read_text(encoding="utf-8"))
            benchmark = json.loads(paths["full_benchmark"].read_text(encoding="utf-8"))
            export = json.loads(paths["full_export_metadata"].read_text(encoding="utf-8"))
            parity = json.loads(paths["full_export_parity"].read_text(encoding="utf-8"))
            manifest_summary = json.loads(
                (root / "manifests" / "v2" / "fold_0_summary.json").read_text(encoding="utf-8")
            )
            import torch
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from src.checkpoint import validate_checkpoint_metadata
            from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig

            final_state = torch.load(paths["full_last_checkpoint"], map_location="cpu")
            validate_checkpoint_metadata(final_state)
            final_model = MobileDeepFilterNet(MobileDeepFilterNetConfig(**final_state["model_cfg"]))
            final_model.load_state_dict(final_state["model"], strict=True)
            runtime = final_state.get("runtime", {})
            training_config = final_state.get("config", {})
            evaluation_metadata = evaluation.get("metadata", {})
            expected_test_fingerprint = manifest_summary.get("fingerprints", {}).get("test")
            required_metrics = {"si_sdr", "snr", "stoi", "pesq"}
            evaluation_outputs = ("noisy", "mask_only", "enhanced")

            def valid_metric(metric: object) -> bool:
                if not isinstance(metric, dict):
                    return False
                mean = metric.get("mean")
                std = metric.get("std")
                return (
                    isinstance(mean, (int, float))
                    and isinstance(std, (int, float))
                    and math.isfinite(float(mean))
                    and math.isfinite(float(std))
                    and float(std) >= 0.0
                )

            def valid_latency(section: object) -> bool:
                if not isinstance(section, dict):
                    return False
                values = [section.get(key) for key in ("mean_ms", "p95_ms", "max_ms", "realtime_factor")]
                return (
                    int(section.get("n", 0)) >= 500
                    and all(isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0 for value in values)
                )

            checks = {
                "evaluation_checkpoint_bound": evaluation.get("metadata", {}).get("checkpoint_sha256") == checkpoint_hash,
                "benchmark_checkpoint_bound": benchmark.get("checkpoint_sha256") == checkpoint_hash,
                "export_checkpoint_bound": export.get("source_checkpoint_sha256") == checkpoint_hash,
                "parity_checkpoint_bound": parity.get("checkpoint_sha256") == checkpoint_hash,
                "parity_torchscript_bound": parity.get("torchscript_sha256") == torchscript_hash,
                "parity_onnx_bound": parity.get("onnx_sha256") == onnx_hash,
                "parity_status_pass": parity.get("status") == "pass",
                "training_global_step": int(final_state.get("global_step", -1)) == 30000,
                "training_device_cuda": runtime.get("device") == "cuda",
                "training_gpu_recorded": bool(runtime.get("gpu")),
                "training_checkpoint_loads_strict": True,
                "training_embedded_config": int(training_config.get("epochs", 0)) == 150
                and int(training_config.get("steps_per_epoch", 0)) == 200,
                "evaluation_config": str(evaluation_metadata.get("config", "")).replace("\\", "/").endswith("configs/train.yaml"),
                "evaluation_device_cuda": evaluation_metadata.get("device") == "cuda",
                "evaluation_test_fingerprint": evaluation_metadata.get("manifest_fingerprint") == expected_test_fingerprint,
                "evaluation_sample_count": evaluation_metadata.get("max_items") == 500
                and all(evaluation.get(name, {}).get("n") == 500 for name in evaluation_outputs),
                "evaluation_required_metrics": all(
                    required_metrics.issubset(evaluation.get(name, {}))
                    and all(valid_metric(evaluation[name][metric]) for metric in required_metrics)
                    for name in evaluation_outputs
                ),
                "benchmark_device_cuda": benchmark.get("device") == "cuda" and bool(benchmark.get("gpu")),
                "benchmark_iterations": int(benchmark.get("iterations", 0)) >= 500,
                "benchmark_sections": valid_latency(benchmark.get("neural_core"))
                and valid_latency(benchmark.get("streaming_end_to_end")),
            }
        except Exception:
            checks = {"metadata_parse": False}
    status = "pass" if files_ok and (not require_full or all(checks.values())) else "fail"
    return {"status": status, "artifacts": artifacts, "checks": checks}


def _all_finite_numeric(value: object) -> bool:
    """Accept containers whose leaves are finite numbers, never JSON lookalikes."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return bool(value) and all(_all_finite_numeric(item) for item in value.values())
    if isinstance(value, list):
        return bool(value) and all(_all_finite_numeric(item) for item in value)
    return False


_LOSS_KEYS = (
    "loss",
    "alpha_loss",
    "beta_si_sdr",
    "loss_eps",
    "sisdr_warmup_start",
    "sisdr_warmup_end",
    "compression_exponent",
    "compression_complex_weight",
)
_EXPECTED_LOSS_CONFIGS: dict[str, dict[str, object]] = {
    "control": {
        "loss": "complex_mse_plus_si_sdr", "alpha_loss": 1.0, "beta_si_sdr": 0.5,
        "loss_eps": 1e-8, "sisdr_warmup_start": 0, "sisdr_warmup_end": 0,
        "compression_exponent": 0.3, "compression_complex_weight": 0.3,
    },
    "complex_nmse": {
        "loss": "complex_nmse", "alpha_loss": 1.0, "beta_si_sdr": 0.0,
        "loss_eps": 1e-8, "sisdr_warmup_start": 0, "sisdr_warmup_end": 0,
        "compression_exponent": 0.3, "compression_complex_weight": 0.3,
    },
    "complex_nmse_sisdr": {
        "loss": "complex_nmse_sisdr", "alpha_loss": 1.0, "beta_si_sdr": 0.01,
        "loss_eps": 1e-8, "sisdr_warmup_start": 500, "sisdr_warmup_end": 1000,
        "compression_exponent": 0.3, "compression_complex_weight": 0.3,
    },
    "complex_nmse_sisdr_beta_0p02": {
        "loss": "complex_nmse_sisdr", "alpha_loss": 1.0, "beta_si_sdr": 0.02,
        "loss_eps": 1e-8, "sisdr_warmup_start": 500, "sisdr_warmup_end": 1000,
        "compression_exponent": 0.3, "compression_complex_weight": 0.3,
    },
    "complex_nmse_sisdr_beta_0p025": {
        "loss": "complex_nmse_sisdr", "alpha_loss": 1.0, "beta_si_sdr": 0.025,
        "loss_eps": 1e-8, "sisdr_warmup_start": 500, "sisdr_warmup_end": 1000,
        "compression_exponent": 0.3, "compression_complex_weight": 0.3,
    },
    "complex_nmse_sisdr_beta_0p03": {
        "loss": "complex_nmse_sisdr", "alpha_loss": 1.0, "beta_si_sdr": 0.03,
        "loss_eps": 1e-8, "sisdr_warmup_start": 500, "sisdr_warmup_end": 1000,
        "compression_exponent": 0.3, "compression_complex_weight": 0.3,
    },
    "compressed_complex": {
        "loss": "compressed_complex", "alpha_loss": 1.0, "beta_si_sdr": 0.0,
        "loss_eps": 1e-8, "sisdr_warmup_start": 0, "sisdr_warmup_end": 0,
        "compression_exponent": 0.3, "compression_complex_weight": 0.3,
    },
}


def gain_calibration_artifact_audit(root: Path) -> dict[str, object]:
    """Verify the full producer schema, bound evidence, exports, and gates."""
    root = Path(root).resolve()
    paths = {
        "full_summary": root / "reports/generated/gain_calibration_full.json",
        "selection_receipt": root / "reports/generated/gain_calibration_selection.json",
        "baseline_checkpoint": root / "checkpoints/best.pth",
        "selected_checkpoint": root / "checkpoints/gain_calibration/best.pth",
        "torchscript": root / "checkpoints/gain_calibration/export.ts",
        "onnx": root / "checkpoints/gain_calibration/export.onnx",
        "export_metadata": root / "checkpoints/gain_calibration/export.json",
        "benchmark": root / "reports/generated/gain_calibration_benchmark.json",
        "export_parity": root / "reports/generated/gain_calibration_export_parity.json",
        "test_manifest": root / "manifests/v2/fold_0_test.jsonl",
        "validation_manifest": root / "manifests/v2/fold_0_val.jsonl",
        "manifest_summary": root / "manifests/v2/fold_0_summary.json",
    }
    artifacts = {
        name: {"path": str(path), "exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0}
        for name, path in paths.items()
    }
    check_names = (
        "required_artifacts", "full_schema", "validation_only_selection", "seed_coverage",
        "screen_promotion_binding", "manifest_fingerprint", "artifact_references",
        "resolved_config_hashes", "run_checkpoint_hashes", "run_checkpoint_strict_load",
        "resolved_yaml_binding", "validation_recomputation", "validation_manifest_binding", "selection_receipt",
        "loss_provenance", "evaluation_hashes", "evaluation_slices", "evaluation_counts",
        "evaluation_finite", "baseline_evaluations", "baseline_checkpoint_binding",
        "baseline_checkpoint_strict_load", "acceptance_gates", "recorded_acceptance", "all_seed_acceptance",
        "selected_acceptance", "selected_checkpoint_hash", "selected_checkpoint_strict_load",
        "selected_checkpoint_provenance", "selected_run_binding", "selection_rationale",
        "cuda_benchmark", "export_hashes", "export_structure", "export_parity",
    )
    checks = {name: False for name in check_names}
    checks["required_artifacts"] = all(item["exists"] and int(item["bytes"]) > 0 for item in artifacts.values())
    if not checks["required_artifacts"]:
        return {"status": "fail", "artifacts": artifacts, "checks": checks}

    errors: list[str] = []
    try:
        import onnxruntime as ort
        import torch

        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from src.checkpoint import validate_checkpoint_metadata
        from src.config import load_train_config
        from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig
        from src.splits import manifest_fingerprint, slice_fingerprint
        from scripts.run_gain_ablation import _evaluation_metrics, summarize_refinement, summarize_screen

        def read_json(path: Path) -> dict[str, object]:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object: {path}")
            return value

        def resolve(value: object) -> Path:
            path = Path(str(value))
            resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
            if resolved != root and root not in resolved.parents:
                raise ValueError(f"Evidence path escapes root: {value}")
            return resolved

        def reference(ref: object) -> tuple[Path | None, bool]:
            if not isinstance(ref, dict) or set(ref) < {"path", "sha256"}:
                return None, False
            try:
                path = resolve(ref["path"])
                return path, path.is_file() and ref["sha256"] == _file_sha256(path)
            except Exception:
                return None, False

        def strict_load(path: Path) -> dict[str, object]:
            state = torch.load(path, map_location="cpu", weights_only=False)
            validate_checkpoint_metadata(state)
            model = MobileDeepFilterNet(MobileDeepFilterNetConfig(**state["model_cfg"]))
            model.load_state_dict(state["model"], strict=True)
            return state

        def same_path(left: object, right: Path) -> bool:
            try:
                return Path(str(left)).samefile(right)
            except OSError:
                return Path(str(left)).resolve() == right.resolve()

        def metric(evaluation: dict[str, object], name: str, statistic: str = "mean") -> float:
            enhanced = evaluation.get("enhanced")
            if not isinstance(enhanced, dict):
                raise ValueError("Missing enhanced metrics")
            value = enhanced.get(name)
            if isinstance(value, dict):
                value = value.get(statistic)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"Invalid metric: {name}.{statistic}")
            return float(value)

        def recompute_acceptance(
            evaluation: dict[str, object], baseline: dict[str, object]
        ) -> dict[str, object]:
            gate_checks = {
                "snri_positive": metric(evaluation, "snr_improvement_mean") > 0.0,
                "si_sdri_delta": metric(evaluation, "si_sdr_improvement_mean")
                >= metric(baseline, "si_sdr_improvement_mean") - 0.3,
                "stoi_delta": metric(evaluation, "stoi") >= metric(baseline, "stoi") - 0.005,
                "pesq_delta": metric(evaluation, "pesq") >= metric(baseline, "pesq") - 0.05,
                "positive_gain": metric(evaluation, "projection_gain", "median") > 0.0,
                "gain_error": abs(metric(evaluation, "gain_error_db", "median")) <= 3.0,
                "polarity": metric(evaluation, "polarity_failure") < 0.01,
            }
            return {"status": "pass" if all(gate_checks.values()) else "fail", "checks": gate_checks}

        full = read_json(paths["full_summary"])
        selected_arm = full.get("selected_arm")
        selected_seed = full.get("selected_seed")
        rows = full.get("runs") if isinstance(full.get("runs"), list) else []
        checks["full_schema"] = (
            full.get("schema_version") == 1
            and full.get("stage") == "full"
            and full.get("status") in {"completed", "rejected"}
            and isinstance(full.get("production_eligible"), bool)
        )
        checks["validation_only_selection"] = full.get("selection_basis") == "validation"
        checks["seed_coverage"] = (
            isinstance(selected_arm, str)
            and selected_arm in _EXPECTED_LOSS_CONFIGS
            and len(rows) == 3
            and {int(row.get("seed", -1)) for row in rows if isinstance(row, dict)} == {17, 29, 43}
            and all(
                isinstance(row, dict)
                and row.get("arm") == selected_arm
                and row.get("status") in {"completed", "rejected"}
                and row.get("execution_status", row.get("status")) in {"completed", "already-complete"}
                for row in rows
            )
        )

        screen_ref = full.get("screen_report")
        screen_path, screen_hash_ok = reference(screen_ref)
        screen = read_json(screen_path) if screen_path is not None and screen_path.is_file() else {}
        screen_rows = screen.get("runs") if isinstance(screen.get("runs"), list) else []
        expected_matrix = {
            (arm, seed)
            for arm in ("control", "complex_nmse", "complex_nmse_sisdr", "compressed_complex")
            for seed in (17, 29, 43)
        }
        actual_matrix = {
            (str(row.get("arm")), int(row.get("seed", -1)))
            for row in screen_rows if isinstance(row, dict)
        }
        recomputed_screen = summarize_screen(screen_rows)
        direct_screen_binding = (
            screen_hash_ok
            and isinstance(screen_ref, dict)
            and screen_ref.get("promoted_arm") == selected_arm
            and screen.get("status") == "completed"
            and screen.get("promoted_arm") == selected_arm
            and len(screen_rows) == 12
            and actual_matrix == expected_matrix
            and all(isinstance(row, dict) and row.get("status") == "completed" for row in screen_rows)
            and recomputed_screen.get("promoted_arm") == selected_arm
        )
        refinement_binding = False
        if (
            screen_hash_ok
            and isinstance(screen_ref, dict)
            and screen.get("stage") == "refine"
            and screen.get("promoted_arm") == selected_arm
        ):
            source_ref = screen.get("screen_report")
            source_path, source_hash_ok = reference(source_ref)
            source_screen = read_json(source_path) if source_path is not None and source_path.is_file() else {}
            source_rows = source_screen.get("runs") if isinstance(source_screen.get("runs"), list) else []
            source_actual = {
                (str(row.get("arm")), int(row.get("seed", -1)))
                for row in source_rows if isinstance(row, dict)
            }
            source_recomputed = summarize_screen(source_rows)
            refinement_rows = screen.get("runs") if isinstance(screen.get("runs"), list) else []
            controls = [row for row in source_rows if isinstance(row, dict) and row.get("arm") == "control"]
            refinement_recomputed = summarize_refinement(refinement_rows, controls)
            refinement_binding = (
                source_hash_ok
                and isinstance(source_ref, dict)
                and source_screen.get("schema_version") == 1
                and source_screen.get("stage") == "screen"
                and source_screen.get("status") == "completed"
                and len(source_rows) == 12
                and source_actual == expected_matrix
                and all(isinstance(row, dict) and row.get("status") == "completed" for row in source_rows)
                and source_recomputed.get("promoted_arm") is None
                and len(refinement_rows) == 3
                and refinement_recomputed.get("promoted_arm") == selected_arm
            )
        checks["screen_promotion_binding"] = direct_screen_binding or refinement_binding

        test_hash = manifest_fingerprint(paths["test_manifest"])
        validation_hash = manifest_fingerprint(paths["validation_manifest"])
        validation_slice_hash = slice_fingerprint(paths["validation_manifest"], 0, 500)
        manifest_summary = read_json(paths["manifest_summary"])
        checks["manifest_fingerprint"] = (
            isinstance(manifest_summary.get("fingerprints"), dict)
            and manifest_summary["fingerprints"].get("test") == test_hash  # type: ignore[union-attr]
            and manifest_summary["fingerprints"].get("val") == validation_hash  # type: ignore[union-attr]
        )
        slices = {
            "comparison": (0, slice_fingerprint(paths["test_manifest"], 0, 500)),
            "audit": (500, slice_fingerprint(paths["test_manifest"], 500, 500)),
        }

        selected_ref = full.get("selected_checkpoint")
        selected_ref_path, selected_ref_ok = reference(selected_ref)
        selected_hash = _file_sha256(paths["selected_checkpoint"])
        checks["selected_checkpoint_hash"] = (
            selected_ref_ok
            and selected_ref_path == paths["selected_checkpoint"]
            and full.get("selected_checkpoint_path") == paths["selected_checkpoint"].relative_to(root).as_posix()
            and full.get("selected_checkpoint_sha256") == selected_hash
            and isinstance(selected_ref, dict)
            and selected_ref.get("sha256") == selected_hash
        )
        try:
            selected_state = strict_load(paths["selected_checkpoint"])
            checks["selected_checkpoint_strict_load"] = True
        except Exception as exc:
            selected_state = {}
            errors.append(f"selected checkpoint: {exc}")
        checks["selected_checkpoint_provenance"] = (
            selected_state.get("loss") == _EXPECTED_LOSS_CONFIGS.get(str(selected_arm), {}).get("loss")
            and selected_state.get("experiment_id") == f"gain_calibration_full_{selected_arm}_seed_{selected_seed}"
        )

        baseline_refs = full.get("baseline_evaluations")
        baseline_payloads: dict[str, dict[str, object]] = {}
        baseline_refs_ok = isinstance(baseline_refs, dict) and set(baseline_refs) == set(slices)
        for name in slices:
            ref = baseline_refs.get(name) if isinstance(baseline_refs, dict) else None
            path, ok = reference(ref)
            baseline_refs_ok &= ok
            if path is not None and path.is_file():
                baseline_payloads[name] = read_json(path)
        checks["baseline_evaluations"] = baseline_refs_ok and set(baseline_payloads) == set(slices)
        baseline_hash = _file_sha256(paths["baseline_checkpoint"])
        checks["baseline_checkpoint_binding"] = all(
            isinstance(payload.get("metadata"), dict)
            and payload["metadata"].get("checkpoint_sha256") == baseline_hash  # type: ignore[union-attr]
            for payload in baseline_payloads.values()
        ) and len(baseline_payloads) == 2
        try:
            strict_load(paths["baseline_checkpoint"])
            checks["baseline_checkpoint_strict_load"] = True
        except Exception as exc:
            errors.append(f"baseline checkpoint: {exc}")

        refs_ok = True
        config_ok = True
        checkpoint_hashes_ok = True
        checkpoint_loads_ok = True
        provenance_ok = True
        evaluation_hashes_ok = True
        slices_ok = True
        counts_ok = True
        finite_ok = True
        acceptance_ok = True
        recorded_acceptance_ok = True
        yaml_binding_ok = True
        validation_recomputation_ok = True
        validation_manifest_binding_ok = True
        validation_rank_rows: list[dict[str, object]] = []
        acceptance_by_seed: dict[int, dict[str, dict[str, object]]] = {}
        selected_row: dict[str, object] | None = None
        for row in rows:
            if not isinstance(row, dict):
                refs_ok = False
                continue
            if int(row.get("seed", -1)) == int(selected_seed):
                selected_row = row
            resolved_path, resolved_ok = reference(row.get("resolved_config"))
            yaml_path, yaml_ok = reference(row.get("resolved_yaml"))
            summary_path, summary_ok = reference(row.get("run_summary"))
            checkpoint_path, checkpoint_ok = reference(row.get("checkpoint"))
            validation_path, validation_ok = reference(row.get("validation_evaluation"))
            refs_ok &= resolved_ok and yaml_ok and summary_ok and checkpoint_ok and validation_ok
            if None in (resolved_path, yaml_path, summary_path, checkpoint_path, validation_path):
                config_ok = checkpoint_hashes_ok = checkpoint_loads_ok = False
                continue
            resolved = read_json(resolved_path)  # type: ignore[arg-type]
            summary = read_json(summary_path)  # type: ignore[arg-type]
            validation = read_json(validation_path)  # type: ignore[arg-type]
            config_hash = _file_sha256(resolved_path)  # type: ignore[arg-type]
            yaml_hash = _file_sha256(yaml_path)  # type: ignore[arg-type]
            checkpoint_hash = _file_sha256(checkpoint_path)  # type: ignore[arg-type]
            config_ok &= (
                summary.get("config_sha256") == config_hash
                and summary.get("resolved_yaml_sha256") == yaml_hash
                and summary.get("experiment_id") == f"gain_calibration_full_{selected_arm}_seed_{row.get('seed')}"
                and int(summary.get("seed", -1)) == int(row.get("seed", -2))
                and int(summary.get("max_steps", -1)) == 30000
                and int(summary.get("scheduler_total_steps", -1)) == 30000
                and all(resolved.get(key) == _EXPECTED_LOSS_CONFIGS.get(str(selected_arm), {}).get(key) for key in _LOSS_KEYS)
            )
            try:
                parsed_yaml = load_train_config(yaml_path, project_root=root).to_dict()  # type: ignore[arg-type]
                expected_run_dir = resolve(row.get("run_dir"))
                expected_experiment = f"gain_calibration_full_{selected_arm}_seed_{row.get('seed')}"
                yaml_binding_ok &= (
                    parsed_yaml.get("loss") == _EXPECTED_LOSS_CONFIGS.get(str(selected_arm), {}).get("loss")
                    and int(parsed_yaml.get("seed", -1)) == int(row.get("seed", -2))
                    and parsed_yaml.get("experiment_id") == expected_experiment
                    and int(parsed_yaml.get("max_steps", -1)) == 30000
                    and int(parsed_yaml.get("scheduler_total_steps", -1)) == 30000
                    and same_path(parsed_yaml.get("checkpoint_dir"), expected_run_dir)
                    and same_path(resolved.get("checkpoint_dir"), expected_run_dir)
                    and all(parsed_yaml.get(key) == _EXPECTED_LOSS_CONFIGS.get(str(selected_arm), {}).get(key) for key in _LOSS_KEYS)
                    and all(resolved.get(key) == parsed_yaml.get(key) for key in (*_LOSS_KEYS, "seed", "experiment_id", "max_steps", "scheduler_total_steps"))
                )
            except Exception as exc:
                yaml_binding_ok = False
                errors.append(f"resolved YAML seed {row.get('seed')}: {exc}")
            try:
                recomputed_validation = _evaluation_metrics(validation)
                validation_metadata = validation.get("metadata")
                claimed_validation_manifest = (
                    resolve(validation_metadata.get("manifest"))
                    if isinstance(validation_metadata, dict)
                    else Path()
                )
                validation_binding_ok = (
                    isinstance(validation_metadata, dict)
                    and claimed_validation_manifest == paths["validation_manifest"]
                    and validation_metadata.get("manifest_fingerprint") == validation_hash
                    and int(validation_metadata.get("slice_offset", -1)) == 0
                    and int(validation_metadata.get("slice_count", -1)) == 500
                    and int(validation_metadata.get("max_items", -1)) == 500
                    and validation_metadata.get("slice_fingerprint") == validation_slice_hash
                )
                validation_manifest_binding_ok &= validation_binding_ok
                validation_recomputation_ok &= (
                    recomputed_validation == row.get("validation_metrics")
                    and recomputed_validation == summary.get("validation_metrics")
                    and isinstance(validation_metadata, dict)
                    and validation_metadata.get("checkpoint_sha256") == checkpoint_hash
                    and validation_metadata.get("config_sha256") == config_hash
                    and summary.get("validation_evaluation_sha256") == _file_sha256(validation_path)  # type: ignore[arg-type]
                    and validation_binding_ok
                )
                if validation_binding_ok:
                    validation_rank_rows.append({"seed": int(row["seed"]), **recomputed_validation})
            except Exception as exc:
                validation_recomputation_ok = False
                validation_manifest_binding_ok = False
                errors.append(f"validation evaluation seed {row.get('seed')}: {exc}")
            checkpoint_hashes_ok &= summary.get("checkpoint_sha256") == checkpoint_hash
            try:
                state = strict_load(checkpoint_path)  # type: ignore[arg-type]
            except Exception as exc:
                checkpoint_loads_ok = False
                errors.append(f"run checkpoint seed {row.get('seed')}: {exc}")
                state = {}
            expected_loss = _EXPECTED_LOSS_CONFIGS.get(str(selected_arm), {})
            provenance_ok &= (
                state.get("loss") == expected_loss.get("loss")
                and state.get("loss_config") == expected_loss
                and state.get("experiment_id") == f"gain_calibration_full_{selected_arm}_seed_{row.get('seed')}"
                and 0 < int(state.get("global_step", -1)) <= 30000
                and isinstance(state.get("manifest_fingerprints"), dict)
                and state["manifest_fingerprints"].get("test") == test_hash  # type: ignore[union-attr]
            )
            eval_refs = row.get("evaluations")
            recomputed_row_acceptance: dict[str, dict[str, object]] = {}
            if not isinstance(eval_refs, dict) or set(eval_refs) != set(slices):
                evaluation_hashes_ok = False
                continue
            for name, (offset, slice_hash) in slices.items():
                eval_path, eval_ok = reference(eval_refs.get(name))
                evaluation_hashes_ok &= eval_ok
                if eval_path is None or not eval_path.is_file() or name not in baseline_payloads:
                    slices_ok = counts_ok = finite_ok = acceptance_ok = False
                    continue
                evaluation = read_json(eval_path)
                baseline = baseline_payloads[name]
                metadata = evaluation.get("metadata")
                baseline_metadata = baseline.get("metadata")
                slices_ok &= (
                    isinstance(metadata, dict) and isinstance(baseline_metadata, dict)
                    and metadata.get("checkpoint_sha256") == checkpoint_hash
                    and metadata.get("config_sha256") == config_hash
                    and metadata.get("manifest_fingerprint") == test_hash
                    and baseline_metadata.get("manifest_fingerprint") == test_hash
                    and int(metadata.get("slice_offset", -1)) == offset
                    and int(baseline_metadata.get("slice_offset", -1)) == offset
                    and int(metadata.get("slice_count", -1)) == 500
                    and int(baseline_metadata.get("slice_count", -1)) == 500
                    and metadata.get("slice_fingerprint") == slice_hash
                    and baseline_metadata.get("slice_fingerprint") == slice_hash
                )
                counts_ok &= all(
                    isinstance(payload.get(output), dict) and int(payload[output].get("n", -1)) == 500  # type: ignore[union-attr]
                    for payload in (evaluation, baseline) for output in ("noisy", "mask_only", "enhanced")
                )
                finite_ok &= all(
                    _all_finite_numeric({output: payload[output] for output in ("noisy", "mask_only", "enhanced")})
                    for payload in (evaluation, baseline)
                )
                try:
                    recomputed_gate = recompute_acceptance(evaluation, baseline)
                    recomputed_row_acceptance[name] = recomputed_gate
                    acceptance_ok &= recomputed_gate["status"] == "pass"
                except Exception:
                    acceptance_ok = False
            acceptance_by_seed[int(row.get("seed", -1))] = recomputed_row_acceptance
            recorded_acceptance_ok &= row.get("acceptance") == recomputed_row_acceptance

        checks.update(
            {
                "artifact_references": refs_ok,
                "resolved_config_hashes": config_ok,
                "resolved_yaml_binding": yaml_binding_ok,
                "validation_recomputation": validation_recomputation_ok,
                "validation_manifest_binding": validation_manifest_binding_ok,
                "run_checkpoint_hashes": checkpoint_hashes_ok,
                "run_checkpoint_strict_load": checkpoint_loads_ok,
                "loss_provenance": provenance_ok,
                "evaluation_hashes": evaluation_hashes_ok,
                "evaluation_slices": slices_ok,
                "evaluation_counts": counts_ok,
                "evaluation_finite": finite_ok,
                "acceptance_gates": acceptance_ok,
                "recorded_acceptance": recorded_acceptance_ok,
            }
        )

        validation_rank_rows.sort(
            key=lambda row: (
                -float(row["snri"]),
                -float(row["si_sdri"]),
                abs(float(row["gain_error_db_median"])),
                int(row["seed"]),
            )
        )
        selection_ref = full.get("selection_receipt")
        selection_path, selection_ref_ok = reference(selection_ref)
        selection = read_json(selection_path) if selection_path is not None and selection_path.is_file() else {}
        selection_core = dict(selection)
        claimed_selection_contract_hash = selection_core.pop("selection_contract_sha256", None)
        computed_selection_contract_hash = hashlib.sha256(
            (json.dumps(selection_core, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        ).hexdigest()
        expected_validation_refs = {
            str(row["seed"]): row["validation_evaluation"]
            for row in rows if isinstance(row, dict)
        }
        checks["selection_receipt"] = (
            selection_ref_ok
            and selection_path == paths["selection_receipt"]
            and selection.get("schema_version") == 1
            and selection.get("selection_basis") == "validation"
            and selection.get("selected_arm") == selected_arm
            and int(selection.get("selected_seed", -1)) == int(selected_seed)
            and selection.get("metric_order") == ["validation_snri", "validation_si_sdri", "absolute_gain_error"]
            and selection.get("ranked_validation_metrics") == validation_rank_rows
            and selection.get("validation_evaluations") == expected_validation_refs
            and claimed_selection_contract_hash == computed_selection_contract_hash
        )
        validation_selected_seed = int(validation_rank_rows[0]["seed"]) if len(validation_rank_rows) == 3 else -1
        checks["validation_recomputation"] &= validation_selected_seed == int(selected_seed)
        selected_gates = acceptance_by_seed.get(int(selected_seed), {})
        checks["selected_acceptance"] = (
            set(selected_gates) == {"comparison", "audit"}
            and all(gate.get("status") == "pass" for gate in selected_gates.values())
        )
        all_seed_acceptance = (
            set(acceptance_by_seed) == {17, 29, 43}
            and all(
                set(seed_gates) == {"comparison", "audit"}
                and all(gate.get("status") == "pass" for gate in seed_gates.values())
                for seed_gates in acceptance_by_seed.values()
            )
        )
        checks["all_seed_acceptance"] = (
            all_seed_acceptance
            and full.get("status") == "completed"
            and full.get("production_eligible") is True
            and all(isinstance(row, dict) and row.get("status") == "completed" for row in rows)
            and isinstance(selected_ref, dict)
            and selected_ref.get("production_eligible") is True
        )

        source_path = resolve(selected_ref.get("source_path")) if isinstance(selected_ref, dict) else Path()
        source_hash = selected_ref.get("source_sha256") if isinstance(selected_ref, dict) else None
        selected_checkpoint_ref = selected_row.get("checkpoint") if isinstance(selected_row, dict) else None
        checks["selected_run_binding"] = (
            isinstance(selected_checkpoint_ref, dict)
            and source_path == resolve(selected_checkpoint_ref.get("path"))
            and source_hash == selected_checkpoint_ref.get("sha256")
            and source_hash == selected_hash
        )
        rationale = full.get("selection_rationale")
        ranked = rationale.get("ranked_validation_metrics") if isinstance(rationale, dict) else None
        checks["selection_rationale"] = (
            isinstance(rationale, dict)
            and int(rationale.get("selected_seed", -1)) == int(selected_seed)
            and rationale.get("metric_order") == ["validation_snri", "validation_si_sdri", "absolute_gain_error"]
            and int(selected_seed) in {int(seed) for seed in rationale.get("eligible_seeds", [])}
            and isinstance(ranked, list) and bool(ranked)
            and isinstance(ranked[0], dict) and int(ranked[0].get("seed", -1)) == int(selected_seed)
            and ranked == validation_rank_rows
            and rationale.get("eligible_seeds") == [int(row["seed"]) for row in validation_rank_rows]
        )

        benchmark = read_json(paths["benchmark"])
        sections = (benchmark.get("neural_core"), benchmark.get("streaming_end_to_end"))
        checks["cuda_benchmark"] = (
            benchmark.get("checkpoint_sha256") == selected_hash
            and benchmark.get("device") == "cuda" and bool(benchmark.get("gpu"))
            and int(benchmark.get("iterations", 0)) >= 500
            and all(
                isinstance(section, dict) and int(section.get("n", 0)) >= 500
                and _all_finite_numeric({key: section.get(key) for key in ("mean_ms", "p95_ms", "max_ms", "realtime_factor")})
                for section in sections
            )
            and isinstance(benchmark.get("streaming_end_to_end"), dict)
            and float(benchmark["streaming_end_to_end"]["p95_ms"]) < 10.0  # type: ignore[index]
        )

        export = read_json(paths["export_metadata"])
        claimed_parity = read_json(paths["export_parity"])
        ts_hash = _file_sha256(paths["torchscript"])
        onnx_hash = _file_sha256(paths["onnx"])
        checks["export_hashes"] = (
            export.get("source_checkpoint_sha256") == selected_hash
            and claimed_parity.get("checkpoint_sha256") == selected_hash
            and claimed_parity.get("torchscript_sha256") == ts_hash
            and claimed_parity.get("onnx_sha256") == onnx_hash
            and claimed_parity.get("status") == "pass"
            and int(claimed_parity.get("steps", 0)) >= 20
        )
        try:
            torch.jit.load(str(paths["torchscript"]), map_location="cpu").eval()
            ort.InferenceSession(str(paths["onnx"]), providers=["CPUExecutionProvider"])
            checks["export_structure"] = True
        except Exception as exc:
            errors.append(f"export structure: {exc}")
        if checks["export_structure"]:
            computed = export_parity_audit(
                root,
                steps=20,
                checkpoint_path=paths["selected_checkpoint"],
                torchscript_path=paths["torchscript"],
                onnx_path=paths["onnx"],
            )
            checks["export_parity"] = computed.get("status") == "pass" and int(computed.get("steps", 0)) >= 20
    except Exception as exc:
        errors.append(f"metadata parse: {type(exc).__name__}: {exc}")

    return {
        "status": "pass" if all(checks.values()) else "fail",
        "artifacts": artifacts,
        "checks": checks,
        **({"errors": errors} if errors else {}),
    }


def gain_calibration_audit(root: Path) -> dict[str, object]:
    """Backward-compatible name for the calibrated artifact audit."""
    return gain_calibration_artifact_audit(root)


def production_readiness(root: Path) -> dict[str, object]:
    """Verify CPU-only runtime evidence and hash-bound deployment bundle."""
    root = Path(root).resolve()
    promoted = (root / "checkpoints/production/best.pth").is_file()
    export_root = root / ("checkpoints/production" if promoted else "checkpoints/gain_calibration")
    paths = {
        "checkpoint": export_root / "best.pth",
        "torchscript": export_root / "export.ts",
        "onnx": export_root / "export.onnx",
        "metadata": export_root / "export.json",
        "parity": root / ("reports/generated/production_export_parity.json" if promoted else "reports/generated/gain_calibration_export_parity.json"),
        "cpu_benchmark": root / "reports/generated/cpu_benchmark.json",
        "cpu_soak": root / "reports/generated/cpu_soak.json",
        "runtime_smoke": root / "reports/generated/cpu_runtime_smoke.json",
        "bundle": root / "artifacts/cpu_bundle/bundle.json",
    }
    checks = {
        "required_artifacts": all(path.is_file() and path.stat().st_size > 0 for path in paths.values()),
        "cpu_benchmark": False,
        "cpu_soak": False,
        "runtime_smoke": False,
        "export_hashes": False,
        "export_structure": False,
        "bundle_integrity": False,
    }
    if not checks["required_artifacts"]:
        return {"schema_version": 1, "status": "fail", "production_eligible": False, "checks": checks, "paths": {k: str(v) for k, v in paths.items()}}

    def read(path: Path) -> dict[str, object]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected object: {path}")
        return value

    errors: list[str] = []
    try:
        benchmark = read(paths["cpu_benchmark"])
        stream = benchmark.get("streaming_end_to_end")
        onnx_core = benchmark.get("onnx_core")
        checks["cpu_benchmark"] = (
            benchmark.get("status") == "pass"
            and benchmark.get("device") == "cpu"
            and int(benchmark.get("iterations", 0)) >= 5000
            and isinstance(stream, dict)
            and isinstance(onnx_core, dict)
            and float(stream.get("p95_ms", float("inf"))) < 10.0
            and float(onnx_core.get("p95_ms", float("inf"))) < 10.0
            and benchmark.get("onnx_sha256") == _file_sha256(paths["onnx"])
            and all(_all_finite_numeric({key: stream.get(key) for key in ("mean_ms", "p95_ms", "p99_ms", "max_ms", "realtime_factor")}) for _ in [0])
        )
        soak = read(paths["cpu_soak"])
        checks["cpu_soak"] = (
            soak.get("status") == "pass"
            and soak.get("device") == "cpu"
            and float(soak.get("requested_seconds", 0.0)) >= 7200.0
            and float(soak.get("elapsed_seconds", 0.0)) >= 7199.0
            and bool(soak.get("reset", {}).get("idempotent"))
            and all(bool(value) for value in soak.get("faults", {}).values())
        )
        runtime = read(paths["runtime_smoke"])
        runtime_output = Path(str(runtime.get("output")))
        if not runtime_output.is_absolute():
            runtime_output = root / runtime_output
        checks["runtime_smoke"] = (
            runtime.get("schema_version") == 1
            and runtime.get("status") == "pass"
            and runtime.get("device") == "cpu"
            and int(runtime.get("input_samples", -1)) == int(runtime.get("output_samples", -2))
            and runtime.get("checkpoint_sha256") == _file_sha256(paths["checkpoint"])
            and runtime_output.is_file()
            and runtime.get("output_sha256") == _file_sha256(runtime_output)
        )
        selected_hash = _file_sha256(paths["checkpoint"])
        metadata = read(paths["metadata"])
        parity = read(paths["parity"])
        checks["export_hashes"] = (
            metadata.get("source_checkpoint_sha256") == selected_hash
            and parity.get("checkpoint_sha256") == selected_hash
            and parity.get("torchscript_sha256") == _file_sha256(paths["torchscript"])
            and parity.get("onnx_sha256") == _file_sha256(paths["onnx"])
            and parity.get("status") == "pass"
        )
        import onnxruntime as ort
        import torch

        torch.jit.load(str(paths["torchscript"]), map_location="cpu").eval()
        ort.InferenceSession(str(paths["onnx"]), providers=["CPUExecutionProvider"])
        checks["export_structure"] = True
        bundle = read(paths["bundle"])
        files = bundle.get("files")
        checks["bundle_integrity"] = (
            isinstance(files, dict)
            and bundle.get("checkpoint_sha256") == selected_hash
            and all(
                isinstance(entry, dict)
                and (paths["bundle"].parent / name).is_file()
                and entry.get("sha256") == _file_sha256(paths["bundle"].parent / name)
                for name, entry in files.items()
            )
        )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    status = "pass" if all(checks.values()) else "fail"
    return {"schema_version": 1, "status": status, "production_eligible": status == "pass", "checks": checks, "paths": {k: str(v) for k, v in paths.items()}, **({"errors": errors} if errors else {})}


def research_readiness(root: Path) -> dict[str, object]:
    """Verify the five-seed research matrix and statistical evidence receipts."""
    root = Path(root).resolve()
    paths = {
        "training": root / "reports/generated/research_training.json",
        "evaluations": root / "reports/generated/research_evaluations.json",
        "robustness": root / "reports/generated/robustness_matrix.json",
        "significance": root / "reports/generated/significance.json",
        "failure_analysis": root / "reports/generated/failure_analysis.json",
        "failure_audio": root / "reports/generated/failure_audio.json",
        "selection": root / "reports/generated/research_selection.json",
    }
    checks = {"required_artifacts": all(path.is_file() and path.stat().st_size > 0 for path in paths.values()), "training_matrix": False, "evaluation_matrix": False, "robustness": False, "significance": False, "failure_analysis": False, "failure_audio": False, "selection": False}
    if not checks["required_artifacts"]:
        return {"schema_version": 1, "status": "fail", "research_eligible": False, "checks": checks, "paths": {k: str(v) for k, v in paths.items()}}

    def read(path: Path) -> dict[str, object]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected object: {path}")
        return value

    errors: list[str] = []
    try:
        training = read(paths["training"])
        training_rows = training.get("runs")
        training_hashes_ok = isinstance(training_rows, list)
        if isinstance(training_rows, list):
            for row in training_rows:
                checkpoint = Path(str(row.get("checkpoint"))) if isinstance(row, dict) else Path()
                if not checkpoint.is_absolute():
                    checkpoint = root / checkpoint
                training_hashes_ok &= checkpoint.is_file() and isinstance(row, dict) and row.get("checkpoint_sha256") == _file_sha256(checkpoint)
        checks["training_matrix"] = (
            training.get("schema_version") == 1
            and training.get("stage") == "research"
            and training.get("status") == "completed"
            and training.get("seeds") == [17, 29, 43, 59, 71]
            and isinstance(training_rows, list)
            and len(training_rows) == 5
            and all(isinstance(row, dict) and row.get("status") in {"completed", "already-complete"} for row in training_rows)
            and training_hashes_ok
        )
        evaluations = read(paths["evaluations"])
        evaluation_rows = evaluations.get("runs")
        checks["evaluation_matrix"] = (
            evaluations.get("schema_version") == 1
            and evaluations.get("stage") == "research-evaluation"
            and evaluations.get("status") == "completed"
            and evaluations.get("seeds") == [17, 29, 43, 59, 71]
            and evaluations.get("all_seed_acceptance") is True
            and isinstance(evaluation_rows, list)
            and len(evaluation_rows) == 5
            and all(isinstance(row, dict) and row.get("status") == "pass" and set(row.get("evaluations", {})) == {"comparison", "audit"} for row in evaluation_rows)
        )
        robustness = read(paths["robustness"])
        robustness_rows = robustness.get("rows")
        checks["robustness"] = (
            robustness.get("schema_version") == 1
            and robustness.get("status") == "pass"
            and int(robustness.get("slices", 0)) >= 5
            and isinstance(robustness_rows, list)
            and all(
                isinstance(row, dict)
                and float(row.get("snri_delta", float("-inf"))) > 0.0
                and float(row.get("si_sdri_delta", float("-inf"))) >= -0.3
                and float(row.get("stoi_delta", float("-inf"))) >= -0.005
                for row in robustness_rows
            )
        )
        significance = read(paths["significance"])
        significance_metrics = significance.get("metrics")
        checks["significance"] = (
            significance.get("schema_version") == 1
            and significance.get("status") == "pass"
            and int(significance.get("paired_items", 0)) >= 500
            and isinstance(significance_metrics, dict)
            and set(significance_metrics) >= {"si_sdr", "snr", "stoi"}
            and all(
                isinstance(metric, dict)
                and isinstance(metric.get("bootstrap"), dict)
                and int(metric["bootstrap"].get("resamples", 0)) >= 10000
                for metric in significance_metrics.values()
            )
            and float(significance_metrics["snr"]["bootstrap"].get("ci95_low", float("-inf"))) > 0.0
            and float(significance_metrics["stoi"]["bootstrap"].get("ci95_low", float("-inf"))) >= -0.005
            and float(significance_metrics["si_sdr"]["bootstrap"].get("ci95_low", float("-inf"))) >= -0.3
        )
        failures = read(paths["failure_analysis"])
        checks["failure_analysis"] = failures.get("schema_version") == 1 and int(failures.get("n", 0)) >= 500 and isinstance(failures.get("worst_cases"), list)
        failure_audio = read(paths["failure_audio"])
        audio_items = failure_audio.get("items")
        audio_refs_ok = True
        if isinstance(audio_items, list):
            for item in audio_items:
                files = item.get("files") if isinstance(item, dict) else None
                if not isinstance(files, dict) or set(files) != {"clean", "noisy", "enhanced"}:
                    audio_refs_ok = False
                    continue
                for ref in files.values():
                    if not isinstance(ref, dict):
                        audio_refs_ok = False
                        continue
                    path = root / str(ref.get("path"))
                    audio_refs_ok &= path.is_file() and root in path.resolve().parents and ref.get("sha256") == _file_sha256(path)
        checks["failure_audio"] = failure_audio.get("schema_version") == 1 and failure_audio.get("status") == "pass" and isinstance(audio_items, list) and len(audio_items) >= 10 and audio_refs_ok
        selection = read(paths["selection"])
        production_checkpoint = Path(str(selection.get("production_checkpoint")))
        if not production_checkpoint.is_absolute():
            production_checkpoint = root / production_checkpoint
        checks["selection"] = (
            selection.get("schema_version") == 1
            and selection.get("status") == "pass"
            and selection.get("selection_basis") == "validation"
            and int(selection.get("selected_seed", -1)) in {17, 29, 43, 59, 71}
            and production_checkpoint.is_file()
            and selection.get("production_checkpoint_sha256") == _file_sha256(production_checkpoint)
        )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    status = "pass" if all(checks.values()) else "fail"
    return {"schema_version": 1, "status": status, "research_eligible": status == "pass", "checks": checks, "paths": {k: str(v) for k, v in paths.items()}, **({"errors": errors} if errors else {})}


def export_parity_audit(
    root: Path,
    *,
    steps: int = 20,
    tolerance: float = 2e-4,
    checkpoint_path: Path | None = None,
    torchscript_path: Path | None = None,
    onnx_path: Path | None = None,
) -> dict[str, object]:
    """Verify exported recurrent state against eager streaming over many hops."""
    checkpoint = checkpoint_path or root / "checkpoints" / "best.pth"
    torchscript = torchscript_path or root / "checkpoints" / "full_best_export.ts"
    onnx = onnx_path or root / "checkpoints" / "full_best_export.onnx"
    try:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import numpy as np
        import onnxruntime as ort
        import torch

        from src.checkpoint import validate_checkpoint_metadata
        from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig

        state = torch.load(checkpoint, map_location="cpu")
        validate_checkpoint_metadata(state)
        model = MobileDeepFilterNet(MobileDeepFilterNetConfig(**state["model_cfg"])).eval()
        model.load_state_dict(state["model"], strict=True)
        scripted = torch.jit.load(str(torchscript), map_location="cpu").eval()
        session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])

        torch.manual_seed(42)
        hidden_eager = torch.zeros(model.cfg.gru_layers, 1, model.cfg.gru_hidden)
        hidden_scripted = hidden_eager.clone()
        hidden_onnx = hidden_eager.numpy()
        ts_errors = [0.0, 0.0, 0.0]
        onnx_errors = [0.0, 0.0, 0.0]
        output_shapes: list[list[int]] = []
        with torch.no_grad():
            for _ in range(steps):
                features = torch.randn(1, 1, model.cfg.freq_bins, model.temporal_receptive_frames)
                eager = model.forward_streaming(features, hidden_eager)
                traced = scripted(features, hidden_scripted)
                ort_outputs = session.run(
                    None,
                    {"feats_logp": features.numpy(), "hidden": hidden_onnx},
                )
                output_shapes = [list(value.shape) for value in eager]
                for index, expected in enumerate(eager):
                    ts_error = _finite_or_inf(
                        float(torch.max(torch.abs(expected - traced[index])).item())
                    )
                    onnx_error = _finite_or_inf(
                        float(np.max(np.abs(expected.numpy() - ort_outputs[index])))
                    )
                    ts_errors[index] = max(
                        ts_errors[index],
                        ts_error,
                    )
                    onnx_errors[index] = max(
                        onnx_errors[index],
                        onnx_error,
                    )
                hidden_eager = eager[2]
                hidden_scripted = traced[2]
                hidden_onnx = ort_outputs[2]
        status = "pass" if max(ts_errors + onnx_errors) <= tolerance else "fail"
        return {
            "schema_version": 2,
            "status": status,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _file_sha256(checkpoint),
            "torchscript": str(torchscript),
            "torchscript_sha256": _file_sha256(torchscript),
            "onnx": str(onnx),
            "onnx_sha256": _file_sha256(onnx),
            "seed": 42,
            "steps": int(steps),
            "tolerance": float(tolerance),
            "input_shape": [1, 1, model.cfg.freq_bins, model.temporal_receptive_frames],
            "output_shapes": output_shapes,
            "eager_vs_torchscript_max_abs_error": ts_errors,
            "eager_vs_onnx_max_abs_error": onnx_errors,
        }

    except Exception as exc:
        return {
            "schema_version": 2,
            "status": "fail",
            "error": f"{type(exc).__name__}: {exc}",
        }


def publish_readiness(root: Path) -> dict[str, object]:
    """Fail-closed audit for the public repository surface.

    This check intentionally reads only curated evidence under ``reports/public``;
    generated/private reports must never be linked from the public README.
    """
    root = Path(root).resolve()
    if __package__:
        from .verify_public_production import verify_public_production
        from .verify_public_research import audit_public_research
    else:
        from verify_public_production import verify_public_production
        from verify_public_research import audit_public_research

    public_production = verify_public_production(root)
    public_research = audit_public_research(root)
    required_docs = (
        "README.md", "LICENSE", "CITATION.cff", "SECURITY.md", "CONTRIBUTING.md",
        "CHANGELOG.md", "MODEL_CARD.md", "docs/research-comparison.md",
    )
    required_receipts = (
        "reports/public/deepfilternet3_reproduction.json",
        "reports/public/model_comparison.json",
        "reports/public/production_readiness_verify.json",
        "reports/public/research_readiness.json",
        "reports/public/rnnoise_build.json",
    )
    checks: dict[str, bool] = {}
    readme = root / "README.md"
    readme_text = readme.read_text(encoding="utf-8", errors="replace") if readme.is_file() else ""
    license_path = root / "LICENSE"
    license_text = license_path.read_text(encoding="utf-8", errors="replace") if license_path.is_file() else ""
    apache_markers = (
        "Apache License",
        "Version 2.0, January 2004",
        "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION",
        "1. Definitions.",
        "2. Grant of Copyright License.",
        "3. Grant of Patent License.",
        "4. Redistribution.",
        "5. Submission of Contributions.",
        "6. Trademarks.",
        "7. Disclaimer of Warranty.",
        "8. Limitation of Liability.",
        "9. Accepting Warranty or Additional Liability.",
        "END OF TERMS AND CONDITIONS",
    )
    checks["license_spdx"] = len(license_text) >= 10_000 and all(marker in license_text for marker in apache_markers)
    checks["required_documents"] = checks["license_spdx"] and all(
        (root / path).is_file() and bool((root / path).read_text(encoding="utf-8", errors="replace").strip())
        for path in required_docs
    )
    checks["public_receipts"] = all((root / path).is_file() for path in required_receipts)
    receipt_hygiene_errors: list[str] = []
    absolute_path = re.compile(r"^[A-Za-z]:[\\/]|^[/\\]{2}")
    for relative in required_receipts:
        candidate = root / relative
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            receipt_hygiene_errors.append(f"{relative}: invalid JSON")
            continue
        pending: list[object] = [payload]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, list):
                pending.extend(value)
            elif isinstance(value, str) and (absolute_path.match(value) or ".venv" in value.replace("\\", "/")):
                receipt_hygiene_errors.append(f"{relative}: local path")
                break
    checks["public_receipt_hygiene"] = not receipt_hygiene_errors
    checks["readme_public_evidence"] = bool(readme_text) and "reports/generated/" not in readme_text and all(
        path in readme_text for path in required_receipts
    )
    placeholder_tokens = ("example.com", "your-org", "your-username", "owner/repository", "TODO", "TBD", "<owner>")
    public_text = readme_text
    for relative in (*required_docs[1:], "pyproject.toml", "configs/desktop_assets.json", ".github/workflows/ci.yml", ".github/workflows/release.yml"):
        candidate = root / relative
        if candidate.is_file():
            public_text += "\n" + candidate.read_text(encoding="utf-8", errors="replace")
    checks["no_placeholders"] = not any(token.lower() in public_text.lower() for token in placeholder_tokens)
    citation_text = (root / "CITATION.cff").read_text(encoding="utf-8", errors="replace") if (root / "CITATION.cff").is_file() else ""
    version = ""
    project = root / "pyproject.toml"
    if project.is_file():
        try:
            import tomllib
            version = str(tomllib.loads(project.read_text(encoding="utf-8")).get("project", {}).get("version", ""))
        except Exception:
            version = ""
    citation_version = ""
    for line in citation_text.splitlines():
        if line.strip().startswith("version:"):
            citation_version = line.split(":", 1)[1].strip().strip("\"'")
            break
    checks["version_consistency"] = bool(version) and version == citation_version
    checks["workflow_surface"] = all((root / path).is_file() for path in (".github/workflows/ci.yml", ".github/workflows/release.yml"))
    checks["package_metadata"] = "clearhop" in (project.read_text(encoding="utf-8", errors="replace").lower() if project.is_file() else "")
    checks["inventory"] = True
    inventory_errors: list[str] = []
    ignored_dirs = {".git", ".venv", "venv", ".artifacts", "raw_data", "data", "train-clean-360", "checkpoints", "runs", "build", "dist", "node_modules"}
    allow_large = {"raw_data.zip"}
    for directory, names, files in os.walk(root):
        base = Path(directory)
        names[:] = [name for name in names if name not in ignored_dirs and (base / name).relative_to(root).as_posix() != "reports/generated"]
        for name in files:
            path = base / name
            rel = path.relative_to(root).as_posix()
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > 50 * 1024 * 1024 and path.name not in allow_large and "reports/public" not in rel:
                inventory_errors.append(rel)
            if path.suffix.lower() in {".pem", ".key", ".p12", ".env"}:
                inventory_errors.append(rel)
    checks["inventory"] = not inventory_errors
    tracked_exclusions: list[str] = []
    if (root / ".git").exists():
        tracked = subprocess.run(
            ["git", "ls-files", "--", "raw_data.zip", "raw_data", "data", "checkpoints", "reports/generated"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        if tracked.returncode != 0:
            tracked_exclusions.append("git ls-files audit failed")
        else:
            tracked_exclusions.extend(line for line in tracked.stdout.splitlines() if line.strip())
    checks["excluded_payloads_untracked"] = not tracked_exclusions
    public_comparison: dict[str, object] = {}
    comparison_path = root / "reports/public/model_comparison.json"
    if comparison_path.is_file():
        try:
            public_comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        except Exception:
            public_comparison = {}
    checks["research_status_separated"] = bool(
        isinstance(public_comparison, dict)
        and isinstance(public_comparison.get("models"), list)
        and all(isinstance(row, dict) and str(row.get("status", "")).lower() in {"pass", "blocked", "reproduced", "reproduced_local", "literature-only"} for row in public_comparison["models"])
    )
    checks["research_receipt_schema"] = bool(
        isinstance(public_comparison, dict)
        and isinstance(public_comparison.get("checks"), dict)
        and all(value is True for value in public_comparison["checks"].values())
    )
    external_rows = [
        row for row in public_comparison.get("models", [])
        if isinstance(row, dict) and row.get("name") not in {"ClearHop", "Frozen historical baseline"}
    ] if isinstance(public_comparison, dict) else []
    research_coverage = {
        "external_total": len(external_rows),
        "external_reproduced": sum(str(row.get("status", "")).lower() in {"pass", "reproduced", "reproduced_local"} for row in external_rows),
        "external_blocked": sum(str(row.get("status", "")).lower() == "blocked" for row in external_rows),
        "comparison_status": public_comparison.get("status") if isinstance(public_comparison, dict) else None,
    }
    audited_coverage = public_research.get("coverage") if isinstance(public_research, dict) else None
    if not isinstance(audited_coverage, dict):
        audited_coverage = {}
    verified_recipes_value = audited_coverage.get("verified_blocker_recipes", 0)
    verified_recipes = verified_recipes_value if isinstance(verified_recipes_value, int) and not isinstance(verified_recipes_value, bool) and verified_recipes_value >= 0 else 0
    if research_coverage["external_reproduced"] >= 2:
        coverage_tier = "two_reproduced"
    elif research_coverage["external_reproduced"] == 1 and verified_recipes >= 1:
        coverage_tier = "one_plus_recipe"
    else:
        coverage_tier = "insufficient"
    research_coverage.update({
        "verified_blocker_recipes": verified_recipes,
        "eligible": coverage_tier != "insufficient",
        "tier": coverage_tier,
    })
    checks["production_receipt"] = public_production.get("status") == "pass"
    checks["research_receipt"] = public_research.get("status") == "pass"
    github_checks = ("required_documents", "license_spdx", "public_receipts", "public_receipt_hygiene", "readme_public_evidence", "no_placeholders", "version_consistency", "workflow_surface", "inventory", "excluded_payloads_untracked")
    production_checks = ("package_metadata", "workflow_surface", "production_receipt", "public_receipts", "version_consistency")
    research_checks = ("research_status_separated", "research_receipt_schema", "research_receipt", "public_receipts", "no_placeholders")
    scores = {name: round(10.0 * sum(checks.get(key, False) for key in group) / len(group), 2) for name, group in (("github", github_checks), ("production", production_checks), ("research", research_checks))}
    research_caps = {"two_reproduced": 10.0, "one_plus_recipe": 9.0, "insufficient": 8.0}
    scores["research"] = min(scores["research"], research_caps.get(str(research_coverage["tier"]), 8.0))
    status = "pass" if all(score >= 9.0 for score in scores.values()) else "fail"
    return {
        "schema_version": 1,
        "status": status,
        "scores": scores,
        "score_scope": "repository and evidence readiness; research score is capped by verified external-baseline coverage",
        "research_coverage": research_coverage,
        "public_evidence": {
            "production": public_production,
            "research": public_research,
        },
        "checks": checks,
        "inventory_errors": inventory_errors,
        "tracked_exclusion_errors": tracked_exclusions,
        "receipt_hygiene_errors": receipt_hygiene_errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.yaml"))
    parser.add_argument("--smoke-train", action="store_true", help="Run one bounded train/eval/export path when dependencies exist.")
    parser.add_argument("--gain-calibration", action="store_true", help="Audit calibrated full-run evidence and acceptance gates.")
    parser.add_argument("--production-readiness", action="store_true", help="Audit CPU-only production evidence and bundle integrity.")
    parser.add_argument("--research-readiness", action="store_true", help="Audit five-seed research evidence and statistics.")
    parser.add_argument("--publish-readiness", action="store_true", help="Audit GitHub, production distribution, and public research surfaces.")
    parser.add_argument("--output", type=Path, default=Path("reports/generated/verify.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    python = Path(sys.executable)

    results: dict[str, object] = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": str(python),
        "torch_available": importlib.util.find_spec("torch") is not None,
        "soundfile_available": importlib.util.find_spec("soundfile") is not None,
    }
    if results["torch_available"]:
        import torch

        results["torch_version"] = torch.__version__
        results["cuda_available"] = bool(torch.cuda.is_available())
        results["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    results["compileall"] = run([str(python), "-m", "compileall", "-q", "src", "scripts", "tests"], root)

    stdlib_modules = ["tests.test_config", "tests.test_dataset", "tests.test_splits", "tests.test_checkpoint"]
    results["stdlib_tests"] = run([str(python), "-m", "unittest", *stdlib_modules, "-v"], root)

    full_tests = run([str(python), "-m", "unittest", "discover", "-s", "tests", "-v"], root)
    if not results["torch_available"]:
        full_tests["status"] = "dependency-gated"
        full_tests["reason"] = "PyTorch-dependent tests were skipped because torch is not installed."
    results["full_tests"] = full_tests

    results["manifest_audit"] = manifest_audit(root)
    require_full = args.config.name == "train.yaml" and not args.gain_calibration
    if require_full:
        parity = export_parity_audit(root)
        parity_path = root / "reports" / "generated" / "full_export_parity.json"
        parity_path.parent.mkdir(parents=True, exist_ok=True)
        parity_path.write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        results["export_parity_audit"] = parity
    if args.publish_readiness:
        results["publish_readiness"] = publish_readiness(root)
    elif args.production_readiness:
        results["production_readiness"] = production_readiness(root)
    elif args.research_readiness:
        results["research_readiness"] = research_readiness(root)
    elif args.gain_calibration:
        results["gain_calibration_audit"] = gain_calibration_audit(root)
    else:
        results["artifact_audit"] = artifact_audit(root, require_full=require_full)

    if args.smoke_train and results["torch_available"]:
        config = str(args.config if args.config.is_absolute() else root / args.config)
        results["smoke_train"] = run([str(python), "scripts/train.py", "--config", config], root)
    elif args.smoke_train:
        results["smoke_train"] = {"status": "blocked", "reason": "PyTorch is not installed in the active runtime."}

    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))

    compile_ok = results["compileall"].get("returncode") == 0  # type: ignore[union-attr]
    tests_ok = results["stdlib_tests"].get("returncode") == 0  # type: ignore[union-attr]
    full_tests_ok = results["full_tests"].get("returncode") == 0  # type: ignore[union-attr]
    manifest_ok = results["manifest_audit"].get("status") == "pass"  # type: ignore[union-attr]
    audit_key = "publish_readiness" if args.publish_readiness else ("production_readiness" if args.production_readiness else ("research_readiness" if args.research_readiness else ("gain_calibration_audit" if args.gain_calibration else "artifact_audit")))
    artifacts_ok = results[audit_key].get("status") == "pass"  # type: ignore[union-attr]
    parity_ok = not require_full or results["export_parity_audit"].get("status") == "pass"  # type: ignore[union-attr]
    return 0 if compile_ok and tests_ok and full_tests_ok and manifest_ok and artifacts_ok and parity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
