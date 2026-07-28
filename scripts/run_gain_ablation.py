"""Plan and execute the bounded gain-calibration experiment sequentially."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from src.config import load_train_config
from src.splits import manifest_fingerprint, slice_fingerprint


DEFAULT_ARMS = ("control", "complex_nmse", "complex_nmse_sisdr", "compressed_complex")
DEFAULT_SEEDS = (17, 29, 43)
RESEARCH_SEEDS = (17, 29, 43, 59, 71)
REFINEMENT_ARMS = (
    "complex_nmse_sisdr_beta_0p015",
    "complex_nmse_sisdr_beta_0p02",
    "complex_nmse_sisdr_beta_0p025",
    "complex_nmse_sisdr_beta_0p03",
)
_CONFIG_DIR = Path("configs/ablations/gain_calibration")
_LOSS_CONFIG_KEYS = (
    "loss",
    "alpha_loss",
    "beta_si_sdr",
    "loss_eps",
    "sisdr_warmup_start",
    "sisdr_warmup_end",
    "compression_exponent",
    "compression_complex_weight",
)
_VALIDATION_OFFSET = 0
_VALIDATION_COUNT = 500


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_new(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(_json_bytes(payload))


def _write_report(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json_bytes(payload))
    temporary.replace(path)


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        if _sha256(temporary) != _sha256(source):
            raise OSError(f"Atomic checkpoint copy hash mismatch: {source}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _csv_values(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("CSV value must not be empty")
    if len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("CSV value contains duplicates")
    return values


def build_plan(
    root: Path,
    stage: str,
    arms: Iterable[str] | None = None,
    seeds: Iterable[int] = DEFAULT_SEEDS,
) -> list[dict[str, Any]]:
    """Resolve a deterministic bounded plan without touching output directories."""
    root = Path(root).resolve()
    arm_list = list(
        arms
        if arms is not None
        else (REFINEMENT_ARMS[:1] if stage == "refine" else DEFAULT_ARMS)
    )
    seed_list = [int(seed) for seed in seeds]
    if stage not in {"screen", "refine", "full", "research"}:
        raise ValueError("stage must be screen, refine, full, or research")
    if not arm_list or not seed_list:
        raise ValueError("At least one arm and seed are required")
    if len(arm_list) != len(set(arm_list)) or len(seed_list) != len(set(seed_list)):
        raise ValueError("Arms and seeds must be unique")
    unknown_arms = sorted(set(arm_list) - set(DEFAULT_ARMS) - set(REFINEMENT_ARMS))
    if unknown_arms:
        raise ValueError(f"Unknown arms: {', '.join(unknown_arms)}")
    allowed_seeds = RESEARCH_SEEDS if stage == "research" else DEFAULT_SEEDS
    unknown_seeds = sorted(set(seed_list) - set(allowed_seeds))
    if unknown_seeds:
        raise ValueError(f"Unsupported seeds: {unknown_seeds}")
    if stage == "full" and len(arm_list) != 1:
        raise ValueError("Full stage requires exactly one arm")
    if stage == "full" and tuple(sorted(seed_list)) != tuple(sorted(DEFAULT_SEEDS)):
        raise ValueError("Full stage requires exactly seeds 17,29,43")
    if stage == "research" and (len(arm_list) != 1 or tuple(sorted(seed_list)) != tuple(sorted(RESEARCH_SEEDS))):
        raise ValueError("Research stage requires exactly one arm and seeds 17,29,43,59,71")
    if stage == "refine" and (
        len(arm_list) != 1
        or arm_list[0] not in REFINEMENT_ARMS
        or tuple(sorted(seed_list)) != tuple(sorted(DEFAULT_SEEDS))
    ):
        raise ValueError("Refine stage requires the locked beta-0.015 arm and exactly seeds 17,29,43")

    max_steps = 1500 if stage in {"screen", "refine"} else 30000
    plan: list[dict[str, Any]] = []
    for arm in arm_list:
        config_path = root / _CONFIG_DIR / f"{arm}.yaml"
        config = load_train_config(config_path, project_root=root).to_dict()
        if int(config.get("scheduler_total_steps") or 0) != 30000:
            raise ValueError(f"{config_path} must lock scheduler_total_steps=30000")
        for seed in seed_list:
            run_dir = root / "checkpoints" / "gain_calibration" / stage / arm / f"seed_{seed}"
            resolved = dict(config)
            resolved.update(
                {
                    "checkpoint_dir": str(run_dir),
                    "experiment_id": f"gain_calibration_{stage}_{arm}_seed_{seed}",
                    "max_steps": max_steps,
                    "scheduler_total_steps": 30000,
                    "seed": seed,
                }
            )
            plan.append(
                {
                    "stage": stage,
                    "arm": arm,
                    "seed": seed,
                    "run_dir": run_dir,
                    "config_path": config_path,
                    "config": resolved,
                    "max_steps": max_steps,
                    "scheduler_total_steps": 30000,
                }
            )
    return plan


def classify_run(run_dir: Path, *, max_steps: int, resume: bool) -> dict[str, Any]:
    """Choose a collision-safe action. Existing data is never deleted or reset."""
    run_dir = Path(run_dir)
    if not run_dir.exists() or not any(run_dir.iterdir()):
        return {"status": "new"}
    summary_path = run_dir / "run_summary.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
        if summary.get("status") == "completed" and int(summary.get("global_step", -1)) == max_steps:
            return {"status": "already-complete"}
    if not resume:
        raise FileExistsError(f"Incomplete run exists at {run_dir}; pass --resume to continue it")
    candidates: list[tuple[int, Path]] = []
    for path in run_dir.glob("step_*.pth"):
        match = re.fullmatch(r"step_(\d+)\.pth", path.name)
        if match and int(match.group(1)) <= max_steps:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No resumable step checkpoint exists in {run_dir}")
    checkpoint = max(candidates, key=lambda item: item[0])[1]
    return {"status": "resume", "checkpoint": checkpoint}


def _write_flat_yaml(path: Path, config: dict[str, Any]) -> None:
    lines = []
    for key, value in config.items():
        # project_root is runtime metadata, not part of TrainConfig's YAML schema.
        if key == "project_root":
            continue
        if isinstance(value, tuple):
            value = list(value)
        lines.append(f"{key}: {json.dumps(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolved_evidence(root: Path, item: dict[str, Any]) -> dict[str, str]:
    """Validate the immutable JSON/YAML pair actually passed to training."""
    run_dir = Path(item["run_dir"])
    resolved_path = run_dir / "resolved_config.json"
    resolved_yaml = run_dir / "resolved_config.yaml"
    if not resolved_path.is_file() or resolved_path.read_bytes() != _json_bytes(item["config"]):
        raise ValueError(f"Resolved JSON evidence drift: {resolved_path}")
    if not resolved_yaml.is_file():
        raise ValueError(f"Resolved YAML evidence missing: {resolved_yaml}")
    loaded = load_train_config(resolved_yaml, project_root=root).to_dict()
    exact_keys = (
        "experiment_id",
        "seed",
        "max_steps",
        "scheduler_total_steps",
        "checkpoint_dir",
        *_LOSS_CONFIG_KEYS,
    )
    mismatches = {
        key: {"expected": item["config"].get(key), "actual": loaded.get(key)}
        for key in exact_keys
        if loaded.get(key) != item["config"].get(key)
    }
    if mismatches:
        raise ValueError(f"Resolved YAML evidence mismatch: {mismatches}")
    return {
        "config_sha256": _sha256(resolved_path),
        "resolved_yaml_sha256": _sha256(resolved_yaml),
    }


def _resolve_inside(root: Path, value: object) -> Path:
    path = Path(str(value))
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Evidence path escapes project root: {value}")
    return resolved


def _artifact_ref(root: Path, path: Path) -> dict[str, str]:
    resolved = _resolve_inside(root, path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Artifact missing: {resolved}")
    return {"path": resolved.relative_to(root).as_posix(), "sha256": _sha256(resolved)}


def _validate_validation_binding(root: Path, evaluation: dict[str, Any]) -> dict[str, Any]:
    manifest = (Path(root).resolve() / "manifests/v2/fold_0_val.jsonl").resolve()
    metadata = evaluation.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Validation evaluation metadata must be an object")
    try:
        claimed_manifest = Path(str(metadata.get("manifest")))
        claimed_manifest = (
            claimed_manifest.resolve()
            if claimed_manifest.is_absolute()
            else (Path(root).resolve() / claimed_manifest).resolve()
        )
        expected = {
            "manifest_fingerprint": manifest_fingerprint(manifest),
            "slice_offset": _VALIDATION_OFFSET,
            "slice_count": _VALIDATION_COUNT,
            "slice_fingerprint": slice_fingerprint(manifest, _VALIDATION_OFFSET, _VALIDATION_COUNT),
            "max_items": _VALIDATION_COUNT,
            "evaluation_profile": "screen",
        }
    except Exception as exc:
        raise ValueError(f"Invalid validation manifest binding: {exc}") from exc
    mismatch = {
        key: (metadata.get(key), value)
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if claimed_manifest != manifest:
        mismatch["manifest"] = (str(claimed_manifest), str(manifest))
    if mismatch:
        raise ValueError(f"Invalid validation manifest binding: {mismatch}")
    return expected


def _validate_completed_evidence(root: Path, item: dict[str, Any]) -> dict[str, Any]:
    run_dir = Path(item["run_dir"])
    hashes = _resolved_evidence(root, item)
    summary_path = run_dir / "run_summary.json"
    try:
        summary = _read_json(summary_path)
    except Exception as exc:
        raise ValueError(f"Completed run evidence is malformed: {summary_path}: {exc}") from exc
    exact = {
        "status": "completed",
        "arm": item["arm"],
        "seed": item["seed"],
        "global_step": item["max_steps"],
        "experiment_id": item["config"]["experiment_id"],
        "max_steps": item["max_steps"],
        "scheduler_total_steps": item["scheduler_total_steps"],
        **hashes,
    }
    mismatch = {key: (summary.get(key), value) for key, value in exact.items() if summary.get(key) != value}
    if mismatch:
        raise ValueError(f"Completed run evidence mismatch: {mismatch}")
    checkpoint = _resolve_inside(root, summary.get("checkpoint"))
    evaluation = run_dir / "validation_evaluation.json"
    if not checkpoint.is_file() or summary.get("checkpoint_sha256") != _sha256(checkpoint):
        raise ValueError("Completed run checkpoint evidence mismatch")
    if not evaluation.is_file() or summary.get("validation_evaluation_sha256") != _sha256(evaluation):
        raise ValueError("Completed run evaluation evidence mismatch")
    parsed = _read_json(evaluation)
    _validate_validation_binding(root, parsed)
    if _evaluation_metrics(parsed) != summary.get("validation_metrics"):
        raise ValueError("Completed run validation metrics are stale")
    return summary


def _manifest_hashes(root: Path) -> dict[str, str]:
    result = {}
    for name in ("train", "val", "test"):
        path = root / "manifests" / "v2" / f"fold_0_{name}.jsonl"
        if path.is_file():
            result[name] = _sha256(path)
    return result


def _gpu_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {"torch_available": importlib.util.find_spec("torch") is not None}
    if not metadata["torch_available"]:
        return metadata
    try:
        import torch

        metadata.update(
            {
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        )
    except Exception as exc:
        metadata["metadata_error"] = f"{type(exc).__name__}: {exc}"
    return metadata


def _run(command: list[str], root: Path) -> tuple[subprocess.CompletedProcess[str], float]:
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
    return completed, time.perf_counter() - started


def _evaluation_metrics(evaluation: dict[str, Any]) -> dict[str, float]:
    enhanced = evaluation.get("enhanced", {})
    metrics = {
        "snri": enhanced.get("snr_improvement_mean"),
        "si_sdri": enhanced.get("si_sdr_improvement_mean"),
        "stoi": enhanced.get("stoi", {}).get("mean"),
        "projection_gain_median": enhanced.get("projection_gain", {}).get("median"),
        "gain_error_db_median": enhanced.get("gain_error_db", {}).get("median"),
    }
    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for value in metrics.values()
    ):
        raise FloatingPointError("Validation evaluation contains missing or non-finite promotion metrics")
    return {key: float(value) for key, value in metrics.items()}


def _checkpoint_metadata(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    try:
        import torch

        state = torch.load(path, map_location="cpu")
        return dict(state.get("manifest_fingerprints", {})), dict(state.get("runtime", {}))
    except Exception as exc:
        return {}, {"metadata_error": f"{type(exc).__name__}: {exc}"}


def execute_run(root: Path, item: dict[str, Any], *, resume: bool) -> dict[str, Any]:
    """Execute one train/evaluate pair. Callers serialize invocations."""
    root = Path(root).resolve()
    run_dir = Path(item["run_dir"])
    action = classify_run(run_dir, max_steps=int(item["max_steps"]), resume=resume)
    if action["status"] == "already-complete":
        existing = _validate_completed_evidence(root, item)
        return {**existing, "status": "already-complete", "execution_status": "already-complete"}

    resolved_path = run_dir / "resolved_config.json"
    resolved_yaml = run_dir / "resolved_config.yaml"
    if action["status"] == "new":
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_new(resolved_path, item["config"])
        _write_flat_yaml(resolved_yaml, item["config"])
    else:
        _resolved_evidence(root, item)

    started_utc = _utc_now()
    command = [str(Path(sys.executable)), "scripts/train.py", "--config", str(resolved_yaml)]
    if action["status"] == "resume":
        command.extend(["--resume", str(action["checkpoint"])])
    try:
        completed, train_elapsed = _run(command, root)
    except Exception as exc:
        completed = subprocess.CompletedProcess(command, 1, "", f"{type(exc).__name__}: {exc}")
        train_elapsed = 0.0
    ended_utc = _utc_now()
    record: dict[str, Any] = {
        "schema_version": 1,
        "stage": item["stage"],
        "arm": item["arm"],
        "seed": item["seed"],
        "status": "failed" if completed.returncode else "training-complete",
        "phase": "train",
        "started_utc": started_utc,
        "ended_utc": ended_utc,
        "elapsed_seconds": train_elapsed,
        "returncode": completed.returncode,
        "command": command,
        "stdout_tail": completed.stdout[-12000:],
        "stderr_tail": completed.stderr[-12000:],
        "config_sha256": _sha256(resolved_path),
        "resolved_yaml_sha256": _sha256(resolved_yaml),
        "manifest_fingerprints": _manifest_hashes(root),
        "gpu_metadata": _gpu_metadata(),
        "max_steps": item["max_steps"],
        "scheduler_total_steps": item["scheduler_total_steps"],
        "experiment_id": item["config"]["experiment_id"],
    }
    if completed.returncode:
        _write_report(run_dir / "run_summary.json", record)
        return record

    trainer_summary_path = run_dir / "run_summary.json"
    if not trainer_summary_path.is_file():
        record.update({"status": "failed", "error": "Trainer did not produce run_summary.json"})
        _write_report(trainer_summary_path, record)
        return record
    try:
        trainer_summary = _read_json(trainer_summary_path)
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "phase": "train-evidence",
                "error": f"Malformed trainer summary: {type(exc).__name__}: {exc}",
            }
        )
        _write_report(trainer_summary_path, record)
        return record
    checkpoint = Path(str(trainer_summary.get("best_path", run_dir / "best.pth")))
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    if not checkpoint.is_file():
        record.update({"status": "failed", "error": f"Missing best checkpoint: {checkpoint}"})
        _write_report(trainer_summary_path, record)
        return record

    validation_path = run_dir / "validation_evaluation.json"
    eval_command = [
        str(Path(sys.executable)),
        "scripts/evaluate.py",
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(resolved_yaml),
        "--manifest",
        str(root / "manifests/v2/fold_0_val.jsonl"),
        "--output",
        str(validation_path),
        "--offset",
        str(_VALIDATION_OFFSET),
        "--max-items",
        str(_VALIDATION_COUNT),
        "--profile",
        "screen",
    ]
    try:
        evaluation, eval_elapsed = _run(eval_command, root)
    except Exception as exc:
        evaluation = subprocess.CompletedProcess(eval_command, 1, "", f"{type(exc).__name__}: {exc}")
        eval_elapsed = 0.0
    record.update(
        {
            "phase": "evaluate",
            "elapsed_seconds": train_elapsed + eval_elapsed,
            "evaluation_returncode": evaluation.returncode,
            "evaluation_command": eval_command,
            "evaluation_stdout_tail": evaluation.stdout[-12000:],
            "evaluation_stderr_tail": evaluation.stderr[-12000:],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
        }
    )
    if evaluation.returncode or not validation_path.is_file():
        record.update({"status": "failed", "error": "Validation evaluation failed"})
        _write_report(trainer_summary_path, record)
        return record
    try:
        validation = _read_json(validation_path)
        metadata = validation.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Validation metadata must be an object")
        _validate_validation_binding(root, validation)
        metadata["config_sha256"] = record["config_sha256"]
        _write_report(validation_path, validation)
        metrics = _evaluation_metrics(validation)
        manifest_hashes, checkpoint_runtime = _checkpoint_metadata(checkpoint)
    except Exception as exc:
        record.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        _write_report(trainer_summary_path, record)
        return record
    record.update(
        {
            "status": "completed",
            "phase": "complete",
            "global_step": int(trainer_summary.get("global_step", -1)),
            "validation_evaluation_sha256": _sha256(validation_path),
            "validation_metrics": metrics,
            "manifest_fingerprints": manifest_hashes or record["manifest_fingerprints"],
            "gpu_metadata": checkpoint_runtime or record["gpu_metadata"],
        }
    )
    if record["global_step"] != int(item["max_steps"]):
        record.update({"status": "failed", "error": "Training stopped at an unexpected global step"})
    _write_report(trainer_summary_path, record)
    return record


def _execute_records(
    root: Path,
    plan: list[dict[str, Any]],
    *,
    resume: bool,
    parallel_workers: int = 1,
) -> list[dict[str, Any]]:
    """Execute independent runs, retaining deterministic plan order."""
    def one(item: dict[str, Any]) -> dict[str, Any]:
        try:
            return execute_run(root, item, resume=resume)
        except Exception as exc:
            return {
                "status": "failed",
                "stage": item["stage"],
                "arm": item["arm"],
                "seed": item["seed"],
                "phase": "runner-evidence",
                "error": f"{type(exc).__name__}: {exc}",
            }

    if parallel_workers <= 1:
        records = []
        for item in plan:
            record = one(item)
            records.append(record)
            if record.get("status") == "failed":
                break
        return records
    with ThreadPoolExecutor(max_workers=parallel_workers) as pool:
        return list(pool.map(one, plan))


def summarize_screen(runs: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Apply paired validation-only gates and select at most one arm."""
    rows = list(runs)
    by_arm_seed = {(str(row.get("arm")), int(row.get("seed", -1))): row for row in rows}
    arms = sorted({str(row.get("arm")) for row in rows})
    expected_matrix = {(arm, seed) for arm in DEFAULT_ARMS for seed in DEFAULT_SEEDS}
    complete_matrix = (
        len(rows) == len(expected_matrix)
        and set(by_arm_seed) == expected_matrix
        and all(row.get("status") == "completed" for row in rows)
    )
    arm_results: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    for arm in arms:
        reasons: list[str] = []
        if arm == "control":
            reasons.append("control-is-baseline")
        arm_rows = [by_arm_seed.get((arm, seed)) for seed in DEFAULT_SEEDS]
        control_rows = [by_arm_seed.get(("control", seed)) for seed in DEFAULT_SEEDS]
        if any(row is None for row in arm_rows):
            reasons.append("missing one or more required seeds")
        if any(row is None for row in control_rows):
            reasons.append("missing paired control seed")
        if any(row is not None and row.get("status") != "completed" for row in arm_rows):
            reasons.append("one or more runs are not complete")

        metrics: list[dict[str, float]] = []
        controls: list[dict[str, float]] = []
        if not any("missing" in reason or "not complete" in reason for reason in reasons):
            try:
                metrics = [dict(row["validation_metrics"]) for row in arm_rows if row is not None]
                controls = [dict(row["validation_metrics"]) for row in control_rows if row is not None]
                required = {"snri", "si_sdri", "stoi", "projection_gain_median", "gain_error_db_median"}
                if any(set(metric) != required for metric in metrics + controls):
                    raise ValueError("promotion metric keys differ from the locked contract")
                if any(isinstance(metric[key], bool) for metric in metrics + controls for key in required):
                    raise ValueError("boolean promotion metric")
                values = [float(metric[key]) for metric in metrics + controls for key in required]
                if not all(math.isfinite(value) for value in values):
                    raise FloatingPointError
            except (KeyError, TypeError, ValueError, FloatingPointError):
                reasons.append("missing or non-finite validation metrics")

        statistics: dict[str, float] = {}
        if metrics and not any("non-finite" in reason for reason in reasons):
            snri_delta = sum(m["snri"] - c["snri"] for m, c in zip(metrics, controls)) / 3.0
            si_sdri_delta = sum(m["si_sdri"] - c["si_sdri"] for m, c in zip(metrics, controls)) / 3.0
            stoi_delta = sum(m["stoi"] - c["stoi"] for m, c in zip(metrics, controls)) / 3.0
            statistics = {
                "mean_snri": sum(m["snri"] for m in metrics) / 3.0,
                "mean_si_sdri": sum(m["si_sdri"] for m in metrics) / 3.0,
                "mean_absolute_gain_error_db": sum(abs(m["gain_error_db_median"]) for m in metrics) / 3.0,
                "mean_paired_snri_delta": snri_delta,
                "mean_paired_si_sdri_delta": si_sdri_delta,
                "mean_paired_stoi_delta": stoi_delta,
            }
            if any(m["projection_gain_median"] <= 0.0 for m in metrics):
                reasons.append("positive gain required for all seeds")
            if snri_delta < 1.0:
                reasons.append(f"snri_delta {snri_delta:.6g} is below +1.0 dB")
            if si_sdri_delta < -0.3:
                reasons.append(f"si_sdri_delta {si_sdri_delta:.6g} is below -0.3 dB")
            if stoi_delta < -0.005:
                reasons.append(f"stoi_delta {stoi_delta:.6g} is below -0.005")
        result = {"arm": arm, "eligible": not reasons, "rejection_reasons": reasons, **statistics}
        arm_results.append(result)
        if result["eligible"]:
            eligible.append(result)

    eligible.sort(
        key=lambda item: (
            -float(item["mean_snri"]),
            -float(item["mean_si_sdri"]),
            float(item["mean_absolute_gain_error_db"]),
            str(item["arm"]),
        )
    )
    promotion_rejection_reasons = [] if complete_matrix else ["exact 4x3 completed matrix required"]
    return {
        "arms": arm_results,
        "promoted_arm": eligible[0]["arm"] if eligible and complete_matrix else None,
        "promotion_rejection_reasons": promotion_rejection_reasons,
    }


def summarize_refinement(
    runs: Iterable[dict[str, Any]], controls: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """Apply the original paired gates to the single locked refinement arm."""
    candidate_rows = list(runs)
    control_rows = list(controls)
    candidate_arms = {str(row.get("arm")) for row in candidate_rows}
    candidate_arm = next(iter(candidate_arms)) if len(candidate_arms) == 1 else ""
    combined = summarize_screen([*candidate_rows, *control_rows])
    candidate = next(
        (row for row in combined["arms"] if row.get("arm") == candidate_arm),
        {
            "arm": candidate_arm or "unknown-refinement",
            "eligible": False,
            "rejection_reasons": ["missing locked refinement arm"],
        },
    )
    exact_candidates = {
        (str(row.get("arm")), int(row.get("seed", -1))) for row in candidate_rows
    } == {(candidate_arm, seed) for seed in DEFAULT_SEEDS}
    exact_controls = {
        (str(row.get("arm")), int(row.get("seed", -1))) for row in control_rows
    } == {("control", seed) for seed in DEFAULT_SEEDS}
    complete = (
        len(candidate_rows) == 3
        and len(control_rows) == 3
        and exact_candidates
        and candidate_arm in REFINEMENT_ARMS
        and exact_controls
        and all(row.get("status") in {"completed", "already-complete"} for row in candidate_rows)
        and all(row.get("status") == "completed" for row in control_rows)
    )
    rejection_reasons = [] if complete else ["exact paired 1x3 refinement and control matrices required"]
    return {
        "arms": [candidate],
        "promoted_arm": candidate_arm if complete and candidate.get("eligible") is True else None,
        "promotion_rejection_reasons": rejection_reasons,
    }


def _serializable_plan(plan: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "stage": item["stage"],
            "arm": item["arm"],
            "seed": item["seed"],
            "run_dir": str(item["run_dir"]),
            "config_path": str(item["config_path"]),
            "max_steps": item["max_steps"],
            "scheduler_total_steps": item["scheduler_total_steps"],
            "loss_config": {key: item["config"][key] for key in _LOSS_CONFIG_KEYS},
        }
        for item in plan
    ]


def _run_evaluation(
    root: Path,
    *,
    checkpoint: Path,
    config: Path,
    manifest: Path,
    output: Path,
    offset: int,
    count: int,
    config_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    command = [
        str(Path(sys.executable)),
        "scripts/evaluate.py",
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(config),
        "--manifest",
        str(manifest),
        "--output",
        str(output),
        "--offset",
        str(offset),
        "--max-items",
        str(count),
    ]
    try:
        completed, _ = _run(command, root)
    except Exception as exc:
        raise RuntimeError(f"Evaluation launch failed: {type(exc).__name__}: {exc}") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"Evaluation failed with return code {completed.returncode}: {completed.stderr[-12000:]}"
        )
    try:
        evaluation = _read_json(output)
    except Exception as exc:
        raise ValueError(f"Malformed evaluation JSON {output}: {type(exc).__name__}: {exc}") from exc
    metadata = evaluation.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Evaluation metadata must be an object: {output}")
    if config_sha256 is not None:
        metadata["config_sha256"] = config_sha256
    _write_report(output, evaluation)
    return evaluation, _artifact_ref(root, output)


def _metric(evaluation: dict[str, Any], name: str, statistic: str = "mean") -> float:
    enhanced = evaluation.get("enhanced")
    if not isinstance(enhanced, dict):
        raise ValueError("Evaluation missing enhanced metrics")
    value = enhanced.get(name)
    if isinstance(value, dict):
        value = value.get(statistic)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"Evaluation metric is missing or non-finite: enhanced.{name}.{statistic}")
    return float(value)


def _acceptance(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "snri_positive": _metric(candidate, "snr_improvement_mean") > 0.0,
        "si_sdri_delta": _metric(candidate, "si_sdr_improvement_mean")
        >= _metric(baseline, "si_sdr_improvement_mean") - 0.3,
        "stoi_delta": _metric(candidate, "stoi") >= _metric(baseline, "stoi") - 0.005,
        "pesq_delta": _metric(candidate, "pesq") >= _metric(baseline, "pesq") - 0.05,
        "positive_gain": _metric(candidate, "projection_gain", "median") > 0.0,
        "gain_error": abs(_metric(candidate, "gain_error_db", "median")) <= 3.0,
        "polarity": _metric(candidate, "polarity_failure") < 0.01,
    }
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks}


def _screen_binding(root: Path, arm: str) -> dict[str, str]:
    path = root / "reports/generated/gain_calibration_screen.json"
    screen = _read_json(path)
    rows = screen.get("runs")
    expected = {(name, seed) for name in DEFAULT_ARMS for seed in DEFAULT_SEEDS}
    actual = {
        (str(row.get("arm")), int(row.get("seed", -1)))
        for row in rows
        if isinstance(row, dict)
    } if isinstance(rows, list) else set()
    recomputed = summarize_screen(rows if isinstance(rows, list) else [])
    source_valid = (
        screen.get("schema_version") != 1
        or screen.get("stage") != "screen"
        or screen.get("status") != "completed"
        or actual != expected
        or len(rows) != 12  # type: ignore[arg-type]
        or any(not isinstance(row, dict) or row.get("status") != "completed" for row in rows)  # type: ignore[union-attr]
    )
    if source_valid:
        raise ValueError("Full stage requires an exact completed 4x3 screen report bound to the promoted arm")
    source_ref = _artifact_ref(root, path)
    if screen.get("promoted_arm") == arm and recomputed.get("promoted_arm") == arm:
        return {**source_ref, "promoted_arm": arm}

    refinement_path = root / f"reports/generated/gain_calibration_refine_{arm}.json"
    if not refinement_path.is_file():
        raise ValueError("Full stage requires an exact completed 4x3 screen report bound to the promoted arm")
    refinement = _read_json(refinement_path)
    refinement_rows = refinement.get("runs")
    controls = [row for row in rows if isinstance(row, dict) and row.get("arm") == "control"]
    recomputed_refinement = summarize_refinement(
        refinement_rows if isinstance(refinement_rows, list) else [],
        controls,
    )
    claimed_source = refinement.get("screen_report")
    if (
        refinement.get("schema_version") != 1
        or refinement.get("stage") != "refine"
        or refinement.get("status") != "completed"
        or refinement.get("promoted_arm") != arm
        or recomputed_refinement.get("promoted_arm") != arm
        or not isinstance(claimed_source, dict)
        or claimed_source.get("sha256") != source_ref.get("sha256")
    ):
        raise ValueError("Full stage requires a completed refinement report bound to the immutable screen report")
    return {
        **_artifact_ref(root, refinement_path),
        "promoted_arm": arm,
        "source_screen_report": source_ref,
    }


def _full_run_row(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    run_dir = _resolve_inside(root, record["checkpoint"]).parent
    return {
        "arm": record["arm"],
        "seed": record["seed"],
        "status": "completed",
        "execution_status": record.get("execution_status", record["status"]),
        "run_dir": run_dir.relative_to(root).as_posix(),
        "run_summary": _artifact_ref(root, run_dir / "run_summary.json"),
        "resolved_config": _artifact_ref(root, run_dir / "resolved_config.json"),
        "resolved_yaml": _artifact_ref(root, run_dir / "resolved_config.yaml"),
        "checkpoint": _artifact_ref(root, _resolve_inside(root, record["checkpoint"])),
        "validation_evaluation": _artifact_ref(root, run_dir / "validation_evaluation.json"),
        "validation_metrics": record["validation_metrics"],
        "evaluations": {},
        "acceptance": {},
    }


def run_experiment(
    root: Path,
    *,
    stage: str,
    arms: Iterable[str],
    seeds: Iterable[int],
    resume: bool,
    parallel_workers: int = 1,
) -> dict[str, Any]:
    """Execute a non-dry stage and persist a complete producer contract."""
    root = Path(root).resolve()
    plan = build_plan(root, stage, arms, seeds)
    report_name = (
        f"gain_calibration_refine_{plan[0]['arm']}.json"
        if stage == "refine"
        else f"gain_calibration_{stage}.json"
    )
    report_path = root / "reports/generated" / report_name
    if report_path.exists() and not resume:
        raise FileExistsError(f"Report already exists; pass --resume to update it: {report_path}")
    screen_ref = _screen_binding(root, str(plan[0]["arm"])) if stage == "full" else None

    records = _execute_records(
        root,
        plan,
        resume=resume,
        parallel_workers=parallel_workers if stage == "full" else 1,
    )

    base: dict[str, Any] = {
        "schema_version": 1,
        "stage": stage,
        "timestamp_utc": _utc_now(),
        "runs": records,
        "planned_run_count": len(plan),
        "estimated_steps": sum(int(item["max_steps"]) for item in plan),
    }
    if stage == "screen":
        base.update(summarize_screen(records))
        base["status"] = (
            "completed"
            if len(records) == len(plan) and all(row.get("status") == "completed" for row in records)
            else "failed"
        )
        _write_report(report_path, base)
        return base
    if stage == "refine":
        screen_path = root / "reports/generated/gain_calibration_screen.json"
        screen = _read_json(screen_path)
        controls = [row for row in screen.get("runs", []) if row.get("arm") == "control"]
        base["screen_report"] = _artifact_ref(root, screen_path)
        base.update(summarize_refinement(records, controls))
        base["status"] = (
            "completed"
            if len(records) == len(plan) and all(row.get("status") == "completed" for row in records)
            else "failed"
        )
        _write_report(report_path, base)
        return base

    assert screen_ref is not None
    base.update(
        {
            "selection_basis": "validation",
            "screen_report": screen_ref,
            "selected_arm": plan[0]["arm"],
            "selected_seed": None,
            "selected_checkpoint_path": None,
            "selected_checkpoint_sha256": None,
            "selected_checkpoint": None,
            "production_eligible": False,
            "baseline_evaluations": {},
            "selection_rationale": {
                "metric_order": ["validation_snri", "validation_si_sdri", "absolute_gain_error"],
                "eligible_seeds": [],
                "selected_seed": None,
            },
        }
    )
    if len(records) != 3 or any(row.get("status") not in {"completed", "already-complete"} for row in records):
        base["status"] = "failed"
        _write_report(report_path, base)
        return base

    full_rows = [_full_run_row(root, record) for record in records]
    base["runs"] = full_rows
    ranked = sorted(
        full_rows,
        key=lambda row: (
            -float(row["validation_metrics"]["snri"]),
            -float(row["validation_metrics"]["si_sdri"]),
            abs(float(row["validation_metrics"]["gain_error_db_median"])),
            int(row["seed"]),
        ),
    )
    winner = ranked[0]
    rationale = {
        "metric_order": ["validation_snri", "validation_si_sdri", "absolute_gain_error"],
        "eligible_seeds": [int(row["seed"]) for row in ranked],
        "selected_seed": int(winner["seed"]),
        "ranked_validation_metrics": [
            {"seed": row["seed"], **row["validation_metrics"]} for row in ranked
        ],
    }
    selection_payload = {
        "schema_version": 1,
        "selection_basis": "validation",
        "selected_arm": plan[0]["arm"],
        "selected_seed": winner["seed"],
        "metric_order": rationale["metric_order"],
        "ranked_validation_metrics": rationale["ranked_validation_metrics"],
        "validation_evaluations": {
            str(row["seed"]): row["validation_evaluation"] for row in full_rows
        },
    }
    selection_payload["selection_contract_sha256"] = hashlib.sha256(
        _json_bytes(selection_payload)
    ).hexdigest()
    selection_path = root / "reports/generated/gain_calibration_selection.json"
    try:
        expected_selection = _json_bytes(selection_payload)
        if selection_path.exists():
            if selection_path.read_bytes() != expected_selection:
                raise FileExistsError(f"Immutable selection receipt collision: {selection_path}")
        else:
            _write_new(selection_path, selection_payload)
        selection_ref = _artifact_ref(root, selection_path)
        base.update(
            {
                "selected_seed": winner["seed"],
                "selection_rationale": rationale,
                "selection_receipt": selection_ref,
            }
        )

        source = root / winner["checkpoint"]["path"]
        selected = root / "checkpoints/gain_calibration/best.pth"
        selected.parent.mkdir(parents=True, exist_ok=True)
        if selected.exists() and _sha256(selected) != _sha256(source):
            raise FileExistsError(f"Selected checkpoint collision: {selected}")
        if not selected.exists():
            _atomic_copy(source, selected)
        selected_ref = _artifact_ref(root, selected)
        base.update(
            {
                "selected_checkpoint_path": selected_ref["path"],
                "selected_checkpoint_sha256": selected_ref["sha256"],
                "selected_checkpoint": {
                    **selected_ref,
                    "source_path": winner["checkpoint"]["path"],
                    "source_sha256": winner["checkpoint"]["sha256"],
                },
            }
        )
    except Exception as exc:
        base["status"] = "failed"
        base["error"] = f"{type(exc).__name__}: {exc}"
        _write_report(report_path, base)
        return base

    manifest = root / "manifests/v2/fold_0_test.jsonl"
    slices = {"comparison": (0, 500), "audit": (500, 500)}
    baseline_checkpoint = root / "checkpoints/best.pth"
    active_row: dict[str, Any] | None = None
    try:
        baseline_payloads: dict[str, dict[str, Any]] = {}
        for name, (offset, count) in slices.items():
            output = root / f"reports/generated/gain_calibration_baseline_{name}.json"
            payload, reference = _run_evaluation(
                root,
                checkpoint=baseline_checkpoint,
                config=root / "configs/train.yaml",
                manifest=manifest,
                output=output,
                offset=offset,
                count=count,
                config_sha256=None,
            )
            baseline_payloads[name] = payload
            base["baseline_evaluations"][name] = reference  # type: ignore[index]

        for row, record in zip(full_rows, records):
            active_row = row
            run_dir = root / row["run_dir"]
            config_hash = row["resolved_config"]["sha256"]
            candidate_payloads = {}
            for name, (offset, count) in slices.items():
                output = run_dir / f"{name}_evaluation.json"
                payload, reference = _run_evaluation(
                    root,
                    checkpoint=root / row["checkpoint"]["path"],
                    config=run_dir / "resolved_config.yaml",
                    manifest=manifest,
                    output=output,
                    offset=offset,
                    count=count,
                    config_sha256=config_hash,
                )
                candidate_payloads[name] = payload
                row["evaluations"][name] = reference
            row["acceptance"] = {
                name: _acceptance(candidate_payloads[name], baseline_payloads[name])
                for name in slices
            }
    except Exception as exc:
        base["status"] = "failed"
        base["error"] = f"{type(exc).__name__}: {exc}"
        if active_row is not None:
            active_row["status"] = "failed"
            active_row["error"] = base["error"]
        _write_report(report_path, base)
        return base

    rejection_reasons: dict[str, list[str]] = {}
    for row in full_rows:
        reasons = [
            f"{slice_name}: {check_name}"
            for slice_name, gate in row["acceptance"].items()
            for check_name, passed in gate.get("checks", {}).items()
            if passed is not True
        ]
        for slice_name, gate in row["acceptance"].items():
            if gate.get("status") != "pass" and not any(reason.startswith(f"{slice_name}: ") for reason in reasons):
                reasons.append(f"{slice_name}: acceptance status {gate.get('status')}")
        row["rejection_reasons"] = reasons
        row["status"] = "completed" if not reasons else "rejected"
        rejection_reasons[str(row["seed"])] = reasons

    all_seeds_pass = all(row["status"] == "completed" for row in full_rows)
    base["production_eligible"] = all_seeds_pass
    base["rejection_reasons"] = rejection_reasons
    if isinstance(base.get("selected_checkpoint"), dict):
        base["selected_checkpoint"]["production_eligible"] = all_seeds_pass
    base["status"] = "completed" if all_seeds_pass else "rejected"
    if not all_seeds_pass:
        base["error"] = "One or more full seeds failed comparison or audit acceptance gates"
    _write_report(report_path, base)
    return base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("screen", "refine", "full"), required=True)
    parser.add_argument("--arms", type=_csv_values, required=True)
    parser.add_argument("--seeds", type=_csv_values, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--parallel-workers", type=int, default=1)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    try:
        seeds = [int(seed) for seed in args.seeds]
    except ValueError as exc:
        parser.error(f"Seeds must be integers: {exc}")
    plan = build_plan(root, args.stage, args.arms, seeds)
    if args.dry_run:
        payload = {
            "schema_version": 1,
            "status": "dry-run",
            "stage": args.stage,
            "runs": _serializable_plan(plan),
            "estimated_steps": sum(int(item["max_steps"]) for item in plan),
            "training_processes_started": 0,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    report = run_experiment(
        root,
        stage=args.stage,
        arms=args.arms,
        seeds=seeds,
        resume=args.resume,
        parallel_workers=args.parallel_workers,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
