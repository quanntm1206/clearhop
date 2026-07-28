"""Fail-closed audit for the research evidence committed to GitHub."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any


SEEDS = [17, 29, 43, 59, 71]
METRICS = {"snri_db", "si_sdri_db", "stoi", "pesq"}
SOURCE_FILES = {
    "failure_analysis.json",
    "failure_audio.json",
    "research_evaluations.json",
    "research_selection.json",
    "research_training.json",
    "robustness_matrix.json",
    "significance.json",
}
ROBUSTNESS_SLICES = {
    "primary_comparison": (0, 500),
    "primary_audit": (500, 500),
    "robustness_low_offset": (1000, 500),
    "robustness_mid_offset": (5000, 500),
    "robustness_tail": (10000, 500),
}
PINNED_EXTERNALS = {"DeepFilterNet3", "RNNoise", "DTLN"}
OPTIONAL_UNPINNED_EXTERNALS = {"WebRTC NS"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


def _sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _commit(value: object) -> bool:
    return isinstance(value, str) and _COMMIT.fullmatch(value) is not None


def _canonical_receipt_sha256(payload: dict) -> str:
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _canonical_value_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_repo_path(root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return None
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _comparison_coverage(root: Path, comparison: object, *, rnnoise_build_ok: bool) -> dict[str, object]:
    models = comparison.get("models") if isinstance(comparison, dict) else None
    if not isinstance(models, list):
        return {"external_reproduced": 0, "verified_blocker_recipes": 0, "eligible": False, "tier": "insufficient"}
    external_names = PINNED_EXTERNALS | OPTIONAL_UNPINNED_EXTERNALS
    external_reproduced = sum(
        1 for row in models
        if isinstance(row, dict) and row.get("name") in external_names and row.get("status") == "reproduced_local"
    )
    verified_blocker_recipes = 0
    for row in models:
        if not isinstance(row, dict) or row.get("status") != "blocked":
            continue
        recipe = row.get("reproduction")
        provenance = row.get("provenance")
        if not isinstance(recipe, dict) or not isinstance(provenance, dict):
            continue
        script = _safe_repo_path(root, recipe.get("setup_script"))
        dockerfile = _safe_repo_path(root, recipe.get("dockerfile"))
        toolchain_path = _safe_repo_path(root, recipe.get("toolchain_manifest"))
        toolchain: object = None
        if toolchain_path is not None and toolchain_path.is_file():
            try:
                toolchain = json.loads(toolchain_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                toolchain = None
        toolchain_ok = (
            isinstance(toolchain, dict)
            and toolchain.get("schema_version") == 1
            and toolchain.get("source_commit") == provenance.get("commit")
            and isinstance(toolchain.get("base_image"), str)
            and re.fullmatch(r"ubuntu@sha256:[0-9a-f]{64}", toolchain["base_image"]) is not None
            and _sha256(toolchain.get("source_archive_sha256"))
            and _sha256(toolchain.get("model_archive_sha256"))
            and isinstance(toolchain.get("model_archive_url"), str)
            and toolchain["model_archive_url"].startswith("https://")
            and isinstance(toolchain.get("packages"), dict)
            and bool(toolchain["packages"])
            and all(isinstance(version, str) and bool(version) for version in toolchain["packages"].values())
        )
        if (
            recipe.get("status") == "blocked"
            and isinstance(recipe.get("command"), str)
            and bool(recipe["command"])
            and recipe.get("pinned_commit") == provenance.get("commit")
            and script is not None
            and script.is_file()
            and _sha256(recipe.get("setup_script_sha256"))
            and _file_sha256(script) == recipe["setup_script_sha256"]
            and dockerfile is not None
            and dockerfile.is_file()
            and _sha256(recipe.get("dockerfile_sha256"))
            and _file_sha256(dockerfile) == recipe["dockerfile_sha256"]
            and toolchain_path is not None
            and _sha256(recipe.get("toolchain_manifest_sha256"))
            and _file_sha256(toolchain_path) == recipe["toolchain_manifest_sha256"]
            and toolchain_ok
            and (row.get("name") != "RNNoise" or rnnoise_build_ok)
        ):
            verified_blocker_recipes += 1
    if external_reproduced >= 2:
        tier = "two_reproduced"
    elif external_reproduced >= 1 and verified_blocker_recipes >= 1:
        tier = "one_plus_recipe"
    else:
        tier = "insufficient"
    return {
        "external_reproduced": external_reproduced,
        "verified_blocker_recipes": verified_blocker_recipes,
        "eligible": tier != "insufficient",
        "tier": tier,
    }


def _comparison_inputs_ok(root: Path, comparison: object) -> bool:
    if not isinstance(comparison, dict):
        return False
    inputs = comparison.get("inputs")
    protocol = comparison.get("protocol")
    if not isinstance(inputs, dict) or not isinstance(protocol, dict):
        return False
    registry = inputs.get("registry")
    manifest = inputs.get("manifest")
    if not isinstance(registry, dict) or not isinstance(manifest, dict):
        return False
    if registry.get("path") != "configs/research_baselines.json":
        return False
    if manifest.get("path") != "manifests/v2/fold_0_test.jsonl" or protocol.get("manifest") != manifest.get("path"):
        return False
    registry_path = _safe_repo_path(root, registry.get("path"))
    manifest_path = _safe_repo_path(root, manifest.get("path"))
    ids = protocol.get("item_ids")
    if registry_path is None or manifest_path is None or not registry_path.is_file() or not manifest_path.is_file() or not isinstance(ids, list):
        return False
    if protocol.get("slice_offset") != 0 or protocol.get("slice_count") != 500 or protocol.get("item_count") != 500:
        return False
    try:
        manifest_rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        actual_ids = [str(row["id"]) for row in manifest_rows[:500] if isinstance(row, dict) and "id" in row]
    except (OSError, UnicodeError, ValueError, TypeError):
        return False
    expected_ids = [str(index) for index in range(500)]
    if len(actual_ids) != 500 or actual_ids != expected_ids or [str(item) for item in ids] != actual_ids:
        return False
    item_ids_hash = hashlib.sha256("\n".join(actual_ids).encode("utf-8")).hexdigest()
    tracked_ok = manifest.get("git_tracked") is True
    if (root / ".git").exists():
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(manifest["path"])],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        tracked_ok = tracked_ok and tracked.returncode == 0 and str(manifest["path"]) in tracked.stdout.splitlines()
    return (
        _sha256(registry.get("sha256"))
        and registry["sha256"] == _file_sha256(registry_path)
        and _sha256(manifest.get("sha256"))
        and manifest["sha256"] == _file_sha256(manifest_path)
        and _sha256(manifest.get("item_ids_sha256"))
        and manifest["item_ids_sha256"] == item_ids_hash
        and tracked_ok
    )


def _finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))


def _ref(value: object) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("path"), str)
        and bool(value["path"])
        and _sha256(value.get("sha256"))
    )


def _rows_have_exact_seeds(rows: object) -> bool:
    return (
        isinstance(rows, list)
        and len(rows) == len(SEEDS)
        and sorted(int(row.get("seed", -1)) for row in rows if isinstance(row, dict)) == SEEDS
    )


def _training_ok(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    rows = value.get("runs")
    return (
        value.get("schema_version") == 1
        and value.get("stage") == "research"
        and value.get("status") == "completed"
        and value.get("seeds") == SEEDS
        and value.get("planned_run_count") == 5
        and _rows_have_exact_seeds(rows)
        and all(
            isinstance(row, dict)
            and row.get("status") == "completed"
            and int(row.get("global_step", -1)) == 30000
            and int(row.get("max_steps", -1)) == 30000
            and int(row.get("scheduler_total_steps", -1)) == 30000
            and int(row.get("returncode", -1)) == 0
            and int(row.get("evaluation_returncode", -1)) == 0
            and all(_sha256(row.get(key)) for key in ("checkpoint_sha256", "config_sha256", "resolved_yaml_sha256", "validation_evaluation_sha256"))
            and isinstance(row.get("manifest_fingerprints"), dict)
            and all(_sha256(row["manifest_fingerprints"].get(key)) for key in ("val", "test"))
            for row in rows
        )
    )


def _evaluations_ok(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    rows = value.get("runs")
    return (
        value.get("schema_version") == 1
        and value.get("stage") == "research-evaluation"
        and value.get("status") == "completed"
        and value.get("seeds") == SEEDS
        and value.get("all_seed_acceptance") is True
        and _rows_have_exact_seeds(rows)
        and all(
            isinstance(row, dict)
            and row.get("status") == "pass"
            and _sha256(row.get("checkpoint_sha256"))
            and _sha256(row.get("config_sha256"))
            and isinstance(row.get("evaluations"), dict)
            and set(row["evaluations"]) == {"comparison", "audit"}
            and all(_ref(ref) for ref in row["evaluations"].values())
            and isinstance(row.get("acceptance"), dict)
            and set(row["acceptance"]) == {"comparison", "audit"}
            and all(
                isinstance(gate, dict)
                and gate.get("status") == "pass"
                and isinstance(gate.get("checks"), dict)
                and gate["checks"]
                and all(check is True for check in gate["checks"].values())
                for gate in row["acceptance"].values()
            )
            for row in rows
        )
    )


def _selection_ok(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    ranked = value.get("ranked_validation_metrics")
    if not _rows_have_exact_seeds(ranked):
        return False
    numeric_keys = {"snri", "si_sdri", "stoi", "projection_gain_median", "gain_error_db_median"}
    return (
        value.get("schema_version") == 1
        and value.get("status") == "pass"
        and value.get("selection_basis") == "validation"
        and value.get("metric_order") == ["validation_snri", "validation_si_sdri", "absolute_gain_error"]
        and int(value.get("selected_seed", -1)) == int(ranked[0]["seed"])
        and int(value.get("selected_seed", -1)) in SEEDS
        and all(all(_finite(row.get(key)) for key in numeric_keys) for row in ranked)
        and all(float(ranked[index]["snri"]) >= float(ranked[index + 1]["snri"]) for index in range(4))
        and _sha256(value.get("source_checkpoint_sha256"))
        and value.get("source_checkpoint_sha256") == value.get("production_checkpoint_sha256")
        and _ref(value.get("parity"))
    )


def _significance_ok(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    metrics = value.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != {"si_sdr", "snr", "stoi"}:
        return False
    for metric in metrics.values():
        if not isinstance(metric, dict) or not isinstance(metric.get("bootstrap"), dict) or not isinstance(metric.get("significance"), dict):
            return False
        bootstrap = metric["bootstrap"]
        significance = metric["significance"]
        if not (
            int(bootstrap.get("n", -1)) == 500
            and int(bootstrap.get("resamples", -1)) == 10000
            and int(bootstrap.get("seed", -1)) == 0
            and all(_finite(bootstrap.get(key)) for key in ("ci95_low", "ci95_high", "mean_delta", "median_delta"))
            and float(bootstrap["ci95_low"]) <= float(bootstrap["ci95_high"])
            and int(significance.get("n", -1)) == 500
            and significance.get("test") == "wilcoxon_signed_rank"
            and all(_finite(significance.get(key)) for key in ("cohen_dz", "p_value", "statistic"))
            and 0.0 <= float(significance["p_value"]) <= 1.0
        ):
            return False
    return (
        value.get("schema_version") == 1
        and value.get("status") == "pass"
        and value.get("paired_items") == 500
        and _ref(value.get("baseline"))
        and _ref(value.get("candidate"))
        and float(metrics["snr"]["bootstrap"]["ci95_low"]) > 0.0
        and float(metrics["stoi"]["bootstrap"]["ci95_low"]) >= -0.005
        and float(metrics["si_sdr"]["bootstrap"]["ci95_low"]) >= -0.3
    )


def _robustness_ok(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    rows = value.get("rows")
    if not isinstance(rows, list) or len(rows) != 5:
        return False
    by_name = {str(row.get("name")): row for row in rows if isinstance(row, dict)}
    if set(by_name) != set(ROBUSTNESS_SLICES):
        return False
    return (
        value.get("schema_version") == 1
        and value.get("status") == "pass"
        and value.get("slices") == 5
        and _sha256(value.get("baseline_checkpoint_sha256"))
        and _sha256(value.get("candidate_checkpoint_sha256"))
        and all(
            int(by_name[name].get("offset", -1)) == offset
            and int(by_name[name].get("count", -1)) == count
            and all(_finite(by_name[name].get(key)) for key in ("snri_delta", "si_sdri_delta", "stoi_delta"))
            and float(by_name[name]["snri_delta"]) > 0.0
            and float(by_name[name]["si_sdri_delta"]) >= -0.3
            and float(by_name[name]["stoi_delta"]) >= -0.005
            for name, (offset, count) in ROBUSTNESS_SLICES.items()
        )
    )


def _failure_analysis_ok(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    worst = value.get("worst_cases")
    return (
        value.get("schema_version") == 1
        and value.get("n") == 500
        and isinstance(value.get("failure_counts"), dict)
        and value["failure_counts"]
        and all(isinstance(count, int) and not isinstance(count, bool) and count >= 0 for count in value["failure_counts"].values())
        and isinstance(value.get("subgroup_failure_counts"), dict)
        and bool(value["subgroup_failure_counts"])
        and isinstance(worst, list)
        and len(worst) >= 25
        and len({int(row.get("index", -1)) for row in worst if isinstance(row, dict)}) == len(worst)
        and all(
            isinstance(row, dict)
            and 0 <= int(row.get("index", -1)) < 500
            and all(_finite(row.get(key)) for key in ("si_sdr_delta", "snr_delta", "stoi_delta"))
            and isinstance(row.get("metadata"), dict)
            for row in worst
        )
    )


def _failure_audio_ok(value: object, analysis: object) -> bool:
    if not isinstance(value, dict) or not isinstance(analysis, dict):
        return False
    items = value.get("items")
    worst = analysis.get("worst_cases")
    if not isinstance(items, list) or not isinstance(worst, list):
        return False
    expected_indices = [int(row.get("index", -1)) for row in worst[:10] if isinstance(row, dict)]
    return (
        value.get("schema_version") == 1
        and value.get("status") == "pass"
        and _sha256(value.get("checkpoint_sha256"))
        and len(items) == 10
        and [int(item.get("index", -1)) for item in items if isinstance(item, dict)] == expected_indices
        and all(
            isinstance(item, dict)
            and isinstance(item.get("files"), dict)
            and set(item["files"]) == {"clean", "noisy", "enhanced"}
            and all(_ref(ref) for ref in item["files"].values())
            for item in items
        )
    )


def _model_checks(value: object) -> dict[str, bool]:
    checks = {"model_item_ids": False, "reproduced_metrics": False, "blocked_metrics": False, "model_provenance": False, "comparison_status": False}
    if not isinstance(value, dict) or value.get("schema_version") != 2 or value.get("status") not in {"partial", "pass"}:
        return checks
    protocol = value.get("protocol")
    models = value.get("models")
    if not isinstance(protocol, dict) or not isinstance(models, list):
        return checks
    ids = protocol.get("item_ids")
    ids_ok = isinstance(ids, list) and bool(ids) and len(ids) == len(set(str(item) for item in ids)) and protocol.get("item_count") == len(ids)
    checks["model_item_ids"] = ids_ok and all(isinstance(row, dict) and row.get("item_ids") == ids for row in models)
    reproduced = [row for row in models if isinstance(row, dict) and row.get("status") == "reproduced_local"]
    blocked = [row for row in models if isinstance(row, dict) and row.get("status") == "blocked"]
    reproduced_external = [row for row in reproduced if row.get("name") in PINNED_EXTERNALS | OPTIONAL_UNPINNED_EXTERNALS]
    statuses = [row.get("status") for row in models if isinstance(row, dict)]
    expected_status = "pass" if statuses and all(status == "reproduced_local" for status in statuses) else "partial" if "blocked" in statuses and "failed" not in statuses else "fail"
    checks["comparison_status"] = len(statuses) == len(models) and value.get("status") == expected_status
    checks["reproduced_metrics"] = len(reproduced) >= 2 and all(
        isinstance(row.get("metrics"), dict)
        and set(row["metrics"]) == METRICS
        and all(
            isinstance(metric, dict)
            and metric.get("available") is True
            and _finite(metric.get("value"))
            and metric.get("source") in {"receipt", "reproduced_local"}
            for metric in row["metrics"].values()
        )
        for row in reproduced
    )
    checks["blocked_metrics"] = (
        {str(row.get("name")) for row in blocked + reproduced_external} == PINNED_EXTERNALS | OPTIONAL_UNPINNED_EXTERNALS
        and all(
            isinstance(row.get("blocked_reason"), str)
            and bool(row["blocked_reason"])
            and isinstance(row.get("metrics"), dict)
            and set(row["metrics"]) == METRICS
            and all(
                isinstance(metric, dict)
                and metric.get("available") is False
                and metric.get("value") is None
                and metric.get("source") == "blocked"
                and isinstance(metric.get("reason"), str)
                and bool(metric["reason"])
                for metric in row["metrics"].values()
            )
            for row in blocked
        )
    )
    provenance_ok = True
    for row in models:
        if not isinstance(row, dict) or not isinstance(row.get("provenance"), dict):
            provenance_ok = False
            continue
        provenance = row["provenance"]
        weights = provenance.get("weights")
        common = (
            isinstance(provenance.get("commit_url"), str)
            and str(provenance["commit_url"]).startswith("https://")
            and isinstance(provenance.get("version"), str)
            and bool(provenance["version"])
            and isinstance(provenance.get("retrieved_at"), str)
            and bool(provenance["retrieved_at"])
            and isinstance(weights, dict)
        )
        if row.get("status") == "reproduced_local":
            external_pin_ok = True
            if row.get("name") in PINNED_EXTERNALS | OPTIONAL_UNPINNED_EXTERNALS:
                commit = provenance.get("commit")
                external_pin_ok = _commit(commit) and str(commit) in str(provenance.get("commit_url"))
            provenance_ok &= common and external_pin_ok and _sha256(provenance.get("weight_sha256")) and weights.get("weight_sha256") == provenance.get("weight_sha256") and weights.get("hash_status") == "verified"
        elif row.get("name") in PINNED_EXTERNALS:
            commit = provenance.get("commit")
            provenance_ok &= common and _commit(commit) and str(commit) in str(provenance.get("commit_url")) and weights.get("hash_status") == "unavailable" and weights.get("weight_sha256") is None
        elif row.get("name") in OPTIONAL_UNPINNED_EXTERNALS:
            explanation = " ".join(str(provenance.get(key, "")) for key in ("version", "weight_sha256_reason"))
            provenance_ok &= common and row.get("status") == "blocked" and (
                (_commit(provenance.get("commit")) and str(provenance["commit"]) in str(provenance.get("commit_url")))
                or (provenance.get("commit") is None and "pin" in explanation.lower() and ("unavailable" in explanation.lower() or "not configured" in (str(row.get("blocked_reason")) + explanation).lower()))
            )
        else:
            provenance_ok = False
    checks["model_provenance"] = provenance_ok
    return checks


def audit_public_research(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    research_path = root / "reports/public/research_readiness.json"
    comparison_path = root / "reports/public/model_comparison.json"
    deepfilter_path = root / "reports/public/deepfilternet3_reproduction.json"
    rnnoise_build_path = root / "reports/public/rnnoise_build.json"
    checks = {
        "required_receipts": research_path.is_file() and comparison_path.is_file() and deepfilter_path.is_file() and rnnoise_build_path.is_file(),
        "rnnoise_build_receipt": False,
        "research_receipt_hash": False,
        "evidence_hashes": False,
        "comparison_receipt_hash": False,
        "comparison_inputs": False,
        "deepfilter_receipt_hash": False,
        "comparison_coverage": False,
        "receipt_contract": False,
        "source_hashes": False,
        "manifest_overlap": False,
        "training_matrix": False,
        "evaluation_matrix": False,
        "selection": False,
        "significance": False,
        "robustness": False,
        "failure_analysis": False,
        "failure_audio": False,
        "model_item_ids": False,
        "reproduced_metrics": False,
        "blocked_metrics": False,
        "model_provenance": False,
        "comparison_status": False,
        "checkpoint_binding": False,
    }
    errors: list[str] = []
    coverage = {"external_reproduced": 0, "verified_blocker_recipes": 0, "eligible": False, "tier": "insufficient"}
    if not checks["required_receipts"]:
        return {"schema_version": 1, "status": "fail", "checks": checks, "coverage": coverage, "errors": ["required public receipts missing"]}
    try:
        research = json.loads(research_path.read_text(encoding="utf-8"))
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        deepfilter = json.loads(deepfilter_path.read_text(encoding="utf-8"))
        try:
            if __package__:
                from .verify_rnnoise_build import audit_rnnoise_build
            else:
                from verify_rnnoise_build import audit_rnnoise_build
            rnnoise_build = audit_rnnoise_build(root)
        except Exception:
            rnnoise_build = {"status": "fail"}
        checks["rnnoise_build_receipt"] = rnnoise_build.get("status") == "pass"
        checks["research_receipt_hash"] = (
            isinstance(research, dict)
            and _sha256(research.get("receipt_sha256"))
            and research["receipt_sha256"] == _canonical_receipt_sha256(research)
        )
        embedded = research.get("embedded_sha256") if isinstance(research, dict) else None
        evidence_payload = research.get("evidence") if isinstance(research, dict) else None
        expected_embedded = {"verifier", "manifest", *{name.removesuffix(".json") for name in SOURCE_FILES}}
        checks["evidence_hashes"] = (
            isinstance(embedded, dict)
            and set(embedded) == expected_embedded
            and isinstance(evidence_payload, dict)
            and embedded.get("verifier") == _canonical_value_sha256(research.get("verifier"))
            and embedded.get("manifest") == _canonical_value_sha256(research.get("manifest"))
            and all(
                _sha256(embedded.get(name))
                and embedded[name] == _canonical_value_sha256(evidence_payload.get(name))
                for name in expected_embedded - {"verifier", "manifest"}
            )
        )
        checks["comparison_receipt_hash"] = (
            isinstance(comparison, dict)
            and _sha256(comparison.get("receipt_sha256"))
            and comparison["receipt_sha256"] == _canonical_receipt_sha256(comparison)
        )
        checks["comparison_inputs"] = _comparison_inputs_ok(root, comparison)
        deepfilter_row = next(
            (row for row in comparison.get("models", []) if isinstance(row, dict) and row.get("name") == "DeepFilterNet3"),
            None,
        ) if isinstance(comparison, dict) else None
        source_receipt = deepfilter_row.get("source_receipt") if isinstance(deepfilter_row, dict) else None
        checks["deepfilter_receipt_hash"] = (
            isinstance(deepfilter, dict)
            and _sha256(deepfilter.get("receipt_sha256"))
            and deepfilter["receipt_sha256"] == _canonical_receipt_sha256(deepfilter)
            and isinstance(source_receipt, dict)
            and source_receipt.get("path") == "reports/public/deepfilternet3_reproduction.json"
            and source_receipt.get("sha256") == deepfilter.get("receipt_sha256")
            and deepfilter.get("model", {}).get("item_ids") == comparison.get("protocol", {}).get("item_ids")
        )
        coverage = _comparison_coverage(root, comparison, rnnoise_build_ok=checks["rnnoise_build_receipt"])
        checks["comparison_coverage"] = coverage["eligible"] is True
        evidence = research.get("evidence") if isinstance(research, dict) else None
        verifier = research.get("verifier") if isinstance(research, dict) else None
        checks["receipt_contract"] = (
            isinstance(research, dict)
            and research.get("schema_version") == 1
            and research.get("receipt_type") == "research_readiness"
            and research.get("status") == "pass"
            and isinstance(verifier, dict)
            and verifier.get("status") == "pass"
            and verifier.get("research_eligible") is True
            and isinstance(evidence, dict)
            and set(evidence) == {name.removesuffix(".json") for name in SOURCE_FILES}
        )
        sources = research.get("source_sha256") if isinstance(research, dict) else None
        checks["source_hashes"] = isinstance(sources, dict) and set(sources) == SOURCE_FILES and all(_sha256(value) for value in sources.values())
        manifest = research.get("manifest") if isinstance(research, dict) else None
        summary = manifest.get("summary") if isinstance(manifest, dict) else None
        overlap = summary.get("speaker_overlap") if isinstance(summary, dict) else None
        fingerprints = summary.get("fingerprints") if isinstance(summary, dict) else None
        checks["manifest_overlap"] = (
            isinstance(manifest, dict)
            and manifest.get("status") == "pass"
            and isinstance(summary, dict)
            and summary.get("schema_version") == 2
            and isinstance(overlap, dict)
            and set(overlap) == {"train_test", "train_val", "val_test"}
            and all(value == 0 for value in overlap.values())
            and isinstance(fingerprints, dict)
            and set(fingerprints) == {"train", "val", "test"}
            and all(_sha256(value) for value in fingerprints.values())
        )
        if isinstance(evidence, dict):
            checks["training_matrix"] = _training_ok(evidence.get("research_training"))
            checks["evaluation_matrix"] = _evaluations_ok(evidence.get("research_evaluations"))
            checks["selection"] = _selection_ok(evidence.get("research_selection"))
            checks["significance"] = _significance_ok(evidence.get("significance"))
            checks["robustness"] = _robustness_ok(evidence.get("robustness_matrix"))
            checks["failure_analysis"] = _failure_analysis_ok(evidence.get("failure_analysis"))
            checks["failure_audio"] = _failure_audio_ok(evidence.get("failure_audio"), evidence.get("failure_analysis"))
        checks.update(_model_checks(comparison))
        if isinstance(evidence, dict) and isinstance(comparison, dict):
            training = evidence.get("research_training")
            evaluations = evidence.get("research_evaluations")
            selection = evidence.get("research_selection")
            robustness = evidence.get("robustness_matrix")
            failure_audio = evidence.get("failure_audio")
            models = comparison.get("models")
            training_rows = training.get("runs") if isinstance(training, dict) else None
            evaluation_rows = evaluations.get("runs") if isinstance(evaluations, dict) else None
            selected_seed = int(selection.get("selected_seed", -1)) if isinstance(selection, dict) else -1
            training_hashes = {
                int(row["seed"]): row.get("checkpoint_sha256")
                for row in training_rows or []
                if isinstance(row, dict) and "seed" in row
            }
            evaluation_hashes = {
                int(row["seed"]): row.get("checkpoint_sha256")
                for row in evaluation_rows or []
                if isinstance(row, dict) and "seed" in row
            }
            clearhop = next(
                (row for row in models or [] if isinstance(row, dict) and row.get("name") == "ClearHop"),
                None,
            )
            clearhop_provenance = clearhop.get("provenance") if isinstance(clearhop, dict) else None
            clearhop_source = clearhop.get("source") if isinstance(clearhop, dict) else None
            selected_hash = training_hashes.get(selected_seed)
            checks["checkpoint_binding"] = (
                set(training_hashes) == set(SEEDS)
                and training_hashes == evaluation_hashes
                and _sha256(selected_hash)
                and isinstance(selection, dict)
                and selection.get("source_checkpoint_sha256") == selected_hash
                and selection.get("production_checkpoint_sha256") == selected_hash
                and isinstance(robustness, dict)
                and robustness.get("candidate_checkpoint_sha256") == selected_hash
                and isinstance(failure_audio, dict)
                and failure_audio.get("checkpoint_sha256") == selected_hash
                and isinstance(clearhop_provenance, dict)
                and clearhop_provenance.get("weight_sha256") == selected_hash
                and isinstance(clearhop_source, dict)
                and clearhop_source.get("checkpoint_sha256") == selected_hash
            )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    status = "pass" if all(checks.values()) else "fail"
    result: dict[str, Any] = {"schema_version": 1, "status": status, "checks": checks, "coverage": coverage, "receipts": {"research": str(research_path), "model_comparison": str(comparison_path), "deepfilternet3": str(deepfilter_path), "rnnoise_build": str(rnnoise_build_path)}}
    if errors:
        result["errors"] = errors
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_public_research(args.root)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
