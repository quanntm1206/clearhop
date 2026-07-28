"""Execute the locked five-seed research training matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_gain_ablation import RESEARCH_SEEDS, _execute_records, _read_json, _serializable_plan, _sha256, _write_report, build_plan


def run_research_training(root: Path, arm: str, *, resume: bool, parallel_workers: int = 1) -> dict[str, object]:
    root = Path(root).resolve()
    plan = build_plan(root, "research", [arm], RESEARCH_SEEDS)
    reused: list[dict[str, object]] = []
    full_path = root / "reports/generated/gain_calibration_full.json"
    full = _read_json(full_path) if full_path.is_file() else {}
    full_rows = full.get("runs") if isinstance(full.get("runs"), list) and full.get("selected_arm") == arm else []
    by_seed = {int(row.get("seed", -1)): row for row in full_rows if isinstance(row, dict)}
    for seed in RESEARCH_SEEDS:
        row = by_seed.get(seed)
        checkpoint_ref = row.get("checkpoint") if isinstance(row, dict) else None
        if not isinstance(checkpoint_ref, dict):
            continue
        checkpoint = root / str(checkpoint_ref.get("path"))
        if not checkpoint.is_file() or checkpoint_ref.get("sha256") != _sha256(checkpoint):
            continue
        reused.append({
            "stage": "research",
            "arm": arm,
            "seed": seed,
            "status": "already-complete",
            "source_stage": "full",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_ref.get("sha256"),
            "validation_metrics": row.get("validation_metrics"),
        })
    reused_seeds = {int(row["seed"]) for row in reused}
    missing_plan = [item for item in plan if int(item["seed"]) not in reused_seeds]
    records = [*reused, *_execute_records(root, missing_plan, resume=resume, parallel_workers=parallel_workers)]
    records.sort(key=lambda row: int(row["seed"]))
    complete = len(records) == len(RESEARCH_SEEDS) and all(row.get("status") in {"completed", "already-complete"} for row in records)
    report = {
        "schema_version": 1,
        "stage": "research",
        "status": "completed" if complete else "failed",
        "arm": arm,
        "seeds": list(RESEARCH_SEEDS),
        "planned_run_count": len(plan),
        "runs": records,
    }
    _write_report(root / "reports/generated/research_training.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", default="complex_nmse_sisdr_beta_0p02")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--parallel-workers", type=int, default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan = build_plan(root, "research", [args.arm], RESEARCH_SEEDS)
    if args.dry_run:
        print(json.dumps({"schema_version": 1, "status": "dry-run", "stage": "research", "runs": _serializable_plan(plan)}, indent=2, sort_keys=True))
        return 0
    report = run_research_training(root, args.arm, resume=args.resume, parallel_workers=args.parallel_workers)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
