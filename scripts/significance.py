"""Generate hash-bound paired bootstrap and significance receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.checkpoint import file_sha256
from src.statistics import paired_bootstrap_delta, paired_significance


def build_significance_receipt(candidate_path: Path, baseline_path: Path, *, resamples: int = 10000) -> dict[str, object]:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    c_rows = candidate.get("items", {}).get("enhanced", [])
    b_rows = baseline.get("items", {}).get("enhanced", [])
    c_by_id = {int(row["index"]): row for row in c_rows}
    b_by_id = {int(row["index"]): row for row in b_rows}
    shared = sorted(set(c_by_id) & set(b_by_id))
    if not shared or len(shared) != len(c_rows) or len(shared) != len(b_rows):
        raise ValueError("candidate and baseline require exact paired per-item indices")
    metrics: dict[str, object] = {}
    for metric in ("si_sdr", "snr", "stoi"):
        c_values = [float(c_by_id[index][metric]) for index in shared]
        b_values = [float(b_by_id[index][metric]) for index in shared]
        metrics[metric] = {
            "bootstrap": paired_bootstrap_delta(c_values, b_values, seed=0, resamples=resamples),
            "significance": paired_significance(c_values, b_values),
        }
    return {
        "schema_version": 1,
        "status": "pass",
        "candidate": {"path": str(candidate_path), "sha256": file_sha256(candidate_path)},
        "baseline": {"path": str(baseline_path), "sha256": file_sha256(baseline_path)},
        "paired_items": len(shared),
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("reports/generated/significance.json"))
    args = parser.parse_args()
    report = build_significance_receipt(args.candidate, args.baseline, resamples=args.resamples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
