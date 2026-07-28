"""Build deterministic paired failure analysis from per-item receipts."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def summarize_failures(items: list[dict[str, object]], top_k: int = 25) -> dict[str, object]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    normalized: list[dict[str, object]] = []
    failures: Counter[str] = Counter()
    subgroup_failures: Counter[str] = Counter()
    for item in items:
        try:
            index = int(item["index"])
            delta = float(item["si_sdr_delta"])
            snr_delta = float(item.get("snr_delta", 0.0))
            stoi_delta = float(item.get("stoi_delta", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("failure items require numeric index, si_sdr_delta, snr_delta, and stoi_delta") from exc
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        row = {"index": index, "si_sdr_delta": delta, "snr_delta": snr_delta, "stoi_delta": stoi_delta, "metadata": metadata}
        failed = False
        if delta < -0.3:
            failures["si_sdr_delta"] += 1
            failed = True
        if snr_delta <= 0:
            failures["snr_non_positive"] += 1
            failed = True
        if stoi_delta < -0.005:
            failures["stoi_delta"] += 1
            failed = True
        if failed:
            for key in ("snr_band", "noise_family", "speaker"):
                subgroup_failures[f"{key}:{metadata.get(key, 'unknown')}"] += 1
        normalized.append(row)
    normalized.sort(key=lambda row: (float(row["si_sdr_delta"]), int(row["index"])))
    return {
        "schema_version": 1,
        "n": len(normalized),
        "failure_counts": dict(sorted(failures.items())),
        "subgroup_failure_counts": dict(sorted(subgroup_failures.items())),
        "worst_cases": normalized[:top_k],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/generated/failure_analysis.json"))
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    c_items = candidate.get("items", {}).get("enhanced", [])
    b_items = baseline.get("items", {}).get("enhanced", [])
    by_index = {int(row["index"]): row for row in b_items}
    paired: list[dict[str, object]] = []
    for row in c_items:
        base = by_index.get(int(row["index"]))
        if base is None:
            continue
        paired.append({
            "index": int(row["index"]),
            "si_sdr_delta": float(row["si_sdr"]) - float(base["si_sdr"]),
            "snr_delta": float(row["snr"]) - float(base["snr"]),
            "stoi_delta": float(row.get("stoi", 0.0)) - float(base.get("stoi", 0.0)),
            "metadata": row.get("metadata", {}),
        })
    report = summarize_failures(paired, args.top_k)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
