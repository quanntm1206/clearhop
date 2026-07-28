"""Evaluate fixed research slices and emit paired robustness deltas."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.research_slices import load_research_slices
from src.checkpoint import file_sha256


def summarize_robustness(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("robustness rows must not be empty")
    required = {"name", "offset", "count", "snri_delta", "si_sdri_delta", "stoi_delta"}
    if any(set(row) != required for row in rows):
        raise ValueError("robustness rows have an unstable schema")
    for row in rows:
        for key in ("snri_delta", "si_sdri_delta", "stoi_delta"):
            value = float(row[key])
            if not (-1e9 < value < 1e9):
                raise ValueError("robustness metric is non-finite")
    return {"schema_version": 1, "status": "pass", "slices": len(rows), "rows": rows}


def run_robustness_matrix(
    root: Path,
    *,
    candidate_checkpoint: Path,
    candidate_config: Path,
    baseline_checkpoint: Path,
    baseline_config: Path,
    manifest: Path,
    slice_spec: Path,
) -> dict[str, object]:
    root = Path(root).resolve()
    slices = load_research_slices(slice_spec)
    with tempfile.TemporaryDirectory(prefix="research_slices_", dir=root / "reports/generated") as tmp:
        temp = Path(tmp)
        rows: list[dict[str, object]] = []
        for item in slices:
            name = str(item["name"])
            candidate_out = temp / f"candidate_{name}.json"
            baseline_out = temp / f"baseline_{name}.json"
            common = ["--manifest", str(manifest), "--offset", str(item["offset"]), "--max-items", str(item["count"]), "--profile", "screen", "--per-item"]
            for checkpoint, config, output in ((candidate_checkpoint, candidate_config, candidate_out), (baseline_checkpoint, baseline_config, baseline_out)):
                python = root / ".venv/Scripts/python.exe" if (root / ".venv/Scripts/python.exe").is_file() else Path(sys.executable)
                command = [str(python), "scripts/evaluate.py", "--checkpoint", str(checkpoint), "--config", str(config), "--output", str(output), *common]
                completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
                if completed.returncode != 0:
                    raise RuntimeError(f"research slice evaluation failed: {name}: {completed.stderr[-2000:]}")
            candidate = json.loads(candidate_out.read_text(encoding="utf-8"))
            baseline = json.loads(baseline_out.read_text(encoding="utf-8"))
            c = candidate["enhanced"]
            b = baseline["enhanced"]
            rows.append({
                "name": name,
                "offset": int(item["offset"]),
                "count": int(item["count"]),
                "snri_delta": float(c["snr_improvement_mean"] - b["snr_improvement_mean"]),
                "si_sdri_delta": float(c["si_sdr_improvement_mean"] - b["si_sdr_improvement_mean"]),
                "stoi_delta": float(c["stoi"]["mean"] - b["stoi"]["mean"]),
            })
    report = summarize_robustness(rows)
    report["candidate_checkpoint_sha256"] = file_sha256(candidate_checkpoint)
    report["baseline_checkpoint_sha256"] = file_sha256(baseline_checkpoint)
    report["slice_spec"] = str(slice_spec)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/v2/fold_0_test.jsonl"))
    parser.add_argument("--slice-spec", type=Path, default=Path("configs/evaluation/research_slices.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/generated/robustness_matrix.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = run_robustness_matrix(root, candidate_checkpoint=args.candidate_checkpoint, candidate_config=args.candidate_config, baseline_checkpoint=args.baseline_checkpoint, baseline_config=args.baseline_config, manifest=args.manifest, slice_spec=args.slice_spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
