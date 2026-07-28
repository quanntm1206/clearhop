"""Evaluate all five research seeds on frozen primary slices."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_gain_ablation import RESEARCH_SEEDS, _acceptance, _read_json, _sha256, _write_report


def run_research_evaluations(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    training = _read_json(root / "reports/generated/research_training.json")
    arm = str(training.get("arm"))
    training_rows = training.get("runs") if isinstance(training.get("runs"), list) else []
    by_seed = {int(row.get("seed", -1)): row for row in training_rows if isinstance(row, dict)}
    if set(by_seed) != set(RESEARCH_SEEDS):
        raise ValueError("research training receipt must contain exactly five locked seeds")
    slices = {"comparison": (0, 500), "audit": (500, 500)}
    baselines = {name: _read_json(root / f"reports/generated/gain_calibration_baseline_{name}.json") for name in slices}
    python = root / ".venv/Scripts/python.exe" if (root / ".venv/Scripts/python.exe").is_file() else Path(sys.executable)
    rows: list[dict[str, object]] = []
    for seed in RESEARCH_SEEDS:
        source = by_seed[seed]
        checkpoint = Path(str(source["checkpoint"]))
        if not checkpoint.is_absolute():
            checkpoint = root / checkpoint
        stage = "full" if source.get("source_stage") == "full" else "research"
        config = root / f"checkpoints/gain_calibration/{stage}/{arm}/seed_{seed}/resolved_config.yaml"
        evaluations: dict[str, object] = {}
        acceptance: dict[str, object] = {}
        for name, (offset, count) in slices.items():
            output = root / f"checkpoints/gain_calibration/research/{arm}/seed_{seed}/{name}_evaluation.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [str(python), "scripts/evaluate.py", "--checkpoint", str(checkpoint), "--config", str(config), "--manifest", "manifests/v2/fold_0_test.jsonl", "--output", str(output), "--offset", str(offset), "--max-items", str(count), "--profile", "full", "--per-item"]
            completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
            if completed.returncode != 0:
                raise RuntimeError(f"seed {seed} {name} evaluation failed: {completed.stderr[-2000:]}")
            payload = _read_json(output)
            evaluations[name] = {"path": output.relative_to(root).as_posix(), "sha256": _sha256(output)}
            acceptance[name] = _acceptance(payload, baselines[name])
        passed = all(isinstance(gate, dict) and gate.get("status") == "pass" for gate in acceptance.values())
        rows.append({"seed": seed, "checkpoint": str(checkpoint), "checkpoint_sha256": _sha256(checkpoint), "config": config.relative_to(root).as_posix(), "config_sha256": _sha256(config), "evaluations": evaluations, "acceptance": acceptance, "status": "pass" if passed else "fail"})
    all_pass = all(row["status"] == "pass" for row in rows)
    report = {"schema_version": 1, "stage": "research-evaluation", "status": "completed", "arm": arm, "seeds": list(RESEARCH_SEEDS), "all_seed_acceptance": all_pass, "runs": rows}
    _write_report(root / "reports/generated/research_evaluations.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = run_research_evaluations(PROJECT_ROOT)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_seed_acceptance"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
