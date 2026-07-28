"""Promote the validation-selected five-seed model after all research gates pass."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify import export_parity_audit
from src.checkpoint import file_sha256, validate_checkpoint_metadata
from src.export import export_model
from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        if file_sha256(source) != file_sha256(temporary):
            raise OSError("promoted checkpoint hash mismatch")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def promote_research_model(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    training = json.loads((root / "reports/generated/research_training.json").read_text(encoding="utf-8"))
    evaluations = json.loads((root / "reports/generated/research_evaluations.json").read_text(encoding="utf-8"))
    if training.get("status") != "completed" or evaluations.get("all_seed_acceptance") is not True:
        raise ValueError("promotion requires completed five-seed training and all-seed acceptance")
    rows = training.get("runs")
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("promotion requires exactly five training rows")
    ranked = sorted(rows, key=lambda row: (-float(row["validation_metrics"]["snri"]), -float(row["validation_metrics"]["si_sdri"]), abs(float(row["validation_metrics"]["gain_error_db_median"])), int(row["seed"])))
    winner = ranked[0]
    source = Path(str(winner["checkpoint"]))
    if not source.is_absolute():
        source = root / source
    destination = root / "checkpoints/production/best.pth"
    _atomic_copy(source, destination)
    state = torch.load(destination, map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(state)
    model = MobileDeepFilterNet(MobileDeepFilterNetConfig(**state["model_cfg"]))
    model.load_state_dict(state["model"], strict=True)
    export_model(model, root / "checkpoints/production/export", export_onnx=True, source_checkpoint=destination, source_checkpoint_sha256=file_sha256(destination))
    parity = export_parity_audit(root, steps=20, checkpoint_path=destination, torchscript_path=root / "checkpoints/production/export.ts", onnx_path=root / "checkpoints/production/export.onnx")
    parity_path = root / "reports/generated/production_export_parity.json"
    parity_path.write_text(json.dumps(parity, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if parity.get("status") != "pass":
        raise RuntimeError("promoted export parity failed")
    receipt = {
        "schema_version": 1,
        "status": "pass",
        "selected_seed": int(winner["seed"]),
        "selection_basis": "validation",
        "metric_order": ["validation_snri", "validation_si_sdri", "absolute_gain_error"],
        "ranked_validation_metrics": [{"seed": int(row["seed"]), **row["validation_metrics"]} for row in ranked],
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": file_sha256(source),
        "production_checkpoint": str(destination),
        "production_checkpoint_sha256": file_sha256(destination),
        "parity": {"path": parity_path.relative_to(root).as_posix(), "sha256": file_sha256(parity_path)},
    }
    output = root / "reports/generated/research_selection.json"
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    argparse.ArgumentParser().parse_args()
    report = promote_research_model(PROJECT_ROOT)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
