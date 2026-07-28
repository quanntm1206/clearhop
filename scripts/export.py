from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.export import export_model
from src.checkpoint import file_sha256, validate_checkpoint_metadata
from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("checkpoints/mobiledeepfilternet_stateful"))
    parser.add_argument("--no-onnx", action="store_true")
    args = parser.parse_args()
    state = torch.load(args.checkpoint, map_location="cpu")
    validate_checkpoint_metadata(state)
    model = MobileDeepFilterNet(MobileDeepFilterNetConfig(**state["model_cfg"]))
    model.load_state_dict(state["model"], strict=True)
    metadata = export_model(
        model,
        args.output,
        export_onnx=not args.no_onnx,
        source_checkpoint=args.checkpoint,
        source_checkpoint_sha256=file_sha256(args.checkpoint),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
