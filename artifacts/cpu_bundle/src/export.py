"""Stateful neural-core export with explicit GRU state."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import torch

from .model import MobileDeepFilterNet


class StatefulExportWrapper(torch.nn.Module):
    def __init__(self, model: MobileDeepFilterNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, feats_logp: torch.Tensor, hidden: torch.Tensor):
        return self.model.forward_streaming(feats_logp, hidden)


def export_model(
    model: MobileDeepFilterNet,
    output_path: Path,
    *,
    example_frames: int | None = None,
    export_onnx: bool = True,
    source_checkpoint: Optional[Path] = None,
    source_checkpoint_sha256: Optional[str] = None,
) -> dict[str, Any]:
    """Export TorchScript and optionally ONNX with explicit hidden state."""
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = StatefulExportWrapper(model).eval()
    device = next(model.parameters()).device
    example_frames = example_frames or model.temporal_receptive_frames
    example = torch.zeros(1, 1, model.cfg.freq_bins, example_frames, device=device)
    hidden = torch.zeros(model.cfg.gru_layers, 1, model.cfg.gru_hidden, device=device)
    traced = torch.jit.trace(wrapper, (example, hidden), strict=False)
    torchscript_path = output_path.with_suffix(".ts")
    traced.save(str(torchscript_path))

    onnx_path = output_path.with_suffix(".onnx")
    onnx_written = False
    if export_onnx:
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            torch.onnx.export(
                wrapper,
                (example, hidden),
                str(onnx_path),
                input_names=["feats_logp", "hidden"],
                output_names=["mask", "w_taps", "hidden_new"],
                opset_version=18,
                dynamo=False,
                dynamic_axes={
                    "feats_logp": {0: "batch", 3: "frames"},
                    "hidden": {1: "batch"},
                    "mask": {0: "batch"},
                    "w_taps": {0: "batch"},
                    "hidden_new": {1: "batch"},
                },
            )
            onnx_written = True
        except Exception as exc:
            (output_path.parent / "export_warnings.txt").write_text(
                f"ONNX export unavailable: {type(exc).__name__}: {exc}\n", encoding="utf-8"
            )

    metadata = {
        "schema_version": 2,
        "input": {"shape": ["B", 1, model.cfg.freq_bins, "T"], "name": "feats_logp"},
        "hidden": {"shape": [model.cfg.gru_layers, "B", model.cfg.gru_hidden], "name": "hidden"},
        "outputs": ["mask", "w_taps", "hidden_new"],
        "temporal_receptive_frames": model.temporal_receptive_frames,
        "torchscript": str(torchscript_path),
        "onnx": str(onnx_path) if onnx_written else None,
        "model_cfg": model.cfg.__dict__,
        "source_checkpoint": str(source_checkpoint) if source_checkpoint is not None else None,
        "source_checkpoint_sha256": source_checkpoint_sha256,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata
