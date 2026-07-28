"""CPU-only WAV enhancement entrypoint for the deployment bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR if (SCRIPT_DIR / "src").is_dir() else Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoint import file_sha256, validate_checkpoint_metadata
from src.inference import StreamingEnhancer
from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig


@torch.no_grad()
def enhance_waveform_cpu(model: MobileDeepFilterNet, waveform: torch.Tensor) -> torch.Tensor:
    if waveform.ndim != 1 or not torch.is_floating_point(waveform) or not torch.isfinite(waveform).all():
        raise ValueError("waveform must be a finite one-dimensional floating-point tensor")
    model = model.cpu().eval()
    enhancer = StreamingEnhancer(model)
    outputs: list[torch.Tensor] = []
    for start in range(0, waveform.numel(), 160):
        hop = waveform[start : start + 160]
        if hop.numel() < 160:
            hop = torch.nn.functional.pad(hop, (0, 160 - hop.numel()))
        outputs.append(enhancer.push(hop.cpu()))
    return torch.cat(outputs)[: waveform.numel()] if outputs else waveform.clone()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    import soundfile as sf

    audio, sr = sf.read(args.input, dtype="float32", always_2d=False)
    if sr != 16000 or audio.ndim != 1:
        raise ValueError("input must be mono 16 kHz WAV")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(state)
    model = MobileDeepFilterNet(MobileDeepFilterNetConfig(**state["model_cfg"]))
    model.load_state_dict(state["model"], strict=True)
    enhanced = enhance_waveform_cpu(model, torch.from_numpy(np.asarray(audio, dtype=np.float32)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, enhanced.numpy(), 16000, subtype="PCM_16")
    receipt = {"schema_version": 1, "status": "pass", "device": "cpu", "checkpoint_sha256": file_sha256(args.checkpoint), "input": str(args.input), "output": str(args.output), "output_sha256": file_sha256(args.output), "input_samples": int(len(audio)), "output_samples": int(enhanced.numel())}
    if args.receipt is not None:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
