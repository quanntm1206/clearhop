"""Benchmark stateful neural and streaming inference on the active machine."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.checkpoint import file_sha256, validate_checkpoint_metadata
from src.inference import StreamingEnhancer
from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig


def summarize_latencies(seconds: Sequence[float], hop_seconds: float) -> dict[str, float | int]:
    values = np.asarray(list(seconds), dtype=np.float64)
    if values.size == 0 or hop_seconds <= 0:
        raise ValueError("latencies must be non-empty and hop_seconds must be positive")
    return {
        "n": int(values.size),
        "mean_ms": float(values.mean() * 1000.0),
        "p50_ms": float(np.percentile(values, 50.0) * 1000.0),
        "p95_ms": float(np.percentile(values, 95.0) * 1000.0),
        "max_ms": float(values.max() * 1000.0),
        "realtime_factor": float(hop_seconds / values.mean()),
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, default=Path("reports/generated/benchmark.json"))
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    validate_checkpoint_metadata(state)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model = MobileDeepFilterNet(MobileDeepFilterNetConfig(**state["model_cfg"]))
    model.load_state_dict(state["model"], strict=True)
    model.to(device).eval()
    enhancer = StreamingEnhancer(model)
    hop = torch.randn(160, device=device)
    features = torch.randn(1, 1, 161, model.temporal_receptive_frames, device=device)
    hidden = torch.zeros(model.cfg.gru_layers, 1, model.cfg.gru_hidden, device=device)

    for _ in range(max(0, args.warmup)):
        model.forward_streaming(features, hidden)
        enhancer.push(hop)
    _sync(device)

    core_times: list[float] = []
    stream_times: list[float] = []
    for _ in range(max(1, args.iterations)):
        start = time.perf_counter()
        model.forward_streaming(features, hidden)
        _sync(device)
        core_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        enhancer.push(hop)
        _sync(device)
        stream_times.append(time.perf_counter() - start)

    memory = {"peak_allocated_mb": None, "peak_reserved_mb": None}
    if device.type == "cuda":
        memory = {
            "peak_allocated_mb": float(torch.cuda.max_memory_allocated(device) / 2**20),
            "peak_reserved_mb": float(torch.cuda.max_memory_reserved(device) / 2**20),
        }
    report = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "checkpoint_bytes": int(args.checkpoint.stat().st_size),
        "hop_samples": 160,
        "hop_seconds": 0.01,
        "warmup": int(args.warmup),
        "iterations": int(args.iterations),
        "neural_core": summarize_latencies(core_times, 0.01),
        "streaming_end_to_end": summarize_latencies(stream_times, 0.01),
        "memory": memory,
    }
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
