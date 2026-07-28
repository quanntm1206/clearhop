"""Deterministic CPU streaming soak and fault-injection receipt."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.benchmark_cpu import _rss_bytes
from src.checkpoint import file_sha256, validate_checkpoint_metadata
from src.inference import StreamingEnhancer
from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig


@torch.no_grad()
def run_cpu_soak(
    root: Path,
    checkpoint: Path,
    *,
    seconds: float = 7200.0,
    max_iterations: int | None = None,
) -> dict[str, object]:
    """Run CPU streaming, reset, and invalid-input checks."""
    root = Path(root).resolve()
    checkpoint = Path(checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    checkpoint = checkpoint.resolve()
    if seconds <= 0 or (max_iterations is not None and max_iterations < 1):
        raise ValueError("seconds must be positive and max_iterations must be positive when provided")
    if not checkpoint.is_file() or root not in checkpoint.parents:
        raise ValueError("checkpoint must be an existing file inside root")
    torch.set_num_threads(1)
    torch.manual_seed(1)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(state)
    model = MobileDeepFilterNet(MobileDeepFilterNetConfig(**state["model_cfg"]))
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    enhancer = StreamingEnhancer(model)
    hop = torch.randn(160)
    start_time = time.monotonic()
    rss_start = _rss_bytes()
    iterations = 0
    failures: list[str] = []
    while (time.monotonic() - start_time) < seconds and (max_iterations is None or iterations < max_iterations):
        output = enhancer.push(hop)
        if not torch.isfinite(output).all():
            failures.append("non-finite output")
            break
        iterations += 1
    before_reset = enhancer.num_frames
    enhancer.reset()
    reset_frames = enhancer.num_frames
    enhancer.reset()
    reset_idempotent = enhancer.num_frames == 0 and enhancer.history_length == 0
    fault_results: dict[str, bool] = {}
    for name, bad in (
        ("empty", torch.empty(0)),
        ("wrong_shape", torch.zeros(159)),
        ("nan", torch.full((160,), float("nan"))),
    ):
        try:
            enhancer.push(bad)
        except ValueError:
            fault_results[name] = enhancer.num_frames == 0
        else:
            fault_results[name] = False
    rss_end = _rss_bytes()
    elapsed_seconds = time.monotonic() - start_time
    bounded_memory = rss_start is None or rss_end is None or (rss_end - rss_start) < 256 * 2**20
    status = "pass" if iterations > 0 and not failures and reset_idempotent and all(fault_results.values()) and bounded_memory else "fail"
    return {
        "schema_version": 1,
        "status": status,
        "device": "cpu",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "requested_seconds": float(seconds),
        "elapsed_seconds": float(elapsed_seconds),
        "iterations": int(iterations),
        "environment": {"os": platform.platform(), "torch_version": torch.__version__, "torch_threads": int(torch.get_num_threads())},
        "reset": {"frames_before": int(before_reset), "frames_after": int(reset_frames), "idempotent": reset_idempotent},
        "faults": fault_results,
        "failures": failures,
        "memory": {"rss_start_bytes": rss_start, "rss_end_bytes": rss_end, "bounded": bounded_memory},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=7200.0)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--output", type=Path, default=Path("reports/generated/cpu_soak.json"))
    args = parser.parse_args()
    report = run_cpu_soak(PROJECT_ROOT, args.checkpoint, seconds=args.seconds, max_iterations=args.max_iterations)
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
