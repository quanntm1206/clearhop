"""Strict CPU-only benchmark and receipt for production readiness."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.checkpoint import file_sha256, validate_checkpoint_metadata
from src.inference import StreamingEnhancer
from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig


def summarize_cpu_latencies(seconds: Sequence[float], hop_seconds: float) -> dict[str, float | int]:
    """Summarize finite CPU timings, including the production p99 gate."""
    values = np.asarray(list(seconds), dtype=np.float64)
    if values.size == 0 or hop_seconds <= 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("latencies must be non-empty, finite, non-negative, and hop_seconds must be positive")
    mean = float(values.mean())
    return {
        "n": int(values.size),
        "mean_ms": mean * 1000.0,
        "p50_ms": float(np.percentile(values, 50.0) * 1000.0),
        "p95_ms": float(np.percentile(values, 95.0) * 1000.0),
        "p99_ms": float(np.percentile(values, 99.0) * 1000.0),
        "max_ms": float(values.max() * 1000.0),
        "realtime_factor": float(hop_seconds / mean) if mean > 0 else float("inf"),
    }


def _rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        if os.name != "nt":
            return None
        try:
            import ctypes
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            get_process = ctypes.windll.kernel32.GetCurrentProcess
            get_process.restype = wintypes.HANDLE
            handle = get_process()
            get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
            get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
            get_memory.restype = wintypes.BOOL
            if not get_memory(handle, ctypes.byref(counters), counters.cb):
                return None
            return int(counters.WorkingSetSize)
        except Exception:
            return None


def select_affinity_cpu(mask: int, preferred: int | None = None) -> int | None:
    """Select a requested logical CPU when allowed, else the lowest allowed CPU."""
    if mask <= 0:
        return None
    if preferred is not None and preferred >= 0 and mask & (1 << preferred):
        return preferred
    return (mask & -mask).bit_length() - 1


def onnx_session_options(ort_module: object) -> object:
    """Prevent ONNX thread-pool contention after pinning the process to one CPU."""
    options = ort_module.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort_module.ExecutionMode.ORT_SEQUENTIAL
    return options


def _pin_single_cpu(preferred_cpu: int | None = None) -> int | None:
    """Best-effort single-core affinity for reproducible latency receipts."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            process_mask = ctypes.c_size_t()
            system_mask = ctypes.c_size_t()
            get_process = ctypes.windll.kernel32.GetCurrentProcess
            get_process.restype = wintypes.HANDLE
            handle = get_process()
            get_affinity = ctypes.windll.kernel32.GetProcessAffinityMask
            get_affinity.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
            get_affinity.restype = wintypes.BOOL
            if not get_affinity(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
                return None
            mask = int(process_mask.value)
            cpu = select_affinity_cpu(mask, preferred_cpu)
            if cpu is None:
                return None
            selected = 1 << cpu
            set_affinity = ctypes.windll.kernel32.SetProcessAffinityMask
            set_affinity.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
            set_affinity.restype = wintypes.BOOL
            if not set_affinity(handle, ctypes.c_size_t(selected)):
                return None
            return cpu
        if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
            allowed = set(os.sched_getaffinity(0))
            cpu = preferred_cpu if preferred_cpu in allowed else min(allowed)
            os.sched_setaffinity(0, {cpu})
            return int(cpu)
    except Exception:
        return None
    return None


@torch.no_grad()
def run_cpu_benchmark(
    root: Path,
    checkpoint: Path,
    *,
    iterations: int = 5000,
    warmup: int = 50,
    threshold_ms: float = 10.0,
    onnx_path: Path | None = None,
    affinity_cpu: int | None = None,
) -> dict[str, object]:
    """Run CPU-only neural and end-to-end latency measurements."""
    root = Path(root).resolve()
    checkpoint = Path(checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    checkpoint = checkpoint.resolve()
    if iterations < 1 or warmup < 0 or threshold_ms <= 0:
        raise ValueError("iterations must be positive, warmup non-negative, threshold positive")
    if not checkpoint.is_file() or root not in checkpoint.parents:
        raise ValueError("checkpoint must be an existing file inside root")
    onnx_path = Path(onnx_path) if onnx_path is not None else root / "checkpoints/gain_calibration/export.onnx"
    if not onnx_path.is_absolute():
        onnx_path = root / onnx_path
    onnx_path = onnx_path.resolve()
    if not onnx_path.is_file() or root not in onnx_path.parents:
        raise ValueError("ONNX export must be an existing file inside root")

    affinity_cpu = _pin_single_cpu(affinity_cpu)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(state)
    model = MobileDeepFilterNet(MobileDeepFilterNetConfig(**state["model_cfg"]))
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    if next(model.parameters()).device.type != "cpu":
        raise RuntimeError("CPU benchmark model is not on CPU")
    enhancer = StreamingEnhancer(model)
    hop = torch.randn(160)
    features = torch.randn(1, 1, 161, model.temporal_receptive_frames)
    hidden = torch.zeros(model.cfg.gru_layers, 1, model.cfg.gru_hidden)
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), sess_options=onnx_session_options(ort), providers=["CPUExecutionProvider"])
    features_np = features.numpy()
    hidden_onnx = hidden.numpy()
    for _ in range(warmup):
        model.forward_streaming(features, hidden)
        enhancer.push(hop)
        hidden_onnx = session.run(None, {"feats_logp": features_np, "hidden": hidden_onnx})[2]

    rss_start = _rss_bytes()
    core_times: list[float] = []
    stream_times: list[float] = []
    onnx_times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        model.forward_streaming(features, hidden)
        core_times.append((time.perf_counter_ns() - start) / 1e9)
        start = time.perf_counter_ns()
        output = enhancer.push(hop)
        stream_times.append((time.perf_counter_ns() - start) / 1e9)
        start = time.perf_counter_ns()
        hidden_onnx = session.run(None, {"feats_logp": features_np, "hidden": hidden_onnx})[2]
        onnx_times.append((time.perf_counter_ns() - start) / 1e9)
        if not torch.isfinite(output).all():
            raise FloatingPointError("CPU streaming output is non-finite")
    rss_end = _rss_bytes()
    neural = summarize_cpu_latencies(core_times, 0.01)
    streaming = summarize_cpu_latencies(stream_times, 0.01)
    onnx_core = summarize_cpu_latencies(onnx_times, 0.01)
    status = "pass" if float(streaming["p95_ms"]) < threshold_ms and float(onnx_core["p95_ms"]) < threshold_ms else "fail"
    return {
        "schema_version": 1,
        "status": status,
        "device": "cpu",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "onnx": str(onnx_path),
        "onnx_sha256": file_sha256(onnx_path),
        "iterations": int(iterations),
        "warmup": int(warmup),
        "thresholds": {"streaming_p95_ms_lt": float(threshold_ms)},
        "environment": {
            "os": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "torch_version": torch.__version__,
            "torch_threads": int(torch.get_num_threads()),
            "affinity_cpu": affinity_cpu,
        },
        "model_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "hop_samples": 160,
        "hop_seconds": 0.01,
        "neural_core": neural,
        "onnx_core": onnx_core,
        "streaming_end_to_end": streaming,
        "memory": {
            "rss_start_bytes": rss_start,
            "rss_end_bytes": rss_end,
            "rss_delta_bytes": None if rss_start is None or rss_end is None else rss_end - rss_start,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--threshold-ms", type=float, default=10.0)
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--affinity-cpu", type=int, help="Pin benchmark to this logical CPU when available")
    parser.add_argument("--output", type=Path, default=Path("reports/generated/cpu_benchmark.json"))
    args = parser.parse_args()
    report = run_cpu_benchmark(PROJECT_ROOT, args.checkpoint, iterations=args.iterations, warmup=args.warmup, threshold_ms=args.threshold_ms, onnx_path=args.onnx, affinity_cpu=args.affinity_cpu)
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
