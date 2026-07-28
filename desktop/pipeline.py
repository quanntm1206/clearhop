"""Dependency-safe local WAV processing contract for the desktop app."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np


class CancellationError(RuntimeError):
    """Raised when a user cancels processing before an atomic commit."""


@dataclass(frozen=True)
class ConversionInfo:
    sample_rate_before: int
    sample_rate_after: int
    channels_before: int
    channels_after: int = 1
    converted: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype(np.float32, copy=False)
    try:
        from scipy.signal import resample_poly

        gcd = int(np.gcd(source_rate, target_rate))
        return np.asarray(resample_poly(samples, target_rate // gcd, source_rate // gcd), dtype=np.float32)
    except ImportError as exc:  # pragma: no cover - package declares scipy
        raise RuntimeError("scipy is required for sample-rate conversion") from exc


def load_wav_mono16k(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, dict[str, Any]]:
    """Read WAV, downmix channels, resample, and report every conversion."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import soundfile as sf

        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception as exc:
        raise ValueError(f"unable to read WAV: {path}") from exc
    channels = int(data.shape[1])
    if channels < 1 or data.shape[0] == 0:
        raise ValueError("WAV must contain at least one sample and one channel")
    mono = np.mean(data, axis=1, dtype=np.float32) if channels > 1 else data[:, 0]
    converted = channels != 1 or int(sample_rate) != int(target_sr)
    output = _resample(np.asarray(mono, dtype=np.float32), int(sample_rate), int(target_sr))
    output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0).astype(np.float32, copy=False)
    return output, ConversionInfo(int(sample_rate), int(target_sr), channels, 1, converted).as_dict()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_processor(checkpoint_path: str | Path) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
    """Load the repository model and return a stateless NumPy processor plus hash.

    Imports Torch/model code lazily so WAV validation and headless tests remain usable
    without loading a checkpoint.
    """
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    import torch
    from src.checkpoint import validate_checkpoint_metadata
    from src.cpu_runtime import enhance_waveform_cpu
    from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(state)
    model = MobileDeepFilterNet(MobileDeepFilterNetConfig(**state["model_cfg"]))
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    def process(samples: np.ndarray) -> np.ndarray:
        enhanced = enhance_waveform_cpu(model, torch.from_numpy(np.asarray(samples, dtype=np.float32)))
        return enhanced.detach().cpu().numpy().astype(np.float32, copy=False)

    return process, file_sha256(checkpoint)


class DenoisePipeline:
    """Single/batch processor. Inject a callable for tests or deployment runtime."""

    def __init__(self, processor: Callable[[np.ndarray], np.ndarray] | None = None, *, checkpoint_sha256: str | None = None, target_sr: int = 16000) -> None:
        self.processor = processor or (lambda samples: samples.copy())
        self.checkpoint_sha256 = checkpoint_sha256
        self.target_sr = int(target_sr)

    def _check_cancel(self, cancel_event: threading.Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise CancellationError("processing cancelled")

    def _reset(self) -> None:
        reset = getattr(self.processor, "reset", None)
        if callable(reset):
            reset()

    def process_file(self, input_path: str | Path, output_path: str | Path, *, receipt_path: str | Path | None = None, cancel_event: threading.Event | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        source = Path(input_path)
        destination = Path(output_path)
        self._check_cancel(cancel_event)
        input_digest = file_sha256(source)
        self._reset()
        samples, conversion = load_wav_mono16k(source, self.target_sr)
        self._check_cancel(cancel_event)
        result = np.asarray(self.processor(samples), dtype=np.float32)
        if result.ndim != 1 or result.size != samples.size or not np.isfinite(result).all():
            raise ValueError("processor must return finite one-dimensional samples of unchanged length")
        self._check_cancel(cancel_event)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            fd, temp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
            os.close(fd)
            temporary = Path(temp_name)
            import soundfile as sf

            sf.write(str(temporary), result, self.target_sr, subtype="PCM_16", format="WAV")
            os.replace(temporary, destination)
            temporary = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        receipt = {
            "schema_version": 1,
            "status": "pass",
            "input": str(source),
            "output": str(destination),
            "input_sha256": input_digest,
            "output_sha256": file_sha256(destination),
            "checkpoint_sha256": self.checkpoint_sha256,
            "sample_rate_conversion": conversion,
            "input_samples": int(samples.size),
            "output_samples": int(result.size),
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }
        if receipt_path is not None:
            path = Path(receipt_path); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return receipt

    def process_batch(self, input_dir: str | Path, output_dir: str | Path, *, cancel_event: threading.Event | None = None, on_progress: Callable[[int, int, dict[str, Any]], None] | None = None) -> list[dict[str, Any]]:
        source_dir, destination_dir = Path(input_dir), Path(output_dir)
        files = sorted(p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() == ".wav")
        receipts: list[dict[str, Any]] = []
        for index, source in enumerate(files):
            if cancel_event is not None and cancel_event.is_set():
                break
            destination = destination_dir / source.name
            try:
                receipt = self.process_file(source, destination, cancel_event=cancel_event)
            except CancellationError:
                break
            except Exception as exc:
                receipt = {"schema_version": 1, "status": "error", "input": str(source), "output": str(destination), "error": f"{type(exc).__name__}: {exc}"}
            receipts.append(receipt)
            if on_progress is not None:
                on_progress(index + 1, len(files), receipt)
        return receipts
