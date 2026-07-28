import hashlib
import threading
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
import pytest

from desktop.pipeline import DenoisePipeline, CancellationError, load_wav_mono16k


def _wav(path: Path, sr: int = 8000, channels: int = 2, n: int = 800):
    t = np.arange(n, dtype=np.float32) / sr
    x = np.stack([0.2 * np.sin(2 * np.pi * 220 * t)] * channels, axis=1) if channels > 1 else 0.2 * np.sin(2 * np.pi * 220 * t)
    sf.write(path, x, sr, subtype="PCM_16")


def test_wav_conversion_reports_mono_and_rate(tmp_path: Path):
    source = tmp_path / "input.wav"
    _wav(source)
    samples, info = load_wav_mono16k(source)
    assert samples.ndim == 1
    assert info["sample_rate_before"] == 8000
    assert info["sample_rate_after"] == 16000
    assert info["channels_before"] == 2
    assert info["converted"] is True
    assert len(samples) == 1600


def test_process_file_atomic_and_receipt_deterministic(tmp_path: Path):
    source = tmp_path / "input.wav"
    output = tmp_path / "out.wav"
    _wav(source, sr=16000, channels=1, n=320)
    pipeline = DenoisePipeline(processor=lambda x: x * 0.5, checkpoint_sha256="a" * 64)
    first = pipeline.process_file(source, output)
    second = pipeline.process_file(source, output)
    assert first["status"] == "pass"
    assert first["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert first["output_sha256"] == second["output_sha256"]
    assert first["checkpoint_sha256"] == "a" * 64
    assert output.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_batch_error_isolation_and_state_reset(tmp_path: Path):
    src = tmp_path / "inputs"
    dst = tmp_path / "outputs"
    src.mkdir()
    _wav(src / "ok.wav", sr=16000, channels=1)
    (src / "broken.wav").write_bytes(b"not wav")
    calls = []

    class Processor:
        def reset(self):
            calls.append("reset")
        def __call__(self, x):
            calls.append(len(x))
            return x

    receipts = DenoisePipeline(processor=Processor()).process_batch(src, dst)
    assert {r["status"] for r in receipts} == {"pass", "error"}
    assert calls.count("reset") == 2
    assert (dst / "ok.wav").exists()


def test_cancellation_before_processing(tmp_path: Path):
    source = tmp_path / "input.wav"
    _wav(source, sr=16000, channels=1)
    event = threading.Event(); event.set()
    with pytest.raises(CancellationError):
        DenoisePipeline(processor=lambda x: x).process_file(source, tmp_path / "out.wav", cancel_event=event)
