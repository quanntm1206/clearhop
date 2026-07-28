from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QFileDialog

from desktop.app import MainWindow
from desktop.pipeline import DenoisePipeline


def test_offscreen_open_process_save_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    app = QApplication.instance() or QApplication([])
    source = tmp_path / "input.wav"
    output_dir = tmp_path / "output"
    sf.write(source, np.linspace(-0.1, 0.1, 1600, dtype=np.float32), 16000)
    pipeline = DenoisePipeline(lambda samples: samples.copy(), checkpoint_sha256="test-checkpoint")
    window = MainWindow(pipeline)
    window.add_paths([source])
    monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args, **kwargs: str(output_dir))

    window.start_processing()
    deadline = time.monotonic() + 10.0
    while window.worker is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)

    output = output_dir / "input_denoised.wav"
    receipt = output.with_suffix(".json")
    assert window.worker is None
    assert output.is_file()
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "pass"
    window.close()
