"""PySide6 desktop application for local WAV denoising.

The UI depends only on the stable :mod:`desktop.pipeline` contract.  Pipeline
imports stay lazy so CLI/test environments without Qt or model dependencies can
still import this module.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Iterable

try:  # Keep headless/CI imports safe when optional desktop extra is absent.
    from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot
    from PySide6.QtGui import QDragEnterEvent, QDropEvent
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QPushButton,
        QProgressBar,
        QVBoxLayout,
        QWidget,
    )
    QT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by minimal installs
    QT_AVAILABLE = False
    QApplication = None  # type: ignore[assignment]
    MainWindow = None  # type: ignore[assignment,misc]
    ProcessWorker = None  # type: ignore[assignment,misc]


def _pipeline_class() -> Any:
    """Resolve pipeline lazily, allowing this module to be imported standalone."""
    try:
        from .pipeline import DenoisePipeline
    except ImportError:  # PyInstaller executes app.py as __main__.
        from desktop.pipeline import DenoisePipeline

    return DenoisePipeline


def _default_pipeline() -> Any:
    """Build the bundled model pipeline, honoring an explicit checkpoint path."""
    try:
        from .pipeline import DenoisePipeline, checkpoint_processor
    except ImportError:  # PyInstaller executes app.py as __main__.
        from desktop.pipeline import DenoisePipeline, checkpoint_processor

    configured = os.environ.get("NOISE_REDUCE_CHECKPOINT")
    candidates = [Path(configured)] if configured else []
    candidates += [
        Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)) / "assets" / "checkpoint.pth",
        Path(sys.prefix).resolve().parent / "assets" / "checkpoint.pth",
        Path.cwd() / "assets" / "checkpoint.pth",
        Path(sys.executable).resolve().parent / "assets" / "checkpoint.pth",
        Path.cwd() / "artifacts" / "cpu_bundle" / "checkpoint.pth",
        Path(__file__).resolve().parents[1] / "artifacts" / "cpu_bundle" / "checkpoint.pth",
    ]
    for checkpoint in candidates:
        if checkpoint.is_file():
            processor, digest = checkpoint_processor(checkpoint)
            return DenoisePipeline(processor, checkpoint_sha256=digest)
    raise FileNotFoundError("No checkpoint found; set NOISE_REDUCE_CHECKPOINT or install model assets")


def _call_pipeline(pipeline: Any, input_path: Path, output_path: Path, cancel: threading.Event, progress) -> Any:
    """Call supported pipeline variants without duplicating model logic."""
    kwargs = {"cancel_event": cancel, "progress_callback": progress}
    if hasattr(pipeline, "process_file"):
        fn = pipeline.process_file
    elif hasattr(pipeline, "process"):
        fn = pipeline.process
    else:
        raise TypeError("DesktopPipeline must expose process_file() or process()")
    for candidate in (kwargs, {"cancel_event": cancel}, {}):
        try:
            return fn(input_path, output_path, **candidate)
        except TypeError as exc:
            if candidate == {}:
                raise
            # Older pipeline implementations may not accept optional callbacks.
            if "unexpected keyword" not in str(exc) and "positional" not in str(exc):
                raise
    raise AssertionError("unreachable")


if QT_AVAILABLE:

    class WorkerSignals(QObject):
        progress = Signal(int, str)
        file_done = Signal(str, object)
        file_error = Signal(str, str)
        finished = Signal()


    class ProcessWorker(QRunnable):
        """Batch worker running outside the GUI thread."""

        def __init__(self, inputs: Iterable[Path], output_dir: Path, pipeline: Any | None = None) -> None:
            super().__init__()
            self.inputs = list(inputs)
            self.output_dir = output_dir
            self.pipeline = pipeline
            self.cancel_event = threading.Event()
            self.signals = WorkerSignals()

        def cancel(self) -> None:
            self.cancel_event.set()

        @Slot()
        def run(self) -> None:
            try:
                pipeline = self.pipeline or _default_pipeline()
                total = max(len(self.inputs), 1)
                for index, source in enumerate(self.inputs):
                    if self.cancel_event.is_set():
                        break
                    destination = self.output_dir / f"{source.stem}_denoised.wav"

                    def update(value: float = 0.0, message: str = "") -> None:
                        # Pipeline callbacks may report either [0,1] or [0,100].
                        fraction = float(value)
                        if fraction <= 1.0:
                            fraction *= 100.0
                        overall = int(((index + max(0.0, min(fraction, 100.0)) / 100.0) / total) * 100)
                        self.signals.progress.emit(overall, message or source.name)

                    try:
                        receipt = _call_pipeline(pipeline, source, destination, self.cancel_event, update)
                        self.signals.file_done.emit(str(source), receipt)
                    except Exception as exc:  # isolate one bad file from the batch
                        self.signals.file_error.emit(str(source), f"{type(exc).__name__}: {exc}")
                    self.signals.progress.emit(int(((index + 1) / total) * 100), source.name)
            finally:
                self.signals.finished.emit()


    class DropList(QListWidget):
        paths_dropped = Signal(list)

        def __init__(self) -> None:
            super().__init__()
            self.setAcceptDrops(True)

        def dragEnterEvent(self, event: QDragEnterEvent) -> None:
            if event.mimeData().hasUrls():
                event.acceptProposedAction()
            else:
                event.ignore()

        def dropEvent(self, event: QDropEvent) -> None:
            paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
            self.paths_dropped.emit(paths)
            event.acceptProposedAction()


    class MainWindow(QMainWindow):
        """WAV/batch denoising window with non-blocking processing."""

        def __init__(self, pipeline: Any | None = None) -> None:
            super().__init__()
            self.setWindowTitle("ClearHop")
            self.resize(760, 520)
            self.pipeline = pipeline
            self.inputs: list[Path] = []
            self.output_dir = Path.cwd() / "outputs"
            self.worker: ProcessWorker | None = None
            self.thread_pool = QThreadPool.globalInstance()
            self._build_ui()

        def _build_ui(self) -> None:
            root = QWidget(self)
            root.setObjectName("root")
            layout = QVBoxLayout(root)
            layout.setContentsMargins(24, 22, 24, 20)
            layout.setSpacing(12)
            intro = QLabel("Local-only WAV denoising (mono / 16 kHz conversion is reported)")
            intro.setObjectName("intro")
            intro.setWordWrap(True)
            layout.addWidget(intro)
            controls = QHBoxLayout()
            self.open_button = QPushButton("Open WAV")
            self.folder_button = QPushButton("Open folder")
            self.process_button = QPushButton("Denoise")
            self.cancel_button = QPushButton("Cancel")
            self.cancel_button.setEnabled(False)
            for button in (self.open_button, self.folder_button, self.process_button, self.cancel_button):
                controls.addWidget(button)
            layout.addLayout(controls)
            self.file_list = DropList()
            self.file_list.setObjectName("fileList")
            self.file_list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
            layout.addWidget(self.file_list)
            self.status = QLabel("Drop WAV files here or choose Open WAV.")
            self.status.setObjectName("status")
            self.progress = QProgressBar()
            self.progress.setRange(0, 100)
            layout.addWidget(self.status)
            layout.addWidget(self.progress)
            playback = QHBoxLayout()
            self.before_button = QPushButton("Play before")
            self.after_button = QPushButton("Play after")
            self.before_button.setEnabled(False)
            self.after_button.setEnabled(False)
            playback.addWidget(self.before_button)
            playback.addWidget(self.after_button)
            layout.addLayout(playback)
            self.setCentralWidget(root)
            self.setStyleSheet(
                """
                #root { background: #0d1721; color: #e8f0f5; }
                QLabel#intro { color: #8ed7c1; font-size: 18px; font-weight: 600; padding-bottom: 4px; }
                QLabel#status { color: #a7b7c4; padding-top: 2px; }
                QListWidget#fileList { background: #142431; border: 1px solid #2b4b5d; border-radius: 8px; padding: 8px; }
                QListWidget#fileList::item { padding: 7px; }
                QListWidget#fileList::item:selected { background: #1d6f73; border-radius: 4px; }
                QPushButton { background: #193746; color: #e8f0f5; border: 1px solid #356477; border-radius: 6px; padding: 9px 14px; }
                QPushButton:hover { background: #245365; }
                QPushButton:pressed { background: #2b7f7f; }
                QPushButton:disabled { color: #6b7c86; background: #17242d; }
                QProgressBar { background: #142431; border: 1px solid #2b4b5d; border-radius: 5px; text-align: center; color: #e8f0f5; }
                QProgressBar::chunk { background: #56c596; border-radius: 4px; }
                """
            )
            self.open_button.clicked.connect(self.open_wav)
            self.folder_button.clicked.connect(self.open_folder)
            self.process_button.clicked.connect(self.start_processing)
            self.cancel_button.clicked.connect(self.cancel_processing)
            self.file_list.paths_dropped.connect(self.add_paths)
            self.before_button.clicked.connect(lambda: self._play_selected(False))
            self.after_button.clicked.connect(lambda: self._play_selected(True))

        @Slot()
        def open_wav(self) -> None:
            names, _ = QFileDialog.getOpenFileNames(self, "Open WAV", "", "WAV files (*.wav)")
            self.add_paths([Path(name) for name in names])

        @Slot()
        def open_folder(self) -> None:
            folder = QFileDialog.getExistingDirectory(self, "Open folder")
            if folder:
                self.add_paths(sorted(Path(folder).glob("*.wav")))

        @Slot(list)
        def add_paths(self, paths: list[Path]) -> None:
            valid = [p for p in paths if p.is_file() and p.suffix.lower() == ".wav"]
            existing = set(self.inputs)
            for path in valid:
                if path not in existing:
                    self.inputs.append(path)
                    self.file_list.addItem(QListWidgetItem(str(path)))
            self.status.setText(f"{len(self.inputs)} file(s) selected")
            enabled = bool(self.inputs) and self.worker is None
            self.process_button.setEnabled(enabled)
            self.before_button.setEnabled(bool(self.inputs))

        @Slot()
        def start_processing(self) -> None:
            if not self.inputs or self.worker is not None:
                return
            output = QFileDialog.getExistingDirectory(self, "Output folder", str(self.output_dir))
            if not output:
                return
            self.output_dir = Path(output)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.progress.setValue(0)
            self.worker = ProcessWorker(self.inputs, self.output_dir, self.pipeline)
            self.worker.signals.progress.connect(lambda value, msg: (self.progress.setValue(value), self.status.setText(msg)))
            self.worker.signals.file_done.connect(self._file_done)
            self.worker.signals.file_error.connect(self._file_error)
            self.worker.signals.finished.connect(self._finished)
            self.process_button.setEnabled(False)
            self.cancel_button.setEnabled(True)
            self.thread_pool.start(self.worker)

        @Slot()
        def cancel_processing(self) -> None:
            if self.worker:
                self.worker.cancel()
                self.status.setText("Cancelling…")

        @Slot(str, object)
        def _file_done(self, source: str, receipt: Any) -> None:
            output = self.output_dir / f"{Path(source).stem}_denoised.wav"
            if isinstance(receipt, dict):
                receipt_path = output.with_suffix(".json")
                receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            self.after_button.setEnabled(output.exists())
            self.status.setText(f"Done: {Path(source).name}")

        @Slot(str, str)
        def _file_error(self, source: str, message: str) -> None:
            self.status.setText(f"Error: {Path(source).name} — {message}")

        @Slot()
        def _finished(self) -> None:
            self.worker = None
            self.cancel_button.setEnabled(False)
            self.process_button.setEnabled(bool(self.inputs))
            if self.progress.value() >= 100:
                self.status.setText("Batch complete")

        def _play_selected(self, after: bool) -> None:
            """Use the OS-associated player; avoids blocking Qt multimedia setup."""
            row = self.file_list.currentRow()
            if row < 0 or row >= len(self.inputs):
                return
            path = self.output_dir / f"{self.inputs[row].stem}_denoised.wav" if after else self.inputs[row]
            if path.exists():
                os.startfile(str(path))  # type: ignore[attr-defined]

        def closeEvent(self, event) -> None:  # noqa: N802
            if self.worker:
                self.worker.cancel()
            event.accept()


def main(argv: list[str] | None = None) -> int:
    if not QT_AVAILABLE:
        print("PySide6 is required. Install with: pip install '.[desktop]'", file=sys.stderr)
        return 2
    parser = argparse.ArgumentParser(description="ClearHop desktop app")
    parser.add_argument("--smoke-input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--smoke-receipt", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    smoke_values = (args.smoke_input, args.smoke_output, args.smoke_receipt)
    if any(smoke_values):
        if not all(smoke_values):
            parser.error("all packaged-smoke paths are required")
        try:
            receipt = _default_pipeline().process_file(
                args.smoke_input,
                args.smoke_output,
                receipt_path=args.smoke_receipt,
            )
            return 0 if receipt.get("status") == "pass" and args.smoke_output.is_file() and args.smoke_receipt.is_file() else 1
        except Exception as exc:
            print(f"Packaged smoke failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
