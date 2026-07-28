from pathlib import Path

import desktop.app as app_module


def test_smoke_cli_processes_and_writes_receipt(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "in.wav"
    output = tmp_path / "out.wav"
    receipt = tmp_path / "receipt.json"
    source.write_bytes(b"fixture")

    class Pipeline:
        def process_file(self, input_path, output_path, *, receipt_path):
            assert Path(input_path) == source
            Path(output_path).write_bytes(b"wav")
            Path(receipt_path).write_text("{}", encoding="utf-8")
            return {"status": "pass"}

    monkeypatch.setattr(app_module, "QT_AVAILABLE", True)
    monkeypatch.setattr(app_module, "_default_pipeline", lambda: Pipeline())
    assert app_module.main(["--smoke-input", str(source), "--smoke-output", str(output), "--smoke-receipt", str(receipt)]) == 0


def test_installed_venv_finds_assets_beside_venv(monkeypatch, tmp_path: Path) -> None:
    install_root = tmp_path / "ClearHop"
    checkpoint = install_root / "assets" / "checkpoint.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    venv = install_root / "venv"
    monkeypatch.setattr(app_module.sys, "prefix", str(venv))
    monkeypatch.setattr(app_module.sys, "executable", str(venv / "Scripts" / "python.exe"))
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)

    import desktop.pipeline as pipeline_module

    sentinel = object()
    monkeypatch.setattr(pipeline_module, "checkpoint_processor", lambda path: (lambda samples: samples, "a" * 64))
    monkeypatch.setattr(pipeline_module, "DenoisePipeline", lambda *args, **kwargs: sentinel)
    assert app_module._default_pipeline() is sentinel
