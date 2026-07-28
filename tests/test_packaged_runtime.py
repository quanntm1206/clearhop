"""Clean-install smoke tests for the packaged desktop inference runtime."""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path


def test_wheel_contains_runtime_and_can_import_outside_checkout(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", str(root), "--no-deps", "--wheel-dir", str(wheelhouse)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    wheel = next(wheelhouse.glob("*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert "src/cpu_runtime.py" in names
        assert "scripts/enhance_cpu.py" not in names

    target = tmp_path / "site"
    install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), str(wheel)],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )
    assert install.returncode == 0, install.stderr
    env = os.environ.copy()
    env["PYTHONPATH"] = str(target)
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import dataclasses,numpy as np,torch; "
                "from desktop.pipeline import checkpoint_processor; "
                "from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig; "
                "cfg=MobileDeepFilterNetConfig(enc_channels=2,num_encoder_blocks=1,gru_hidden=2,k_tap=1); "
                "m=MobileDeepFilterNet(cfg); ck='fixture.pth'; "
                "torch.save({'schema_version':2,'model':m.state_dict(),'model_cfg':dataclasses.asdict(cfg),"
                "'audio_cfg':{'sr':16000,'n_fft':320,'hop':160,'freq_bins':161},'config':{}},ck); "
                "process,digest=checkpoint_processor(ck); y=process(np.zeros(321,dtype=np.float32)); "
                "assert y.shape == (321,) and np.isfinite(y).all() and len(digest)==64; print('ok')"
            ),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "ok"
