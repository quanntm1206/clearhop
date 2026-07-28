import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.augment import mix_clean_with_noises
from src.utils import compute_rms, snr_db


def _make_clean_sine(sr: int, seconds: float, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(sr * seconds), dtype=np.float32) / float(sr)
    x = 0.2 * np.sin(2.0 * np.pi * freq * t).astype(np.float32)
    # Add a small envelope to mimic speech dynamics.
    env = (0.6 + 0.4 * np.sin(2.0 * np.pi * 1.5 * t)).astype(np.float32)
    return (x * env).astype(np.float32)


def _write_temp_wav(path: Path, x: np.ndarray, sr: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), x, sr)


class TestMixing(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import soundfile  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("soundfile is required for on-disk mixing tests")
        self.sr = 16000
        self.seg_len = 4.0
        self.clean = _make_clean_sine(self.sr, self.seg_len)

        # Create 3 deterministic noise files on disk for mixing tests.
        rng = np.random.RandomState(0)
        self.tmp = Path(tempfile.mkdtemp(prefix="noise-reduce-mixing-"))
        self.noise_paths = []
        for i in range(3):
            n = (0.1 * rng.randn(self.clean.size)).astype(np.float32)
            p = self.tmp / f"noise_{i}.wav"
            _write_temp_wav(p, n, self.sr)
            self.noise_paths.append(str(p))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_target_snr_accuracy_single_noise(self) -> None:
        targets = [-10, -5, 0, 5, 10]
        for t_snr in targets:
            with (
                patch("src.augment._add_hum", side_effect=lambda x, sr, rng: (x, {"hum": None})),
                patch("src.augment._apply_device_effects", side_effect=lambda x, sr, rng: (x, {})),
                patch("src.augment.soft_clip", side_effect=lambda x, threshold: x),
            ):
                clean_target, noisy, meta = mix_clean_with_noises(
                    clean=self.clean,
                    noise_paths=self.noise_paths[:1],
                    sr=self.sr,
                    segment_len=self.seg_len,
                    snr_config={"seed": 123, "snr_db": float(t_snr), "project_root": str(Path.cwd())},
                    rir=None,
                    time_varying=False,
                    multi_noise=(1, 1),
                )
            actual = snr_db(clean_target, noisy)
            self.assertLessEqual(abs(actual - float(t_snr)), 0.5, msg=f"target={t_snr}, actual={actual}, meta={meta}")

    def test_multi_noise_rms_reasonable(self) -> None:
        clean_target, noisy, meta = mix_clean_with_noises(
            clean=self.clean,
            noise_paths=self.noise_paths,
            sr=self.sr,
            segment_len=self.seg_len,
            snr_config={"seed": 7, "snr_db": 0.0, "project_root": str(Path.cwd())},
            rir=None,
            time_varying=False,
            multi_noise=(3, 3),
        )
        # Mixed should have RMS larger than clean at 0 dB SNR, but not explode.
        rms_clean = compute_rms(clean_target)
        rms_noisy = compute_rms(noisy)
        self.assertGreater(rms_noisy, rms_clean)
        self.assertLess(rms_noisy, 5.0 * rms_clean)
        self.assertEqual(len(meta["noise_files"]), 3)

    def test_time_varying_snr_segments(self) -> None:
        # Force a deterministic 3-segment SNR pattern for testability.
        bounds = [0, int(self.sr * 1.5), int(self.sr * 3.0), int(self.sr * 4.0)]
        snr_list = [-5.0, 0.0, 5.0]
        with (
            patch("src.augment._add_hum", side_effect=lambda x, sr, rng: (x, {"hum": None})),
            patch("src.augment._apply_device_effects", side_effect=lambda x, sr, rng: (x, {})),
            patch("src.augment.soft_clip", side_effect=lambda x, threshold: x),
        ):
            clean_target, noisy, meta = mix_clean_with_noises(
                clean=self.clean,
                noise_paths=self.noise_paths[:1],
                sr=self.sr,
                segment_len=self.seg_len,
                snr_config={
                    "seed": 999,
                    "project_root": str(Path.cwd()),
                    "vary_boundaries": bounds,
                    "vary_snr_db_list": snr_list,
                },
                rir=None,
                time_varying=True,
                multi_noise=(1, 1),
            )
        info = meta["snr_db_segments"]
        self.assertIsNotNone(info)
        # Verify segment-wise SNR is close (±1 dB tolerance).
        for i, target in enumerate(snr_list):
            a, b = bounds[i], bounds[i + 1]
            actual = snr_db(clean_target[a:b], noisy[a:b])
            self.assertLessEqual(abs(actual - target), 1.0, msg=f"seg={i}, target={target}, actual={actual}, meta={meta}")

    def test_configured_snr_range_is_respected(self) -> None:
        noise = np.ones(self.clean.size, dtype=np.float32)
        with patch("src.augment.load_audio_mono", return_value=noise):
            _, _, meta = mix_clean_with_noises(
                clean=self.clean,
                noise_paths=["unused.wav"],
                sr=self.sr,
                segment_len=self.seg_len,
                snr_config={"seed": 9, "snr_range": [7.0, 7.0]},
                rir=None,
                time_varying=False,
                multi_noise=(1, 1),
            )
        self.assertEqual(meta["snr_db"], 7.0)


if __name__ == "__main__":
    unittest.main()
