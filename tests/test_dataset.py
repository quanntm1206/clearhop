import unittest
from unittest.mock import patch

import numpy as np

from src.augment import mix_clean_with_noises
from src.utils import snr_db


class TestPairedMixtureContract(unittest.TestCase):
    def test_processed_clean_target_preserves_requested_snr(self) -> None:
        sr = 16000
        samples = sr
        t = np.arange(samples, dtype=np.float32) / sr
        clean = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        noise = np.random.RandomState(1).normal(0, 0.1, samples).astype(np.float32)

        with (
            patch("src.augment.load_audio_mono", return_value=noise),
            patch("src.augment._maybe_time_or_pitch_perturb", side_effect=lambda x, sr, rng: (x, {})),
            patch("src.augment._add_hum", side_effect=lambda x, sr, rng: (x, {"hum": None})),
            patch("src.augment._apply_device_effects", side_effect=lambda x, sr, rng: (x, {})),
            patch("src.augment._butter_bandpass", side_effect=lambda x, sr, low_hz, high_hz: x),
            patch("src.augment.soft_clip", side_effect=lambda x, threshold: x),
        ):
            clean_target, noisy, _ = mix_clean_with_noises(
                clean=clean,
                noise_paths=["noise.wav"],
                sr=sr,
                segment_len=1.0,
                snr_config={"seed": 7, "snr_db": 0.0},
                rir=None,
                time_varying=False,
                multi_noise=(1, 1),
            )

        self.assertAlmostEqual(snr_db(clean_target, noisy), 0.0, delta=0.05)
        self.assertFalse(np.array_equal(clean_target, clean))

    def test_train_seed_changes_by_epoch_and_is_reproducible(self) -> None:
        from src.utils import derive_item_seed

        epoch_3_a = derive_item_seed(42, epoch=3, item_index=9, worker_id=0)
        epoch_3_b = derive_item_seed(42, epoch=3, item_index=9, worker_id=0)
        epoch_4 = derive_item_seed(42, epoch=4, item_index=9, worker_id=0)

        self.assertEqual(epoch_3_a, epoch_3_b)
        self.assertNotEqual(epoch_3_a, epoch_4)

    def test_snr_sampler_honors_configured_range(self) -> None:
        from src.augment import _sample_snr_db

        rng = np.random.RandomState(3)
        values = [_sample_snr_db(rng, {"snr_range": [4.0, 6.0]}, False) for _ in range(32)]
        self.assertTrue(all(4.0 <= value <= 6.0 for value in values))

    def test_noisy_only_soft_clip_does_not_change_clean_target(self) -> None:
        clean = np.linspace(-0.8, 0.8, 16000, dtype=np.float32)
        noise = np.random.RandomState(5).normal(0, 0.1, clean.size).astype(np.float32)
        with (
            patch("src.augment.load_audio_mono", return_value=noise),
            patch("src.augment._maybe_time_or_pitch_perturb", side_effect=lambda x, sr, rng: (x, {})),
            patch("src.augment._add_hum", side_effect=lambda x, sr, rng: (x, {"hum": None})),
            patch("src.augment._apply_device_effects", side_effect=lambda x, sr, rng: (x, {})),
            patch("src.augment._butter_bandpass", side_effect=lambda x, sr, low_hz, high_hz: x),
        ):
            for seed in range(100):
                clean_actual, _, meta = mix_clean_with_noises(
                    clean, ["noise.wav"], 16000, 1.0, {"seed": seed, "snr_db": 0.0}, multi_noise=(1, 1)
                )
                if "soft_clip_post" not in meta["aug"]:
                    continue
                with patch("src.augment.soft_clip", side_effect=lambda x, threshold: x):
                    clean_without_clip, _, _ = mix_clean_with_noises(
                        clean, ["noise.wav"], 16000, 1.0, {"seed": seed, "snr_db": 0.0}, multi_noise=(1, 1)
                    )
                self.assertTrue(np.allclose(clean_actual, clean_without_clip))
                return
        self.fail("No deterministic seed exercised post-mix soft clipping.")


if __name__ == "__main__":
    unittest.main()
