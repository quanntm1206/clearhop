import tempfile
import unittest
from pathlib import Path


class TestProjectConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.root_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.root_dir.cleanup)
        self.root = Path(self.root_dir.name)
        self.required_paths = {
            "clean_root": "data/clean",
            "noise_root": "data/noise",
        }

    def test_audio_contract_rejects_incompatible_fft(self) -> None:
        from src.config import AudioConfig

        with self.assertRaisesRegex(ValueError, "n_fft=320"):
            AudioConfig(n_fft=512)

    def test_relative_data_roots_resolve_from_project(self) -> None:
        from src.config import load_train_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data" / "clean").mkdir(parents=True)
            (root / "data" / "noise").mkdir(parents=True)
            config_path = root / "train.yaml"
            config_path.write_text(
                "clean_root: data/clean\n"
                "noise_root: data/noise\n"
                "sr: 16000\n"
                "segment_len: 4.0\n",
                encoding="utf-8",
            )

            cfg = load_train_config(config_path, project_root=root)

            self.assertEqual(cfg.clean_root, (root / "data" / "clean").resolve())
            self.assertEqual(cfg.noise_root, (root / "data" / "noise").resolve())
            cfg.validate_data_paths()

    def test_unknown_config_key_fails_fast(self) -> None:
        from src.config import TrainConfig

        with self.assertRaisesRegex(ValueError, "Unknown config keys: mystery"):
            TrainConfig.from_mapping({"mystery": 1}, project_root=Path.cwd())

    def test_missing_data_path_reports_exact_root(self) -> None:
        from src.config import TrainConfig

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = TrainConfig.from_mapping(
                {"clean_root": "missing-clean", "noise_root": "missing-noise"},
                project_root=root,
            )

            with self.assertRaisesRegex(FileNotFoundError, "missing-clean"):
                cfg.validate_data_paths()

    def test_optional_training_metric_flags_parse(self) -> None:
        from src.config import TrainConfig

        cfg = TrainConfig.from_mapping(
            {"compute_stoi": True, "compute_pesq": True, "metric_max_items": 3},
            project_root=Path.cwd(),
        )
        self.assertTrue(cfg.compute_stoi)
        self.assertTrue(cfg.compute_pesq)
        self.assertEqual(cfg.metric_max_items, 3)

    def test_gain_calibration_fields_round_trip_and_resolve_checkpoint_dir(self) -> None:
        from src.config import TrainConfig

        values = {
            **self.required_paths,
            "loss": "compressed_complex",
            "loss_eps": 1.0e-7,
            "sisdr_warmup_start": 100,
            "sisdr_warmup_end": 500,
            "compression_exponent": 0.25,
            "compression_complex_weight": 0.4,
            "checkpoint_dir": "checkpoints/gain_calibration",
            "max_steps": 1500,
            "scheduler_total_steps": 30000,
            "selection_metric": "si_sdr_improvement",
            "experiment_id": "compressed_complex_seed17",
        }

        cfg = TrainConfig.from_mapping(values, project_root=self.root)
        result = cfg.to_dict()

        self.assertEqual(cfg.checkpoint_dir, (self.root / "checkpoints/gain_calibration").resolve())
        for key in (
            "loss", "loss_eps", "sisdr_warmup_start", "sisdr_warmup_end",
            "compression_exponent", "compression_complex_weight", "checkpoint_dir",
            "max_steps", "scheduler_total_steps", "selection_metric", "experiment_id",
        ):
            value = values[key]
            expected = str(cfg.checkpoint_dir) if key == "checkpoint_dir" else value
            self.assertEqual(result[key], expected)

    def test_gain_calibration_validation_rejects_invalid_values(self) -> None:
        from src.config import TrainConfig

        invalid = [
            {"loss": "unknown"},
            {"loss_eps": 0.0},
            {"beta_si_sdr": -0.1},
            {"sisdr_warmup_start": 10, "sisdr_warmup_end": 9},
            {"compression_exponent": 0.0},
            {"compression_exponent": 1.1},
            {"compression_complex_weight": -0.1},
            {"compression_complex_weight": 1.1},
            {"max_steps": 0},
            {"scheduler_total_steps": 0},
            {"selection_metric": "loss"},
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                TrainConfig.from_mapping(
                    {**self.required_paths, **values}, project_root=self.root
                )

    def test_gain_calibration_validation_rejects_non_finite_values(self) -> None:
        from src.config import TrainConfig

        invalid = [
            {"loss_eps": float("nan")},
            {"loss_eps": float("inf")},
            {"loss_eps": float("-inf")},
            {"alpha_loss": float("nan")},
            {"alpha_loss": float("inf")},
            {"alpha_loss": float("-inf")},
            {"beta_si_sdr": float("nan")},
            {"beta_si_sdr": float("inf")},
            {"beta_si_sdr": float("-inf")},
            {"compression_exponent": float("nan")},
            {"compression_exponent": float("inf")},
            {"compression_exponent": float("-inf")},
            {"compression_complex_weight": float("nan")},
            {"compression_complex_weight": float("inf")},
            {"compression_complex_weight": float("-inf")},
            {"snr_range": [float("nan"), 1.0]},
            {"snr_range": [0.0, float("inf")]},
            {"snr_range": [float("-inf"), 1.0]},
        ]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                TrainConfig.from_mapping(
                    {**self.required_paths, **values}, project_root=self.root
                )


if __name__ == "__main__":
    unittest.main()
