import unittest

try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest("torch is required for neural runtime tests")

from src.dataset import collate_audio_batch
from src.trainer import _compute_objective, _resume_position, _selection_value


class TestTrainingContracts(unittest.TestCase):
    def test_resume_position_preserves_exact_global_step(self) -> None:
        self.assertEqual(_resume_position(1000, 200), (5, 0))
        self.assertEqual(_resume_position(1500, 200), (7, 100))

    def test_collate_keeps_metadata_out_of_default_tensor_collation(self) -> None:
        batch = [
            {"clean": torch.zeros(4), "noisy": torch.ones(4), "meta": {"snr_db": 0}},
            {"clean": torch.zeros(2), "noisy": torch.ones(2), "meta": {"snr_db": 5}},
        ]

        result = collate_audio_batch(batch)

        self.assertEqual(tuple(result["clean"].shape), (2, 4))
        self.assertEqual(len(result["meta"]), 2)
        self.assertEqual(result["meta"][1]["snr_db"], 5)

    def test_objective_exposes_all_component_losses(self) -> None:
        y_ri = torch.full((1, 2, 2, 2), 0.5, requires_grad=True)
        clean_ri = torch.ones_like(y_ri)
        enhanced = torch.tensor([[0.25, -0.25, 0.5, -0.5]], requires_grad=True)
        clean = torch.tensor([[1.0, -1.0, 1.0, -1.0]])
        cfg = {
            "loss": "complex_nmse_sisdr",
            "loss_eps": 1e-8,
            "alpha_loss": 1.0,
            "beta_si_sdr": 0.5,
            "sisdr_warmup_start": 0,
            "sisdr_warmup_end": 10,
            "compression_exponent": 0.3,
            "compression_complex_weight": 0.3,
        }

        result = _compute_objective(cfg, y_ri, clean_ri, enhanced, clean, global_step=5)

        self.assertEqual(
            set(result),
            {"total", "complex_mse", "complex_nmse", "compressed_complex", "si_sdr", "si_sdr_weight"},
        )
        self.assertAlmostEqual(float(result["si_sdr_weight"]), 0.25)
        self.assertTrue(result["total"].requires_grad)

    def test_selection_value_routes_validation_metric(self) -> None:
        self.assertEqual(_selection_value("si_sdr_improvement", 1.5, 7.0), 1.5)
        self.assertEqual(_selection_value("snr_improvement", 1.5, 7.0), 7.0)


if __name__ == "__main__":
    unittest.main()
