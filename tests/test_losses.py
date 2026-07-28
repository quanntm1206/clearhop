import unittest

try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest("torch is required for neural runtime tests")

from src.losses import (
    complex_mse,
    complex_nmse,
    compressed_complex_loss,
    scheduled_weight,
    si_sdr_loss,
)
from src.trainer import _complex_mse


class TestCalibrationLosses(unittest.TestCase):
    def setUp(self) -> None:
        self.target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

    def test_complex_nmse_exact_scale_values(self) -> None:
        self.assertAlmostEqual(complex_nmse(self.target, self.target).item(), 0.0, places=7)
        self.assertAlmostEqual(complex_nmse(0.5 * self.target, self.target).item(), 0.25, places=6)
        self.assertAlmostEqual(complex_nmse(2.0 * self.target, self.target).item(), 1.0, places=6)
        self.assertAlmostEqual(complex_nmse(-self.target, self.target).item(), 4.0, places=6)

    def test_complex_nmse_silence_and_mixed_batch(self) -> None:
        silence = torch.zeros_like(self.target)
        self.assertEqual(complex_nmse(silence, silence).item(), 0.0)
        estimate = torch.cat((self.target, 0.5 * self.target), dim=0)
        reference = torch.cat((self.target, self.target), dim=0)
        self.assertAlmostEqual(complex_nmse(estimate, reference).item(), 0.125, places=6)

    def test_complex_mse_identity_and_scale(self) -> None:
        self.assertEqual(complex_mse(self.target, self.target).item(), 0.0)
        self.assertAlmostEqual(complex_mse(2.0 * self.target, self.target).item(), 0.5, places=7)

    def test_compressed_loss_identity_and_gradients(self) -> None:
        estimate = (0.5 * self.target).clone().requires_grad_()
        self.assertEqual(compressed_complex_loss(self.target, self.target, 0.3, 0.5).item(), 0.0)
        loss = compressed_complex_loss(estimate, self.target, 0.3, 0.5)
        loss.backward()
        self.assertTrue(torch.isfinite(estimate.grad).all())

    def test_compressed_loss_scales_silence_and_mixed_batch(self) -> None:
        exponent = 0.3
        complex_weight = 0.5
        scale_loss = (2.0**exponent - 1.0) ** 2
        self.assertAlmostEqual(
            compressed_complex_loss(2.0 * self.target, self.target, exponent, complex_weight).item(),
            scale_loss,
            places=6,
        )
        self.assertAlmostEqual(
            compressed_complex_loss(-self.target, self.target, exponent, complex_weight).item(),
            4.0 * complex_weight,
            places=6,
        )
        silence = torch.zeros_like(self.target)
        self.assertEqual(compressed_complex_loss(silence, silence, exponent, complex_weight).item(), 0.0)
        estimate = torch.cat((self.target, 2.0 * self.target), dim=0)
        reference = torch.cat((self.target, self.target), dim=0)
        self.assertAlmostEqual(
            compressed_complex_loss(estimate, reference, exponent, complex_weight).item(),
            scale_loss / 2.0,
            places=6,
        )

    def test_si_sdr_identity_silence_and_gradients(self) -> None:
        signal = torch.tensor([[1.0, -1.0, 0.5, -0.5]])
        self.assertLess(si_sdr_loss(signal, signal).item(), -70.0)
        self.assertTrue(torch.isfinite(si_sdr_loss(torch.zeros_like(signal), torch.zeros_like(signal))))
        estimate = (0.5 * signal).clone().requires_grad_()
        si_sdr_loss(estimate, signal).backward()
        self.assertTrue(torch.isfinite(estimate.grad).all())

    def test_silent_losses_have_finite_gradients(self) -> None:
        for loss_fn in (
            lambda est, ref: compressed_complex_loss(est, ref, 0.3, 0.5),
            complex_nmse,
        ):
            estimate = torch.zeros_like(self.target, requires_grad=True)
            loss_fn(estimate, torch.zeros_like(self.target)).backward()
            self.assertTrue(torch.isfinite(estimate.grad).all())
        waveform = torch.zeros((1, 4), requires_grad=True)
        si_sdr_loss(waveform, torch.zeros_like(waveform)).backward()
        self.assertTrue(torch.isfinite(waveform.grad).all())

    def test_legacy_trainer_complex_mse_compatibility(self) -> None:
        estimate = torch.ones((2, 3, 2))
        reference = torch.zeros_like(estimate)
        self.assertEqual(_complex_mse(estimate, reference).item(), 1.0)
        self.assertEqual(_complex_mse(a_ri=estimate, b_ri=reference).item(), 1.0)

    def test_invalid_shapes_and_compression_values(self) -> None:
        with self.assertRaises(ValueError):
            complex_nmse(self.target[..., 0], self.target[..., 0])
        with self.assertRaises(ValueError):
            complex_mse(self.target, torch.zeros(2, 1, 2, 2))
        with self.assertRaises(ValueError):
            si_sdr_loss(torch.zeros(1, 2, 3), torch.zeros(1, 2, 3))
        with self.assertRaises(ValueError):
            compressed_complex_loss(self.target, self.target, 0.0, 0.5)
        with self.assertRaises(ValueError):
            compressed_complex_loss(self.target, self.target, 0.3, 1.1)

    def test_scheduled_weight(self) -> None:
        self.assertEqual(scheduled_weight(500, 0.01, 500, 1000), 0.0)
        self.assertAlmostEqual(scheduled_weight(750, 0.01, 500, 1000), 0.005)
        self.assertEqual(scheduled_weight(1000, 0.01, 500, 1000), 0.01)


if __name__ == "__main__":
    unittest.main()
