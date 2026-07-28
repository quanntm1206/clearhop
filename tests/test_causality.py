import unittest

try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest("torch is required for neural runtime tests")

from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig


class TestCausalModel(unittest.TestCase):
    def test_future_frames_do_not_change_past_outputs(self) -> None:
        torch.manual_seed(0)
        model = MobileDeepFilterNet(
            MobileDeepFilterNetConfig(
                enc_channels=4,
                num_encoder_blocks=1,
                gru_hidden=4,
                k_tap=2,
            )
        ).eval()
        first = torch.randn(1, 1, 161, 8)
        second = first.clone()
        second[:, :, :, 5:] += 10.0

        with torch.no_grad():
            mask_a, taps_a, _ = model(first)
            mask_b, taps_b, _ = model(second)

        self.assertTrue(torch.allclose(mask_a[..., :5], mask_b[..., :5], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(taps_a[..., :5, :], taps_b[..., :5, :], atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
