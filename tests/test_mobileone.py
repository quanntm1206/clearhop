import unittest

try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest("torch is required for neural runtime tests")

from src.mobileone import MobileOneBlock, MobileOneConfig


class TestMobileOne(unittest.TestCase):
    def test_reparameterization_equivalence(self) -> None:
        torch.manual_seed(0)

        cfg = MobileOneConfig(
            in_channels=8,
            out_channels=8,
            stride=1,
            padding=1,
            num_conv_branches=3,
            use_scale_branch=True,
            use_identity_branch=True,
            activation="none",
        )
        block = MobileOneBlock(cfg, inference_mode=False).eval()

        x = torch.randn(2, 8, 32, 16)
        with torch.no_grad():
            y_train = block(x)
            block.reparameterize().eval()
            y_deploy = block(x)

        max_err = (y_train - y_deploy).abs().max().item()
        self.assertLess(max_err, 1e-5, msg=f"Max error too high: {max_err}")

    def test_causal_reparameterization_preserves_current_frame_alignment(self) -> None:
        torch.manual_seed(1)
        cfg = MobileOneConfig(
            in_channels=4,
            out_channels=4,
            stride=1,
            padding=1,
            num_conv_branches=2,
            use_scale_branch=True,
            use_identity_branch=True,
            activation="none",
            causal_time=True,
        )
        block = MobileOneBlock(cfg, inference_mode=False).eval()
        x = torch.randn(1, 4, 7, 9)
        with torch.no_grad():
            expected = block(x)
            actual = block.reparameterize()(x)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
