import unittest

try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest("torch is required for neural runtime tests")

from src.filtering import causal_deep_filter, stft_analysis_ri, stft_synthesis


class TestFiltering(unittest.TestCase):
    def test_identity_tap_preserves_complex_stft(self) -> None:
        x = torch.randn(1, 161, 5, 2)
        mask = torch.ones(1, 161, 5)
        taps = torch.zeros(1, 161, 1, 5, 2)
        taps[..., 0] = 1.0

        y = causal_deep_filter(x, mask, taps)

        self.assertTrue(torch.allclose(x, y))

    def test_ola_identity_reconstructs_frame_sequence(self) -> None:
        torch.manual_seed(1)
        waveform = torch.randn(1, 640)
        window = torch.hann_window(320)
        frames = stft_analysis_ri(waveform, 320, 160, window)
        reconstructed = stft_synthesis(frames, 320, 160, window, length=640)

        max_error = float((waveform - reconstructed).abs().max())
        self.assertTrue(
            torch.allclose(waveform, reconstructed, atol=1e-4, rtol=1e-4),
            msg=f"max reconstruction error: {max_error}",
        )

    def test_causal_synthesis_has_finite_gradients(self) -> None:
        waveform = torch.randn(1, 640, requires_grad=True)
        window = torch.hann_window(320)
        frames = stft_analysis_ri(waveform, 320, 160, window)
        reconstructed = stft_synthesis(frames, 320, 160, window, length=640)
        reconstructed.square().mean().backward()
        self.assertTrue(torch.isfinite(waveform.grad).all())


if __name__ == "__main__":
    unittest.main()
