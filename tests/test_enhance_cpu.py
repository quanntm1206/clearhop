import unittest

import torch

from scripts.enhance_cpu import enhance_waveform_cpu
from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig


class TestEnhanceCpu(unittest.TestCase):
    def test_output_is_finite_and_length_preserving(self) -> None:
        model = MobileDeepFilterNet(MobileDeepFilterNetConfig(enc_channels=2, num_encoder_blocks=1, gru_hidden=2, k_tap=1)).eval()
        waveform = torch.randn(321)
        output = enhance_waveform_cpu(model, waveform)
        self.assertEqual(output.shape, waveform.shape)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
