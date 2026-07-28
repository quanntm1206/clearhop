import unittest

import torch

from src.inference import StreamingEnhancer
from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig


class TestCpuSoakContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.model = MobileDeepFilterNet(MobileDeepFilterNetConfig(enc_channels=2, num_encoder_blocks=1, gru_hidden=2, k_tap=1)).eval()
        self.enhancer = StreamingEnhancer(self.model)

    def test_invalid_inputs_do_not_advance_state(self) -> None:
        for bad in (torch.empty(0), torch.zeros(159), torch.full((160,), float("nan"))):
            with self.subTest(shape=tuple(bad.shape)):
                with self.assertRaises(ValueError):
                    self.enhancer.push(bad)
                self.assertEqual(self.enhancer.num_frames, 0)

    def test_reset_is_idempotent(self) -> None:
        self.enhancer.push(torch.zeros(160))
        self.enhancer.reset()
        self.enhancer.reset()
        self.assertEqual(self.enhancer.num_frames, 0)
        self.assertEqual(self.enhancer.history_length, 0)


if __name__ == "__main__":
    unittest.main()
