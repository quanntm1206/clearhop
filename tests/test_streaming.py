import unittest

try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest("torch is required for neural runtime tests")

from src.inference import StreamingEnhancer, enhance_offline
from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig


class TestStreamingEnhancer(unittest.TestCase):
    def test_push_advances_one_frame_and_returns_one_hop(self) -> None:
        model = MobileDeepFilterNet(
            MobileDeepFilterNetConfig(
                enc_channels=4,
                num_encoder_blocks=1,
                gru_hidden=4,
                k_tap=1,
            )
        ).eval()
        enhancer = StreamingEnhancer(model)
        hop = torch.zeros(160)

        output = enhancer.push(hop)

        self.assertEqual(tuple(output.shape), (160,))
        self.assertEqual(enhancer.num_frames, 1)
        self.assertEqual(enhancer.history_length, 1)

    def test_offline_and_streaming_match_after_causal_latency(self) -> None:
        torch.manual_seed(4)
        model = MobileDeepFilterNet(
            MobileDeepFilterNetConfig(
                enc_channels=4,
                num_encoder_blocks=1,
                gru_hidden=4,
                k_tap=2,
            )
        ).eval()
        waveform = torch.randn(1, 640) * 0.05
        offline = enhance_offline(model, waveform)
        enhancer = StreamingEnhancer(model)
        chunks = [enhancer.push(chunk) for chunk in waveform[0].split(160)]
        chunks.append(enhancer.flush())
        streamed_with_latency = torch.cat(chunks)
        streamed = streamed_with_latency[160:800]

        self.assertTrue(
            torch.allclose(offline[0], streamed, atol=1e-4, rtol=1e-4),
            msg=f"max parity error: {float((offline[0] - streamed).abs().max())}",
        )


if __name__ == "__main__":
    unittest.main()
