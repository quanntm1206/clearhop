import unittest
import tempfile
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    raise unittest.SkipTest("torch is required for neural runtime tests")

from src.export import StatefulExportWrapper, export_model
from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig


class TestStatefulExport(unittest.TestCase):
    def test_wrapper_matches_streaming_newest_frame(self) -> None:
        model = MobileDeepFilterNet(
            MobileDeepFilterNetConfig(enc_channels=4, num_encoder_blocks=1, gru_hidden=4, k_tap=1)
        ).eval()
        wrapper = StatefulExportWrapper(model)
        features = torch.randn(1, 1, 161, model.temporal_receptive_frames)
        hidden = torch.randn(1, 1, 4)

        with torch.no_grad():
            expected = model.forward_streaming(features, hidden)
            actual = wrapper(features, hidden)

        for eager, exported in zip(expected, actual):
            self.assertEqual(tuple(exported.shape), tuple(eager.shape))
            self.assertTrue(torch.allclose(exported, eager, atol=1e-6, rtol=1e-6))

    def test_wrapper_returns_next_hidden_state(self) -> None:
        model = MobileDeepFilterNet(
            MobileDeepFilterNetConfig(enc_channels=4, num_encoder_blocks=1, gru_hidden=4, k_tap=1)
        ).eval()
        wrapper = StatefulExportWrapper(model)
        features = torch.zeros(1, 1, 161, model.temporal_receptive_frames)
        hidden = torch.zeros(1, 1, 4)

        mask, taps, hidden_new = wrapper(features, hidden)

        self.assertEqual(tuple(mask.shape), (1, 161))
        self.assertEqual(tuple(taps.shape), (1, 161, 1, 2))
        self.assertEqual(tuple(hidden_new.shape), (1, 1, 4))

    def test_onnx_parity_when_runtime_is_installed(self) -> None:
        try:
            import onnxruntime as ort
        except ModuleNotFoundError:
            self.skipTest("onnxruntime is optional")
        model = MobileDeepFilterNet(
            MobileDeepFilterNetConfig(enc_channels=2, num_encoder_blocks=1, gru_hidden=2, k_tap=1)
        ).eval()
        with tempfile.TemporaryDirectory() as tmp:
            metadata = export_model(model, Path(tmp) / "core", export_onnx=True)
            if metadata["onnx"] is None:
                self.fail("ONNX export did not produce an artifact")
            features = torch.randn(1, 1, 161, model.temporal_receptive_frames)
            hidden = torch.randn(1, 1, 2)
            with torch.no_grad():
                eager = StatefulExportWrapper(model)(features, hidden)
            session = ort.InferenceSession(str(Path(tmp) / "core.onnx"), providers=["CPUExecutionProvider"])
            actual = session.run(
                None,
                {"feats_logp": features.numpy(), "hidden": hidden.numpy()},
            )
            for expected, observed in zip(eager, actual):
                self.assertTrue(torch.allclose(expected, torch.from_numpy(observed), atol=2e-4, rtol=2e-4))

    def test_torchscript_parity(self) -> None:
        model = MobileDeepFilterNet(
            MobileDeepFilterNetConfig(enc_channels=2, num_encoder_blocks=1, gru_hidden=2, k_tap=1)
        ).eval()
        with tempfile.TemporaryDirectory() as tmp:
            export_model(model, Path(tmp) / "core", export_onnx=False)
            scripted = torch.jit.load(str(Path(tmp) / "core.ts")).eval()
            features = torch.randn(1, 1, 161, model.temporal_receptive_frames)
            hidden = torch.randn(1, 1, 2)
            with torch.no_grad():
                eager = StatefulExportWrapper(model)(features, hidden)
                actual = scripted(features, hidden)
            for expected, observed in zip(eager, actual):
                self.assertTrue(torch.allclose(expected, observed, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
