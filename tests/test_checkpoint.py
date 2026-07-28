import tempfile
import unittest
from pathlib import Path

from src.checkpoint import file_sha256, manifest_fingerprints, validate_checkpoint_metadata


class TestCheckpointContract(unittest.TestCase):
    def test_metadata_requires_schema_and_audio_contract(self) -> None:
        base = {
            "schema_version": 2,
            "model": {},
            "model_cfg": {},
            "audio_cfg": {"sr": 16000, "n_fft": 320, "hop": 160, "freq_bins": 161},
            "config": {},
        }
        validate_checkpoint_metadata(base)
        invalid = dict(base)
        invalid["audio_cfg"] = dict(base["audio_cfg"], hop=80)
        with self.assertRaisesRegex(ValueError, "audio contract"):
            validate_checkpoint_metadata(invalid)

    def test_manifest_fingerprint_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "train.jsonl"
            path.write_text('{"id": 1}\n', encoding="utf-8")
            first = manifest_fingerprints({"train": path})
            second = manifest_fingerprints({"train": path})
            self.assertEqual(first, second)
            self.assertEqual(len(first["train"]), 64)

    def test_file_sha256_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.bin"
            path.write_bytes(b"artifact")
            digest = file_sha256(path)
            self.assertEqual(len(digest), 64)
            self.assertEqual(digest, file_sha256(path))


if __name__ == "__main__":
    unittest.main()
