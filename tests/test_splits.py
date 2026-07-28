import json
import hashlib
import tempfile
import unittest
from pathlib import Path


class TestSpeakerAwareSplits(unittest.TestCase):
    def test_known_corpora_speaker_ids(self) -> None:
        from src.splits import infer_speaker_id

        self.assertEqual(infer_speaker_id(Path("100-121669-0001.wav")), "libri:100")
        self.assertEqual(infer_speaker_id(Path("VIVOSSPK28_103.wav")), "vivos:VIVOSSPK28")
        self.assertEqual(infer_speaker_id(Path("vctk_p282_p282_100.wav")), "vctk:p282")

    def test_grouped_folds_have_no_speaker_overlap(self) -> None:
        from src.splits import build_grouped_splits, infer_speaker_id

        paths = [Path(f"{speaker}-10-{utterance:04d}.wav") for speaker in range(10, 20) for utterance in range(3)]
        folds = build_grouped_splits(paths, n_folds=5, val_fraction=0.2, seed=42)

        self.assertEqual(len(folds), 5)
        all_test = []
        for fold in folds:
            train_groups = {infer_speaker_id(p) for p in fold.train}
            val_groups = {infer_speaker_id(p) for p in fold.val}
            test_groups = {infer_speaker_id(p) for p in fold.test}
            self.assertFalse(train_groups & val_groups)
            self.assertFalse(train_groups & test_groups)
            self.assertFalse(val_groups & test_groups)
            all_test.extend(fold.test)
        self.assertCountEqual(all_test, paths)

    def test_manifest_paths_are_relative_and_fingerprint_is_stable(self) -> None:
        from src.splits import manifest_fingerprint, write_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            clean = root / "data" / "clean" / "speaker.wav"
            noise = root / "data" / "noise" / "room.wav"
            clean.parent.mkdir(parents=True)
            noise.parent.mkdir(parents=True)
            clean.touch()
            noise.touch()
            entries = [{"clean_path": clean, "noise_paths": [noise], "seed": 3, "segment_len": 4.0}]
            out = root / "manifests" / "test.jsonl"

            first = write_manifest(entries, out, project_root=root)
            second = manifest_fingerprint(out)
            row = json.loads(out.read_text(encoding="utf-8"))

            self.assertEqual(first, second)
            self.assertEqual(row["clean_path"], "data/clean/speaker.wav")
            self.assertEqual(row["noise_paths"], ["data/noise/room.wav"])

    def test_manifest_mixtures_are_deterministic(self) -> None:
        from src.splits import build_manifest_entries

        clean = [Path(f"clean-{index}.wav") for index in range(4)]
        noise = [Path(f"noise-{index}.wav") for index in range(3)]

        first = build_manifest_entries(clean, noise, seed=8, segment_len=4.0)
        second = build_manifest_entries(clean, noise, seed=8, segment_len=4.0)

        self.assertEqual(first, second)
        self.assertEqual([row["clean_path"] for row in first], clean)
        self.assertTrue(all(1 <= len(row["noise_paths"]) <= 3 for row in first))

    def test_manifest_slice_fingerprint_is_stable_and_offset_sensitive(self) -> None:
        from src.splits import manifest_slice_fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "test.jsonl"
            manifest.write_text('{"id": 0}\n{"id": 1}\n{"id": 2}\n', encoding="utf-8")

            first = manifest_slice_fingerprint(manifest, offset=0, count=2)
            repeated = manifest_slice_fingerprint(manifest, offset=0, count=2)
            shifted = manifest_slice_fingerprint(manifest, offset=1, count=2)

            self.assertEqual(first, repeated)
            self.assertNotEqual(first, shifted)
            with self.assertRaises(ValueError):
                manifest_slice_fingerprint(manifest, offset=2, count=2)
            with self.assertRaises(ValueError):
                manifest_slice_fingerprint(manifest, offset=0, count=0)

    def test_manifest_slice_fingerprint_uses_logical_nonblank_entries(self) -> None:
        from src.splits import manifest_slice_fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "test.jsonl"
            manifest.write_text(
                '  \n{"id": 0}\n\n\t\n{"id": 1}\n{"id": 2}\n',
                encoding="utf-8",
            )

            fingerprint = manifest_slice_fingerprint(manifest, offset=1, count=1)

            self.assertEqual(
                fingerprint,
                hashlib.sha256(b'{"id": 1}\n').hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
