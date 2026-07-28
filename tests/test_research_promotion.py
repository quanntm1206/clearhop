import unittest

from scripts.promote_research_model import _atomic_copy


class TestResearchPromotion(unittest.TestCase):
    def test_atomic_copy_replaces_destination(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            destination = root / "destination"
            source.write_bytes(b"new")
            destination.write_bytes(b"old")
            _atomic_copy(source, destination)
            self.assertEqual(destination.read_bytes(), b"new")


if __name__ == "__main__":
    unittest.main()
