import json
import tempfile
import unittest
from pathlib import Path

from scripts.research_slices import load_research_slices, resolve_research_slice


class TestResearchSlices(unittest.TestCase):
    def test_loads_unique_nonoverlapping_slices(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "slices.json"
            path.write_text(json.dumps({"schema_version": 1, "slices": [{"name": "a", "offset": 0, "count": 2}, {"name": "b", "offset": 3, "count": 2}]}), encoding="utf-8")
            self.assertEqual(resolve_research_slice(path, "b", manifest_length=5)["offset"], 3)

    def test_rejects_overlap_and_duplicate_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "slices.json"
            path.write_text(json.dumps({"schema_version": 1, "slices": [{"name": "a", "offset": 0, "count": 2}, {"name": "a", "offset": 2, "count": 2}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_research_slices(path)

    def test_project_spec_is_valid(self) -> None:
        root = Path(__file__).resolve().parents[1]
        self.assertGreaterEqual(len(load_research_slices(root / "configs/evaluation/research_slices.json")), 5)


if __name__ == "__main__":
    unittest.main()
