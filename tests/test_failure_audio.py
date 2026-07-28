import unittest

from pathlib import Path

from scripts.export_failure_audio import resolve_from_root, selected_failure_ids


class TestFailureAudio(unittest.TestCase):
    def test_selects_locked_worst_case_order(self) -> None:
        payload = {"worst_cases": [{"index": 7}, {"index": 3}]}
        self.assertEqual(selected_failure_ids(payload, 2), [7, 3])

    def test_resolves_relative_output_against_project_root(self) -> None:
        root = Path.cwd().resolve()
        self.assertEqual(resolve_from_root(root, Path("artifacts/failure_audio")), root / "artifacts/failure_audio")


if __name__ == "__main__":
    unittest.main()
