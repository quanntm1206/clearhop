import unittest
from pathlib import Path


class TestCpuSmokeEntrypoints(unittest.TestCase):
    def test_wrappers_expose_dry_run_and_checkpoint(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for name in ("scripts/run_cpu_smoke.ps1", "scripts/run_cpu_smoke.sh"):
            text = (root / name).read_text(encoding="utf-8")
            self.assertIn("checkpoint", text.lower())
            self.assertIn("dry", text.lower())


if __name__ == "__main__":
    unittest.main()
