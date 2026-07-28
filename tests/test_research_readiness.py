import unittest
from pathlib import Path

from scripts.run_gain_ablation import RESEARCH_SEEDS, build_plan
from scripts.robustness_matrix import summarize_robustness


class TestResearchReadinessPlan(unittest.TestCase):
    def test_research_plan_locks_five_seeds(self) -> None:
        root = Path(__file__).resolve().parents[1]
        plan = build_plan(root, "research", ["complex_nmse_sisdr_beta_0p02"], RESEARCH_SEEDS)
        self.assertEqual([row["seed"] for row in plan], list(RESEARCH_SEEDS))
        self.assertTrue(all(row["max_steps"] == 30000 for row in plan))

    def test_research_plan_rejects_missing_seed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with self.assertRaises(ValueError):
            build_plan(root, "research", ["complex_nmse_sisdr_beta_0p02"], RESEARCH_SEEDS[:-1])

    def test_robustness_summary_requires_exact_rows(self) -> None:
        report = summarize_robustness([{"name": "a", "offset": 0, "count": 500, "snri_delta": 1.0, "si_sdri_delta": 0.1, "stoi_delta": 0.0}])
        self.assertEqual(report["status"], "pass")


if __name__ == "__main__":
    unittest.main()
