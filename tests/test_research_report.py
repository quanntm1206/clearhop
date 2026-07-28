import unittest


class TestResearchReport(unittest.TestCase):
    def test_render_accepts_verifier_wrapper(self) -> None:
        from scripts.build_research_report import render_research_report

        report = render_research_report({"research_readiness": {"status": "pass", "checks": {"significance": True}}})
        self.assertIn("Status: **pass**", report)
        self.assertIn("`significance`: pass", report)


if __name__ == "__main__":
    unittest.main()
