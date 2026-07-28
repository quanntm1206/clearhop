"""Render the machine-readable research gate as a concise Markdown report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def render_research_report(readiness: dict[str, object]) -> str:
    # `verify.py` stores the gate under a top-level result key; accept both
    # that receipt and the standalone gate object for reproducible reporting.
    wrapped = readiness.get("research_readiness")
    if "checks" not in readiness and isinstance(wrapped, dict):
        readiness = wrapped
    checks = readiness.get("checks", {})
    rows = ["# Research Readiness", "", f"Status: **{readiness.get('status', 'unknown')}**", "", "## Checks", ""]
    if isinstance(checks, dict):
        rows.extend(f"- `{name}`: {'pass' if value else 'fail'}" for name, value in sorted(checks.items()))
    reasons = readiness.get("reasons", [])
    if reasons:
        rows.extend(["", "## Limitations", "", *[f"- {reason}" for reason in reasons]])
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("reports/generated/research_readiness.json"))
    parser.add_argument("--output", type=Path, default=Path("docs/research-readiness.md"))
    args = parser.parse_args()
    readiness = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_research_report(readiness), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
