from __future__ import annotations

import sys
import types
import unittest

import numpy as np

from src.evaluate import _optional_pesq


class TestEvaluateMetrics(unittest.TestCase):
    def test_optional_pesq_uses_library_and_returns_float(self) -> None:
        fake = types.SimpleNamespace(pesq=lambda sr, clean, estimate, mode: 2.75)
        previous = sys.modules.get("pesq")
        sys.modules["pesq"] = fake  # type: ignore[assignment]
        try:
            value = _optional_pesq(np.zeros(1600), np.zeros(1600), 16000)
        finally:
            if previous is None:
                sys.modules.pop("pesq", None)
            else:
                sys.modules["pesq"] = previous
        self.assertEqual(value, 2.75)


if __name__ == "__main__":
    unittest.main()
