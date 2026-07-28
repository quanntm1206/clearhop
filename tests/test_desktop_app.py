from __future__ import annotations

import importlib


def test_desktop_module_imports_without_qt() -> None:
    module = importlib.import_module("desktop.app")
    assert hasattr(module, "main")


def test_desktop_main_reports_missing_qt_when_unavailable() -> None:
    module = importlib.import_module("desktop.app")
    if not module.QT_AVAILABLE:
        assert module.main([]) == 2
