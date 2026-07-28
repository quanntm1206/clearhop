from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_ci_installs_desktop_and_runs_fail_closed_gates() -> None:
    text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    for token in (
        ".[desktop,dev,export]", "QT_QPA_PLATFORM", "test_desktop_e2e.py",
        "--production-readiness", "--research-readiness", "--publish-readiness",
    ):
        assert token in text


def test_release_gates_before_upload_and_limits_permissions() -> None:
    text = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    for token in (
        "permissions:", "contents: read", "contents: write", "pytest", "--publish-readiness",
        "smoke-packaged-desktop.ps1", "Get-FileHash", "action-gh-release",
    ):
        assert token in text
    assert text.index("pytest") < text.index("action-gh-release")
