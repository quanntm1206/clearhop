from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_installer_uses_clearhop_identity_and_cpu_runtime() -> None:
    text = (ROOT / "scripts/install-desktop.ps1").read_text(encoding="utf-8")
    for token in (
        "ClearHop", "download.pytorch.org/whl/cpu", "torch==2.13.0+cpu",
        "pip==26.1.2", "setuptools==83.0.0", "$env:PIP_CONSTRAINT", "constraints\\release.txt", "[clearhop]",
    ):
        assert token in text
    assert "Noise Reduce.lnk" not in text
    assert "AbsoluteUri" in text


def test_installer_can_skip_user_shortcuts_for_isolated_validation() -> None:
    text = (ROOT / "scripts/install-desktop.ps1").read_text(encoding="utf-8")
    assert "[switch]$NoShortcuts" in text
    assert "if (-not $NoShortcuts)" in text


def test_installer_requires_the_python_minor_used_by_the_release_lock() -> None:
    installer = (ROOT / "scripts/install-desktop.ps1").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for token in ("py -3.11", "sys.version_info[:2] == (3, 11)", "Python 3.11 is required"):
        assert token in installer
    assert "Python 3.10 or newer" not in installer
    assert "Python 3.11" in readme


def test_release_lock_is_complete_and_has_no_local_project_reference() -> None:
    lines = [
        line.strip()
        for line in (ROOT / "constraints/release.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert "pip==26.1.2" in lines
    assert "torch==2.13.0+cpu" in lines
    assert len(lines) >= 40
    assert all("clearhop" not in line.lower() and " @ " not in line and "file:" not in line for line in lines)
