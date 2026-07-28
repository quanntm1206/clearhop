from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_desktop_build_copies_and_verifies_project_license() -> None:
    text = (ROOT / "scripts/build-desktop.ps1").read_text(encoding="utf-8")
    for token in (
        'Join-Path $ReleaseRoot "LICENSE"',
        "Copy-Item -LiteralPath $LicenseSource",
        "Get-FileHash -LiteralPath $LicenseSource",
        "Get-FileHash -LiteralPath $LicenseTarget",
        "Packaged license hash mismatch",
    ):
        assert token in text


def test_ci_installs_desktop_and_runs_fail_closed_gates() -> None:
    text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    for token in (
        ".[desktop,dev,export]", "QT_QPA_PLATFORM", "test_desktop_e2e.py",
        "verify_public_production.py", "verify_public_research.py", "--publish-readiness",
    ):
        assert token in text


def test_ci_runs_clean_offline_installer_and_real_denoise_outside_repo() -> None:
    text = (ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    installer_step = text[text.index("Clean offline installer integration") :]
    for token in (
        "-Offline",
        "-NoShortcuts",
        "-InstallDir $installRoot",
        "$env:RUNNER_TEMP",
        "noise-reduce-desktop.exe",
        "--smoke-input",
        "--smoke-output",
        "--smoke-receipt",
        "checkpoint_sha256",
        "output_sha256",
        "Get-FileHash",
        "Remove-Item -LiteralPath $installRoot -Recurse -Force",
    ):
        assert token in installer_step
    assert "-DryRun" not in installer_step


def test_release_gates_before_upload_and_limits_permissions() -> None:
    text = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    for token in (
        "permissions:", "contents: read", "contents: write", "pytest", "--publish-readiness",
        "smoke-packaged-desktop.ps1", "Get-FileHash", "action-gh-release",
    ):
        assert token in text
    assert text.index("pytest") < text.index("action-gh-release")
    for token in (
        "download.pytorch.org/whl/cpu", "torch==2.13.0+cpu", "pip==26.1.2", "setuptools==83.0.0", "PIP_CONSTRAINT",
        "${{ github.workspace }}", "constraints/release.txt", "torch.version.cuda is None", "refs/tags/",
        "actions/upload-artifact@", "actions/download-artifact@", "ExpectedCheckpointSha256",
        "Final release asset validation failed", "needs: verify-build",
    ):
        assert token in text
    assert text.index("contents: write") > text.index("publish:")
    assert "actions/checkout@v" not in text
    assert "actions/setup-python@v" not in text
    assert "softprops/action-gh-release@v" not in text
