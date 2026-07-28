# ClearHop Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `quanntm1206/clearhop` score at least 9/10 for GitHub publishing, Windows production distribution, and public research evidence.

**Architecture:** Move all installed inference dependencies into packaged modules, then enforce clean-install and packaged-launch contracts in CI/release. Separate generated/private evidence from curated public receipts. Extend the comparison runner with source-provenanced external adapters and fail-closed readiness scoring.

**Tech Stack:** Python 3.10+, PyTorch, PySide6, PyInstaller, pytest, PowerShell, GitHub Actions, JSON receipts.

## Task 1: Package-Safe Production Runtime

**Files:** `src/cpu_runtime.py`, `scripts/enhance_cpu.py`, `desktop/pipeline.py`, `pyproject.toml`, `tests/test_cpu_runtime.py`, `tests/test_desktop_wheel.py`

- [x] Add a failing test proving a built wheel lacks the current `scripts.enhance_cpu` runtime dependency.
- [x] Add a failing clean-install test that loads the pinned checkpoint and denoises a generated WAV outside repository cwd.
- [x] Move reusable CPU enhancement into `src/cpu_runtime.py`; retain the script as a thin wrapper.
- [x] Update desktop imports and package discovery; remove runtime imports from `scripts`.
- [x] Build the wheel and run focused tests until both contracts pass.

## Task 2: Real Desktop and Installer Acceptance

**Files:** `tests/test_desktop_e2e.py`, `scripts/install-desktop.ps1`, `scripts/build-desktop.ps1`, `scripts/smoke-packaged-desktop.ps1`, `configs/desktop_assets.json`

- [x] Add a failing offscreen Open/Process/Save/receipt test with PySide6 installed.
- [x] Add a failing isolated offline installer smoke using a disposable install directory.
- [x] Add a packaged executable smoke that runs outside repository cwd.
- [x] Fix installer/version/asset behavior required by these tests.
- [x] Verify checkpoint and ONNX hashes before shortcut activation.

## Task 3: Publish-Readiness Gate and Repository Hygiene

**Files:** `scripts/verify.py`, `scripts/doctor.ps1`, `scripts/doctor.sh`, `tests/test_publish_readiness.py`, `.gitignore`, `reports/public/**`

- [x] Add failing tests for ignored README evidence, large-file inventory, placeholder URLs, version drift, and missing public documents.
- [x] Extend `scripts/verify.py` with `--publish-readiness` and a machine-readable score for all three pillars.
- [x] Add cross-platform doctor wrappers invoking the new audit.
- [x] Curate immutable receipts into `reports/public/`; keep generated evidence ignored.
- [x] Ignore exploratory root artifacts and reject unapproved large/secret-like files.

## Task 4: GitHub CI and Release Fail-Closed Gates

**Files:** `.github/workflows/ci.yml`, `.github/workflows/release.yml`, `.github/dependabot.yml`, `.github/ISSUE_TEMPLATE/**`, `CONTRIBUTING.md`, `CHANGELOG.md`, `MODEL_CARD.md`

- [x] Add validation tests for required workflow commands and permissions.
- [x] Install `.[desktop,dev]` in Windows CI and run actual offscreen E2E.
- [x] Run production, research, and publish verifiers in CI.
- [x] Gate release on tests, clean wheel install, packaged launch, asset hashes, and release inventory.
- [x] Add contribution, release, model-card, dependency-update, and issue-reporting surfaces.

## Task 5: Public Research Comparison

**Files:** `scripts/model_comparison.py`, `configs/research_baselines.json`, `tests/test_model_comparison.py`, `reports/public/model_comparison.json`, `docs/research-comparison.md`, `MODEL_CARD.md`

- [x] Research official source, commit/version, weight hash, and redistribution license for candidate baselines.
- [x] Add failing tests for provenance, license, command, environment, sample-rate, latency, and identical item IDs.
- [ ] Reproduce at least two free baselines when executable on this Windows machine; blocked by missing runnable dependencies/weights, with verified recipes recorded.
- [x] Regenerate the public report; separate reproduced, literature-only, and blocked rows.
- [x] Run research rigor/schema checks and record exact limitations.

## Task 6: ClearHop Branding and Visual Evidence

**Files:** `pyproject.toml`, `desktop/app.py`, `README.md`, `CITATION.cff`, `SECURITY.md`, `docs/assets/desktop-ui.png`

- [x] Replace generic/project-placeholder metadata with `quanntm1206/clearhop`.
- [x] Render the Windows UI with explicit supported fonts and capture readable pixels.
- [x] Review hierarchy, empty/loading/error/disabled states, contrast, and spacing.
- [x] Link README only to tracked public receipts and verified commands.

## Task 7: Final Verification

- [x] Run focused tests for every changed subsystem.
- [x] Run full `pytest` suite.
- [x] Run production, research, and publish readiness verifiers.
- [x] Build wheel; clean-install and denoise outside repo cwd.
- [x] Run real offline installer smoke in disposable directory.
- [x] Build PyInstaller onedir; run packaged smoke outside repo cwd.
- [x] Validate public inventory and README links.
- [x] Confirm each readiness score is at least 9/10; report honest blockers if not.
