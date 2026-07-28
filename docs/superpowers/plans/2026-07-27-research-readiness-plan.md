# Research Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Produce a reproducible, statistically defensible five-seed study with robustness and failure analysis.

**Architecture:** Preserve the frozen primary comparison/audit contract. Add deterministic slice specs, per-item paired metrics, bootstrap/significance receipts, and targeted ablations; rejected seeds remain visible.

**Tech Stack:** Python, NumPy, SciPy if already installed, existing evaluator/trainer, JSON receipts, `unittest`.

## Global Constraints

- Baseline, manifests, preprocessing, thresholds, and primary slices remain frozen.
- Final candidate uses five independent seeds; no seed is silently removed.
- Missing metric dependencies produce `blocked`, never an imputed pass.
- Report confidence intervals, effect sizes, and failure cases before conclusions.
- Existing 122-test suite must remain green.

### Task 1: Deterministic robustness slice contract

**Files:**
- Create: `configs/evaluation/research_slices.json`
- Modify: `scripts/evaluate.py`
- Create: `tests/test_research_slices.py`

**Interfaces:**
- `load_research_slices(path: Path) -> list[dict[str, object]]`
- `slice_fingerprint(manifest: Path, offset: int, count: int) -> str` remains the binding primitive.

- [ ] Test schema validation, duplicate slice rejection, stable ordering, and manifest-fingerprint binding.
- [ ] Implement slices for primary comparison/audit, SNR bands, noise families, speaker groups, and stream-position buckets using existing metadata only.
- [ ] Emit exact item IDs and fingerprints in every evaluation receipt.
- [ ] Run focused tests and compileall.

### Task 2: Per-item metrics and failure analysis

**Files:**
- Modify: `src/evaluate.py`
- Modify: `scripts/evaluate.py`
- Create: `scripts/analyze_failures.py`
- Create: `tests/test_failure_analysis.py`

**Interfaces:**
- `evaluate_items(...) -> list[dict[str, float | str]]`
- `summarize_failures(items: list[dict[str, object]], top_k: int = 25) -> dict[str, object]`

- [ ] Test finite per-item output, deterministic worst-case ordering, and subgroup counts.
- [ ] Persist per-item SI-SDR/SNR/STOI/gain/polarity values plus manifest item IDs without changing primary aggregate keys.
- [ ] Classify failures by metric gate, SNR/noise/speaker/duration bucket; save top-k worst items and reproducible paths.
- [ ] Run focused tests and confirm existing evaluator tests remain green.

### Task 3: Bootstrap and significance receipt

**Files:**
- Create: `src/statistics.py`
- Create: `scripts/significance.py`
- Create: `tests/test_statistics.py`

**Interfaces:**
- `paired_bootstrap_delta(candidate: Sequence[float], baseline: Sequence[float], seed: int = 0, resamples: int = 10000) -> dict[str, float]`
- `paired_significance(candidate: Sequence[float], baseline: Sequence[float]) -> dict[str, float | str]`

- [ ] Test deterministic bootstrap, length mismatch, non-finite rejection, and zero-variance behavior.
- [ ] Implement paired deltas, 95% percentile CI, median/mean effect size, and a predeclared paired test available in the installed environment.
- [ ] Emit `reports/generated/significance.json` bound to exact candidate/baseline evaluation hashes.
- [ ] Run focused tests and compileall.

### Task 4: Five-seed controlled experiment

**Files:**
- Modify: `scripts/run_gain_ablation.py`
- Modify: `scripts/verify.py`
- Create: `tests/test_research_readiness.py`

- [ ] Add an explicit five-seed plan (`17, 29, 43, 59, 71`) without changing the frozen three-seed historical receipt.
- [ ] Add targeted ablation receipts for beta/refinement choices; bind each to the same manifests and baseline.
- [ ] Require all five seeds, all primary slices, and all robustness slices in the research gate.
- [ ] Add negative fixtures for one rejected seed, slice drift, and missing per-item data.
- [ ] Run focused tests before any long training.

### Task 5: Research report and reproduction bundle

**Files:**
- Create: `scripts/build_research_report.py`
- Create: `docs/research-readiness.md`
- Generate: `reports/generated/robustness_matrix.json`
- Generate: `reports/generated/significance.json`
- Generate: `reports/generated/failure_analysis.json`
- Generate: `reports/generated/research_readiness.json`

- [ ] Build a report with research question, protocol, primary/robustness tables, CIs, effect sizes, negative results, and limitations.
- [ ] Include commands, environment, configs, seeds, manifest hashes, checkpoint/export hashes, and raw receipt pointers.
- [ ] Require `research_eligible=true` only when every declared research check passes.
- [ ] Run the complete test suite and the final research verifier.
