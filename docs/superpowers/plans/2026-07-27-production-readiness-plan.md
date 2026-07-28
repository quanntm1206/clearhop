# Production Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Make the exported enhancer verifiably safe for Windows desktop CPU-only real-time use.

**Architecture:** Keep the existing stateful streaming model and ONNX contract. Add deterministic CPU benchmark, soak, fault-injection, and bundle-integrity receipts; promotion stays conjunctive and never falls back to CUDA.

**Tech Stack:** Python, PyTorch, ONNX Runtime, NumPy, existing `scripts/verify.py`, `unittest`.

## Global Constraints

- CPU-only benchmark; missing CPU prerequisites produce `blocked`, never `pass`.
- Reference hop is 10 ms; production p95 threshold is `< 10 ms`.
- Every receipt binds checkpoint/config/export hashes and environment metadata.
- Existing 122-test suite must remain green.
- No production label when any gate fails.

### Task 1: CPU benchmark contract

**Files:**
- Create: `scripts/benchmark_cpu.py`
- Create: `tests/test_benchmark_cpu.py`
- Modify: `scripts/verify.py`

**Interfaces:**
- `run_cpu_benchmark(root: Path, checkpoint: Path, iterations: int = 5000) -> dict[str, object]`
- Receipt keys: `schema_version`, `status`, `device="cpu"`, `iterations`, `checkpoint_sha256`, `environment`, `neural_core`, `streaming_end_to_end`.

- [ ] Write tests for deterministic receipt shape, CPU-only execution, p95 threshold, and hash mismatch rejection.
- [ ] Run `python -m unittest tests.test_benchmark_cpu -v`; confirm failures before implementation.
- [ ] Implement benchmark with `torch.set_num_threads(1)`, fixed seed, warmup, `time.perf_counter_ns`, p50/p95/p99/max, realtime factor, RSS delta.
- [ ] Ensure CUDA is never selected even when available; return `blocked` when the exported runtime cannot load on CPU.
- [ ] Re-run focused tests, then `python -m compileall -q src scripts tests`.

### Task 2: CPU soak and fault contract

**Files:**
- Create: `scripts/soak_cpu.py`
- Create: `tests/test_soak_cpu.py`
- Modify: `src/streaming.py`

**Interfaces:**
- `run_cpu_soak(root: Path, checkpoint: Path, seconds: int = 7200) -> dict[str, object]`
- `StreamingEnhancer.reset() -> None` remains idempotent; invalid frame input raises a typed `ValueError` without mutating state.

- [ ] Add tests for reset idempotence, invalid shape rejection, finite outputs, and state isolation after reset.
- [ ] Run focused tests and confirm baseline failures.
- [ ] Implement deterministic short-fixture soak mode plus production-duration mode; record iteration count, failures, RSS start/end/peak, and state hash before/after reset.
- [ ] Add fault cases: empty frame, wrong frequency bins, NaN input, repeated start/stop; mark unsupported device-loss simulation as `blocked`.
- [ ] Re-run focused tests and compileall.

### Task 3: Bundle integrity and production gate

**Files:**
- Create: `scripts/package_cpu_bundle.py`
- Create: `tests/test_package_cpu_bundle.py`
- Modify: `scripts/verify.py`

**Interfaces:**
- `package_cpu_bundle(root: Path, output: Path, checkpoint: Path, torchscript: Path, onnx: Path, config: Path) -> dict[str, object]`
- `production_readiness(root: Path) -> dict[str, object]` returns `status`, `production_eligible`, `checks`, and `reasons`.

- [ ] Test path containment, hash binding, missing-file rejection, and atomic destination replacement.
- [ ] Implement bundle layout `model.onnx`, `model.ts`, `model.json`, `config.yaml`, `README.txt`, each with SHA-256 metadata.
- [ ] Require CPU benchmark, soak, parity, structure, and bundle checks before `production_eligible=true`.
- [ ] Add `--production-readiness` to `scripts/verify.py`; emit `reports/generated/production_readiness.json`.
- [ ] Run focused tests and verify a deliberate hash mutation yields `status=fail`.

### Task 4: Windows smoke entrypoint

**Files:**
- Create: `scripts/run_cpu_smoke.ps1`
- Create: `scripts/run_cpu_smoke.sh`
- Create: `tests/test_cpu_smoke_entrypoint.py`
- Create: `docs/production-cpu.md`

- [ ] Test dry-run command construction and non-zero propagation.
- [ ] Implement PowerShell and POSIX wrappers with explicit Python path, output directory, and dry-run support.
- [ ] Document reference hardware, thread settings, receipt paths, rollback procedure, and failure semantics.
- [ ] Run wrapper dry-run plus full tests.

### Task 5: Production acceptance run

**Files:**
- Generate: `reports/generated/cpu_benchmark.json`
- Generate: `reports/generated/cpu_soak.json`
- Generate: `reports/generated/production_readiness.json`

- [ ] Run CPU benchmark for 5000 hops.
- [ ] Run deterministic soak fixture, then launch the 2-hour soak if the fixture passes.
- [ ] Package the selected export and run production verifier.
- [ ] Record pass/fail without relaxing thresholds; production label only if every check is true.
