# Trustworthy Streaming Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one causal, reproducible, end-to-end MobileDeepFilterNet training and streaming pipeline with trustworthy evaluation artifacts.

**Architecture:** Promote `src/` to the only production pipeline. Introduce typed audio/data configuration, portable speaker-aware manifests, a causal neural encoder, explicit streaming state, and CLI-driven train/evaluate/export workflows. Historical notebooks and checkpoints remain read-only evidence.

**Tech Stack:** Python, PyTorch, NumPy, SciPy, SoundFile, PyYAML, unittest/pytest-compatible tests, optional ONNX Runtime, optional PESQ/STOI/DNSMOS.

## Global Constraints

- Sample rate 16000 Hz, frame 320 samples, hop 160 samples, 161 bins.
- No production code before a regression test fails for the intended reason.
- Paths are project-relative or supplied explicitly; no drive-letter defaults.
- Training and streaming use identical feature and filtering definitions.
- No benchmark claim without a machine-readable result and generating command.
- Existing datasets, notebooks, checkpoints, and user artifacts are preserved.

---

### Task 1: Reproducible Project Contract

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/config.py`
- Create: `tests/test_config.py`
- Modify: `configs/train.yaml`
- Modify: `README.md`

**Interfaces:**
- Produces: `AudioConfig`, `DataConfig`, `TrainConfig`, `load_train_config(path, project_root)`.

- [ ] Write failing tests for project-relative defaults, STFT invariants, unknown config keys, and missing paths.
- [ ] Run `python -m unittest tests.test_config -v`; expect failures because `src.config` is absent.
- [ ] Implement typed config parsing with fail-fast validation.
- [ ] Run the config tests; expect all pass.
- [ ] Add dependency metadata, ignored artifact rules, and CLI documentation.

### Task 2: Correct Paired Mixture Dataset

**Files:**
- Modify: `src/augment.py`
- Modify: `src/dataset.py`
- Create: `tests/test_dataset.py`
- Modify: `tests/test_mixing.py`

**Interfaces:**
- Produces: `mix_clean_with_noises(...) -> (clean_target, noisy, meta)`.
- Produces: `NoiseSuppressionDataset.set_epoch(epoch: int) -> None`.

- [ ] Write failing tests proving the returned clean target shares RIR, gain, and normalization with the mixture construction.
- [ ] Write a failing test proving train mixtures change across epochs and reproduce when epoch/seed match.
- [ ] Run dataset tests; confirm target-alignment and epoch-variation failures.
- [ ] Return the processed clean target, use it in dataset output, and derive worker/epoch seeds.
- [ ] Keep validation deterministic and make metadata safely collatable.
- [ ] Run mixing and dataset tests; expect all pass.

### Task 3: Portable Speaker-Aware Splits

**Files:**
- Create: `src/splits.py`
- Create: `scripts/build_manifests.py`
- Create: `tests/test_splits.py`

**Interfaces:**
- Produces: `infer_speaker_id(path)`, `build_grouped_splits(...)`, and relative JSONL manifests with fingerprints.

- [ ] Write failing tests for LibriSpeech, VIVOS, VCTK naming and zero speaker overlap.
- [ ] Run split tests; expect failure because the module is absent.
- [ ] Implement deterministic grouped splits and manifest serialization.
- [ ] Add corpus/SNR distribution reports and overlap assertions.
- [ ] Run split tests; expect all pass.
- [ ] Generate new manifests under `manifests/v2/` without deleting historical manifests.

### Task 4: Causal Neural Core

**Files:**
- Modify: `src/mobileone.py`
- Modify: `src/model.py`
- Create: `tests/test_causality.py`
- Modify: `tests/test_mobileone.py`

**Interfaces:**
- Produces causal time-padding modules while preserving MobileOne deploy fusion.
- Preserves model outputs `mask`, `w_taps`, `h_new`.

- [ ] Write a failing prefix-invariance test: changing future frames must not alter earlier outputs.
- [ ] Run the causality test; confirm the symmetric time convolutions fail it.
- [ ] Implement left-only time padding for initial, MobileOne, and depthwise convolutions.
- [ ] Adapt reparameterization tests to the causal padding contract.
- [ ] Run causality and MobileOne tests; expect all pass.

### Task 5: Verified Spectral and Streaming Runtime

**Files:**
- Modify: `src/filtering.py`
- Modify: `src/streaming_stft.py`
- Create: `src/inference.py`
- Create: `tests/test_filtering.py`
- Create: `tests/test_streaming.py`

**Interfaces:**
- Produces `enhance_offline(model, noisy, state=None)`.
- Produces `StreamingEnhancer.push(hop_samples) -> enhanced_hop` and explicit reset/state behavior.

- [ ] Write failing identity-filter, reconstruction, and offline/streaming-prefix tests.
- [ ] Run tests; confirm missing synthesis/state behavior fails.
- [ ] Implement stable analysis/synthesis overlap-add and deep-filter history.
- [ ] Ensure each streaming call advances GRU state exactly once.
- [ ] Run filtering/streaming tests; expect all pass.

### Task 6: Train/Evaluate/Checkpoint Pipeline

**Files:**
- Modify: `src/trainer.py`
- Create: `src/evaluate.py`
- Create: `scripts/train.py`
- Create: `scripts/evaluate.py`
- Create: `tests/test_training_step.py`
- Create: `tests/test_checkpoint.py`

**Interfaces:**
- Produces CLI train/evaluate commands and checkpoint schema version 2.

- [ ] Write failing integration tests for DataLoader collation, one train step, AMP unscale-before-clip, and checkpoint round trip.
- [ ] Run integration tests; confirm current trainer fails the contract.
- [ ] Refactor trainer around shared inference/config functions and safe collation.
- [ ] Implement noisy, mask-only, and deep-filter evaluation baselines.
- [ ] Run training/checkpoint tests; expect all pass.

### Task 7: Stateful Export and Parity

**Files:**
- Create: `src/export.py`
- Create: `scripts/export.py`
- Create: `tests/test_export.py`

**Interfaces:**
- Produces neural-core artifacts with explicit GRU state input/output and metadata sidecar.

- [ ] Write failing eager/TorchScript parity tests for output and next hidden state.
- [ ] Run export tests; confirm exporter is absent.
- [ ] Implement versioned export wrapper and metadata validation.
- [ ] Add optional ONNX parity when ONNX Runtime is installed.
- [ ] Run export tests; expect required tests pass and optional tests skip explicitly.

### Task 8: Verification, Smoke Training, Benchmark

**Files:**
- Create: `scripts/verify.py`
- Create: `configs/smoke.yaml`
- Create: `reports/README.md`

**Interfaces:**
- Produces one verification command and machine-readable smoke/evaluation reports.

- [ ] Run the full correctness suite and syntax/static checks.
- [ ] Build v2 manifests and run leakage audit.
- [ ] Run bounded smoke training using available compute.
- [ ] Evaluate the smoke checkpoint against noisy and mask-only baselines.
- [ ] Export and run parity/latency checks supported by the environment.
- [ ] Record exact commands, environment limitations, metrics, and next full-training command.
