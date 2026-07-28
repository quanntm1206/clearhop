# MobileDeepFilterNet Gain Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add scale- and polarity-aware training, run a controlled three-seed ablation, and publish a verified calibrated streaming checkpoint without overwriting the historical baseline.

**Architecture:** Keep `MobileDeepFilterNet` and its streaming/export interfaces unchanged. Add pure loss functions and strict config, route training into isolated run directories, extend evaluation with calibration and slice provenance, then let a bounded runner execute and summarize the locked experiment. Validation promotes one arm; two fixed 500-item test slices confirm it.

**Tech Stack:** Python 3, PyTorch, NumPy, unittest, existing STFT/filtering pipeline, TorchScript, ONNX Runtime, CUDA.

## Global Constraints

- Preserve `checkpoints/best.pth` and all `full_*` artifacts byte-for-byte.
- Screening: arms `control`, `complex_nmse`, `complex_nmse_sisdr`, `compressed_complex`; seeds `17,29,43`; exactly 1,500 steps on a 30,000-step scheduler.
- Full confirmation: one promoted arm, seeds `17,29,43`, exactly 30,000 steps each.
- New output root: `checkpoints/gain_calibration/`; new reports use `gain_calibration_*` names.
- Selection uses validation only. Test comparison slice is offset `0`, count `500`; untouched audit slice is offset `500`, count `500`.
- Reject non-finite loss immediately. Never silently skip a failed run or seed.
- No new runtime dependency. No model/state/export signature change unless every loss arm fails.
- Workspace has no Git repository; omit commit commands and retain deterministic test/report receipts instead.

---

### Task 1: Strict loss and run configuration

**Files:**
- Modify: `src/config.py:32`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces: `TrainConfig.loss`, `loss_eps`, `sisdr_warmup_start`, `sisdr_warmup_end`, `compression_exponent`, `compression_complex_weight`, `checkpoint_dir`, `max_steps`, `scheduler_total_steps`, `selection_metric`, `experiment_id`.
- Preserves: all existing `configs/train.yaml` values and default checkpoint directory `checkpoints/`.

- [ ] **Step 1: Write failing config tests**

Add tests that construct `TrainConfig.from_mapping` with every new field, assert round-trip through `to_dict`, assert relative `checkpoint_dir` resolves from `project_root`, and assert these invalid mappings raise `ValueError`:

```python
invalid = [
    {"loss": "unknown"},
    {"loss_eps": 0.0},
    {"beta_si_sdr": -0.1},
    {"sisdr_warmup_start": 10, "sisdr_warmup_end": 9},
    {"compression_exponent": 0.0},
    {"compression_exponent": 1.1},
    {"compression_complex_weight": -0.1},
    {"compression_complex_weight": 1.1},
    {"max_steps": 0},
    {"scheduler_total_steps": 0},
    {"selection_metric": "loss"},
]
for values in invalid:
    with self.subTest(values=values), self.assertRaises(ValueError):
        TrainConfig.from_mapping({**self.required_paths, **values}, project_root=self.root)
```

- [ ] **Step 2: Confirm RED**

Run:

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.test_config -v
```

Expected: new-field and validation tests fail; existing tests remain green.

- [ ] **Step 3: Implement fields and validation**

Add exact defaults:

```python
loss_eps: float = 1e-8
sisdr_warmup_start: int = 0
sisdr_warmup_end: int = 0
compression_exponent: float = 0.3
compression_complex_weight: float = 0.3
checkpoint_dir: Path | None = None
max_steps: int | None = None
scheduler_total_steps: int | None = None
selection_metric: str = "si_sdr_improvement"
experiment_id: str = "baseline"
```

Allow loss modes `complex_mse_plus_si_sdr`, `complex_nmse`, `complex_nmse_sisdr`, and `compressed_complex`. Validate numeric ranges in `TrainConfig.__post_init__`. Resolve `checkpoint_dir` in `from_mapping`; default it to `project_root / "checkpoints"`. Serialize all fields in `to_dict` with `checkpoint_dir` as a string.

- [ ] **Step 4: Confirm GREEN and backward compatibility**

Run:

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.test_config tests.test_checkpoint -v
& .\.venv\Scripts\python.exe -c "from pathlib import Path; from src.config import load_train_config; c=load_train_config(Path('configs/train.yaml'),Path.cwd()); assert c.loss=='complex_mse_plus_si_sdr'; assert c.checkpoint_dir==Path.cwd()/'checkpoints'; print('config-backcompat-pass')"
```

Expected: all tests pass; output includes `config-backcompat-pass`.

---

### Task 2: Pure calibration losses

**Files:**
- Create: `src/losses.py`
- Create: `tests/test_losses.py`
- Modify: `src/trainer.py:50`

**Interfaces:**
- Produces: `si_sdr_loss(est, ref, eps)`, `complex_mse(est, ref)`, `complex_nmse(est, ref, eps)`, `compressed_complex_loss(est, ref, exponent, complex_weight, eps)`, `scheduled_weight(step, target, start, end)`.
- Consumes tensors shaped `(B,F,T,2)` for spectral losses and `(B,T)` for SI-SDR.

- [ ] **Step 1: Write exact-value and gradient tests**

Cover identity, `0.5*S`, `2*S`, `-S`, silence, mixed-level batch, invalid shapes, invalid compression values, and finite gradients. Required NMSE assertions:

```python
target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
self.assertAlmostEqual(complex_nmse(target, target).item(), 0.0, places=7)
self.assertAlmostEqual(complex_nmse(0.5 * target, target).item(), 0.25, places=6)
self.assertAlmostEqual(complex_nmse(2.0 * target, target).item(), 1.0, places=6)
self.assertAlmostEqual(complex_nmse(-target, target).item(), 4.0, places=6)
```

Assert `scheduled_weight(500, 0.01, 500, 1000) == 0`, step `750 == 0.005`, step `1000 == 0.01`.

- [ ] **Step 2: Confirm RED**

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.test_losses -v
```

Expected: import failure for `src.losses`.

- [ ] **Step 3: Implement pure functions**

Use per-example sums before batch mean:

```python
def complex_nmse(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    _validate_complex_pair(est, ref)
    reduce_dims = tuple(range(1, est.ndim))
    error = (est - ref).square().sum(dim=reduce_dims)
    target = ref.square().sum(dim=reduce_dims).clamp_min(eps)
    return (error / target).mean()
```

For compressed loss, compute `magnitude = sqrt(re**2 + im**2 + eps)`, `compressed = z * magnitude**(exponent - 1)`, normalize magnitude and complex errors independently by target energy, then return `(1-complex_weight)*magnitude_loss + complex_weight*complex_loss`. Reject non-finite outputs with `FloatingPointError`.

Move the existing trainer SI-SDR and complex-MSE helpers into this module; re-export private aliases from `src.trainer` temporarily so existing imports remain compatible.

- [ ] **Step 4: Confirm GREEN**

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.test_losses tests.test_training_step -v
```

Expected: all loss and existing training contract tests pass.

---

### Task 3: Trainer routing, bounded stops, isolated artifacts

**Files:**
- Modify: `src/trainer.py:145`
- Modify: `scripts/train.py:12`
- Modify: `tests/test_training_step.py`
- Create: `tests/test_trainer_control.py`

**Interfaces:**
- Consumes Task 1 config and Task 2 losses.
- Produces checkpoints under `TrainConfig.checkpoint_dir`; every checkpoint embeds `experiment_id`, loss config, `scheduler_total_steps`, and `max_steps`.
- Produces final `run_summary.json` with status, stop reason, best metric, best path, last path, global step, seed, and component-loss means.

- [ ] **Step 1: Write failing trainer-control tests**

Use mocks for dataset/model-heavy boundaries. Assert:

1. scheduler total steps remains `30000` while `max_steps=3` stops at global step `3`;
2. the final partial epoch still invokes validation;
3. `checkpoint_dir` receives all files and root `checkpoints/best.pth` is untouched;
4. `selection_metric=snr_improvement` compares validation SNRi;
5. resume from step `2` reaches exactly step `3` without repeating scheduler steps;
6. non-finite component loss raises `FloatingPointError` before optimizer step.

- [ ] **Step 2: Confirm RED**

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.test_trainer_control -v
```

Expected: failures for unsupported bounded-stop and isolated-output behavior.

- [ ] **Step 3: Extract objective and validation helpers**

Add these internal interfaces:

```python
def _compute_objective(
    cfg: dict[str, Any],
    y_ri: torch.Tensor,
    clean_ri: torch.Tensor,
    enhanced: torch.Tensor,
    clean: torch.Tensor,
    global_step: int,
) -> dict[str, torch.Tensor]:
    raw = complex_mse(y_ri, clean_ri)
    normalized = complex_nmse(y_ri, clean_ri, float(cfg["loss_eps"]))
    sisdr = si_sdr_loss(enhanced, clean, float(cfg["loss_eps"]))
    compressed = y_ri.new_zeros(())
    weight = y_ri.new_zeros(())
    mode = str(cfg["loss"])
    if mode == "complex_mse_plus_si_sdr":
        weight = y_ri.new_tensor(float(cfg["beta_si_sdr"]))
        total = float(cfg["alpha_loss"]) * raw + weight * sisdr
    elif mode == "complex_nmse":
        total = float(cfg["alpha_loss"]) * normalized
    elif mode == "complex_nmse_sisdr":
        weight = y_ri.new_tensor(
            scheduled_weight(
                global_step,
                float(cfg["beta_si_sdr"]),
                int(cfg["sisdr_warmup_start"]),
                int(cfg["sisdr_warmup_end"]),
            )
        )
        total = float(cfg["alpha_loss"]) * normalized + weight * sisdr
    elif mode == "compressed_complex":
        compressed = compressed_complex_loss(
            y_ri,
            clean_ri,
            exponent=float(cfg["compression_exponent"]),
            complex_weight=float(cfg["compression_complex_weight"]),
            eps=float(cfg["loss_eps"]),
        )
        total = float(cfg["alpha_loss"]) * compressed
    else:
        raise ValueError(f"Unsupported loss mode: {mode}")
    if not torch.isfinite(total):
        raise FloatingPointError(f"Non-finite training loss for mode {mode}")
    return {
        "total": total,
        "complex_mse": raw.detach(),
        "complex_nmse": normalized.detach(),
        "compressed_complex": compressed.detach(),
        "si_sdr": sisdr.detach(),
        "si_sdr_weight": weight.detach(),
    }

def _selection_value(selection_metric: str, si_sdri: float, snri: float) -> float:
    if selection_metric == "si_sdr_improvement":
        return si_sdri
    if selection_metric == "snr_improvement":
        return snri
    raise ValueError(f"Unsupported selection_metric: {selection_metric}")
```

Return keys `total`, `complex_mse`, `complex_nmse`, `compressed_complex`, `si_sdr`, and `si_sdr_weight`; unused components are scalar zeros on the same device.

- [ ] **Step 4: Implement exact stop and scheduler contracts**

Set scheduler total to `scheduler_total_steps or epochs*steps_per_epoch`. Stop only after completing optimizer step `max_steps`. Run validation once at the final stop even inside an epoch. Save `step_{global_step}.pth`, `best.pth`, and `run_summary.json` beneath `checkpoint_dir`. Include `status="completed"` only after writer close and final checkpoint serialization.

- [ ] **Step 5: Add CLI overrides without changing defaults**

Add `--max-steps`, `--checkpoint-dir`, and `--experiment-id` to `scripts/train.py`; apply them to a loaded config mapping before calling `train_model`. Existing `python scripts/train.py --config configs/train.yaml` must behave unchanged.

- [ ] **Step 6: Confirm GREEN and run a tiny CUDA smoke**

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.test_training_step tests.test_trainer_control -v
& .\.venv\Scripts\python.exe scripts/train.py --config configs/smoke.yaml --max-steps 2 --checkpoint-dir checkpoints/gain_calibration/dev_smoke --experiment-id dev_smoke
```

Expected: tests pass; `run_summary.json` reports global step `2`; historical hashes remain unchanged.

---

### Task 4: Calibration metrics and deterministic evaluation slices

**Files:**
- Modify: `src/evaluate.py:45`
- Modify: `scripts/evaluate.py:22`
- Modify: `src/splits.py`
- Create: `tests/test_evaluate_calibration.py`
- Modify: `tests/test_splits.py`

**Interfaces:**
- Produces row metrics `projection_gain`, `gain_error_db`, `polarity_failure`.
- Produces metadata `slice_offset`, `slice_count`, `slice_fingerprint`, `manifest_fingerprint`.
- CLI adds `--offset` while preserving `--max-items` semantics.

- [ ] **Step 1: Write failing metric tests**

For clean `[1,-1]`, assert projection gain is `1` for clean, `0.5` for half scale, `-1` for inverted; polarity failure is `1.0` only for non-positive gain; gain error is `-6.0206 dB` for half scale and `0 dB` for both `1` and `-1` magnitude. Assert aggregate output contains mean, std, median, min, and max.

- [ ] **Step 2: Write failing slice-fingerprint tests**

Create a three-line temporary manifest; assert offsets `0` and `1` have different fingerprints, repeated calls are stable, and an out-of-range or non-positive count raises `ValueError`.

- [ ] **Step 3: Confirm RED**

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.test_evaluate_calibration tests.test_splits -v
```

Expected: missing metric/fingerprint interfaces fail.

- [ ] **Step 4: Implement metrics and slice hashing**

Add:

```python
def calibration_metrics(clean: np.ndarray, output: np.ndarray, eps: float = 1e-12) -> dict[str, float]:
    gain = float(np.dot(output, clean) / max(float(np.dot(clean, clean)), eps))
    return {
        "projection_gain": gain,
        "gain_error_db": float(20.0 * np.log10(max(abs(gain), eps))),
        "polarity_failure": float(gain <= 0.0),
    }
```

Hash the canonical UTF-8 bytes of the selected ordered manifest rows, including their newline separators. Wrap the dataset in `Subset(dataset, range(offset, offset+count))`; fail instead of truncating an invalid slice.

- [ ] **Step 5: Confirm GREEN and baseline compatibility**

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.test_evaluate_calibration tests.test_splits tests.test_verify -v
& .\.venv\Scripts\python.exe scripts/evaluate.py --checkpoint checkpoints/best.pth --manifest manifests/v2/fold_0_test.jsonl --offset 0 --max-items 8 --output reports/generated/gain_calibration_eval_smoke.json
```

Expected: tests pass; all three outputs report calibration metrics and slice metadata.

---

### Task 5: Bounded ablation runner and verifier contract

**Files:**
- Create: `configs/ablations/gain_calibration/control.yaml`
- Create: `configs/ablations/gain_calibration/complex_nmse.yaml`
- Create: `configs/ablations/gain_calibration/complex_nmse_sisdr.yaml`
- Create: `configs/ablations/gain_calibration/compressed_complex.yaml`
- Create: `scripts/run_gain_ablation.py`
- Create: `tests/test_gain_ablation_runner.py`
- Modify: `scripts/verify.py:48`
- Modify: `tests/test_verify.py`

**Interfaces:**
- CLI: `run_gain_ablation.py --stage {screen,full} --arms ARMS_CSV --seeds SEEDS_CSV [--dry-run] [--resume]`.
- Produces per-run `resolved_config.json`, `run_summary.json`, `validation_evaluation.json`.
- Produces `reports/generated/gain_calibration_screen.json` or `gain_calibration_full.json`.

- [ ] **Step 1: Write failing dry-run and collision tests**

Assert the default screen plan contains twelve unique run directories, each has `max_steps=1500`, `scheduler_total_steps=30000`, and the exact arm coefficients from the design. Assert full stage rejects multiple arms. Assert a completed run is skipped with status `already-complete`; an incomplete run requires `--resume`; no code path overwrites without explicit force support.

- [ ] **Step 2: Confirm RED**

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.test_gain_ablation_runner -v
```

Expected: runner import fails.

- [ ] **Step 3: Implement deterministic planning and subprocess execution**

Use `sys.executable`; never invoke a different Python. Run one process at a time to avoid duplicate GPU jobs. Capture start/end UTC, return code, stdout/stderr tail, config hash, manifest hashes, checkpoint hash, GPU metadata, and elapsed seconds. On nonzero exit, record `failed` and stop promotion; keep prior run directories.

- [ ] **Step 4: Implement screening summary and promotion**

Require all three seeds per arm. Compare paired arm/control validation metrics. Eligibility is positive gain for all seeds, SNRi delta at least `+1.0 dB`, SI-SDRi delta at least `-0.3 dB`, STOI delta at least `-0.005`, and finite values. Rank eligible arms by mean SNRi, mean SI-SDRi, then smallest absolute gain error. Emit exactly one `promoted_arm` or `null` with explicit rejection reasons.

- [ ] **Step 5: Add gain-calibration verification mode**

Add `--gain-calibration` to `scripts/verify.py`. In this mode audit seed coverage, resolved config hashes, manifest/slice fingerprints, checkpoint strict load, loss provenance, both 500-item evaluations, acceptance gates, 500-iteration CUDA benchmark, export hashes, and parity. Preserve existing smoke and `--config configs/train.yaml` behavior.

- [ ] **Step 6: Confirm GREEN and dry-run receipt**

```powershell
& .\.venv\Scripts\python.exe -m unittest tests.test_gain_ablation_runner tests.test_verify -v
& .\.venv\Scripts\python.exe scripts/run_gain_ablation.py --stage screen --arms control,complex_nmse,complex_nmse_sisdr,compressed_complex --seeds 17,29,43 --dry-run
```

Expected: tests pass; dry-run lists twelve runs, estimated steps `18000`, no training processes, no checkpoints.

---

### Task 6: Execute screening and select one arm

**Files:**
- Generate: `checkpoints/gain_calibration/screen/**`
- Generate: `reports/generated/gain_calibration_screen.json`

- [ ] **Step 1: Record historical artifact hashes**

```powershell
Get-FileHash checkpoints/best.pth, checkpoints/step_30000.pth, checkpoints/full_best_export.ts, checkpoints/full_best_export.onnx -Algorithm SHA256 | ConvertTo-Json | Set-Content reports/generated/gain_calibration_historical_hashes.json
```

- [ ] **Step 2: Run twelve paired screens**

```powershell
& .\.venv\Scripts\python.exe scripts/run_gain_ablation.py --stage screen --arms control,complex_nmse,complex_nmse_sisdr,compressed_complex --seeds 17,29,43
```

Expected: twelve completed `run_summary.json` files or an explicit failed-run record; no duplicate Python training processes.

- [ ] **Step 3: Verify promotion evidence**

```powershell
& .\.venv\Scripts\python.exe -c "import json; p=json.load(open('reports/generated/gain_calibration_screen.json')); assert p['schema_version']==1; assert len(p['runs'])==12; print(p['promoted_arm'])"
```

Expected: one promoted arm. If output is `None`, stop full training, document the negative result, and return to the design's gain-head fallback review.

- [ ] **Step 4: Recheck immutable historical hashes**

Compare current SHA-256 values with `gain_calibration_historical_hashes.json`; any difference is a hard failure requiring investigation before continuing.

---

### Task 7: Full three-seed confirmation and final selection

**Files:**
- Generate: `checkpoints/gain_calibration/full/**`
- Generate: `reports/generated/gain_calibration_full.json`
- Generate: comparison and audit evaluation JSON per seed and baseline.

- [ ] **Step 1: Run promoted arm to 30,000 steps**

Read the exact promoted arm from the screen report:

```powershell
$arm = (Get-Content reports/generated/gain_calibration_screen.json -Raw | ConvertFrom-Json).promoted_arm
if ([string]::IsNullOrWhiteSpace($arm)) { throw 'No promoted arm; full training is forbidden.' }
& .\.venv\Scripts\python.exe scripts/run_gain_ablation.py --stage full --arms $arm --seeds 17,29,43
```

Expected: three completed runs at global step `30000`, each with strict-loadable `best.pth` and `step_30000.pth`.

- [ ] **Step 2: Evaluate frozen baseline and all winner seeds on both slices**

For each checkpoint, run `scripts/evaluate.py` with `(offset,count)=(0,500)` and `(500,500)`. Output names must include arm, seed, and slice. Do not change the slice after seeing results.

- [ ] **Step 3: Select production candidate from validation only**

Among the three winner seeds, require all full acceptance gates. Rank passing seeds by validation SNRi, validation SI-SDRi, then absolute validation gain error. Copy, do not move, the winner to `checkpoints/gain_calibration/best.pth`; record source path and both hashes in `gain_calibration_full.json`.

- [ ] **Step 4: Audit statistical reporting**

Require individual seed values plus mean, population SD, median, range, and paired utterance bootstrap 95% intervals. Label the evidence `bounded engineering evidence`; do not claim powered significance or algorithmic superiority.

---

### Task 8: Export, benchmark, verification, and report

**Files:**
- Generate: `checkpoints/gain_calibration/export.ts`
- Generate: `checkpoints/gain_calibration/export.onnx`
- Generate: `checkpoints/gain_calibration/export.json`
- Generate: `reports/generated/gain_calibration_export_parity.json`
- Generate: `reports/generated/gain_calibration_benchmark.json`
- Generate: `reports/generated/gain_calibration_verify.json`
- Create: `reports/gain_calibration_report.md`

- [ ] **Step 1: Export selected checkpoint**

```powershell
& .\.venv\Scripts\python.exe scripts/export.py --checkpoint checkpoints/gain_calibration/best.pth --output checkpoints/gain_calibration/export
```

Expected: TorchScript, ONNX, and metadata bind to the selected checkpoint SHA-256.

- [ ] **Step 2: Benchmark 500 streaming hops**

```powershell
& .\.venv\Scripts\python.exe scripts/benchmark.py --checkpoint checkpoints/gain_calibration/best.pth --iterations 500 --warmup 50 --device cuda --output reports/generated/gain_calibration_benchmark.json
```

Expected: finite core and end-to-end latency; streaming p95 below `10 ms`.

- [ ] **Step 3: Run full verification**

```powershell
& .\.venv\Scripts\python.exe scripts/verify.py --gain-calibration --output reports/generated/gain_calibration_verify.json
```

Expected: compileall, all unit tests, manifests, seed coverage, two-slice metrics, checkpoint hashes, export parity, and benchmark gates pass.

- [ ] **Step 4: Write evidence report**

Create `reports/gain_calibration_report.md` with locked hypothesis, root-cause evidence, exact configs, compute per run and total, failures, screen table, per-seed full table, slice fingerprints, bootstrap method, baseline comparison, export/latency results, limitations, and artifact SHA-256 values. State any failed gate before conclusions.

- [ ] **Step 5: Final completion audit**

Re-run:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
& .\.venv\Scripts\python.exe -m compileall -q src scripts tests
& .\.venv\Scripts\python.exe scripts/verify.py --config configs/train.yaml --output reports/generated/full_verify.json
& .\.venv\Scripts\python.exe scripts/verify.py --gain-calibration --output reports/generated/gain_calibration_verify.json
```

Expected: all commands return `0`; historical and calibrated artifact audits both pass; every objective requirement has direct current-state evidence.
