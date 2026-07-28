# MobileDeepFilterNet Gain Calibration Design

Date: 2026-07-26

## Goal

Correct the model's output polarity and gain without sacrificing speech-quality
improvement, causal streaming behavior, export parity, or real-time latency.
Use controlled multi-seed evidence rather than replacing the production
checkpoint after a single favorable run.

## Observed failure

The current objective is raw complex MSE plus `0.5 * SI-SDR`. A deterministic
100-item diagnostic on `fold_0_test` found:

- median enhanced RMS is 58.8 dB below clean;
- the clean-to-enhanced least-squares gain is negative for every item;
- the weighted SI-SDR gradient norm with respect to the enhanced STFT is about
  9,514 times the complex-MSE gradient norm;
- the failure is already present at step 500 and grows through step 30,000.

A fixed scalar estimated from 100 validation items (`-809.4`, 58.16 dB) raises
the 500-item test SNR from -0.008 dB to 7.707 dB while leaving SI-SDRi at
4.163 dB. This is diagnostic evidence that useful waveform shape survived, not
a production solution: the scalar is checkpoint-specific, leaves extremely
small internal values, and is unsafe under distribution shift or quantization.

The mechanism matches the SI-SDR definition: projection removes scale as an
error and therefore does not identify amplitude or polarity. Raw complex MSE
is scale-aware but its gradient is too weak in the current reduction and
weighting.

## Decision

Use loss-only calibration first. Keep the architecture, streaming interface,
state shape, and export interface unchanged.

### Primary loss

Add per-utterance target-energy-normalized complex MSE:

```text
L_nmse = mean_b( sum_f,t,ri (Y - S)^2 / max(sum_f,t,ri S^2, eps) )
```

For `Y = cS`, this term has its unique minimum at `c = 1`; polarity inversion
and attenuation are errors. Per-utterance normalization prevents high-energy
utterances from dominating the batch. SI-SDR remains a logged metric and a
selection guard; it is not required in the primary arm's gradient.

### Controlled alternatives

Screen four arms:

1. `control`: raw complex MSE weight `1.0` plus SI-SDR weight `0.5`.
2. `complex_nmse`: normalized complex MSE weight `1.0`, SI-SDR weight `0.0`;
   this is the primary candidate.
3. `complex_nmse_sisdr`: normalized complex MSE weight `1.0`; SI-SDR weight is
   zero through step 500, then ramps linearly to `0.01` at step 1,000 and stays
   there. This tests shape supervision only after gain is anchored.
4. `compressed_complex`: per-utterance target-normalized compressed magnitude
   and complex terms, compression exponent `0.3`, complex-term weight `0.3`,
   SI-SDR weight `0.0`.

All normalized losses use epsilon `1e-8`. For the compressed arm, define
`M(Z) = sqrt(Z_re^2 + Z_im^2 + eps)` and
`C(Z) = Z * M(Z)^(c - 1)`. The magnitude and complex squared errors are each
divided by their corresponding target energy before the weighted mean. This
keeps the phase-aware term polarity-sensitive and makes the implementation
independent of batch level.

All coefficients, epsilon values, warm-up length, and compression parameters
are explicit config fields embedded in checkpoints. No implicit defaults may
differ between arms.

An explicit causal gain head is deferred. It is allowed only if every loss arm
fails the promotion gate; adding a head changes runtime behavior and introduces
pumping, clipping, and smoothing decisions. Validation-derived fixed gain is
retained only as an oracle diagnostic baseline.

## Code boundaries

- `src/config.py`: validate and serialize loss mode and its parameters.
- `src/trainer.py`: pure loss helpers, configured objective, component logging,
  finite-value checks, and calibration validation metrics.
- `configs/ablations/gain_calibration/`: immutable per-arm/per-seed configs.
- `scripts/run_gain_ablation.py`: bounded runner with dry-run, resume, arm/seed
  filters, collision protection, and machine-readable summaries.
- `src/evaluate.py`: add signed projection gain, gain error in dB, and polarity
  failure rate without changing existing metric keys.
- `scripts/verify.py`: audit provenance, seed coverage, acceptance gates,
  recurrent export parity, and finite metrics for the selected checkpoint.

The original checkpoint and reports remain immutable. New artifacts use a
separate `gain_calibration` namespace.

For calibration metrics, define output projection gain as
`a = <enhanced, clean> / max(||clean||^2, eps)`. Ideal gain is `a = 1`;
polarity failure is `a <= 0`; signed gain error is `20*log10(max(|a|, eps))`.

## Data and experiment protocol

### Locked controls

- Manifests and fingerprints remain unchanged.
- Model architecture, preprocessing, augmentations, batch size, optimizer,
  scheduler, AMP mode, steps, and evaluator remain equal across arms.
- Screening seeds are `17`, `29`, and `43`.
- For a given seed, arms use the same data order and stochastic augmentation
  sequence.
- Validation selects the arm and checkpoint. The test split is evaluated only
  after selection.
- Screening uses the full-run 30,000-step learning-rate schedule and stops via
  an independent `max_steps=1500`; it must not compress OneCycle into 1,500
  steps. A final validation pass runs at the exact stop.
- Failed, divergent, and interrupted runs remain in the report.

### Stage 1: screening

Run four arms by three seeds for 1,500 optimizer steps. Estimated cost is about
1.34 GPU-hours from the measured 30,000-step runtime.

Promote exactly one arm only if all three seeds are finite, have positive signed
gain, improve mean validation SNRi by at least 1.0 dB over paired controls, lose
no more than 0.3 dB SI-SDRi, and lose no more than 0.005 STOI. Rank eligible
arms by SNRi, SI-SDRi, then absolute gain error. If no arm passes, stop and
revisit the mechanism; do not spend the full-run budget.

### Stage 2: full confirmation

Train the promoted arm for 30,000 steps on all three seeds. Estimated total
screen-plus-confirm cost is about 8 GPU-hours. Compare every seed with the
frozen current checkpoint on the identical first 500 test entries. Because
those entries were already used to diagnose the historical failure, also run a
predeclared untouched audit on entries 500 through 999. Record offset, count,
ordered-entry fingerprint, and full manifest fingerprint for both slices.
This supports a product-improvement claim, not algorithmic superiority, because
the historical full baseline has only one training seed.

Report each seed, mean, standard deviation, median, and range. Use paired
utterance bootstrap intervals for checkpoint-level metric differences; do not
present them as a substitute for training-seed uncertainty.

## Acceptance gates

Every selected full-run seed must satisfy the metric gates on both the original
comparison slice and the untouched audit slice:

- held-out SNRi greater than 0 dB;
- mean SI-SDRi no more than 0.3 dB below the frozen baseline on that slice
  (`3.8629 dB` minimum on the original comparison slice);
- enhanced STOI no more than 0.005 below the frozen baseline on that slice
  (`0.7740` minimum on the original comparison slice);
- enhanced PESQ no more than 0.05 below the frozen baseline on that slice
  (`1.5156` minimum on the original comparison slice);
- positive median signed projection gain;
- median absolute gain error no greater than 3 dB;
- polarity failure rate below 1%;
- streaming p95 below 10 ms on the current benchmark host;
- finite eager, streaming, TorchScript, and ONNX outputs;
- 20-hop recurrent export parity within the existing tolerance;
- all repository tests and artifact verification pass.

The production candidate is selected by validation performance only. Failure of
any acceptance gate is reported honestly and leaves the current artifact set as
the production reference.

## Artifacts

- `reports/generated/gain_calibration_screen.json`
- `reports/generated/gain_calibration_full.json`
- `reports/generated/gain_calibration_export_parity.json`
- `reports/generated/gain_calibration_benchmark.json`
- `reports/generated/gain_calibration_verify.json`
- `reports/gain_calibration_report.md`
- `checkpoints/gain_calibration/best.pth`
- `checkpoints/gain_calibration/export.ts`
- `checkpoints/gain_calibration/export.onnx`
- `checkpoints/gain_calibration/export.json`

Every report binds config, manifest, checkpoint, export, device, and software
metadata by content hash where applicable. Evaluation reports also bind slice
offset, count, and ordered-entry fingerprint.

## Error handling and safety

- Reject unknown loss modes and invalid coefficients before allocating the
  dataset or GPU.
- Clamp only denominators and fractional-power magnitudes with documented
  epsilon values; reject non-finite component or total losses immediately.
- Preserve resumable optimizer, scheduler, scaler, global-step, seed, arm, and
  provenance state.
- Refuse to overwrite a completed arm/seed unless an explicit force flag is
  supplied.
- Support dry-run planning and bounded arm/seed selection.
- Never overwrite the historical `best.pth` or its generated reports.

## Testing

Unit tests cover exact loss values for identity, attenuation, amplification,
polarity inversion, silence, batched level imbalance, finite gradients, config
validation, metric aggregation, and artifact schemas. Integration tests cover a
tiny deterministic training run, resume equivalence, arm isolation, dry-run,
and strict checkpoint loading. Existing causal filtering, streaming, export,
and verification tests remain mandatory.

## Evidence and limitations

- SI-SDR scale behavior: Le Roux et al., ICASSP 2019,
  https://arxiv.org/abs/1811.02508
- Level-normalized spectral losses: Braun and Tashev, SPECOM 2020,
  https://arxiv.org/abs/2008.06412
- Phase-aware compressed spectral loss: Schröter et al., ICASSP 2022,
  https://arxiv.org/abs/2110.05588
- Seed variability: Reimers and Gurevych, EMNLP 2017,
  https://aclanthology.org/D17-1035/
- Statistical reporting with few runs: Agarwal et al., NeurIPS 2021,
  https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html

The short-run ranking may not predict 30,000-step ranking, three seeds do not
constitute a powered statistical study, and the current benchmark proves only
performance on the local GPU. These limitations remain explicit in the final
report.
