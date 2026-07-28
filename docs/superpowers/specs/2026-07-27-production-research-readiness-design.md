# Production + Research Readiness Design

Date: 2026-07-27
Scope: Windows desktop, CPU-only real-time inference; reproducible paper-quality study.

## Research question

Can the gain-calibration enhancement pipeline deliver a statistically reliable improvement over the frozen historical baseline while meeting a strict CPU real-time contract across representative noise, SNR, speaker, and stream-duration conditions?

Current evidence is promising but insufficient: the selected `complex_nmse_sisdr_beta_0p02` run improves quality, yet one of three full-training seeds fails the audit SI-SDR gate. The existing verifier correctly refuses promotion.

## Decision rules

Production-ready requires every production gate to pass:

- CPU-only streaming p95 latency `< 10 ms` per 10 ms hop on the declared reference machine.
- No NaN/Inf, shape drift, state leak, or unbounded memory during a 2-hour soak.
- Eager/TorchScript/ONNX outputs remain within the existing numerical tolerance.
- Repeated start/stop, underrun, malformed input, and device-loss paths fail closed and recover without process restart where supported.
- Versioned model/config/runtime bundle reproduces the verified hashes.

Research-ready requires every research gate to pass:

- Frozen baseline, candidate, manifests, preprocessing, and decision thresholds.
- At least five independent seeds for the final candidate and baseline-comparable evaluation.
- Paired bootstrap 95% confidence intervals, effect sizes, and a predeclared paired significance test.
- Main comparison plus OOD robustness slices: SNR bands, noise families, speaker groups, and duration/stream-position slices.
- Controlled ablations isolating loss terms and calibration/refinement choices.
- Error taxonomy with saved worst-case examples and qualitative listening artifacts.
- Reproduction manifest: configs, seed list, hashes, environment, commands, and raw metric tables.

Promotion is conjunctive. A metric average cannot override a failed seed, subgroup, latency, or integrity gate.

## Architecture

### 1. Contract layer

Add explicit schemas for production and research receipts under `reports/generated/`:

- `production_readiness.json`
- `research_readiness.json`
- `cpu_benchmark.json`
- `robustness_matrix.json`
- `significance.json`
- `failure_analysis.json`

Each receipt stores schema version, source artifact hashes, command/config hashes, manifest fingerprints, environment metadata, thresholds, raw measurements, and a deterministic `status`.

### 2. CPU production harness

Reuse the existing stateful streaming wrapper and export contract. Add a CPU benchmark driver that:

- Pins one process and records CPU model, OS, Python/runtime versions, thread settings, and warmup.
- Measures per-hop neural and end-to-end latency over at least 5000 hops.
- Reports p50/p95/p99/max, realtime factor, allocations, and RSS delta.
- Runs deterministic parity against eager inference for fixed random inputs.
- Runs soak scenarios with reset, repeated stream boundaries, short buffers, and invalid-input rejection.

The harness must not silently fall back to CUDA. Missing CPU prerequisites produce `blocked`, not `pass`.

### 3. Research evaluation layer

Extend the existing evaluator without changing the frozen primary contract:

- Keep comparison/audit slices immutable.
- Add deterministic slice generation keyed by manifest fingerprint and a checked-in slice specification.
- Emit per-item metrics, aggregate metrics, subgroup aggregates, and worst-case item IDs.
- Compute paired bootstrap CIs and effect sizes from the same ordered item pairs.
- Record all seeds, including rejected seeds; never discard failures before aggregation.

### 4. Experiment ladder

Run in this order:

1. Reproduce the current selected arm and historical baseline.
2. Evaluate CPU production gate on the selected export.
3. Run robustness and failure analysis to identify the dominant failure mode.
4. Execute only targeted ablations justified by that failure mode.
5. Train the final candidate and baseline comparison with five seeds.
6. Freeze receipts, run both verifiers, then package the Windows CPU runtime.

### 5. Packaging

Produce a versioned CPU bundle containing ONNX, metadata, config, hashes, a minimal CLI smoke test, and rollback-safe model replacement. No production label is emitted unless the production receipt is green.

## Testing and acceptance

- Existing full suite remains green.
- New unit tests cover receipt schemas, hash binding, deterministic slice fingerprints, bootstrap determinism, invalid-input handling, and threshold boundaries.
- Integration tests run the CPU harness against the exported model and a tiny fixture.
- A negative-fixture test proves that one failed seed, one failed subgroup, CPU fallback, or hash drift blocks promotion.
- Final command set is recorded in the report and rerunnable offline.

## Risks and limits

- CPU latency depends on the declared reference machine; the receipt must bind the exact hardware and thread settings.
- PESQ/listening quality may be unavailable in minimal environments; missing metrics remain `blocked`, never imputed.
- Five-seed training may still be underpowered for publication claims; report uncertainty and avoid causal claims beyond the controlled ablations.
- No new architecture is justified until failure analysis identifies a mechanism; this prevents metric chasing.

## Deliverables

- Design and implementation plan.
- CPU benchmark and soak receipts.
- Robustness/significance/failure-analysis receipts.
- Five-seed final training/evaluation bundle.
- Production and research readiness reports with explicit pass/fail reasons.
