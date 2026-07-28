# Gain calibration - final report

## Result

- Selected arm: `complex_nmse_sisdr_beta_0p02`.
- Selected seed: `29` (validation-only ranking).
- Selected checkpoint: `checkpoints/gain_calibration/best.pth`.
- Production status: **rejected**; `production_eligible=false`.
- Reason: seed `43` fails the audit `si_sdri_delta` gate only. Its SI-SDR improvement is `3.1233 dB`; baseline `3.4513 dB`; allowed floor `3.1513 dB`.

## Evidence

- Full training: 3 seeds, `30,000` scheduled steps each; all run artifacts complete.
- Seed 17: comparison/audit pass. Validation SNRi `3.9915 dB`, SI-SDRi `3.5755 dB`, STOI `0.7923`.
- Seed 29: comparison/audit pass. Validation SNRi `4.0242 dB`, SI-SDRi `3.6108 dB`, STOI `0.7902`.
- Seed 43: comparison pass; audit fails SI-SDR gate. Validation SNRi `3.9447 dB`, SI-SDRi `3.5434 dB`, STOI `0.7891`.
- Export parity: pass, 20 recurrent hops; TorchScript max error `0`; ONNX max error `1.43e-6`.
- Benchmark: 500 CUDA iterations on `NVIDIA GeForce RTX 5060 Ti`; neural p95 `1.6951 ms`; streaming p95 `2.4231 ms`; realtime factor `4.91x`.
- Full tests: `122` tests pass.

## Verifier

- `reports/generated/gain_calibration_verify.json` records all artifact, hash, manifest, export, and selection checks.
- Exit status `1` is intentional: `acceptance_gates=false`, `all_seed_acceptance=false`; verifier does not promote a rejected model.
- Verifier now accepts the immutable refinement binding and beta-`0.02` loss provenance, while retaining strict production gates.

## Next improvement

- No promotion claimed. Further training requires a new experiment; current evidence does not justify changing the frozen production decision.
