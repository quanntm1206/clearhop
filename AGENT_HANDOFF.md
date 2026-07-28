# ClearHop Agent Handoff

This file is the zero-context continuation packet for a fresh engineering/research agent. It is intentionally detailed because the originating workstation may be deleted. Treat this branch as a durable handoff snapshot, not as proof that mutable external services are still healthy.

## Snapshot identity

- Project: ClearHop
- Repository: https://github.com/quanntm1206/clearhop
- Handoff branch: `continue`
- Source branch: `main`
- Source commit: `df65abadbc19d8d16ff585a11fe3a023598a4d4e`
- Latest verified release: https://github.com/quanntm1206/clearhop/releases/tag/v0.1.1
- Snapshot date: `2026-07-29 02:48:09 +07:00` (Asia/Saigon)
- License: Apache-2.0
- Public posture: local-only speech denoising; no telemetry; no audio upload.

## Executive state

The approved v0.1.1 scope is complete and publicly released. The repository-level readiness verifier scored GitHub publish readiness 10.0/10, production distribution 10.0/10, and public research evidence 9.0/10. Research is intentionally capped because only DeepFilterNet3 has reproduced local external-model metrics; RNNoise, DTLN, and WebRTC NS remain numerically blocked. Do not convert blocked rows into pass rows without exact, locally reproduced receipts.

Fresh evidence before this handoff:

- Local pytest: 232 passed, 60 warnings, 52 subtests passed.
- `scripts/verify_public_production.py`: pass.
- `scripts/verify_public_research.py`: pass.
- `scripts/verify.py --publish-readiness`: pass.
- Main CI: https://github.com/quanntm1206/clearhop/actions/runs/30391310937
- Release workflow: https://github.com/quanntm1206/clearhop/actions/runs/30391710800
- Release v0.1.1 was downloaded again, checksum-verified, extracted, and denoised a WAV outside the repository.
- The one-command online installer was run into a clean short path using Python 3.11 CPU-only PyTorch; asset hashes and real denoising passed.
- GitHub recognized the repository license as Apache-2.0.
- `main` protection required strict `linux` and `windows` checks, enforced for admins, linear history, resolved conversations, no force pushes, no deletion.

## Product direction

ClearHop targets reproducible causal speech denoising that can be used both as:

1. A Windows-first desktop application for ordinary local WAV cleanup.
2. A Python training/evaluation/export pipeline for controlled research.
3. A CPU-deployable streaming model with hash-bound production evidence.

V1 is deliberately WAV and batch focused. Live microphone processing and broad multi-platform desktop distribution are not part of the released scope.

## User-visible desktop behavior

- Open a WAV or drag/drop it.
- Preview before/after audio.
- Denoise without blocking the UI.
- Save output atomically.
- Process a folder as a batch.
- Show progress, cancellation, and per-file errors.
- Convert supported input to mono 16 kHz explicitly; record conversion in the receipt.
- Keep all processing local.

Stable desktop entrypoints:

- `desktop/app.py`: PySide6 UI, worker orchestration, playback, smoke entrypoint.
- `desktop/pipeline.py`: stable WAV/directory processing contract and receipts.
- `desktop/assets.py`: manifest loading, asset lookup, SHA-256 validation.
- Console script: `noise-reduce-desktop`.

## Signal and inference contract

- Mono PCM WAV output.
- 16,000 Hz model rate.
- 20 ms analysis frame.
- 10 ms hop.
- 161 STFT bins.
- Stateful causal recurrent inference.
- Desktop code reuses `src` inference/streaming logic; do not fork model behavior into the UI.
- Each file must start with reset model state. Cancellation or an invalid file must not leak state into the next file.
- Output writes must be atomic.

Receipts bind at least:

- input and output paths/status;
- input/output SHA-256;
- original and model sample rates;
- mono/resample conversion details;
- input/output sample counts;
- checkpoint SHA-256;
- elapsed time;
- error information when failed.

## Architecture map

```text
WAV/file batch
  -> validation + decode
  -> explicit mono/16 kHz conversion
  -> causal framing/STFT
  -> stateful ClearHop recurrent enhancer
  -> inverse transform / waveform reconstruction
  -> atomic WAV write
  -> hash-bound JSON receipt

Training manifests
  -> deterministic mixture generation
  -> speaker-aware train/val/test split
  -> bounded/full training
  -> validation selection
  -> checkpoint + metadata hashes
  -> TorchScript/ONNX export + parity audit
  -> CPU benchmark + soak
  -> public production/research receipts
```

Important implementation surfaces:

- `src/model.py`: model contract and recurrent state.
- `src/streaming.py`: causal streaming enhancer.
- `src/export.py`: TorchScript/ONNX export and parity surfaces.
- `src/evaluate.py`: objective metrics and optional PESQ handling.
- `scripts/train.py`: training CLI.
- `scripts/enhance_cpu.py`: CPU enhancement CLI.
- `scripts/verify.py`: umbrella verifier and readiness scoring.
- `scripts/verify_public_production.py`: fail-closed public production audit.
- `scripts/verify_public_research.py`: fail-closed public research audit.

## Data and manifests

Raw corpora, generated mixtures, personal audio, exploratory runs, and most checkpoints are intentionally excluded from Git. Deleting the original workstation removes those local bytes. GitHub preserves code, manifests, configs, public receipts, the release checkpoint, and ONNX export; it does not preserve the complete training corpora.

Observed during the completed run:

- A local `train-clean-360` source directory existed (LibriSpeech directory naming).
- Manifest audit counted 171,315 clean files and 11,537 noise files.
- Split files: train 115,694; validation 13,352; test 42,269.
- Speaker counts: train 790; validation 88; test 220.
- Speaker overlap was zero for train/validation, train/test, and validation/test.
- Fold: 0; split seed: 42; segment length: 4.0 seconds.
- Frozen manifest fingerprints:
  - train: `e38622e3e99c6b01d18f0c864f7c331b01d60bf148e1be73dfa2385badad03a6`
  - validation: `6aaf80943520536c1a56788cab089393ffa1845b511a625a317179725f8e500c`
  - test: `8e8e0958a96b28b0492cbc0e17be808c138e355fd65b103fabba04050e8b1cf3`

Tracked source evidence captured at handoff creation:

```text
rg: configs/full.yaml: The system cannot find the file specified. (os error 2)
MODEL_CARD.md:5:ClearHop is a causal 16 kHz speech denoiser using a MobileOne-style encoder and stateful GRU mask/deep-filter path. The selected production model uses beta `0.03`, seed `71`, checkpoint SHA-256 `1305037b59438b1679f6202762001f52f5fb5bd80d6a7ee0e184bc7af46e4789`.
MODEL_CARD.md:22:also binds the registry file SHA-256, tracked manifest path/SHA-256, ordered
MODEL_CARD.md:45:Inference is local-only with no telemetry. Project code is Apache-2.0; datasets and third-party models retain their own licenses.
```

Rules for future dataset work:

- Treat manifests and their fingerprints as the experimental identity.
- Never regenerate a public slice silently.
- Preserve speaker isolation.
- Bind every report to tracked manifest path, file SHA-256, and item-ID fingerprint.
- Do not commit raw audio or generated datasets.
- Re-check every corpus license before redistribution; project Apache-2.0 does not relicense datasets or third-party weights.
- If raw corpora are unavailable after local deletion, reacquire them from their official sources and rebuild manifests deterministically. Compare fingerprints before claiming reproduction.

## Training and research protocol

The repository contains bounded smoke training plus full artifact verification. Public research evidence uses a frozen five-seed protocol. Selection, evaluation, robustness, significance, and failure-analysis receipts are cross-hash bound.

Canonical evidence:

- `reports/public/research_readiness.json`
- `reports/public/model_comparison.json`
- `reports/public/deepfilternet3_reproduction.json`
- `reports/public/rnnoise_build.json`
- `docs/research-comparison.md`
- `MODEL_CARD.md`
- research plan/config files under `configs/`
- frozen slices/manifests under `manifests/`

Frozen historical-baseline result:

- SNR delta: +8.0161 dB; 95% CI [7.6781, 8.3618].
- STOI delta: +0.01064; 95% CI [0.00927, 0.01202].
- SI-SDR delta: +0.02756 dB; 95% CI [-0.04418, 0.09995]; not statistically significant.

DeepFilterNet3 reproduced locally on the same frozen 500-item slice:

- SI-SDRi: 6.390235 dB.
- SNRi: 4.362339 dB.
- STOI: 0.856658.
- PESQ: 2.018783.
- Reproduction explicitly records 16 -> 48 -> 16 kHz conversion, timings, environment, pinned source/archive/checkpoint hashes, and canonical receipt hash.

External baseline state:

- DeepFilterNet3: `reproduced_local`.
- RNNoise: numeric evaluation `blocked`; pinned build recipe and rebuilt-artifact receipt verified.
- DTLN: `blocked`.
- WebRTC NS: `blocked`.
- Third-party weights are not bundled until license/redistribution review passes.
- Missing dependencies must remain `blocked`, never pass.
- Literature/model-card claims must remain separate from reproduced local results.

## Production evidence

Published evidence in `reports/public/production_readiness_verify.json`:

- CPU streaming p95 over 5,000 hops: 3.96217 ms.
- CPU ONNX p95 over 5,000 hops: 2.147705 ms.
- CPU soak: 7,200 seconds, 2,292,681 iterations, 0 failures.
- Production eligibility: pass.

Pinned assets:

- Checkpoint SHA-256: `1305037b59438b1679f6202762001f52f5fb5bd80d6a7ee0e184bc7af46e4789`.
- ONNX SHA-256: `806fb5f6bf23c4e74c781bea5544e63bc5307902f0ec7f4dbd667640853c441b`.
- Release ZIP SHA-256: `e25496a8e8534ff43a5d2f8bdaa06044b5c3d54e49d9c19d09005cbb351e7329`.
- Release ZIP license SHA-256: `313605fbac6945e9324d4825470796b5b7dbc012f523fdc181f6e6fd234eb88f`.

Do not replace an asset without updating the manifest, receipts, release evidence, tests, and version together. Hash mismatch must fail closed.

## Installation and release

One-command Windows install after clone:

```powershell
pwsh -ExecutionPolicy Bypass -File .\scripts\install-desktop.ps1 -Launch
```

Offline install uses the tracked CPU bundle:

```powershell
pwsh -ExecutionPolicy Bypass -File .\scripts\install-desktop.ps1 -Offline -Launch
```

Installer properties:

- Requires Windows, PowerShell 7, Python 3.11.
- Creates an isolated environment.
- Installs pinned CPU-only dependencies.
- Installs `.[desktop]`.
- Downloads release assets or uses `artifacts/cpu_bundle` offline.
- Verifies SHA-256 before activation.
- Creates Start Menu/Desktop shortcuts unless disabled.
- Supports `-DryRun`, `-Offline`, `-NoShortcuts`, and `-Launch`.

Release workflow properties:

- Triggered by immutable version tags.
- Re-checks tag target before publishing.
- Runs full tests and public evidence gates.
- Tests a clean wheel entrypoint.
- Builds PyInstaller `onedir` on Windows.
- Executes a real packaged denoise outside repository cwd.
- Adds the full Apache-2.0 `LICENSE` to the onedir root and verifies its hash.
- Publishes ZIP, checksum, checkpoint, and ONNX assets.

Do not move or overwrite release tags. `v0.1.0` and `v0.1.1` are immutable history. Make a new semantic version for any changed release bytes.

## GitHub history relevant to this handoff

- Repository: https://github.com/quanntm1206/clearhop
- v0.1.0 release: https://github.com/quanntm1206/clearhop/releases/tag/v0.1.0
- v0.1.1 release: https://github.com/quanntm1206/clearhop/releases/tag/v0.1.1
- PR #4 restored the full Apache-2.0 license and added fail-closed SPDX gating: https://github.com/quanntm1206/clearhop/pull/4
- PR #5 bundled and hash-verified the license in the desktop release, bumped 0.1.1, and removed duplicate feature-branch CI work: https://github.com/quanntm1206/clearhop/pull/5
- Main release commit: `df65abadbc19d8d16ff585a11fe3a023598a4d4e`.

## Verification commands

Create/activate an environment compatible with repository constraints, then run:

```powershell
python -m pytest -q
python scripts/verify_public_production.py
python scripts/verify_public_research.py
python scripts/verify.py --publish-readiness
```

Expected repository readiness scores at this snapshot:

```text
GitHub:    10.0
Production: 10.0
Research:   9.0
Status: pass
```

Additional release checks:

```powershell
pwsh -ExecutionPolicy Bypass -File .\scripts\install-desktop.ps1 -DryRun
pwsh -ExecutionPolicy Bypass -File .\scripts\install-desktop.ps1 -Offline -NoShortcuts
```

Headless Qt smoke uses `QT_QPA_PLATFORM=offscreen`. CI is the canonical cross-platform check; never infer Linux/Windows health from a single local OS.

## Known warnings and limitations

- Research is 9/10, not 10/10, because three external numeric baselines remain blocked.
- PyTorch emits TorchScript/legacy ONNX deprecation warnings. These were warnings, not failures; migration to `torch.export` needs controlled parity work, not blind replacement.
- GitHub Actions emitted Node 20 action-runtime deprecation annotations even though jobs passed. Update pinned action SHAs only after verifying upstream releases and rerunning all gates.
- Windows-first packaging is verified; macOS/Linux desktop packages are not released.
- Live microphone denoising is not implemented.
- WAV/batch is the supported V1 surface.
- Raw datasets and historical full training artifacts are not recoverable from GitHub after local deletion.
- External snapshot/archive services may be transient. RNNoise Ubuntu snapshot previously returned HTTP 500/502; retry an unchanged pinned job instead of weakening hashes or silently changing sources.

## High-value continuation goals

Prioritize evidence, not cosmetic score inflation:

1. Reproduce RNNoise numerically on the exact frozen item IDs, recording native rate conversion, latency, memory, CPU percentiles, source/model hashes, and licenses.
2. Reproduce DTLN under the same contract; keep blocked if its dependencies or weights cannot be verified.
3. Reproduce WebRTC NS as an optional classical reference.
4. Raise research readiness from 9 to 10 only when coverage policy legitimately passes; never edit the score directly.
5. Migrate TorchScript/legacy ONNX export carefully with stateful multi-hop parity tests and benchmark comparison.
6. Add code signing and provenance/attestation if public Windows distribution risk justifies it; do not claim signing before an actual trusted certificate pipeline exists.
7. Consider live microphone support only as a separately specified V2 feature with bounded latency, device lifecycle, cancellation, and privacy tests.
8. Consider macOS/Linux packaging only with clean-machine CI evidence and native audio/playback testing.

## Fresh-agent startup checklist

1. Clone the repository and check out `continue`.
2. Read this file, `README.md`, `MODEL_CARD.md`, `docs/research-comparison.md`, and public JSON receipts before reading implementation broadly.
3. Confirm `git rev-parse v0.1.1^{commit}` equals the recorded release commit.
4. Run the four verification commands above.
5. Query current GitHub CI/release status; URLs in this file are historical evidence, not a substitute for a fresh check.
6. Check whether raw datasets/full artifacts exist. If absent, do not promise retraining until official sources are reacquired and fingerprints match.
7. Before changing research outputs, trace every number to a canonical receipt and hash.
8. Work on a new feature branch from the desired base; do not develop directly on `continue` or rewrite immutable release history.
9. Preserve fail-closed behavior: missing, stale, nonfinite, mismatched, or unlicensed evidence must fail or remain blocked.
10. Update this handoff only when durable state changes; place raw logs under ignored artifact directories, not in chat or Git.

## Files intentionally absent from GitHub

The following categories were excluded on purpose and may disappear when the workstation is deleted:

- raw corpora and downloaded dataset archives;
- generated mixtures and dataset caches;
- personal/source audio;
- exploratory notebooks/outputs not promoted to tracked evidence;
- most checkpoints and training run directories;
- temporary PyInstaller/install smoke directories;
- third-party model weights without redistribution approval.

Do not interpret their absence as repository corruption. Recreate them through documented scripts and official sources. Never commit them merely to make a future agent's environment look complete.

## Final integrity rule

A future agent may improve ClearHop, but must not rewrite history, invent benchmark numbers, weaken hash checks, silently change frozen manifests, or mark unavailable baselines as reproduced. Correctness and reproducible evidence outrank a perfect-looking score.