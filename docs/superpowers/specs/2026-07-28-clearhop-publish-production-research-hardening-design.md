# ClearHop Publish, Production, and Research Hardening Design

**Repository:** `https://github.com/quanntm1206/clearhop`

**Objective:** Raise GitHub publish readiness, production distribution, and
public research evidence to at least 9/10 using executable acceptance gates.

## Scope

ClearHop remains a Windows-first, local-only causal speech denoiser. V1 accepts
WAV files and folders, converts inputs explicitly to mono/16 kHz, runs the
existing causal model, writes output atomically, and emits a hash-bound receipt.
No microphone mode, telemetry, audio upload, paid API, or unlicensed third-party
weight redistribution is added.

## Distribution Architecture

Production inference helpers move from `scripts/` into an installable runtime
module under `src/`. Both the desktop pipeline and CLI scripts call that module;
scripts no longer form a runtime dependency. The wheel contains `src` and
`desktop`, while research tooling may remain an executable script when it is not
needed by the installed application.

The installer supports online release assets and the checked local CPU bundle.
Every asset is versioned and SHA-256 verified before shortcuts are activated.
Release builds use PyInstaller onedir. The release workflow must run tests,
production/research/publish verifiers, build the application, start the packaged
executable outside the repository working directory, package it, and publish the
ZIP, checksum, checkpoint, and ONNX asset only after all gates pass.

## GitHub Surface

The public project name is ClearHop. Package metadata, desktop title, README,
`CITATION.cff`, release URLs, and security documentation use
`quanntm1206/clearhop`. Repository hygiene excludes datasets, archives,
checkpoints outside the curated CPU bundle, build products, debug logs, and
exploratory root artifacts.

Evidence is split into two classes:

- Private/generated evidence stays under ignored `reports/generated/`.
- Curated immutable public receipts live under `reports/public/` and are tracked.

README links target only public receipts. A release inventory gate rejects large
or secret-like files outside an allowlist. GitHub CI runs on Windows and Linux.
Windows CI performs an actual wheel install and offscreen open/process/save
desktop smoke, not an optional-import substitute. The public screenshot must be
captured from the rendered Windows UI and pass visual review before use.

## Public Research Evidence

All compared systems receive identical manifest item IDs and slice definitions.
Native sample rate, resampling method, algorithmic latency, runtime command,
source commit, weight hash, license, hardware, dependency environment, and raw
timings are recorded. Metrics remain SI-SDRi, SNRi, STOI, PESQ when available,
CPU p50/p95/p99, memory, and realtime factor.

The target is at least two reproduced external baselines from DeepFilterNet3,
RNNoise, DTLN, or WebRTC NS, selected only after source and weight license review.
Third-party artifacts are cached outside Git and are never redistributed unless
their license explicitly permits it. A failed or unavailable runtime is
`blocked` with command/error evidence. Literature/model-card claims stay in a
separate table and never substitute for local measurements.

Public research readiness reaches 9/10 only when:

1. Current ClearHop and frozen historical results are hash-bound.
2. At least two external baselines are reproduced on the same public protocol,
   or one is reproduced and a second has a verified platform/license blocker
   plus a reproducible container/CI recipe.
3. Metrics, latency, manifests, versions, and licenses pass schema validation.
4. The report separates reproduced, literature-only, and blocked results.

## Verification Gates

The existing `scripts/verify.py` gains a publish-readiness audit rather than a
parallel verifier. It checks installable wheel contents, public evidence links,
release/version consistency, asset hashes, repository inventory, research
comparison status, and required documentation. Cross-platform doctor wrappers
invoke this audit because a new readiness gate must be exercised by the project
health command.

Required acceptance sequence:

1. Focused tests fail before each implementation change, then pass.
2. Full Python test suite passes.
3. Production and research readiness verifiers pass.
4. Wheel contains no runtime import outside packaged modules.
5. Clean Windows wheel install denoises a real fixture and writes a receipt.
6. Offline installer performs a real isolated install and launch smoke.
7. PyInstaller artifact launches outside repository cwd.
8. GitHub workflow syntax and release inventory validate locally.
9. Public evidence links resolve to tracked files.
10. Rendered desktop screenshot passes visual inspection.

## Failure Handling

No unavailable dependency is converted to pass. Release jobs fail closed on
hash mismatch, missing evidence, untracked README targets, wheel runtime import
failure, or packaged launch failure. Clean-install tests use disposable paths.
Audio fixtures contain generated silence/tone only. Destructive cleanup is
restricted to paths created by the test itself.

## Success Rubric

- GitHub publish readiness: at least 9/10; public inventory, docs, evidence, and
  CI checks all pass.
- Production distribution: at least 9/10; clean install and packaged denoise
  smoke pass on Windows.
- Public research evidence: at least 9/10; controlled comparison and provenance
  requirements pass without substituting literature claims.

Residual non-blocking limits may include unsigned Windows binaries and external
baselines whose redistribution is prohibited. Both must be disclosed clearly.
