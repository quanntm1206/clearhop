# ClearHop model card

## Model details

ClearHop is a causal 16 kHz speech denoiser using a MobileOne-style encoder and stateful GRU mask/deep-filter path. The selected production model uses beta `0.03`, seed `71`, checkpoint SHA-256 `1305037b59438b1679f6202762001f52f5fb5bd80d6a7ee0e184bc7af46e4789`.

## Intended use

Local enhancement of mono or automatically converted WAV speech on CPU. V1 does not support live microphone input, speaker separation, dereverberation guarantees, forensic restoration, or safety-critical communication.

## Inputs and outputs

Input is WAV. The desktop pipeline explicitly converts to mono/16 kHz. Output is a WAV plus JSON receipt containing input/output hashes, conversion metadata, checkpoint hash, elapsed time, and status.

## Evaluation

Reproduced results and hardware-bound latency appear only in `reports/public/`. Literature claims remain separate in `docs/research-comparison.md`. External baselines without an executable, license-reviewed setup remain `blocked`.

The machine-readable comparison receipt is schema v2 and binds each baseline to
`configs/research_baselines.json` (source URL, pinned commit/version, native
sample rate, license, weight hash status, command/recipe, and environment).
DeepFilterNet3, RNNoise, DTLN, and WebRTC NS are currently `blocked`: this
Windows environment has no configured third-party runtime or redistribution-
reviewed weights. No external metric is claimed. A baseline may be promoted
only after rerunning the same item IDs with a pinned executable, dependency
lock, model SHA-256, and captured latency receipt.

## Limitations

Training/evaluation corpora may not represent every language, microphone, room, or noise type. SI-SDR improvement versus the frozen baseline was not statistically significant. Third-party weights are not redistributed.

## Privacy and license

Inference is local-only with no telemetry. Project code is Apache-2.0; datasets and third-party models retain their own licenses.
