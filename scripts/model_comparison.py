"""Reproducible, license-safe model comparison protocol.

The external adapters intentionally do not ship weights.  A missing runtime or
unconfigured command is represented as ``blocked``; it can never be scored as
a passing model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Callable, Iterable, Sequence

import numpy as np

try:  # scipy is optional for report-only / blocked comparisons
    from scipy.signal import resample_poly as _scipy_resample_poly
except Exception:  # pragma: no cover - exercised on minimal installs
    _scipy_resample_poly = None


def canonical_receipt_sha256(payload: dict) -> str:
    """Hash a receipt canonically while excluding its embedded self-hash."""
    body = {key: value for key, value in payload.items() if key != "receipt_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_tracked(root: Path, relative: str) -> bool:
    try:
        completed = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return False
    return completed.returncode == 0 and relative in completed.stdout.splitlines()


METRIC_NAMES = ("snri_db", "si_sdri_db", "stoi", "pesq")
DEFAULT_NATIVE_RATES = {"DeepFilterNet3": 48000, "RNNoise": 48000, "DTLN": 16000, "WebRTC NS": 16000}


def load_research_baselines(path: Path) -> dict:
    """Load and validate the immutable baseline provenance registry."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("baselines"), list):
        raise ValueError("research baseline registry must use schema_version=1 with baselines")
    names: set[str] = set()
    required = {"name", "source_url", "license", "native_sample_rate", "provenance", "recipe"}
    for row in payload["baselines"]:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError("baseline row missing required provenance fields")
        if row["name"] in names:
            raise ValueError(f"duplicate baseline name: {row['name']}")
        names.add(str(row["name"]))
        if not str(row["source_url"]).startswith("https://"):
            raise ValueError(f"baseline source URL must be HTTPS: {row['name']}")
        if int(row["native_sample_rate"]) <= 0 or not row["license"]:
            raise ValueError(f"invalid baseline contract: {row['name']}")
        provenance = row["provenance"]
        for key in ("commit", "commit_url", "version", "retrieved_at", "weight_sha256"):
            if key not in provenance:
                raise ValueError(f"baseline provenance missing {key}: {row['name']}")
    return payload


@dataclass(frozen=True)
class BenchmarkItem:
    item_id: str
    noisy: np.ndarray
    clean: np.ndarray
    sample_rate: int
    slice_name: str = "primary"


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    native_sample_rate: int = 16000
    source_url: str | None = None
    license: str | None = None
    runtime: str | None = None
    weights_bundled: bool = False


class ModelAdapter:
    spec: AdapterSpec

    def availability(self) -> tuple[bool, str | None]:
        return True, None

    def reset(self) -> None:
        return None

    def enhance(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        raise NotImplementedError


class ArrayAdapter(ModelAdapter):
    """Small in-process adapter used by the protocol and unit tests."""

    def __init__(self, name: str, fn: Callable[[np.ndarray, int], np.ndarray], *, native_sample_rate: int = 16000):
        self.spec = AdapterSpec(name=name, native_sample_rate=native_sample_rate, runtime="in-process")
        self._fn = fn

    def enhance(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return np.asarray(self._fn(audio, sample_rate), dtype=np.float32)


class CommandAdapter(ModelAdapter):
    """Adapter for an explicitly configured third-party CLI.

    ``command`` may contain ``{input}``, ``{output}``, and ``{sample_rate}``
    placeholders. No shell is used, preventing accidental command injection.
    """

    def __init__(self, spec: AdapterSpec, command: Sequence[str] | None = None):
        self.spec = spec
        self.command = tuple(str(part) for part in command) if command else None

    def availability(self) -> tuple[bool, str | None]:
        if not self.command:
            return False, "runtime command not configured; third-party weights are not bundled"
        executable = shutil.which(self.command[0])
        if executable is None:
            return False, f"runtime executable not found: {self.command[0]}"
        return True, None

    def enhance(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        available, reason = self.availability()
        if not available:
            raise RuntimeError(reason or "adapter unavailable")
        import soundfile as sf

        with tempfile.TemporaryDirectory(prefix="noise_reduce_compare-") as tmp:
            root = Path(tmp)
            input_path, output_path = root / "input.wav", root / "output.wav"
            sf.write(str(input_path), np.asarray(audio, dtype=np.float32), sample_rate, subtype="FLOAT")
            command = [part.format(input=str(input_path), output=str(output_path), sample_rate=sample_rate) for part in self.command]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"{self.spec.name} exited {completed.returncode}: {completed.stderr[-500:]}")
            if not output_path.is_file():
                raise RuntimeError(f"{self.spec.name} did not create output WAV")
            output, output_rate = sf.read(str(output_path), dtype="float32", always_2d=False)
            if np.asarray(output).ndim > 1:
                output = np.asarray(output).mean(axis=1)
            return _resample(np.asarray(output, dtype=np.float32), int(output_rate), sample_rate)


class DeepFilterNet3Adapter(CommandAdapter):
    def __init__(self, command: Sequence[str] | None = None, *, native_sample_rate: int = 48000):
        super().__init__(AdapterSpec("DeepFilterNet3", native_sample_rate, "https://github.com/Rikorose/DeepFilterNet", "Apache-2.0 OR MIT", "external-cli"), command)


class RNNoiseAdapter(CommandAdapter):
    def __init__(self, command: Sequence[str] | None = None, *, native_sample_rate: int = 48000):
        super().__init__(AdapterSpec("RNNoise", native_sample_rate, "https://github.com/xiph/rnnoise", "BSD-3-Clause", "external-cli"), command)


class DTLNAdapter(CommandAdapter):
    def __init__(self, command: Sequence[str] | None = None, *, native_sample_rate: int = 16000):
        super().__init__(AdapterSpec("DTLN", native_sample_rate, "https://github.com/breizhn/DTLN", "MIT", "external-cli"), command)


class WebRTCNSAdapter(CommandAdapter):
    def __init__(self, command: Sequence[str] | None = None, *, native_sample_rate: int = 16000):
        super().__init__(AdapterSpec("WebRTC NS", native_sample_rate, "https://webrtc.org/", "BSD-3-Clause", "external-cli"), command)


class FrozenHistoricalAdapter(ModelAdapter):
    """Import an existing local evaluation receipt without re-running weights."""

    def __init__(self, receipt_path: Path, manifest_path: Path, *, name: str = "Frozen historical baseline"):
        self.receipt_path = Path(receipt_path)
        self.manifest_path = Path(manifest_path)
        self.spec = AdapterSpec(name, 16000, "local receipt", "project data", "receipt", False)

    def availability(self) -> tuple[bool, str | None]:
        if not self.receipt_path.is_file():
            return False, f"receipt not found: {self.receipt_path}"
        if not self.manifest_path.is_file():
            return False, f"manifest not found: {self.manifest_path}"
        try:
            self._load()
        except Exception as exc:
            return False, f"invalid historical receipt: {exc}"
        return True, None

    def _load(self) -> tuple[dict, list[str]]:
        payload = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        if not isinstance(payload.get("enhanced"), dict):
            raise ValueError("missing enhanced aggregate")
        offset = int(metadata.get("slice_offset", 0))
        count = int(metadata.get("slice_count", 0))
        ids = []
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines()[offset : offset + count]:
            if line.strip():
                row = json.loads(line)
                if "id" not in row:
                    raise ValueError("manifest row missing id")
                ids.append(str(row["id"]))
        if count and len(ids) != count:
            raise ValueError("manifest slice shorter than receipt")
        return payload, ids

    def summary_result(self, item_ids: list[str]) -> dict:
        payload, expected_ids = self._load()
        if expected_ids and expected_ids != item_ids:
            raise ValueError("manifest item IDs differ from historical receipt")
        enhanced = payload["enhanced"]

        def aggregate(name: str, nested: bool = False):
            value = enhanced.get(name)
            if nested and isinstance(value, dict):
                value = value.get("mean")
            return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else None

        metrics = {
            "snri_db": _metric(aggregate("snr_improvement_mean"), "receipt"),
            "si_sdri_db": _metric(aggregate("si_sdr_improvement_mean"), "receipt"),
            "stoi": _metric(aggregate("stoi", True), "receipt"),
            "pesq": _metric(aggregate("pesq", True), "receipt"),
        }
        try:
            receipt_ref = self.receipt_path.resolve().relative_to(self.manifest_path.resolve().parents[2]).as_posix()
        except (IndexError, ValueError):
            receipt_ref = self.receipt_path.name
        receipt_sha256 = hashlib.sha256(self.receipt_path.read_bytes()).hexdigest()
        return {
            "status": "reproduced_local",
            "item_ids": item_ids,
            "metrics": metrics,
            "sample_rate": {"reference": 16000, "native": 16000, "input": {"from": 16000, "to": 16000}, "method": "none"},
            "source": {
                "receipt": receipt_ref,
                "receipt_sha256": receipt_sha256,
                "checkpoint_sha256": payload.get("metadata", {}).get("checkpoint_sha256"),
            },
            "latency": {"available": False, "reason": "historical receipt has no per-item timing"},
        }

    def enhance(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:  # pragma: no cover - receipt-only adapter
        raise RuntimeError("historical adapter is receipt-only")


class ReproducedExternalAdapter(FrozenHistoricalAdapter):
    """Import a public external-baseline receipt without loading its weights."""

    def __init__(self, receipt_path: Path, *, name: str):
        self.receipt_path = Path(receipt_path)
        self.manifest_path = self.receipt_path
        self.spec = AdapterSpec(name, DEFAULT_NATIVE_RATES.get(name, 16000), runtime="external-receipt")

    def _load(self) -> tuple[dict, list[str]]:
        payload = json.loads(self.receipt_path.read_text(encoding="utf-8"))
        if payload.get("receipt_sha256") != canonical_receipt_sha256(payload):
            raise ValueError("external receipt canonical hash mismatch")
        row = payload.get("model")
        if payload.get("schema_version") != 1 or payload.get("status") != "reproduced_local" or not isinstance(row, dict):
            raise ValueError("external receipt is not a reproduced_local schema-v1 result")
        if row.get("name") != self.spec.name or row.get("status") != "reproduced_local":
            raise ValueError("external receipt model identity/status mismatch")
        ids = row.get("item_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError("external receipt has no item IDs")
        return payload, [str(item_id) for item_id in ids]

    def summary_result(self, item_ids: list[str]) -> dict:
        payload, expected_ids = self._load()
        if expected_ids != item_ids:
            raise ValueError("manifest item IDs differ from external receipt")
        row = json.loads(json.dumps(payload["model"], allow_nan=False))
        receipt_path = self.receipt_path.as_posix()
        marker = "/reports/public/"
        if marker in receipt_path:
            receipt_path = "reports/public/" + receipt_path.split(marker, 1)[1]
        row["source_receipt"] = {
            "path": receipt_path,
            "sha256": payload["receipt_sha256"],
        }
        return row


def _metric(value: float | None, source: str, reason: str | None = None) -> dict:
    payload = {"available": value is not None, "value": value, "source": source}
    if value is None:
        payload["reason"] = reason or "metric unavailable"
    return payload


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return np.asarray(audio, dtype=np.float32)
    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if _scipy_resample_poly is not None:
        gcd = math.gcd(int(source_rate), int(target_rate))
        return np.asarray(_scipy_resample_poly(audio, target_rate // gcd, source_rate // gcd), dtype=np.float32)
    # Report-only environments may not install scipy; linear interpolation keeps
    # the adapter contract usable while production installs use polyphase.
    positions = np.linspace(0.0, max(0, len(audio) - 1), max(1, round(len(audio) * target_rate / source_rate)))
    return np.interp(positions, np.arange(len(audio), dtype=np.float64), audio.astype(np.float64)).astype(np.float32)


def _si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = reference.astype(np.float64) - float(np.mean(reference))
    est = estimate.astype(np.float64) - float(np.mean(estimate))
    target = np.dot(est, ref) / (np.dot(ref, ref) + 1e-8) * ref
    error = est - target
    return float(10.0 * math.log10((np.dot(target, target) + 1e-8) / (np.dot(error, error) + 1e-8)))


def _snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    error = estimate.astype(np.float64) - reference.astype(np.float64)
    return float(10.0 * math.log10((np.mean(reference.astype(np.float64) ** 2) + 1e-12) / (np.mean(error**2) + 1e-12)))


def _optional_metric(name: str, clean: np.ndarray, output: np.ndarray, sr: int) -> float | None:
    try:
        if name == "stoi":
            from pystoi.stoi import stoi

            return float(stoi(clean, output, sr, extended=False))
        if name == "pesq":
            from pesq import pesq

            mode = "wb" if sr == 16000 else "nb" if sr == 8000 else None
            return float(pesq(sr, clean, output, mode)) if mode else None
    except Exception:
        return None
    return None


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {name: float(np.quantile(arr, quantile)) for name, quantile in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))}


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_model_comparison(
    items: Iterable[BenchmarkItem],
    adapters: Sequence[ModelAdapter],
    *,
    include_optional_metrics: bool = True,
    clock: Callable[[], float] = time.perf_counter,
) -> dict:
    rows = list(items)
    if not rows:
        raise ValueError("comparison requires at least one manifest item")
    item_ids = [str(item.item_id) for item in rows]
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("manifest item IDs must be unique")
    models: list[dict] = []
    for adapter in adapters:
        available, reason = adapter.availability()
        if isinstance(adapter, FrozenHistoricalAdapter):
            if not available:
                models.append(_blocked_row(adapter, item_ids, reason))
                continue
            try:
                row = adapter.summary_result(item_ids)
            except Exception as exc:
                models.append(_failed_row(adapter, item_ids, str(exc)))
                continue
            row["name"] = adapter.spec.name
            row["spec"] = _spec_dict(adapter.spec)
            models.append(row)
            continue
        if not available:
            models.append(_blocked_row(adapter, item_ids, reason))
            continue

        timings: list[float] = []
        rtfs: list[float] = []
        rows_metrics: list[dict[str, float]] = []
        latency_items: list[dict] = []
        memory_peak = 0
        error: str | None = None
        tracemalloc.start()
        try:
            for item in rows:
                noisy = np.asarray(item.noisy, dtype=np.float32).reshape(-1)
                clean = np.asarray(item.clean, dtype=np.float32).reshape(-1)
                if not np.all(np.isfinite(noisy)) or not np.all(np.isfinite(clean)):
                    raise ValueError(f"non-finite input for item {item.item_id}")
                native = int(adapter.spec.native_sample_rate)
                native_noisy = _resample(noisy, int(item.sample_rate), native)
                adapter.reset()
                start = clock()
                native_output = np.asarray(adapter.enhance(native_noisy, native), dtype=np.float32).reshape(-1)
                elapsed = max(0.0, float(clock() - start))
                if not np.all(np.isfinite(native_output)):
                    raise ValueError(f"non-finite output for item {item.item_id}")
                output = _resample(native_output, native, int(item.sample_rate))
                if output.size < clean.size:
                    output = np.pad(output, (0, clean.size - output.size))
                output = output[: clean.size]
                snr_noisy, snr_output = _snr(clean, noisy[: clean.size]), _snr(clean, output)
                sdr_noisy, sdr_output = _si_sdr(clean, noisy[: clean.size]), _si_sdr(clean, output)
                metric_row: dict[str, float] = {"snri_db": snr_output - snr_noisy, "si_sdri_db": sdr_output - sdr_noisy}
                if include_optional_metrics:
                    for name in ("stoi", "pesq"):
                        value = _optional_metric(name, clean, output, int(item.sample_rate))
                        if value is not None:
                            metric_row[name] = value
                if not all(math.isfinite(float(value)) for value in metric_row.values()):
                    raise ValueError(f"non-finite metric for item {item.item_id}")
                rows_metrics.append(metric_row)
                duration = max(float(clean.size) / float(item.sample_rate), 1e-9)
                timings.append(elapsed * 1000.0)
                rtfs.append(elapsed / duration)
                latency_items.append({"item_id": str(item.item_id), "elapsed_ms": elapsed * 1000.0, "realtime_factor": elapsed / duration})
            _, peak = tracemalloc.get_traced_memory()
            memory_peak = int(peak)
        except Exception as exc:
            error = str(exc)
        finally:
            tracemalloc.stop()
        if error:
            models.append(_failed_row(adapter, item_ids, error))
            continue
        metrics = {}
        for name in METRIC_NAMES:
            values = [row[name] for row in rows_metrics if name in row]
            metrics[name] = _metric(float(mean(values)), "reproduced_local") if values else _metric(None, "runtime", "optional dependency unavailable")
        latency_receipt = {"schema_version": 1, "model": adapter.spec.name, "item_ids": item_ids, "items": latency_items, "cpu_ms": _quantiles(timings), "realtime_factor": _quantiles(rtfs), "memory_peak_mb": memory_peak / (1024 * 1024)}
        # Memory is allocator-dependent; exclude it from the deterministic latency hash.
        latency_hash_payload = {key: value for key, value in latency_receipt.items() if key != "memory_peak_mb"}
        models.append({"name": adapter.spec.name, "status": "reproduced_local", "item_ids": item_ids, "spec": _spec_dict(adapter.spec), "sample_rate": {"reference": rows[0].sample_rate, "native": adapter.spec.native_sample_rate, "input": {"from": rows[0].sample_rate, "to": adapter.spec.native_sample_rate}, "method": "none" if rows[0].sample_rate == adapter.spec.native_sample_rate else "scipy.resample_poly"}, "metrics": metrics, "latency": latency_receipt, "latency_receipt_sha256": _canonical_hash(latency_hash_payload), "memory_peak_mb": memory_peak / (1024 * 1024)})
    statuses = [row["status"] for row in models]
    report = {"schema_version": 1, "protocol": {"item_ids": item_ids, "item_count": len(item_ids), "metrics": list(METRIC_NAMES), "no_network": True}, "models": models, "status": "pass" if all(status == "reproduced_local" for status in statuses) else "partial" if any(status == "blocked" for status in statuses) else "fail"}
    report["checks"] = {"same_item_ids": all(row.get("item_ids") == item_ids for row in models), "missing_dependencies_blocked": all(row.get("status") != "pass" for row in models), "metric_finiteness": all(_row_metrics_finite(row) for row in models if row.get("status") == "reproduced_local")}
    return report


def _spec_dict(spec: AdapterSpec) -> dict:
    return {"native_sample_rate": spec.native_sample_rate, "source_url": spec.source_url, "license": spec.license, "runtime": spec.runtime, "weights_bundled": spec.weights_bundled}


def _blocked_row(adapter: ModelAdapter, item_ids: list[str], reason: str | None) -> dict:
    return {"name": adapter.spec.name, "status": "blocked", "blocked_reason": reason or "runtime unavailable", "item_ids": item_ids, "spec": _spec_dict(adapter.spec), "metrics": {name: _metric(None, "blocked", reason or "runtime unavailable") for name in METRIC_NAMES}, "sample_rate": {"reference": 16000, "native": adapter.spec.native_sample_rate, "input": {"from": 16000, "to": adapter.spec.native_sample_rate}, "method": "scipy.resample_poly" if adapter.spec.native_sample_rate != 16000 else "none"}}


def _failed_row(adapter: ModelAdapter, item_ids: list[str], reason: str) -> dict:
    return {"name": adapter.spec.name, "status": "failed", "error": reason, "item_ids": item_ids, "spec": _spec_dict(adapter.spec)}


def _row_metrics_finite(row: dict) -> bool:
    for metric in row.get("metrics", {}).values():
        value = metric.get("value") if isinstance(metric, dict) else None
        if metric.get("available") and (not isinstance(value, (int, float)) or not math.isfinite(float(value))):
            return False
    return True


def _attach_registry_metadata(report: dict, registry: dict) -> dict:
    by_name = {str(row["name"]): row for row in registry["baselines"]}
    for row in report.get("models", []):
        meta = by_name.get(str(row.get("name")))
        if meta is None:
            continue
        if row.get("status") != "reproduced_local" or not isinstance(row.get("provenance"), dict):
            provenance = dict(meta["provenance"])
            provenance["weights"] = {
                "bundled": bool(meta.get("weights_bundled", False)),
                "weight_sha256": provenance.get("weight_sha256"),
                "hash_status": "verified" if provenance.get("weight_sha256") else "unavailable",
            }
            row["provenance"] = provenance
        if not isinstance(row.get("environment"), dict):
            row["environment"] = {
                "runtime": meta.get("runtime"),
                "command": meta.get("command"),
                "recipe": meta.get("recipe"),
                "python": meta.get("python"),
            }
        row["evidence_class"] = meta.get("evidence_class", row.get("status"))
        if isinstance(meta.get("reproduction"), dict):
            row["reproduction"] = dict(meta["reproduction"])
        row.setdefault("spec", {})["license"] = meta["license"]
    return report


def validate_model_comparison(report: dict) -> bool:
    models = report.get("models")
    protocol = report.get("protocol", {})
    ids = protocol.get("item_ids")
    if report.get("schema_version") not in (1, 2):
        return False
    if not isinstance(models, list) or not isinstance(ids, list) or not ids:
        return False
    for row in models:
        if row.get("item_ids") != ids or row.get("status") == "failed":
            return False
        if row.get("status") == "reproduced_local" and not _row_metrics_finite(row):
            return False
    return True


def _load_manifest_ids(path: Path, offset: int, count: int) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()[offset : offset + count]
    return [str(json.loads(line)["id"]) for line in lines if line.strip()]


def build_default_report(root: Path, *, output: Path | None = None) -> dict:
    root = Path(root).resolve()
    registry = load_research_baselines(root / "configs/research_baselines.json")
    config = json.loads((root / "configs/evaluation/research_slices.json").read_text(encoding="utf-8"))
    primary = next(slice_ for slice_ in config["slices"] if slice_["name"] == "primary_comparison")
    manifest = root / str(config["manifest"])
    ids = _load_manifest_ids(manifest, int(primary["offset"]), int(primary["count"]))
    # Receipt adapters preserve exact manifest identity without re-running old weights.
    current = FrozenHistoricalAdapter(root / "reports/generated/research_candidate_primary.json", manifest, name="ClearHop")
    historical = FrozenHistoricalAdapter(root / "reports/generated/research_baseline_primary.json", manifest)
    deepfilter_receipt = root / "reports/public/deepfilternet3_reproduction.json"
    deepfilter: ModelAdapter = ReproducedExternalAdapter(deepfilter_receipt, name="DeepFilterNet3") if deepfilter_receipt.is_file() else DeepFilterNet3Adapter()
    adapters: list[ModelAdapter] = [current, historical, deepfilter, RNNoiseAdapter(), DTLNAdapter(), WebRTCNSAdapter()]
    # Receipt-only rows need no waveforms; use zero arrays solely to carry IDs.
    items = [BenchmarkItem(item_id, np.zeros(160, dtype=np.float32), np.zeros(160, dtype=np.float32), 16000, str(primary["name"])) for item_id in ids]
    report = run_model_comparison(items, adapters, include_optional_metrics=False)
    _attach_registry_metadata(report, registry)
    report["schema_version"] = 2
    report["protocol"].update({"manifest": str(Path(config["manifest"]).as_posix()), "slice": primary["name"], "slice_offset": primary["offset"], "slice_count": primary["count"]})
    registry_path = root / "configs/research_baselines.json"
    manifest_path = root / str(config["manifest"])
    report["inputs"] = {
        "registry": {
            "path": "configs/research_baselines.json",
            "sha256": _file_sha256(registry_path),
        },
        "manifest": {
            "path": str(Path(config["manifest"]).as_posix()),
            "sha256": _file_sha256(manifest_path),
            "item_ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
            "git_tracked": _git_tracked(root, str(Path(config["manifest"]).as_posix())),
        },
    }
    report["receipt_sha256"] = canonical_receipt_sha256(report)
    if output is None:
        output = root / "reports/generated/model_comparison.json"
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = build_default_report(args.root, output=args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if validate_model_comparison(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
