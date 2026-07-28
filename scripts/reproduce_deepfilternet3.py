"""Reproduce DeepFilterNet3 on ClearHop's frozen 500-item primary slice.

Run this script in an isolated Python 3.11 environment containing
``deepfilternet==0.5.6``. The model directory must be downloaded outside Git
from the pinned upstream commit; this script never downloads or bundles it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.model_comparison import AdapterSpec, BenchmarkItem, ModelAdapter, canonical_receipt_sha256, run_model_comparison


SOURCE_COMMIT = "d375b2d8309e0935d165700c91da9de862a99c31"
SOURCE_URL = "https://github.com/Rikorose/DeepFilterNet"
MODEL_ARCHIVE_URL = f"https://raw.githubusercontent.com/Rikorose/DeepFilterNet/{SOURCE_COMMIT}/models/DeepFilterNet3.zip"


def installed_package_version(name: str) -> str:
    version = importlib.metadata.version(name)
    if version:
        return str(version)
    module_version = getattr(importlib.import_module(name), "__version__", None)
    if not module_version:
        raise RuntimeError(f"installed package has no version metadata: {name}")
    return str(module_version)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def unpack_init_result(value: tuple):
    if len(value) == 3:
        return value[0], value[1], value[2], None
    if len(value) == 4:
        return value
    raise ValueError(f"unexpected DeepFilterNet init_df result length: {len(value)}")


class DeepFilterNet3Runtime(ModelAdapter):
    def __init__(self, model_dir: Path):
        from df.enhance import enhance, init_df

        import psutil
        import torch

        self.spec = AdapterSpec(
            "DeepFilterNet3",
            native_sample_rate=48000,
            source_url=SOURCE_URL,
            license="Apache-2.0 OR MIT",
            runtime="deepfilternet-python",
        )
        self._enhance = enhance
        self._torch = torch
        self._process = psutil.Process()
        self.rss_samples_mb: list[float] = []
        self.model, self.df_state, _, self.epoch = unpack_init_result(init_df(
            str(Path(model_dir).resolve()),
            log_level="ERROR",
            log_file=None,
            config_allow_defaults=True,
            epoch="best",
        ))

    def enhance(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate != 48000:
            raise ValueError("DeepFilterNet3 requires 48 kHz input")
        tensor = self._torch.from_numpy(np.asarray(audio, dtype=np.float32)).reshape(1, -1)
        enhanced = self._enhance(self.model, self.df_state, tensor, pad=True)
        self.rss_samples_mb.append(self._process.memory_info().rss / (1024 * 1024))
        return enhanced.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=False)


def build_benchmark_items(root: Path, config_path: Path, manifest_path: Path, *, offset: int, count: int) -> list[BenchmarkItem]:
    from src.config import load_train_config
    from src.dataset import NoiseSuppressionDataset
    from src.utils import load_manifest

    root = Path(root).resolve()
    config_path = config_path if config_path.is_absolute() else root / config_path
    manifest_path = manifest_path if manifest_path.is_absolute() else root / manifest_path
    cfg = load_train_config(config_path, root)
    entries = load_manifest(manifest_path)
    if offset < 0 or count <= 0 or offset + count > len(entries):
        raise ValueError("requested manifest slice is out of bounds")
    clean_paths = [str((root / row["clean_path"]).resolve()) if not Path(row["clean_path"]).is_absolute() else row["clean_path"] for row in entries]
    noise_paths = [str((root / row["noise_paths"][0]).resolve()) if not Path(row["noise_paths"][0]).is_absolute() else row["noise_paths"][0] for row in entries]
    dataset = NoiseSuppressionDataset(
        clean_paths,
        noise_paths,
        segment_len=cfg.segment_len,
        sr=cfg.audio.sr,
        mix_config={
            "seed": cfg.seed,
            "snr_range": cfg.snr_range,
            "project_root": str(root),
            "val_use_rir": False,
            "val_time_vary": False,
        },
        mode="val",
        manifest=manifest_path,
    )
    items: list[BenchmarkItem] = []
    for index in range(offset, offset + count):
        item = dataset[index]
        manifest_id = str(item["meta"].get("manifest_id", entries[index]["id"]))
        items.append(
            BenchmarkItem(
                manifest_id,
                item["noisy"].detach().cpu().numpy().astype(np.float32, copy=False),
                item["clean"].detach().cpu().numpy().astype(np.float32, copy=False),
                cfg.audio.sr,
                "primary_comparison",
            )
        )
    return items


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {name: float(np.quantile(array, q)) for name, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99), ("peak", 1.0))}


def build_reproduction_receipt(
    model_row: dict,
    *,
    source_commit: str,
    model_archive_sha256: str,
    model_checkpoint_sha256: str,
    manifest_sha256: str,
    package_versions: dict[str, str],
    rss_samples_mb: list[float] | None = None,
) -> dict:
    if model_row.get("name") != "DeepFilterNet3" or model_row.get("status") != "reproduced_local":
        raise ValueError("receipt requires a reproduced DeepFilterNet3 row")
    if len(source_commit) != 40 or not all(_is_sha256(value) for value in (model_archive_sha256, model_checkpoint_sha256, manifest_sha256)):
        raise ValueError("receipt provenance hashes are invalid")
    row = json.loads(json.dumps(model_row, allow_nan=False))
    row["provenance"] = {
        "source_url": SOURCE_URL,
        "commit": source_commit,
        "commit_url": f"{SOURCE_URL}/commit/{source_commit}",
        "version": "deepfilternet==0.5.6 / DeepFilterNet3",
        "retrieved_at": datetime.now(timezone.utc).date().isoformat(),
        "model_archive_url": MODEL_ARCHIVE_URL,
        "model_archive_sha256": model_archive_sha256,
        "weight_sha256": model_checkpoint_sha256,
        "weights": {"bundled": False, "hash_status": "verified", "weight_sha256": model_checkpoint_sha256},
    }
    row["environment"] = {
        "runtime": "deepfilternet-python",
        "command": "python scripts/reproduce_deepfilternet3.py --model-dir <external-cache>/DeepFilterNet3 --model-archive <external-cache>/DeepFilterNet3.zip",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": dict(sorted(package_versions.items())),
        "network_during_inference": False,
    }
    row["sample_rate"]["output"] = {
        "from": int(row["sample_rate"]["native"]),
        "to": int(row["sample_rate"]["reference"]),
    }
    row["sample_rate"]["latency_scope"] = "DeepFilterNet3 inference at 48 kHz; explicit 16->48->16 kHz resampling excluded"
    row["memory_rss_mb"] = _quantiles(rss_samples_mb or [])
    row["evidence_class"] = "reproduced_local"
    receipt = {
        "schema_version": 1,
        "status": "reproduced_local",
        "protocol": {
            "manifest": "manifests/v2/fold_0_test.jsonl",
            "manifest_sha256": manifest_sha256,
            "slice": "primary_comparison",
            "slice_offset": 0,
            "slice_count": len(row["item_ids"]),
            "item_ids": row["item_ids"],
        },
        "model": row,
    }
    receipt["receipt_sha256"] = canonical_receipt_sha256(receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-archive", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("checkpoints/gain_calibration/research/complex_nmse_sisdr_beta_0p03/seed_71/resolved_config.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/v2/fold_0_test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/public/deepfilternet3_reproduction.json"))
    parser.add_argument("--count", type=int, default=500)
    args = parser.parse_args()
    root = args.root.resolve()
    model_dir = args.model_dir.resolve()
    archive = args.model_archive.resolve()
    checkpoint_files = sorted((model_dir / "checkpoints").glob("model*.ckpt.best"))
    if len(checkpoint_files) != 1:
        raise ValueError("expected exactly one pinned DeepFilterNet3 best checkpoint")
    items = build_benchmark_items(root, args.config, args.manifest, offset=0, count=args.count)
    adapter = DeepFilterNet3Runtime(model_dir)
    comparison = run_model_comparison(items, [adapter], include_optional_metrics=True)
    row = comparison["models"][0]
    if row.get("status") != "reproduced_local":
        raise RuntimeError(f"DeepFilterNet3 reproduction failed: {row.get('error', row)}")
    packages = {}
    for name in ("deepfilternet", "deepfilterlib", "numpy", "torch", "scipy", "pystoi", "pesq", "psutil"):
        packages[name] = installed_package_version(name)
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    receipt = build_reproduction_receipt(
        row,
        source_commit=SOURCE_COMMIT,
        model_archive_sha256=_sha256(archive),
        model_checkpoint_sha256=_sha256(checkpoint_files[0]),
        manifest_sha256=_sha256(manifest),
        package_versions=packages,
        rss_samples_mb=adapter.rss_samples_mb,
    )
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "output": str(output), "metrics": receipt["model"]["metrics"], "latency": receipt["model"]["latency"], "memory_rss_mb": receipt["model"]["memory_rss_mb"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
