"""Export reproducible clean/noisy/enhanced WAV triplets for worst cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.checkpoint import file_sha256, validate_checkpoint_metadata
from src.config import load_train_config
from src.dataset import NoiseSuppressionDataset
from src.inference import enhance_offline
from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig
from src.utils import load_manifest


def selected_failure_ids(failure_analysis: dict[str, object], count: int = 10) -> list[int]:
    rows = failure_analysis.get("worst_cases")
    if not isinstance(rows, list) or count < 1:
        raise ValueError("failure analysis requires worst_cases and positive count")
    result = [int(row["index"]) for row in rows[:count] if isinstance(row, dict) and "index" in row]
    if len(result) < count:
        raise ValueError("not enough worst cases")
    return result


def resolve_from_root(root: Path, path: Path) -> Path:
    """Resolve CLI paths consistently when invoked outside the project root."""
    path = Path(path)
    return (path if path.is_absolute() else root / path).resolve()


@torch.no_grad()
def export_failure_audio(root: Path, checkpoint: Path, config: Path, failure_path: Path, output_dir: Path, *, count: int = 10) -> dict[str, object]:
    root = Path(root).resolve()
    checkpoint = resolve_from_root(root, checkpoint)
    config = resolve_from_root(root, config)
    failure_path = resolve_from_root(root, failure_path)
    output_dir = resolve_from_root(root, output_dir)
    failures = json.loads(failure_path.read_text(encoding="utf-8"))
    ids = selected_failure_ids(failures, count)
    cfg = load_train_config(config, root)
    entries = load_manifest(root / "manifests/v2/fold_0_test.jsonl")
    dataset = NoiseSuppressionDataset(
        [str((root / row["clean_path"]).resolve()) for row in entries],
        [str((root / row["noise_paths"][0]).resolve()) for row in entries],
        segment_len=cfg.segment_len,
        sr=cfg.audio.sr,
        mix_config={"seed": cfg.seed, "snr_range": cfg.snr_range, "project_root": str(root), "val_use_rir": False, "val_time_vary": False},
        mode="val",
        manifest=root / "manifests/v2/fold_0_test.jsonl",
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint_metadata(state)
    model = MobileDeepFilterNet(MobileDeepFilterNetConfig(**state["model_cfg"]))
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, object]] = []
    for item_id in ids:
        item = dataset[item_id]
        clean = item["clean"]
        noisy = item["noisy"]
        enhanced = enhance_offline(model, noisy.unsqueeze(0))[0]
        files: dict[str, object] = {}
        for name, audio in (("clean", clean), ("noisy", noisy), ("enhanced", enhanced)):
            path = output_dir / f"item_{item_id}_{name}.wav"
            sf.write(path, audio.numpy(), cfg.audio.sr, subtype="PCM_16")
            files[name] = {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)}
        items.append({"index": item_id, "files": files})
    return {"schema_version": 1, "status": "pass", "checkpoint_sha256": file_sha256(checkpoint), "items": items}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--failure-analysis", type=Path, default=Path("reports/generated/failure_analysis.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/generated/failure_audio"))
    parser.add_argument("--receipt", type=Path, default=Path("reports/generated/failure_audio.json"))
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    report = export_failure_audio(PROJECT_ROOT, args.checkpoint, args.config, args.failure_analysis, args.output_dir, count=args.count)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
