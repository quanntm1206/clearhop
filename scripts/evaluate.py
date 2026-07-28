from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader, Subset

from src.config import load_train_config
from src.checkpoint import file_sha256, validate_checkpoint_metadata
from src.splits import manifest_fingerprint, slice_fingerprint
from src.dataset import NoiseSuppressionDataset, collate_audio_batch
from src.evaluate import evaluate_model
from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig
from src.utils import load_manifest
from scripts.research_slices import resolve_research_slice


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--manifest", type=Path, default=Path("manifests/v2/fold_0_test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/generated/evaluation.json"))
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--profile", choices=("full", "screen"), default="full")
    parser.add_argument("--slice-spec", type=Path)
    parser.add_argument("--slice-name")
    parser.add_argument("--per-item", action="store_true")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    cfg = load_train_config(args.config, root)
    state = torch.load(args.checkpoint, map_location="cpu")
    validate_checkpoint_metadata(state)
    model_cfg = MobileDeepFilterNetConfig(**state["model_cfg"])
    model = MobileDeepFilterNet(model_cfg)
    model.load_state_dict(state["model"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    entries = load_manifest(manifest)
    if (args.slice_spec is None) != (args.slice_name is None):
        raise ValueError("--slice-spec and --slice-name must be provided together")
    if args.slice_spec is not None:
        spec_path = args.slice_spec if args.slice_spec.is_absolute() else root / args.slice_spec
        selected_slice = resolve_research_slice(spec_path, str(args.slice_name), manifest_length=len(entries))
        args.offset = int(selected_slice["offset"])
        args.max_items = int(selected_slice["count"])
    clean_paths = [str((root / row["clean_path"]).resolve()) if not Path(row["clean_path"]).is_absolute() else row["clean_path"] for row in entries]
    noise_paths = [str((root / row["noise_paths"][0]).resolve()) if not Path(row["noise_paths"][0]).is_absolute() else row["noise_paths"][0] for row in entries]
    dataset = NoiseSuppressionDataset(
        clean_paths,
        noise_paths,
        segment_len=cfg.segment_len,
        sr=cfg.audio.sr,
        mix_config={"seed": cfg.seed, "snr_range": cfg.snr_range, "project_root": str(root), "val_use_rir": False, "val_time_vary": False},
        mode="val",
        manifest=manifest,
    )
    slice_count = len(dataset) - args.offset if args.max_items is None else args.max_items
    slice_digest = slice_fingerprint(manifest, args.offset, slice_count)
    dataset_slice = Subset(dataset, range(args.offset, args.offset + slice_count))
    loader = DataLoader(dataset_slice, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_audio_batch)
    result = evaluate_model(
        model,
        loader,
        device=device,
        sr=cfg.audio.sr,
        max_items=args.max_items,
        profile=args.profile,
        return_items=args.per_item,
    )
    if args.per_item and isinstance(result.get("items"), dict):
        for item_rows in result["items"].values():
            for item in item_rows:
                if item.get("metadata", {}).get("snr_db") is None:
                    item["index"] = int(item["index"]) + int(args.offset)
    result["metadata"] = {
        "schema_version": 1,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "checkpoint_schema_version": int(state["schema_version"]),
        "config": str(args.config),
        "manifest": str(manifest),
        "manifest_fingerprint": manifest_fingerprint(manifest),
        "slice_offset": args.offset,
        "slice_count": slice_count,
        "slice_fingerprint": slice_digest,
        "slice_name": args.slice_name,
        "max_items": args.max_items,
        "evaluation_profile": args.profile,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
