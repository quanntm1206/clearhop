"""Build portable speaker-disjoint manifests without touching historical files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_train_config
from src.splits import (
    build_grouped_splits,
    build_manifest_entries,
    infer_speaker_id,
    write_manifest,
)

AUDIO_SUFFIXES = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}


def list_audio(root: Path) -> list[Path]:
    return sorted(
        (path.resolve() for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES),
        key=lambda path: path.as_posix(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--output", type=Path, default=Path("manifests/v2"))
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    cfg = load_train_config(config_path, project_root=root)
    cfg.validate_data_paths()

    clean_paths = list_audio(cfg.clean_root)
    noise_paths = list_audio(cfg.noise_root)
    folds = build_grouped_splits(
        clean_paths,
        n_folds=args.n_folds,
        val_fraction=args.val_fraction,
        seed=cfg.seed,
    )
    if not 0 <= args.fold_index < len(folds):
        raise ValueError(f"fold-index must be in [0, {len(folds) - 1}].")
    fold = folds[args.fold_index]

    fingerprints: dict[str, str] = {}
    for offset, (name, paths) in enumerate(
        (("train", fold.train), ("val", fold.val), ("test", fold.test))
    ):
        entries = build_manifest_entries(
            paths,
            noise_paths,
            seed=cfg.seed + 10000 * args.fold_index + offset,
            segment_len=cfg.segment_len,
        )
        manifest_path = output / f"fold_{args.fold_index}_{name}.jsonl"
        fingerprints[name] = write_manifest(entries, manifest_path, project_root=root)

    groups = {
        name: {infer_speaker_id(path) for path in paths}
        for name, paths in (("train", fold.train), ("val", fold.val), ("test", fold.test))
    }
    overlaps = {
        "train_val": len(groups["train"] & groups["val"]),
        "train_test": len(groups["train"] & groups["test"]),
        "val_test": len(groups["val"] & groups["test"]),
    }
    if any(overlaps.values()):
        raise RuntimeError(f"Speaker leakage detected: {overlaps}")

    summary = {
        "schema_version": 2,
        "fold": args.fold_index,
        "seed": cfg.seed,
        "segment_len": cfg.segment_len,
        "clean_files": len(clean_paths),
        "noise_files": len(noise_paths),
        "split_files": {"train": len(fold.train), "val": len(fold.val), "test": len(fold.test)},
        "split_speakers": {name: len(value) for name, value in groups.items()},
        "speaker_overlap": overlaps,
        "fingerprints": fingerprints,
    }
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / f"fold_{args.fold_index}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
