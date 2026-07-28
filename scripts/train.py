from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_train_config
from src.trainer import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--experiment-id")
    args = parser.parse_args()

    config_obj = load_train_config(args.config, project_root=Path.cwd())
    config = config_obj.to_dict()
    config["project_root"] = str(config_obj.project_root)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.checkpoint_dir is not None:
        config["checkpoint_dir"] = str(args.checkpoint_dir)
    if args.experiment_id is not None:
        config["experiment_id"] = args.experiment_id
    train_model(config=config, resume=args.resume)


if __name__ == "__main__":
    main()
