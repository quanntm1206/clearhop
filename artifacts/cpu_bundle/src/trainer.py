"""
Training loop for MobileDeepFilterNet.

Implements:
- Config-driven training from configs/train.yaml
- On-the-fly mixing train dataset + deterministic manifest-based val dataset
- STFT -> model -> causal deep filtering -> iSTFT reconstruction
- Loss: alpha * complex MSE + beta * SI-SDR loss (configurable)
- Metrics: loss, SI-SDR improvement, SNR improvement, STOI (+ optional PESQ)
- Mixed precision (CUDA), gradient clipping, checkpointing, TensorBoard
- Resume support via checkpoint path
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard is optional at runtime
    class SummaryWriter:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def add_scalar(self, *args: Any, **kwargs: Any) -> None:
            pass

        def close(self) -> None:
            pass

from .config import TrainConfig, load_train_config
from .checkpoint import manifest_fingerprints
from .dataset import NoiseSuppressionDataset, collate_audio_batch
from .filtering import causal_deep_filter, stft_analysis_ri, stft_synthesis
from .losses import (
    complex_mse,
    complex_nmse,
    compressed_complex_loss,
    scheduled_weight,
    si_sdr_loss,
)
from .losses import si_sdr_loss as _si_sdr_loss
from .model import MobileDeepFilterNet, MobileDeepFilterNetConfig
from .utils import load_manifest, list_audio_files, seed_all, snr_db


# Prevent corrupt/unstable AMP runs from consuming an unbounded stream of batches.
_MAX_CONSECUTIVE_AMP_SKIPS = 8
_INITIAL_BEST_METRIC = -1e9


def _complex_mse(a_ri: torch.Tensor, b_ri: torch.Tensor) -> torch.Tensor:
    """Legacy complex MSE helper accepting any matching real/imag shape."""
    return F.mse_loss(a_ri, b_ri)


def _stft_ri(x: torch.Tensor, n_fft: int, hop: int, window: torch.Tensor) -> torch.Tensor:
    """
    STFT returning real/imag channel.
    Returns: (B, F, T, 2)
    """
    return stft_analysis_ri(x, n_fft, hop, window)


def _istft_from_ri(X_ri: torch.Tensor, n_fft: int, hop: int, window: torch.Tensor, length: int) -> torch.Tensor:
    """iSTFT from real/imag tensor (B,F,T,2) -> (B,T)."""
    return stft_synthesis(X_ri, n_fft=n_fft, hop=hop, window=window, length=length)


def _log_power_feats(X_ri: torch.Tensor) -> torch.Tensor:
    """Compute log power magnitude. Input: (B,F,T,2) -> (B,1,F,T)."""
    mag2 = X_ri[..., 0] ** 2 + X_ri[..., 1] ** 2
    logp = torch.log(mag2 + 1e-12)
    return logp.permute(0, 2, 1).unsqueeze(1)  # (B,1,T,F) -> wait


def _logp_b1ft(X_ri: torch.Tensor) -> torch.Tensor:
    """(B,F,T,2) -> (B,1,F,T)"""
    mag2 = X_ri[..., 0] ** 2 + X_ri[..., 1] ** 2
    logp = torch.log(mag2 + 1e-12)
    return logp.unsqueeze(1)  # (B,1,F,T)


@torch.no_grad()
def _compute_stoi(clean: np.ndarray, test: np.ndarray, sr: int) -> Optional[float]:
    try:
        from pystoi.stoi import stoi  # type: ignore

        return float(stoi(clean, test, sr, extended=False))
    except Exception:
        return None


@torch.no_grad()
def _compute_pesq(clean: np.ndarray, test: np.ndarray, sr: int) -> Optional[float]:
    try:
        from pesq import pesq  # type: ignore

        mode = "wb" if sr >= 16000 else "nb"
        return float(pesq(sr, clean, test, mode))
    except Exception:
        return None


def _resolve_default_paths(project_root: Path) -> Tuple[Path, Path]:
    clean_root = project_root / "data" / "clean_16k"
    noise_root = project_root / "data" / "noise_16k"
    return clean_root, noise_root


def _load_config(config_path: Path) -> Dict[str, Any]:
    return load_train_config(config_path, project_root=config_path.resolve().parents[1]).to_dict()


def _save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _resume_position(global_step: int, steps_per_epoch: int) -> tuple[int, int]:
    """Map an optimizer step count to the next epoch and in-epoch step."""
    if global_step < 0 or steps_per_epoch <= 0:
        raise ValueError("global_step must be non-negative and steps_per_epoch positive")
    return divmod(global_step, steps_per_epoch)


def _compute_objective(
    cfg: dict[str, Any],
    y_ri: torch.Tensor,
    clean_ri: torch.Tensor,
    enhanced: torch.Tensor,
    clean: torch.Tensor,
    global_step: int,
) -> dict[str, torch.Tensor]:
    raw = complex_mse(y_ri, clean_ri)
    normalized = complex_nmse(y_ri, clean_ri, float(cfg["loss_eps"]))
    sisdr = si_sdr_loss(enhanced, clean, float(cfg["loss_eps"]))
    compressed = y_ri.new_zeros(())
    weight = y_ri.new_zeros(())
    mode = str(cfg["loss"])
    if mode == "complex_mse_plus_si_sdr":
        weight = y_ri.new_tensor(float(cfg["beta_si_sdr"]))
        total = float(cfg["alpha_loss"]) * raw + weight * sisdr
    elif mode == "complex_nmse":
        total = float(cfg["alpha_loss"]) * normalized
    elif mode == "complex_nmse_sisdr":
        weight = y_ri.new_tensor(
            scheduled_weight(
                global_step,
                float(cfg["beta_si_sdr"]),
                int(cfg["sisdr_warmup_start"]),
                int(cfg["sisdr_warmup_end"]),
            )
        )
        total = float(cfg["alpha_loss"]) * normalized + weight * sisdr
    elif mode == "compressed_complex":
        compressed = compressed_complex_loss(
            y_ri,
            clean_ri,
            exponent=float(cfg["compression_exponent"]),
            complex_weight=float(cfg["compression_complex_weight"]),
            eps=float(cfg["loss_eps"]),
        )
        total = float(cfg["alpha_loss"]) * compressed
    else:
        raise ValueError(f"Unsupported loss mode: {mode}")
    if not torch.isfinite(total):
        raise FloatingPointError(f"Non-finite training loss for mode {mode}")
    component_values = {
        "complex_mse": raw,
        "complex_nmse": normalized,
        "compressed_complex": compressed,
        "si_sdr": sisdr,
        "si_sdr_weight": weight,
    }
    for name, value in component_values.items():
        if not torch.isfinite(value):
            raise FloatingPointError(
                f"Non-finite training loss component {name} for mode {mode}"
            )
    return {
        "total": total,
        "complex_mse": raw.detach(),
        "complex_nmse": normalized.detach(),
        "compressed_complex": compressed.detach(),
        "si_sdr": sisdr.detach(),
        "si_sdr_weight": weight.detach(),
    }


def _selection_value(selection_metric: str, si_sdri: float, snri: float) -> float:
    if selection_metric == "si_sdr_improvement":
        return si_sdri
    if selection_metric == "snr_improvement":
        return snri
    raise ValueError(f"Unsupported selection_metric: {selection_metric}")


def train_model(config: Dict[str, Any] | None = None, *, config_path: Optional[Path] = None, resume: Optional[Path] = None) -> None:
    """Train MobileDeepFilterNet using a validated configuration."""
    project_root = Path(config.get("project_root", Path.cwd()) if config else Path.cwd()).resolve()
    config_path = config_path or (project_root / "configs" / "train.yaml")
    if config is None:
        config_obj = load_train_config(config_path, project_root=project_root)
    else:
        raw_config = dict(config)
        raw_config.pop("project_root", None)
        config_obj = TrainConfig.from_mapping(raw_config, project_root=project_root)
    cfg = config_obj.to_dict()
    config_obj.validate_data_paths()

    seed = int(cfg.get("seed", 42))
    seed_all(seed)

    sr = int(cfg["sr"])
    segment_len = float(cfg["segment_len"])
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg["num_workers"])
    epochs = int(cfg["epochs"])
    steps_per_epoch = int(cfg.get("steps_per_epoch", 200))
    max_steps = int(cfg["max_steps"]) if cfg.get("max_steps") is not None else None
    scheduler_total_steps = int(
        cfg["scheduler_total_steps"]
        if cfg.get("scheduler_total_steps") is not None
        else epochs * steps_per_epoch
    )
    selection_metric = str(cfg.get("selection_metric", "si_sdr_improvement"))
    experiment_id = str(cfg.get("experiment_id", "baseline"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.get("mixed_precision", True)) and device.type == "cuda"
    runtime_meta = {
        "device": str(device),
        "torch_version": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }

    clean_root = config_obj.clean_root
    noise_root = config_obj.noise_root
    clean_list = list_audio_files(clean_root)
    noise_list = list_audio_files(noise_root)
    if not clean_list:
        raise RuntimeError(f"No clean audio files found under {clean_root}")
    if not noise_list:
        raise RuntimeError(f"No noise audio files found under {noise_root}")

    manifest_root = project_root / "manifests" / "v2"
    train_manifest = manifest_root / "fold_0_train.jsonl"
    manifest_path = manifest_root / "fold_0_val.jsonl"
    if not train_manifest.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "Portable manifests missing. Run scripts/build_manifests.py first: "
            f"{train_manifest}"
        )
    train_entries = load_manifest(train_manifest)
    clean_list = [
        str((project_root / entry["clean_path"]).resolve())
        if not Path(entry["clean_path"]).is_absolute()
        else entry["clean_path"]
        for entry in train_entries
    ]

    mix_cfg = {
        "seed": seed,
        "p_rir": float(cfg.get("p_rir", 0.35)),
        "p_vary": float(cfg.get("p_vary", 0.4)),
        "snr_range": tuple(float(v) for v in cfg.get("snr_range", (-10.0, 20.0))),
        "project_root": str(project_root),
        "return_clean_spec": False,
        "val_use_rir": True,
        "val_time_vary": True,
    }
    train_ds = NoiseSuppressionDataset(
        clean_list=clean_list,
        noise_list=noise_list,
        segment_len=segment_len,
        sr=sr,
        mix_config=mix_cfg,
        mode="train",
        manifest=None,
    )
    val_ds = NoiseSuppressionDataset(
        clean_list=clean_list,
        noise_list=noise_list,
        segment_len=segment_len,
        sr=sr,
        mix_config=mix_cfg,
        mode="val",
        manifest=manifest_path,
    )
    val_limit = min(int(cfg.get("n_val", len(val_ds))), len(val_ds))
    val_view = Subset(val_ds, range(val_limit))
    val_loader = DataLoader(
        val_view,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, num_workers // 2),
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_audio_batch,
    )

    def make_train_loader(epoch: int) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(seed + epoch)
        return DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_audio_batch,
            generator=generator,
        )

    model_cfg = MobileDeepFilterNetConfig(
        freq_bins=161,
        enc_channels=int(cfg.get("enc_channels", 32)),
        num_encoder_blocks=int(cfg.get("num_encoder_blocks", 2)),
        gru_hidden=int(cfg.get("gru_hidden", 64)),
        gru_layers=int(cfg.get("gru_layers", 1)),
        k_tap=int(cfg.get("k_tap", 3)),
        activation=str(cfg.get("activation", "silu")),
        causal_time=True,
    )
    model = MobileDeepFilterNet(model_cfg).to(device)
    opt = AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-2)),
    )

    sched_name = str(cfg.get("scheduler", "onecycle")).lower()
    if sched_name == "onecycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=float(cfg.get("lr", 3e-4)),
            total_steps=scheduler_total_steps,
            pct_start=0.1,
            anneal_strategy="cos",
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, scheduler_total_steps)
        )

    def scheduler_horizon(value: Any) -> Optional[int]:
        for key in ("total_steps", "T_max"):
            if isinstance(value, dict) and key in value:
                return int(value[key])
            if hasattr(value, key):
                return int(getattr(value, key))
        return None

    initial_horizon = scheduler_horizon(scheduler)
    if initial_horizon != scheduler_total_steps:
        raise ValueError(
            f"Scheduler horizon {initial_horizon} does not match configured "
            f"scheduler_total_steps {scheduler_total_steps}"
        )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    checkpoint_value = cfg.get("checkpoint_dir")
    ckpt_dir = (
        Path(str(checkpoint_value)).resolve()
        if checkpoint_value not in (None, "")
        else project_root / "checkpoints"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    resume_step_in_epoch = 0
    global_step = 0
    best_val_metric = _INITIAL_BEST_METRIC
    best_path = ckpt_dir / "best.pth"
    has_best_checkpoint = False
    resume_state: Optional[dict[str, Any]] = None
    resume_consumed_batches_in_epoch = 0
    manifest_meta = manifest_fingerprints(
        {"train": train_manifest, "val": manifest_path, "test": manifest_root / "fold_0_test.jsonl"}
    )
    if resume is not None and Path(resume).exists():
        resume_path = Path(resume).resolve()
        if resume_path.parent != ckpt_dir.resolve():
            raise ValueError(
                f"Resume checkpoint parent {resume_path.parent} does not match "
                f"configured checkpoint_dir {ckpt_dir.resolve()}"
            )
        state = torch.load(str(resume_path), map_location="cpu")
        resume_state = state
        global_step = int(state.get("global_step", 0))
        if max_steps is not None and global_step > max_steps:
            raise ValueError(
                f"Resume global_step {global_step} exceeds max_steps {max_steps}"
            )
        stored_metadata_horizon = state.get("scheduler_total_steps")
        if (
            stored_metadata_horizon is not None
            and int(stored_metadata_horizon) != scheduler_total_steps
        ):
            raise ValueError(
                f"Checkpoint scheduler_total_steps {stored_metadata_horizon} does not "
                f"match configured value {scheduler_total_steps}"
            )
        stored_state_horizon = scheduler_horizon(state["sched"])
        if stored_state_horizon is not None and stored_state_horizon != scheduler_total_steps:
            raise ValueError(
                f"Checkpoint scheduler state horizon {stored_state_horizon} does not "
                f"match configured value {scheduler_total_steps}"
            )
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        scheduler.load_state_dict(state["sched"])
        restored_horizon = scheduler_horizon(scheduler)
        if restored_horizon != scheduler_total_steps:
            raise ValueError(
                f"Restored scheduler horizon {restored_horizon} does not match "
                f"configured value {scheduler_total_steps}"
            )
        scaler.load_state_dict(state.get("scaler", scaler.state_dict()))
        start_epoch, resume_step_in_epoch = _resume_position(global_step, steps_per_epoch)
        best_val_metric = float(state.get("best_val_metric", best_val_metric))
        if "has_best_checkpoint" in state:
            has_best_checkpoint = bool(state["has_best_checkpoint"])
        else:
            has_best_checkpoint = (
                math.isfinite(best_val_metric)
                and best_val_metric > _INITIAL_BEST_METRIC
                and best_path.is_file()
            )
        if has_best_checkpoint and not best_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint requires a local best checkpoint but it "
                f"is missing: {best_path}"
            )
        resume_consumed_batches_in_epoch = int(
            state.get("consumed_batches_in_epoch", resume_step_in_epoch)
        )
        if resume_consumed_batches_in_epoch < resume_step_in_epoch:
            raise ValueError(
                "consumed_batches_in_epoch cannot be less than the successful "
                "step cursor"
            )
    writer = SummaryWriter(log_dir=str(ckpt_dir / "runs"))

    n_fft, hop = config_obj.audio.n_fft, config_obj.audio.hop
    window = torch.hann_window(n_fft, device=device)
    save_every = int(cfg.get("save_every_n_steps", 500))
    loss_config_keys = (
        "loss",
        "alpha_loss",
        "beta_si_sdr",
        "loss_eps",
        "sisdr_warmup_start",
        "sisdr_warmup_end",
        "compression_exponent",
        "compression_complex_weight",
    )
    loss_config = {key: cfg[key] for key in loss_config_keys}
    component_keys = (
        "total",
        "complex_mse",
        "complex_nmse",
        "compressed_complex",
        "si_sdr",
        "si_sdr_weight",
    )
    component_sums = {key: 0.0 for key in component_keys}
    successful_update_count = 0
    component_history_complete = True
    if resume_state is not None:
        stored_sums = resume_state.get("component_sums")
        stored_count = resume_state.get("successful_update_count")
        if stored_sums is None or stored_count is None:
            # Legacy checkpoints cannot reconstruct pre-resume component history;
            # the counter below therefore covers post-resume updates only.
            component_history_complete = False
        else:
            missing_components = set(component_keys) - set(stored_sums)
            if missing_components:
                raise ValueError(
                    f"Checkpoint component_sums missing keys: {sorted(missing_components)}"
                )
            component_sums = {
                key: float(stored_sums[key])
                for key in component_keys
            }
            if not all(math.isfinite(value) for value in component_sums.values()):
                raise FloatingPointError("Checkpoint component_sums contain non-finite values")
            successful_update_count = int(stored_count)
            if successful_update_count < 0:
                raise ValueError("successful_update_count must be non-negative")
            component_history_complete = bool(
                resume_state.get("component_history_complete", True)
            )

    def forward_enhance(noisy_wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        X_ri = _stft_ri(noisy_wav, n_fft=n_fft, hop=hop, window=window)
        feats = _logp_b1ft(X_ri)
        mask, w_taps, _ = model(feats, None)
        Y_ri = causal_deep_filter(X_ri, mask, w_taps)
        enh = _istft_from_ri(
            Y_ri,
            n_fft=n_fft,
            hop=hop,
            window=window,
            length=noisy_wav.size(-1),
        )
        return enh, X_ri, Y_ri

    def checkpoint_payload(epoch: int, step_in_epoch: int) -> dict[str, Any]:
        actual_scheduler_horizon = scheduler_horizon(scheduler)
        if actual_scheduler_horizon != scheduler_total_steps:
            raise ValueError(
                f"Scheduler horizon {actual_scheduler_horizon} does not match configured "
                f" value {scheduler_total_steps}"
            )
        return {
            "schema_version": 2,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "sched": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "global_step": global_step,
            "best_val_metric": best_val_metric,
            "has_best_checkpoint": has_best_checkpoint,
            "config": cfg,
            "model_cfg": asdict(model_cfg),
            "audio_cfg": asdict(config_obj.audio),
            "manifest_fingerprints": manifest_meta,
            "runtime": runtime_meta,
            "experiment_id": experiment_id,
            "loss": str(cfg["loss"]),
            "loss_config": loss_config,
            "scheduler_total_steps": actual_scheduler_horizon,
            "max_steps": max_steps,
            "selection_metric": selection_metric,
            "consumed_batches_in_epoch": consumed_batches_in_epoch,
            "component_sums": component_sums,
            "successful_update_count": successful_update_count,
            "component_history_complete": component_history_complete,
            "successful_update_count_scope": (
                "full_run" if component_history_complete else "post_resume"
            ),
        }

    def run_validation() -> dict[str, float]:
        model.eval()
        val_loss = 0.0
        sisdr_impr: list[float] = []
        snr_impr: list[float] = []
        stoi_scores: list[float] = []
        pesq_scores: list[float] = []
        metric_seen = 0
        metric_limit = int(cfg.get("metric_max_items", 0))
        compute_stoi = bool(cfg.get("compute_stoi", False))
        compute_pesq = bool(cfg.get("compute_pesq", False))

        with torch.no_grad():
            for batch in val_loader:
                clean = batch["clean"].to(device)
                noisy = batch["noisy"].to(device)
                enh, _, _ = forward_enhance(noisy)
                vloss = _si_sdr_loss(enh, clean)
                val_loss += float(vloss.item()) * clean.size(0)

                clean_np = clean.detach().cpu().numpy()
                noisy_np = noisy.detach().cpu().numpy()
                enh_np = enh.detach().cpu().numpy()
                for i in range(clean_np.shape[0]):
                    noisy_tensor = torch.from_numpy(noisy_np[i : i + 1])
                    clean_tensor = torch.from_numpy(clean_np[i : i + 1])
                    enhanced_tensor = torch.from_numpy(enh_np[i : i + 1])
                    si_in = -float(_si_sdr_loss(noisy_tensor, clean_tensor).item())
                    si_out = -float(_si_sdr_loss(enhanced_tensor, clean_tensor).item())
                    sisdr_impr.append(si_out - si_in)
                    snr_impr.append(
                        snr_db(clean_np[i], enh_np[i]) - snr_db(clean_np[i], noisy_np[i])
                    )
                    if (compute_stoi or compute_pesq) and (
                        metric_limit <= 0 or metric_seen < metric_limit
                    ):
                        if compute_stoi:
                            score = _compute_stoi(clean_np[i], enh_np[i], sr=sr)
                            if score is not None:
                                stoi_scores.append(score)
                        if compute_pesq:
                            score = _compute_pesq(clean_np[i], enh_np[i], sr=sr)
                            if score is not None:
                                pesq_scores.append(score)
                        metric_seen += 1

        n_val = len(val_view)
        return {
            "loss": val_loss / max(1, n_val),
            "si_sdri": float(np.mean(sisdr_impr)) if sisdr_impr else float("nan"),
            "snri": float(np.mean(snr_impr)) if snr_impr else float("nan"),
            "stoi": float(np.mean(stoi_scores)) if stoi_scores else float("nan"),
            "pesq": float(np.mean(pesq_scores)) if pesq_scores else float("nan"),
        }

    def validate_and_select() -> dict[str, float]:
        nonlocal best_val_metric, has_best_checkpoint
        validation = run_validation()
        selection_value = _selection_value(
            selection_metric,
            validation["si_sdri"],
            validation["snri"],
        )
        if not math.isfinite(selection_value):
            raise FloatingPointError(
                f"Non-finite validation selection metric {selection_metric}"
            )

        writer.add_scalar("val/loss_si_sdr", validation["loss"], global_step=global_step)
        writer.add_scalar("val/si_sdr_impr", validation["si_sdri"], global_step=global_step)
        writer.add_scalar("val/snr_impr", validation["snri"], global_step=global_step)
        if not math.isnan(validation["stoi"]):
            writer.add_scalar("val/stoi", validation["stoi"], global_step=global_step)
        if not math.isnan(validation["pesq"]):
            writer.add_scalar("val/pesq", validation["pesq"], global_step=global_step)

        if selection_value > best_val_metric:
            best_val_metric = selection_value
            has_best_checkpoint = True
            best_epoch, best_step_in_epoch = _resume_position(global_step, steps_per_epoch)
            torch.save(
                checkpoint_payload(best_epoch, best_step_in_epoch),
                str(best_path),
            )
        return validation

    last_epoch = start_epoch
    last_step_in_epoch = resume_step_in_epoch
    last_validation = {
        "loss": float("nan"),
        "si_sdri": float("nan"),
        "snri": float("nan"),
        "stoi": float("nan"),
        "pesq": float("nan"),
    }
    stop_training = max_steps is not None and global_step >= max_steps
    validation_ran = False
    consumed_batches_in_epoch = resume_consumed_batches_in_epoch
    consecutive_amp_skips = 0

    try:
        for epoch in range(start_epoch, epochs):
            if stop_training:
                break
            model.train()
            t_epoch0 = time.time()
            train_ds.set_epoch(epoch)
            train_loader = make_train_loader(epoch)
            iterator = iter(train_loader)
            first_step = resume_step_in_epoch if epoch == start_epoch else 0
            if epoch != start_epoch:
                consumed_batches_in_epoch = 0
            steps_run = 0
            epoch_component_sums = {key: 0.0 for key in component_keys}

            for _ in range(consumed_batches_in_epoch):
                try:
                    next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    next(iterator)

            successful_step_in_epoch = first_step
            while successful_step_in_epoch < steps_per_epoch:
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(train_loader)
                    batch = next(iterator)
                consumed_batches_in_epoch += 1
                clean = batch["clean"].to(device)
                noisy = batch["noisy"].to(device)

                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device.type, enabled=use_amp):
                    enh, _, Y_ri = forward_enhance(noisy)
                    S_ri = _stft_ri(clean, n_fft=n_fft, hop=hop, window=window)
                    objective = _compute_objective(
                        cfg, Y_ri, S_ri, enh, clean, global_step
                    )
                    loss = objective["total"]

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scale_before = scaler.get_scale()
                scaler.step(opt)
                scaler.update()
                if scaler.get_scale() < scale_before:
                    consecutive_amp_skips += 1
                    if consecutive_amp_skips >= _MAX_CONSECUTIVE_AMP_SKIPS:
                        raise FloatingPointError(
                            f"AMP skipped {_MAX_CONSECUTIVE_AMP_SKIPS} consecutive "
                            "optimizer steps"
                        )
                    continue
                consecutive_amp_skips = 0
                scheduler.step()

                log_step = global_step
                global_step += 1
                successful_step_in_epoch += 1
                steps_run += 1
                last_epoch = epoch
                last_step_in_epoch = successful_step_in_epoch
                successful_update_count += 1
                for key in component_keys:
                    value = float(objective[key].item())
                    component_sums[key] += value
                    epoch_component_sums[key] += value
                    writer.add_scalar(f"train/{key}", value, global_step=log_step)
                writer.add_scalar(
                    "train/loss", float(objective["total"].item()), global_step=log_step
                )
                writer.add_scalar(
                    "train/lr", float(opt.param_groups[0]["lr"]), global_step=log_step
                )

                if successful_step_in_epoch == steps_per_epoch:
                    consumed_batches_in_epoch = 0

                if global_step % save_every == 0:
                    torch.save(
                        checkpoint_payload(epoch, successful_step_in_epoch),
                        str(ckpt_dir / f"step_{global_step}.pth"),
                    )
                if max_steps is not None and global_step >= max_steps:
                    stop_training = True
                    break

            if steps_run == 0:
                continue

            last_validation = validate_and_select()
            validation_ran = True

            epoch_components = {
                key: epoch_component_sums[key] / max(1, steps_run)
                for key in component_keys
            }
            _save_json(
                ckpt_dir / "epoch_summaries" / f"epoch_{epoch:03d}.json",
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss_mean": epoch_components["total"],
                    "train_steps": steps_run,
                    "val_loss_si_sdr": last_validation["loss"],
                    "val_si_sdr_impr": last_validation["si_sdri"],
                    "val_snr_impr": last_validation["snri"],
                    "val_stoi": last_validation["stoi"],
                    "val_pesq": last_validation["pesq"],
                    "best_val_metric": best_val_metric,
                    "component_loss_means": epoch_components,
                    "elapsed_sec": float(time.time() - t_epoch0),
                },
            )
            if stop_training:
                break

        if stop_training and not validation_ran:
            last_validation = validate_and_select()
            validation_ran = True

        final_epoch, final_step_in_epoch = _resume_position(global_step, steps_per_epoch)
        last_path = ckpt_dir / f"step_{global_step}.pth"
        torch.save(
            checkpoint_payload(final_epoch, final_step_in_epoch),
            str(last_path),
        )
        writer.close()
    except Exception:
        writer.close()
        raise

    stop_reason = (
        "max_steps"
        if max_steps is not None and global_step >= max_steps
        else "epochs_completed"
    )
    component_means = (
        {
            key: component_sums[key] / successful_update_count
            for key in component_keys
        }
        if component_history_complete and successful_update_count > 0
        else None
    )
    _save_json(
        ckpt_dir / "run_summary.json",
        {
            "status": "completed",
            "stop_reason": stop_reason,
            "best_metric": best_val_metric,
            "has_best_checkpoint": has_best_checkpoint,
            "best_path": str(best_path),
            "last_path": str(last_path),
            "global_step": global_step,
            "seed": seed,
            "component_loss_means": component_means,
            "successful_update_count": successful_update_count,
            "component_history_complete": component_history_complete,
            "successful_update_count_scope": (
                "full_run" if component_history_complete else "post_resume"
            ),
            "selection_metric": selection_metric,
            "experiment_id": experiment_id,
            "scheduler_total_steps": scheduler_total_steps,
            "max_steps": max_steps,
            "val_si_sdr_impr": last_validation["si_sdri"],
            "val_snr_impr": last_validation["snri"],
        },
    )
