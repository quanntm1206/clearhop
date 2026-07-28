"""Typed, portable configuration for the production pipeline."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class AudioConfig:
    sr: int = 16000
    n_fft: int = 320
    hop: int = 160
    freq_bins: int = 161

    def __post_init__(self) -> None:
        if (self.sr, self.n_fft, self.hop, self.freq_bins) != (16000, 320, 160, 161):
            raise ValueError(
                "Audio contract requires sr=16000, n_fft=320, hop=160, "
                "freq_bins=161."
            )


@dataclass(frozen=True)
class DataConfig:
    clean_root: Path
    noise_root: Path
    segment_len: float = 4.0


@dataclass
class TrainConfig:
    project_root: Path
    audio: AudioConfig = field(default_factory=AudioConfig)
    data: DataConfig | None = None
    batch_size: int = 8
    num_workers: int = 0
    epochs: int = 150
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-2
    optimizer: str = "AdamW"
    scheduler: str = "onecycle"
    mixed_precision: bool = True
    enc_channels: int = 32
    num_encoder_blocks: int = 2
    gru_hidden: int = 64
    gru_layers: int = 1
    k_tap: int = 3
    activation: str = "silu"
    loss: str = "complex_mse_plus_si_sdr"
    alpha_loss: float = 1.0
    beta_si_sdr: float = 0.5
    loss_eps: float = 1e-8
    sisdr_warmup_start: int = 0
    sisdr_warmup_end: int = 0
    compression_exponent: float = 0.3
    compression_complex_weight: float = 0.3
    checkpoint_dir: Path | None = None
    max_steps: int | None = None
    scheduler_total_steps: int | None = None
    selection_metric: str = "si_sdr_improvement"
    experiment_id: str = "baseline"
    snr_range: tuple[float, float] = (-10.0, 20.0)
    p_rir: float = 0.35
    p_vary: float = 0.4
    cache_mixtures: bool = False
    seed: int = 42
    n_val: int = 500
    steps_per_epoch: int = 200
    save_every_n_steps: int = 500
    compute_stoi: bool = False
    compute_pesq: bool = False
    metric_max_items: int = 0

    def __post_init__(self) -> None:
        calibration_values = (
            self.loss_eps,
            self.alpha_loss,
            self.beta_si_sdr,
            self.compression_exponent,
            self.compression_complex_weight,
            *self.snr_range,
        )
        if not all(math.isfinite(value) for value in calibration_values):
            raise ValueError("Calibration values must be finite.")
        if self.loss not in {
            "complex_mse_plus_si_sdr",
            "complex_nmse",
            "complex_nmse_sisdr",
            "compressed_complex",
        }:
            raise ValueError(f"Unsupported loss: {self.loss}")
        if self.loss_eps <= 0:
            raise ValueError("loss_eps must be positive.")
        if self.beta_si_sdr < 0:
            raise ValueError("beta_si_sdr must be non-negative.")
        if self.sisdr_warmup_start < 0 or self.sisdr_warmup_end < self.sisdr_warmup_start:
            raise ValueError("SI-SDR warmup steps must be non-negative and ordered.")
        if not 0 < self.compression_exponent <= 1:
            raise ValueError("compression_exponent must be in (0, 1].")
        if not 0 <= self.compression_complex_weight <= 1:
            raise ValueError("compression_complex_weight must be in [0, 1].")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be positive when set.")
        if self.scheduler_total_steps is not None and self.scheduler_total_steps <= 0:
            raise ValueError("scheduler_total_steps must be positive when set.")
        if self.selection_metric != "si_sdr_improvement":
            raise ValueError("selection_metric must be si_sdr_improvement.")

    @property
    def clean_root(self) -> Path:
        assert self.data is not None
        return self.data.clean_root

    @property
    def noise_root(self) -> Path:
        assert self.data is not None
        return self.data.noise_root

    @property
    def segment_len(self) -> float:
        assert self.data is not None
        return self.data.segment_len

    def validate_data_paths(self) -> None:
        if not self.clean_root.is_dir():
            raise FileNotFoundError(f"Clean dataset root not found: {self.clean_root}")
        if not self.noise_root.is_dir():
            raise FileNotFoundError(f"Noise dataset root not found: {self.noise_root}")

    def to_dict(self) -> dict[str, Any]:
        out = {
            "project_root": str(self.project_root),
            "sr": self.audio.sr,
            "n_fft": self.audio.n_fft,
            "hop": self.audio.hop,
            "freq_bins": self.audio.freq_bins,
            "clean_root": str(self.clean_root),
            "noise_root": str(self.noise_root),
            "segment_len": self.segment_len,
        }
        for key in (
            "batch_size", "num_workers", "epochs", "lr", "weight_decay", "optimizer",
            "scheduler", "mixed_precision", "enc_channels", "num_encoder_blocks",
            "gru_hidden", "gru_layers", "k_tap", "activation", "loss", "alpha_loss",
            "beta_si_sdr", "snr_range", "p_rir", "p_vary", "cache_mixtures", "seed",
            "n_val", "steps_per_epoch", "save_every_n_steps",
            "compute_stoi", "compute_pesq", "metric_max_items",
            "loss_eps", "sisdr_warmup_start", "sisdr_warmup_end",
            "compression_exponent", "compression_complex_weight", "checkpoint_dir",
            "max_steps", "scheduler_total_steps", "selection_metric", "experiment_id",
        ):
            value = getattr(self, key)
            out[key] = str(value) if key == "checkpoint_dir" and value is not None else value
        return out

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any], project_root: Path) -> "TrainConfig":
        known = {
            "sr", "n_fft", "hop", "freq_bins", "clean_root", "noise_root",
            "segment_len", "batch_size", "num_workers", "epochs", "lr",
            "weight_decay", "optimizer", "scheduler", "mixed_precision",
            "enc_channels", "num_encoder_blocks", "gru_hidden", "gru_layers",
            "k_tap", "activation", "loss", "alpha_loss", "beta_si_sdr",
            "loss_eps", "sisdr_warmup_start", "sisdr_warmup_end",
            "compression_exponent", "compression_complex_weight", "checkpoint_dir",
            "max_steps", "scheduler_total_steps", "selection_metric", "experiment_id",
            "snr_range", "p_rir", "p_vary", "cache_mixtures", "seed", "n_val",
            "steps_per_epoch", "save_every_n_steps",
            "compute_stoi", "compute_pesq", "metric_max_items",
        }
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown config keys: {', '.join(unknown)}")

        root = Path(project_root).resolve()

        def resolve_path(value: Any, default: str) -> Path:
            path = Path(default if value in (None, "") else str(value))
            return (root / path).resolve() if not path.is_absolute() else path.resolve()

        audio = AudioConfig(
            sr=int(values.get("sr", 16000)),
            n_fft=int(values.get("n_fft", 320)),
            hop=int(values.get("hop", 160)),
            freq_bins=int(values.get("freq_bins", 161)),
        )
        data = DataConfig(
            clean_root=resolve_path(values.get("clean_root"), "data/clean_16k"),
            noise_root=resolve_path(values.get("noise_root"), "data/noise_16k"),
            segment_len=float(values.get("segment_len", 4.0)),
        )
        kwargs = {
            key: values[key]
            for key in known
            if key in values
            and key not in {
                "sr", "n_fft", "hop", "freq_bins", "clean_root", "noise_root",
                "segment_len", "checkpoint_dir",
            }
        }
        kwargs["checkpoint_dir"] = resolve_path(
            values.get("checkpoint_dir"), "checkpoints"
        )
        if "snr_range" in kwargs:
            snr = tuple(float(x) for x in kwargs["snr_range"])
            if len(snr) != 2 or snr[0] > snr[1]:
                raise ValueError("snr_range must contain [minimum, maximum].")
            kwargs["snr_range"] = snr
        return cls(project_root=root, audio=audio, data=data, **kwargs)


def _parse_flat_yaml(text: str) -> dict[str, Any]:
    """Parse the flat scalar/list subset used by this project's config files."""
    values: dict[str, Any] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Invalid config line {line_number}: {raw_line}")
        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key or not raw_value:
            raise ValueError(f"Invalid config line {line_number}: {raw_line}")
        try:
            value = json.loads(raw_value)
        except json.JSONDecodeError:
            lowered = raw_value.lower()
            if lowered in {"true", "false"}:
                value = lowered == "true"
            elif lowered in {"null", "none"}:
                value = None
            else:
                value = raw_value.strip("'\"")
        values[key] = value
    return values


def load_train_config(path: Path, project_root: Path | None = None) -> TrainConfig:
    config_path = Path(path)
    values = _parse_flat_yaml(config_path.read_text(encoding="utf-8"))
    root = Path(project_root) if project_root is not None else config_path.resolve().parents[1]
    return TrainConfig.from_mapping(values, project_root=root)
