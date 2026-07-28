"""
Dataset and on-the-fly mixing logic for training MobileDeepFilterNet.

Requirements covered:
- torch.utils.data.Dataset with train/val modes
- Train: on-the-fly mixing = virtually unlimited mixtures
- Val: deterministic mixtures via manifest + stored per-item seed
- Optional precomputed STFT in returned dict (for debug / profiling)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except Exception as e:  # pragma: no cover
    raise ImportError("This module requires PyTorch. Install torch first.") from e

from .augment import load_or_generate_rir, mix_clean_with_noises
from .utils import crop_or_repeat, derive_item_seed, load_audio_mono, load_manifest


def collate_audio_batch(batch: list[dict]) -> dict:
    """Pad waveforms while keeping heterogeneous metadata as a Python list."""
    max_len = max(max(item["clean"].numel(), item["noisy"].numel()) for item in batch)
    clean = [torch.nn.functional.pad(item["clean"], (0, max_len - item["clean"].numel())) for item in batch]
    noisy = [torch.nn.functional.pad(item["noisy"], (0, max_len - item["noisy"].numel())) for item in batch]
    return {"clean": torch.stack(clean), "noisy": torch.stack(noisy), "meta": [item.get("meta") for item in batch]}


class NoiseSuppressionDataset(Dataset):
    """
    On-the-fly mixture dataset for noise suppression.

    Each item returns:
        {
          "clean": FloatTensor(L,),
          "noisy": FloatTensor(L,),
          "clean_spec": optional STFT,
          "meta": {...}
        }
    """

    def __init__(
        self,
        clean_list: list[str],
        noise_list: list[str],
        segment_len: float = 4.0,
        sr: int = 16000,
        mix_config: dict | None = None,
        mode: str = "train",  # "train" or "val"
        manifest: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.clean_list = list(clean_list)
        self.noise_list = list(noise_list)
        self.segment_len = float(segment_len)
        self.sr = int(sr)
        self.mix_config: Dict = dict(mix_config or {})
        self.mode = str(mode)
        self.manifest = manifest
        self._epoch = 0

        if not self.clean_list:
            raise ValueError("clean_list is empty.")
        if not self.noise_list:
            raise ValueError("noise_list is empty.")
        if self.mode not in {"train", "val"}:
            raise ValueError("mode must be 'train' or 'val'.")

        self.project_root = Path(self.mix_config.get("project_root", Path.cwd()))

        self._manifest_entries: Optional[List[Dict]] = None
        if self.mode == "val":
            if self.manifest is None:
                raise ValueError("Validation requires a manifest path.")
            self._manifest_entries = load_manifest(Path(self.manifest))
            if not self._manifest_entries:
                raise ValueError(f"Manifest is empty: {self.manifest}")

        # Optional STFT precompute in dataset output (not used by default training loop).
        self.return_clean_spec = bool(self.mix_config.get("return_clean_spec", False))

    def __len__(self) -> int:
        if self.mode == "val":
            assert self._manifest_entries is not None
            return len(self._manifest_entries)
        # Train is conceptually unbounded; use clean_list length as an epoch proxy.
        return len(self.clean_list)

    def set_epoch(self, epoch: int) -> None:
        """Select the reproducible mixture generation epoch for train mode."""
        self._epoch = int(epoch)

    def _choose_train_paths(self, idx: int, rng: np.random.RandomState) -> tuple[str, list[str], int]:
        clean_path = self.clean_list[idx % len(self.clean_list)]
        n_noise = int(rng.randint(1, 4))  # 1..3
        noise_paths = [self.noise_list[int(rng.randint(0, len(self.noise_list)))] for _ in range(n_noise)]
        seed = int(rng.randint(0, 2**31 - 1))
        return clean_path, noise_paths, seed

    def _resolve_manifest_path(self, value: str) -> str:
        path = Path(value)
        return str(path if path.is_absolute() else (self.project_root / path).resolve())

    def __getitem__(self, idx: int) -> dict:
        # Deterministic RNG for each item in val; stochastic in train.
        base_seed = int(self.mix_config.get("seed", 42))
        if self.mode == "val":
            assert self._manifest_entries is not None
            entry = self._manifest_entries[idx]
            clean_path = self._resolve_manifest_path(entry["clean_path"])
            noise_paths = [self._resolve_manifest_path(value) for value in entry["noise_paths"]]
            item_seed = int(entry["seed"])
            seg_len = float(entry.get("segment_len", self.segment_len))
            rng = np.random.RandomState(item_seed)
        else:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = 0 if worker_info is None else int(worker_info.id)
            item_seed = derive_item_seed(base_seed, self._epoch, idx, worker_id)
            rng = np.random.RandomState(item_seed)
            clean_path, noise_paths, _ = self._choose_train_paths(idx, rng)
            seg_len = self.segment_len

        clean = load_audio_mono(clean_path, sr=self.sr)
        length = int(round(seg_len * self.sr))
        clean_seg = crop_or_repeat(clean, length, rng)

        # Decide time-varying SNR.
        p_vary = float(self.mix_config.get("p_vary", 0.4))
        time_vary = bool(rng.rand() < p_vary) if self.mode == "train" else bool(self.mix_config.get("val_time_vary", True))

        # Decide RIR on clean.
        p_rir = float(self.mix_config.get("p_rir", 0.35))
        rir = None
        rir_meta = None
        if (self.mode == "train" and rng.rand() < p_rir) or (self.mode == "val" and bool(self.mix_config.get("val_use_rir", True))):
            rir, rir_meta = load_or_generate_rir(self.project_root, sr=self.sr, rng=rng)

        snr_cfg = dict(self.mix_config)
        snr_cfg["seed"] = int(item_seed)
        snr_cfg["project_root"] = self.project_root

        clean_target, noisy, meta = mix_clean_with_noises(
            clean=clean_seg,
            noise_paths=list(noise_paths),
            sr=self.sr,
            segment_len=seg_len,
            snr_config=snr_cfg,
            rir=rir,
            time_varying=time_vary,
            multi_noise=(1, 3),
        )
        if self.mode == "val":
            meta["manifest_id"] = int(entry.get("id", idx))
            meta["clean_path"] = str(entry["clean_path"])
            meta["noise_paths"] = [str(value) for value in entry["noise_paths"]]
        if rir_meta is not None and meta.get("rir") is not None:
            meta["rir"].update(rir_meta)

        # Return tensors
        clean_t = torch.from_numpy(clean_target.astype(np.float32)).float()
        noisy_t = torch.from_numpy(noisy.astype(np.float32)).float()

        out = {"clean": clean_t, "noisy": noisy_t, "meta": meta}

        if self.return_clean_spec:
            # Optional debug: precompute clean STFT.
            window = torch.hann_window(320)
            spec = torch.stft(clean_t, n_fft=320, hop_length=160, win_length=320, window=window, center=False, return_complex=True)
            out["clean_spec"] = spec

        return out
