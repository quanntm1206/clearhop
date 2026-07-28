from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

import src.trainer as trainer
from src.config import AudioConfig
import scripts.train as train_script


class _FakeConfig:
    def __init__(self, root: Path, **overrides: object) -> None:
        self.project_root = root
        self.clean_root = root / "clean"
        self.noise_root = root / "noise"
        self.clean_root.mkdir(exist_ok=True)
        self.noise_root.mkdir(exist_ok=True)
        self.audio = AudioConfig()
        self.checkpoint_dir = Path(overrides.get("checkpoint_dir", root / "checkpoints"))
        self.values = {
            "project_root": str(root),
            "sr": 16000,
            "segment_len": 0.001,
            "batch_size": 1,
            "num_workers": 0,
            "epochs": 1,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "scheduler": "onecycle",
            "mixed_precision": False,
            "steps_per_epoch": 5,
            "save_every_n_steps": 99,
            "n_val": 1,
            "loss": "complex_mse_plus_si_sdr",
            "alpha_loss": 1.0,
            "beta_si_sdr": 0.5,
            "loss_eps": 1e-8,
            "sisdr_warmup_start": 0,
            "sisdr_warmup_end": 10,
            "compression_exponent": 0.3,
            "compression_complex_weight": 0.3,
            "checkpoint_dir": str(self.checkpoint_dir),
            "max_steps": 3,
            "scheduler_total_steps": 30000,
            "selection_metric": "si_sdr_improvement",
            "experiment_id": "trainer-control",
            "seed": 17,
        }
        self.values.update(overrides)
        self.values["checkpoint_dir"] = str(self.checkpoint_dir)

    def to_dict(self) -> dict[str, object]:
        return dict(self.values)

    def validate_data_paths(self) -> None:
        return None


class _TinyDataset:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.epochs: list[int] = []

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return _batch()

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


class _TinyModel(torch.nn.Module):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, feats: torch.Tensor, state: object) -> tuple[torch.Tensor, None, None]:
        return self.gain, None, None


class _Scheduler:
    instances: list["_Scheduler"] = []

    def __init__(self, optimizer: object, **kwargs: object) -> None:
        self.optimizer = optimizer
        self.total_steps = int(kwargs.get("total_steps", kwargs.get("T_max", 0)))
        self.calls = 0
        self.__class__.instances.append(self)

    def step(self) -> None:
        self.calls += 1

    def state_dict(self) -> dict[str, int]:
        return {"calls": self.calls, "total_steps": self.total_steps}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.calls = int(state["calls"])


class _Writer:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.closed = False

    def add_scalar(self, tag: str, value: float, global_step: int) -> None:
        self.scalars.append((tag, value, global_step))

    def close(self) -> None:
        self.closed = True


class _TrackingOptimizer:
    instances: list["_TrackingOptimizer"] = []

    def __init__(self, params: object, **kwargs: object) -> None:
        self.param_groups = [{"lr": float(kwargs.get("lr", 0.0))}]
        self.steps = 0
        self.__class__.instances.append(self)

    def zero_grad(self, set_to_none: bool = True) -> None:
        return None

    def step(self) -> None:
        self.steps += 1

    def state_dict(self) -> dict[str, object]:
        return {"steps": self.steps}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.steps = int(state.get("steps", 0))


class _OverflowScaler:
    instances: list["_OverflowScaler"] = []

    def __init__(self, *args: object, **kwargs: object) -> None:
        self.current_scale = 8.0
        self.attempts = 0
        self.__class__.instances.append(self)

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss

    def unscale_(self, optimizer: object) -> None:
        return None

    def step(self, optimizer: _TrackingOptimizer) -> None:
        self.attempts += 1
        if self.attempts > 1:
            optimizer.step()

    def update(self) -> None:
        if self.attempts == 1:
            self.current_scale /= 2.0

    def get_scale(self) -> float:
        return self.current_scale

    def state_dict(self) -> dict[str, float]:
        return {"scale": self.current_scale, "attempts": float(self.attempts)}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.current_scale = float(state.get("scale", self.current_scale))
        self.attempts = int(state.get("attempts", self.attempts))


class _AlwaysOverflowScaler(_OverflowScaler):
    instances: list["_AlwaysOverflowScaler"] = []

    def step(self, optimizer: _TrackingOptimizer) -> None:
        self.attempts += 1
        if self.attempts > 10:
            raise RuntimeError("trainer did not terminate repeated AMP skips")

    def update(self) -> None:
        self.current_scale /= 2.0


def _batch() -> dict[str, torch.Tensor]:
    return {
        "clean": torch.tensor([[1.0, -1.0, 0.5, -0.5]]),
        "noisy": torch.tensor([[0.75, -0.75, 0.25, -0.25]]),
    }


def _stft(x: torch.Tensor, **kwargs: object) -> torch.Tensor:
    return torch.stack((x, torch.zeros_like(x)), dim=-1).unsqueeze(1)


def _istft(x: torch.Tensor, **kwargs: object) -> torch.Tensor:
    return x[:, 0, :, 0]


class TestTrainerControl(unittest.TestCase):
    def setUp(self) -> None:
        _Scheduler.instances.clear()
        _TrackingOptimizer.instances.clear()
        _OverflowScaler.instances.clear()
        _AlwaysOverflowScaler.instances.clear()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        manifests = self.root / "manifests" / "v2"
        manifests.mkdir(parents=True)
        for name in ("fold_0_train.jsonl", "fold_0_val.jsonl", "fold_0_test.jsonl"):
            (manifests / name).write_text("{}\n", encoding="utf-8")
        self.historical_best = self.root / "checkpoints" / "best.pth"
        self.historical_best.parent.mkdir()
        self.historical_best.write_bytes(b"historical-best")
        self.historical_full = self.root / "checkpoints" / "full_export.ts"
        self.historical_full.write_bytes(b"historical-full")
        self.output = self.root / "checkpoints" / "gain_calibration" / "arm"
        self.writer = _Writer()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _resume_path(self, *, with_best: bool = True) -> Path:
        self.output.mkdir(parents=True, exist_ok=True)
        if with_best:
            (self.output / "best.pth").touch()
        resume = self.output / "resume.pth"
        resume.touch()
        return resume

    def _patches(self, cfg: _FakeConfig, *, optimizer: object = torch.optim.AdamW):
        def make_loader(dataset: object, **kwargs: object):
            return [_batch()]

        return (
            patch.object(trainer.TrainConfig, "from_mapping", return_value=cfg),
            patch.object(trainer, "list_audio_files", return_value=["audio.wav"]),
            patch.object(trainer, "load_manifest", return_value=[{"clean_path": "clean.wav"}]),
            patch.object(trainer, "manifest_fingerprints", return_value={}),
            patch.object(trainer, "NoiseSuppressionDataset", _TinyDataset),
            patch.object(trainer, "DataLoader", side_effect=make_loader),
            patch.object(trainer, "MobileDeepFilterNet", _TinyModel),
            patch.object(trainer, "_stft_ri", side_effect=_stft),
            patch.object(trainer, "_logp_b1ft", side_effect=lambda x: x[..., 0].unsqueeze(1)),
            patch.object(trainer, "causal_deep_filter", side_effect=lambda x, gain, taps: x * gain),
            patch.object(trainer, "_istft_from_ri", side_effect=_istft),
            patch.object(trainer, "SummaryWriter", return_value=self.writer),
            patch.object(trainer, "AdamW", optimizer),
            patch.object(torch.optim.lr_scheduler, "OneCycleLR", _Scheduler),
            patch.object(torch.optim.lr_scheduler, "CosineAnnealingLR", _Scheduler),
        )

    def _train(self, cfg: _FakeConfig, *, resume: Path | None = None, optimizer: object = torch.optim.AdamW) -> None:
        patches = self._patches(cfg, optimizer=optimizer)
        for item in patches:
            item.start()
        try:
            trainer.train_model({"project_root": str(self.root)}, resume=resume)
        finally:
            for item in reversed(patches):
                item.stop()

    def test_bounded_stop_keeps_scheduler_horizon_and_validates_partial_epoch(self) -> None:
        cfg = _FakeConfig(self.root, checkpoint_dir=self.output, max_steps=3, scheduler_total_steps=30000)

        self._train(cfg)

        summary = json.loads((self.output / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["global_step"], 3)
        self.assertEqual(summary["stop_reason"], "max_steps")
        self.assertEqual(_Scheduler.instances[0].total_steps, 30000)
        self.assertEqual(_Scheduler.instances[0].calls, 3)
        self.assertTrue(any(tag.startswith("val/") and step == 3 for tag, _, step in self.writer.scalars))
        self.assertTrue(any(tag == "train/loss" for tag, _, _ in self.writer.scalars))
        epoch_summary = json.loads(
            (self.output / "epoch_summaries" / "epoch_000.json").read_text(encoding="utf-8")
        )
        self.assertIn("train_loss_mean", epoch_summary)

    def test_all_artifacts_stay_under_configured_checkpoint_dir(self) -> None:
        cfg = _FakeConfig(self.root, checkpoint_dir=self.output)

        self._train(cfg)

        self.assertEqual(self.historical_best.read_bytes(), b"historical-best")
        self.assertEqual(self.historical_full.read_bytes(), b"historical-full")
        self.assertTrue((self.output / "step_3.pth").is_file())
        self.assertTrue((self.output / "best.pth").is_file())
        self.assertTrue((self.output / "run_summary.json").is_file())
        for name in ("step_3.pth", "best.pth"):
            state = torch.load(self.output / name, map_location="cpu", weights_only=False)
            self.assertEqual(state["experiment_id"], "trainer-control")
            self.assertEqual(state["loss_config"]["loss"], "complex_mse_plus_si_sdr")
            self.assertEqual(state["scheduler_total_steps"], 30000)
            self.assertEqual(state["max_steps"], 3)
            self.assertEqual(state["consumed_batches_in_epoch"], 3)
            self.assertEqual(state["successful_update_count"], 3)
            self.assertTrue(state["component_history_complete"])
            self.assertEqual(
                set(state["component_sums"]),
                {"total", "complex_mse", "complex_nmse", "compressed_complex", "si_sdr", "si_sdr_weight"},
            )
        best = torch.load(self.output / "best.pth", map_location="cpu", weights_only=False)
        self.assertEqual((best["epoch"], best["step_in_epoch"]), (0, 3))

    def test_snr_selection_uses_validation_snri(self) -> None:
        cfg = _FakeConfig(
            self.root,
            checkpoint_dir=self.output,
            steps_per_epoch=1,
            max_steps=1,
            selection_metric="snr_improvement",
        )
        patches = self._patches(cfg)
        patches += (patch.object(trainer, "snr_db", side_effect=[5.0, 1.0]),)
        for item in patches:
            item.start()
        try:
            trainer.train_model({"project_root": str(self.root)})
        finally:
            for item in reversed(patches):
                item.stop()

        state = torch.load(self.output / "best.pth", map_location="cpu", weights_only=False)
        self.assertEqual(state["best_val_metric"], 4.0)
        self.assertEqual(state["selection_metric"], "snr_improvement")

    def test_resume_runs_only_missing_optimizer_step(self) -> None:
        cfg = _FakeConfig(self.root, checkpoint_dir=self.output, max_steps=3)
        resume = self._resume_path()
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        state = {
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "sched": {"calls": 2, "total_steps": 30000},
            "scaler": {},
            "global_step": 2,
            "best_val_metric": -1e9,
        }
        patches = self._patches(cfg)
        patches += (patch.object(torch, "load", return_value=state),)
        for item in patches:
            item.start()
        try:
            trainer.train_model({"project_root": str(self.root)}, resume=resume)
        finally:
            for item in reversed(patches):
                item.stop()

        self.assertEqual(_Scheduler.instances[0].calls, 3)
        summary = json.loads((self.output / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["global_step"], 3)

    def test_nonfinite_component_raises_before_optimizer_step(self) -> None:
        cfg = _FakeConfig(self.root, checkpoint_dir=self.output, loss="complex_nmse")
        patches = self._patches(cfg, optimizer=_TrackingOptimizer)
        patches += (
            patch.object(
                trainer,
                "complex_nmse",
                return_value=torch.tensor(float("nan")),
                create=True,
            ),
        )
        for item in patches:
            item.start()
        try:
            with self.assertRaises(FloatingPointError):
                trainer.train_model({"project_root": str(self.root)})
        finally:
            for item in reversed(patches):
                item.stop()

        self.assertEqual(_TrackingOptimizer.instances[0].steps, 0)

    def test_nonfinite_unselected_component_raises_before_optimizer_step(self) -> None:
        cfg = _FakeConfig(self.root, checkpoint_dir=self.output, loss="complex_nmse")
        patches = self._patches(cfg, optimizer=_TrackingOptimizer)
        patches += (
            patch.object(
                trainer,
                "si_sdr_loss",
                return_value=torch.tensor(float("nan")),
            ),
        )
        for item in patches:
            item.start()
        try:
            with self.assertRaises(FloatingPointError):
                trainer.train_model({"project_root": str(self.root)})
        finally:
            for item in reversed(patches):
                item.stop()

        self.assertEqual(_TrackingOptimizer.instances[0].steps, 0)

    def test_amp_overflow_does_not_consume_successful_step_budget(self) -> None:
        cfg = _FakeConfig(self.root, checkpoint_dir=self.output, max_steps=3)
        patches = self._patches(cfg, optimizer=_TrackingOptimizer)
        patches += (patch.object(torch.amp, "GradScaler", _OverflowScaler),)
        for item in patches:
            item.start()
        try:
            trainer.train_model({"project_root": str(self.root)})
        finally:
            for item in reversed(patches):
                item.stop()

        self.assertEqual(_OverflowScaler.instances[0].attempts, 4)
        self.assertEqual(_TrackingOptimizer.instances[0].steps, 3)
        self.assertEqual(_Scheduler.instances[0].calls, 3)
        summary = json.loads((self.output / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["global_step"], 3)

    def test_repeated_amp_overflow_terminates(self) -> None:
        cfg = _FakeConfig(self.root, checkpoint_dir=self.output, max_steps=1)
        patches = self._patches(cfg, optimizer=_TrackingOptimizer)
        patches += (patch.object(torch.amp, "GradScaler", _AlwaysOverflowScaler),)
        for item in patches:
            item.start()
        try:
            with self.assertRaises(FloatingPointError):
                trainer.train_model({"project_root": str(self.root)})
        finally:
            for item in reversed(patches):
                item.stop()

        self.assertEqual(_AlwaysOverflowScaler.instances[0].attempts, 8)
        self.assertEqual(_TrackingOptimizer.instances[0].steps, 0)
        self.assertEqual(_Scheduler.instances[0].calls, 0)

    def test_overflow_checkpoint_resume_uses_consumed_batch_cursor(self) -> None:
        def distinct_loader(dataset: object, **kwargs: object):
            if isinstance(dataset, torch.utils.data.Subset):
                return [_batch()]
            return [
                {"clean": torch.full((1, 4), float(value)), "noisy": torch.full((1, 4), 0.5)}
                for value in (1, 2, 3, 4, 5)
            ]

        objective = trainer._compute_objective
        first_seen: list[float] = []

        def record_first(*args: object, **kwargs: object):
            clean = args[4]
            assert isinstance(clean, torch.Tensor)
            first_seen.append(float(clean[0, 0]))
            return objective(*args, **kwargs)

        first_cfg = _FakeConfig(self.root, checkpoint_dir=self.output, max_steps=1)
        patches = self._patches(first_cfg, optimizer=_TrackingOptimizer)
        patches += (
            patch.object(trainer, "DataLoader", side_effect=distinct_loader),
            patch.object(trainer, "_compute_objective", side_effect=record_first),
            patch.object(torch.amp, "GradScaler", _OverflowScaler),
        )
        for item in patches:
            item.start()
        try:
            trainer.train_model({"project_root": str(self.root)})
        finally:
            for item in reversed(patches):
                item.stop()

        resume = self.output / "step_1.pth"
        state = torch.load(resume, map_location="cpu", weights_only=False)
        self.assertEqual(first_seen, [1.0, 2.0])
        self.assertEqual(state["consumed_batches_in_epoch"], 2)
        self.assertEqual(state["successful_update_count"], 1)

        second_seen: list[float] = []

        def record_second(*args: object, **kwargs: object):
            clean = args[4]
            assert isinstance(clean, torch.Tensor)
            second_seen.append(float(clean[0, 0]))
            return objective(*args, **kwargs)

        second_cfg = _FakeConfig(self.root, checkpoint_dir=self.output, max_steps=2)
        patches = self._patches(second_cfg, optimizer=_TrackingOptimizer)
        patches += (
            patch.object(trainer, "DataLoader", side_effect=distinct_loader),
            patch.object(trainer, "_compute_objective", side_effect=record_second),
            patch.object(torch.amp, "GradScaler", _OverflowScaler),
        )
        for item in patches:
            item.start()
        try:
            trainer.train_model({"project_root": str(self.root)}, resume=resume)
        finally:
            for item in reversed(patches):
                item.stop()

        self.assertEqual(second_seen, [3.0])
        summary = json.loads((self.output / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["successful_update_count"], 2)
        self.assertTrue(summary["component_history_complete"])
        self.assertIsInstance(summary["component_loss_means"], dict)

    def test_resume_consumes_prior_epoch_batches_before_next_update(self) -> None:
        cfg = _FakeConfig(self.root, checkpoint_dir=self.output, max_steps=3)
        resume = self._resume_path()
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        state = {
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "sched": {"calls": 2, "total_steps": 30000},
            "scaler": {},
            "global_step": 2,
            "best_val_metric": -1e9,
            "scheduler_total_steps": 30000,
        }
        seen_clean: list[float] = []
        generator_seeds: list[int] = []
        objective = trainer._compute_objective

        def distinct_loader(dataset: object, **kwargs: object):
            if isinstance(dataset, torch.utils.data.Subset):
                return [_batch()]
            generator = kwargs.get("generator")
            if generator is not None:
                generator_seeds.append(generator.initial_seed())
            return [
                {"clean": torch.full((1, 4), float(value)), "noisy": torch.full((1, 4), 0.5)}
                for value in (1, 2, 3, 4, 5)
            ]

        def record_objective(*args: object, **kwargs: object):
            clean = args[4]
            assert isinstance(clean, torch.Tensor)
            seen_clean.append(float(clean[0, 0]))
            return objective(*args, **kwargs)

        patches = self._patches(cfg)
        patches += (
            patch.object(trainer, "DataLoader", side_effect=distinct_loader),
            patch.object(trainer, "_compute_objective", side_effect=record_objective),
            patch.object(torch, "load", return_value=state),
        )
        for item in patches:
            item.start()
        try:
            trainer.train_model({"project_root": str(self.root)}, resume=resume)
        finally:
            for item in reversed(patches):
                item.stop()

        self.assertEqual(seen_clean, [3.0])
        self.assertEqual(generator_seeds, [17])

    def test_resume_above_max_steps_is_rejected(self) -> None:
        cfg = _FakeConfig(self.root, checkpoint_dir=self.output, max_steps=3)
        resume = self._resume_path(with_best=False)
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        state = {
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "sched": {"calls": 4, "total_steps": 30000},
            "scaler": {},
            "global_step": 4,
            "scheduler_total_steps": 30000,
        }
        patches = self._patches(cfg)
        patches += (patch.object(torch, "load", return_value=state),)
        for item in patches:
            item.start()
        try:
            with self.assertRaises(ValueError):
                trainer.train_model({"project_root": str(self.root)}, resume=resume)
        finally:
            for item in reversed(patches):
                item.stop()

    def test_resume_at_max_steps_validates_without_optimizer_update(self) -> None:
        cfg = _FakeConfig(self.root, checkpoint_dir=self.output, max_steps=3)
        resume = self._resume_path()
        model = _TinyModel()
        state = {
            "model": model.state_dict(),
            "opt": {"steps": 3},
            "sched": {"calls": 3, "total_steps": 30000},
            "scaler": {},
            "global_step": 3,
            "best_val_metric": -1e9,
            "scheduler_total_steps": 30000,
        }
        patches = self._patches(cfg, optimizer=_TrackingOptimizer)
        patches += (patch.object(torch, "load", return_value=state),)
        for item in patches:
            item.start()
        try:
            trainer.train_model({"project_root": str(self.root)}, resume=resume)
        finally:
            for item in reversed(patches):
                item.stop()

        self.assertEqual(_TrackingOptimizer.instances[0].steps, 3)
        summary = json.loads((self.output / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["global_step"], 3)
        self.assertTrue(math.isfinite(summary["best_metric"]))
        self.assertTrue(math.isfinite(summary["val_si_sdr_impr"]))
        self.assertFalse(summary["component_history_complete"])
        self.assertIsNone(summary["component_loss_means"])
        self.assertEqual(summary["successful_update_count_scope"], "post_resume")

    def test_resume_scheduler_horizon_mismatch_is_rejected(self) -> None:
        cfg = _FakeConfig(self.root, checkpoint_dir=self.output, max_steps=3)
        resume = self._resume_path(with_best=False)
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        state = {
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "sched": {"calls": 2, "total_steps": 99},
            "scaler": {},
            "global_step": 2,
            "scheduler_total_steps": 99,
        }
        patches = self._patches(cfg)
        patches += (patch.object(torch, "load", return_value=state),)
        for item in patches:
            item.start()
        try:
            with self.assertRaises(ValueError):
                trainer.train_model({"project_root": str(self.root)}, resume=resume)
        finally:
            for item in reversed(patches):
                item.stop()

    def test_cross_directory_resume_is_rejected(self) -> None:
        cfg = _FakeConfig(self.root, checkpoint_dir=self.output, max_steps=3)
        resume = self.root / "resume.pth"
        resume.touch()
        state = {
            "global_step": 2,
            "scheduler_total_steps": 30000,
            "best_val_metric": 1.0,
        }
        patches = self._patches(cfg)
        patches += (patch.object(torch, "load", return_value=state),)
        for item in patches:
            item.start()
        try:
            with self.assertRaisesRegex(ValueError, "checkpoint_dir"):
                trainer.train_model({"project_root": str(self.root)}, resume=resume)
        finally:
            for item in reversed(patches):
                item.stop()

    def test_resume_with_finite_best_requires_local_best_checkpoint(self) -> None:
        cfg = _FakeConfig(self.root, checkpoint_dir=self.output, max_steps=3)
        resume = self._resume_path(with_best=False)
        model = _TinyModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        state = {
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "sched": {"calls": 2, "total_steps": 30000},
            "scaler": {},
            "global_step": 2,
            "best_val_metric": 1.0,
            "has_best_checkpoint": True,
            "scheduler_total_steps": 30000,
        }
        patches = self._patches(cfg)
        patches += (patch.object(torch, "load", return_value=state),)
        for item in patches:
            item.start()
        try:
            with self.assertRaises(FileNotFoundError):
                trainer.train_model({"project_root": str(self.root)}, resume=resume)
        finally:
            for item in reversed(patches):
                item.stop()

    def test_prevalidation_periodic_checkpoint_resumes_without_best_file(self) -> None:
        cfg = _FakeConfig(
            self.root,
            checkpoint_dir=self.output,
            steps_per_epoch=2,
            max_steps=2,
            save_every_n_steps=1,
        )
        patches = self._patches(cfg)
        patches += (
            patch.object(
                trainer,
                "_selection_value",
                side_effect=RuntimeError("stop before best checkpoint"),
            ),
        )
        for item in patches:
            item.start()
        try:
            with self.assertRaisesRegex(RuntimeError, "before best"):
                trainer.train_model({"project_root": str(self.root)})
        finally:
            for item in reversed(patches):
                item.stop()

        resume = self.output / "step_1.pth"
        state = torch.load(resume, map_location="cpu", weights_only=False)
        self.assertFalse(state["has_best_checkpoint"])
        self.assertEqual(state["best_val_metric"], -1e9)
        self.assertFalse((self.output / "best.pth").exists())

        resume_cfg = _FakeConfig(
            self.root,
            checkpoint_dir=self.output,
            steps_per_epoch=2,
            max_steps=2,
            save_every_n_steps=1,
        )
        self._train(resume_cfg, resume=resume)

        summary = json.loads((self.output / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["global_step"], 2)
        self.assertTrue(summary["has_best_checkpoint"])
        self.assertEqual(summary["successful_update_count_scope"], "full_run")

    def test_nonfinite_selected_validation_metric_is_rejected(self) -> None:
        cfg = _FakeConfig(
            self.root,
            checkpoint_dir=self.output,
            steps_per_epoch=1,
            max_steps=1,
            selection_metric="snr_improvement",
        )
        for invalid in (float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                patches = self._patches(cfg)
                patches += (patch.object(trainer, "snr_db", side_effect=[invalid, 0.0]),)
                for item in patches:
                    item.start()
                try:
                    with self.assertRaises(FloatingPointError):
                        trainer.train_model({"project_root": str(self.root)})
                finally:
                    for item in reversed(patches):
                        item.stop()

    def test_completed_summary_is_written_after_final_checkpoint_and_writer_close(self) -> None:
        cfg = _FakeConfig(self.root, checkpoint_dir=self.output)
        events: list[str] = []
        save = torch.save
        save_json = trainer._save_json

        def record_save(obj: object, path: str) -> None:
            save(obj, path)
            events.append(f"checkpoint:{Path(path).name}")

        def record_json(path: Path, obj: dict[str, object]) -> None:
            if path.name == "run_summary.json":
                self.assertTrue(self.writer.closed)
                self.assertIn("checkpoint:step_3.pth", events)
                events.append("completed-summary")
            save_json(path, obj)

        patches = self._patches(cfg)
        patches += (
            patch.object(torch, "save", side_effect=record_save),
            patch.object(trainer, "_save_json", side_effect=record_json),
        )
        for item in patches:
            item.start()
        try:
            trainer.train_model({"project_root": str(self.root)})
        finally:
            for item in reversed(patches):
                item.stop()

        self.assertEqual(events[-1], "completed-summary")


class TestTrainCli(unittest.TestCase):
    def test_cli_applies_only_explicit_training_overrides(self) -> None:
        loaded = Mock()
        loaded.project_root = Path("D:/project")
        loaded.to_dict.return_value = {"epochs": 10, "max_steps": None}
        argv = [
            "train.py",
            "--config",
            "configs/train.yaml",
            "--max-steps",
            "3",
            "--checkpoint-dir",
            "checkpoints/gain_calibration/cli",
            "--experiment-id",
            "cli-test",
        ]
        with (
            patch.object(train_script, "load_train_config", return_value=loaded, create=True),
            patch.object(train_script, "train_model") as train_model,
            patch("sys.argv", argv),
        ):
            train_script.main()

        config = train_model.call_args.kwargs["config"]
        self.assertEqual(config["epochs"], 10)
        self.assertEqual(config["max_steps"], 3)
        self.assertEqual(Path(config["checkpoint_dir"]), Path("checkpoints/gain_calibration/cli"))
        self.assertEqual(config["experiment_id"], "cli-test")

    def test_cli_without_overrides_preserves_loaded_defaults(self) -> None:
        loaded = Mock()
        loaded.project_root = Path("D:/project")
        loaded.to_dict.return_value = {
            "max_steps": None,
            "checkpoint_dir": "D:/project/checkpoints",
            "experiment_id": "baseline",
        }
        with (
            patch.object(train_script, "load_train_config", return_value=loaded),
            patch.object(train_script, "train_model") as train_model,
            patch("sys.argv", ["train.py", "--config", "configs/train.yaml"]),
        ):
            train_script.main()

        config = train_model.call_args.kwargs["config"]
        self.assertIsNone(config["max_steps"])
        self.assertEqual(config["checkpoint_dir"], "D:/project/checkpoints")
        self.assertEqual(config["experiment_id"], "baseline")


if __name__ == "__main__":
    unittest.main()
