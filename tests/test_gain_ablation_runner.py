from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.run_gain_ablation import (
    DEFAULT_ARMS,
    DEFAULT_SEEDS,
    _write_flat_yaml,
    build_plan,
    classify_run,
    summarize_screen,
)


class TestGainAblationRunner(unittest.TestCase):
    def test_refine_plan_is_exact_three_seed_beta_0p015_screen(self) -> None:
        root = Path(__file__).resolve().parents[1]

        plan = build_plan(root, "refine")

        self.assertEqual(len(plan), 3)
        self.assertEqual({item["seed"] for item in plan}, set(DEFAULT_SEEDS))
        self.assertEqual({item["arm"] for item in plan}, {"complex_nmse_sisdr_beta_0p015"})
        self.assertTrue(all(item["max_steps"] == 1500 for item in plan))
        self.assertTrue(all(item["scheduler_total_steps"] == 30000 for item in plan))
        self.assertTrue(all(item["config"]["beta_si_sdr"] == 0.015 for item in plan))
        self.assertTrue(all(Path(item["run_dir"]).parts[-3] == "refine" for item in plan))

    def test_refine_requires_exact_locked_arm_and_seeds(self) -> None:
        root = Path(__file__).resolve().parents[1]

        with self.assertRaisesRegex(ValueError, "Refine stage requires"):
            build_plan(root, "refine", ["complex_nmse"], DEFAULT_SEEDS)
        with self.assertRaisesRegex(ValueError, "Refine stage requires"):
            build_plan(root, "refine", ["complex_nmse_sisdr_beta_0p015"], [17, 29])

    def test_refine_allows_predeclared_beta_0p02_followup(self) -> None:
        root = Path(__file__).resolve().parents[1]

        plan = build_plan(
            root,
            "refine",
            ["complex_nmse_sisdr_beta_0p02"],
            DEFAULT_SEEDS,
        )

        self.assertEqual(len(plan), 3)
        self.assertTrue(all(item["config"]["beta_si_sdr"] == 0.02 for item in plan))
        self.assertTrue(all(item["arm"] == "complex_nmse_sisdr_beta_0p02" for item in plan))

    def test_refine_allows_predeclared_beta_0p03_followup(self) -> None:
        root = Path(__file__).resolve().parents[1]

        plan = build_plan(
            root,
            "refine",
            ["complex_nmse_sisdr_beta_0p03"],
            DEFAULT_SEEDS,
        )

        self.assertEqual(len(plan), 3)
        self.assertTrue(all(item["config"]["beta_si_sdr"] == 0.03 for item in plan))
        self.assertTrue(all(item["arm"] == "complex_nmse_sisdr_beta_0p03" for item in plan))

    def test_refine_promotion_reuses_original_paired_control_gates(self) -> None:
        from scripts.run_gain_ablation import summarize_refinement

        controls = [
            self._completed("control", seed, -4.5, 1.5, 0.77, -0.05, -25.0)
            for seed in DEFAULT_SEEDS
        ]
        candidates = [
            self._completed(
                "complex_nmse_sisdr_beta_0p015",
                seed,
                1.8,
                1.25,
                0.766,
                0.7,
                -3.0,
            )
            for seed in DEFAULT_SEEDS
        ]

        result = summarize_refinement(candidates, controls)

        self.assertEqual(result["promoted_arm"], "complex_nmse_sisdr_beta_0p015")
        self.assertTrue(result["arms"][0]["eligible"])
        self.assertAlmostEqual(result["arms"][0]["mean_paired_stoi_delta"], -0.004)

    def test_full_binding_accepts_passing_refinement_receipt(self) -> None:
        from scripts.run_gain_ablation import _artifact_ref, _screen_binding

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generated = root / "reports/generated"
            generated.mkdir(parents=True)
            controls = [
                self._completed("control", seed, -4.5, 1.5, 0.77, -0.05, -25.0)
                for seed in DEFAULT_SEEDS
            ]
            failed_candidate = [
                self._completed("complex_nmse_sisdr", seed, -1.0, 0.0, 0.7, 0.7, -3.0)
                for seed in DEFAULT_SEEDS
            ]
            screen_path = generated / "gain_calibration_screen.json"
            screen_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "stage": "screen",
                        "status": "completed",
                        "promoted_arm": None,
                        "runs": controls
                        + failed_candidate
                        + [
                            self._completed(arm, seed, -2.0, 0.0, 0.7, 0.7, -3.0)
                            for arm in ("complex_nmse", "compressed_complex")
                            for seed in DEFAULT_SEEDS
                        ],
                    }
                ),
                encoding="utf-8",
            )
            passing = [
                self._completed("complex_nmse_sisdr_beta_0p02", seed, 1.8, 1.25, 0.766, 0.7, -3.0)
                for seed in DEFAULT_SEEDS
            ]
            refine_path = generated / "gain_calibration_refine_complex_nmse_sisdr_beta_0p02.json"
            refine_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "stage": "refine",
                        "status": "completed",
                        "promoted_arm": "complex_nmse_sisdr_beta_0p02",
                        "runs": passing,
                        "screen_report": _artifact_ref(root, screen_path),
                    }
                ),
                encoding="utf-8",
            )

            binding = _screen_binding(root, "complex_nmse_sisdr_beta_0p02")

            self.assertEqual(binding["promoted_arm"], "complex_nmse_sisdr_beta_0p02")
            self.assertEqual(binding["path"], "reports/generated/gain_calibration_refine_complex_nmse_sisdr_beta_0p02.json")

    def test_parallel_record_executor_preserves_plan_order(self) -> None:
        from scripts.run_gain_ablation import _execute_records

        items = [{"arm": "a", "seed": seed} for seed in DEFAULT_SEEDS]

        def fake_execute(_root: Path, item: dict[str, object], *, resume: bool) -> dict[str, object]:
            return {"status": "completed", "seed": item["seed"], "resume": resume}

        with patch("scripts.run_gain_ablation.execute_run", side_effect=fake_execute) as execute:
            records = _execute_records(Path("."), items, resume=True, parallel_workers=3)

        self.assertEqual([row["seed"] for row in records], list(DEFAULT_SEEDS))
        self.assertEqual(execute.call_count, 3)

    def test_default_screen_plan_is_bounded_and_has_exact_arm_configs(self) -> None:
        root = Path(__file__).resolve().parents[1]

        plan = build_plan(root, "screen")

        self.assertEqual(len(plan), 12)
        self.assertEqual(len({str(item["run_dir"]) for item in plan}), 12)
        self.assertEqual(sum(int(item["max_steps"]) for item in plan), 18000)
        expected = {
            "control": {
                "loss": "complex_mse_plus_si_sdr",
                "alpha_loss": 1.0,
                "beta_si_sdr": 0.5,
                "loss_eps": 1e-8,
                "sisdr_warmup_start": 0,
                "sisdr_warmup_end": 0,
                "compression_exponent": 0.3,
                "compression_complex_weight": 0.3,
            },
            "complex_nmse": {
                "loss": "complex_nmse",
                "alpha_loss": 1.0,
                "beta_si_sdr": 0.0,
                "loss_eps": 1e-8,
                "sisdr_warmup_start": 0,
                "sisdr_warmup_end": 0,
                "compression_exponent": 0.3,
                "compression_complex_weight": 0.3,
            },
            "complex_nmse_sisdr": {
                "loss": "complex_nmse_sisdr",
                "alpha_loss": 1.0,
                "beta_si_sdr": 0.01,
                "loss_eps": 1e-8,
                "sisdr_warmup_start": 500,
                "sisdr_warmup_end": 1000,
                "compression_exponent": 0.3,
                "compression_complex_weight": 0.3,
            },
            "compressed_complex": {
                "loss": "compressed_complex",
                "alpha_loss": 1.0,
                "beta_si_sdr": 0.0,
                "loss_eps": 1e-8,
                "sisdr_warmup_start": 0,
                "sisdr_warmup_end": 0,
                "compression_exponent": 0.3,
                "compression_complex_weight": 0.3,
            },
        }
        for item in plan:
            self.assertEqual(item["max_steps"], 1500)
            self.assertEqual(item["scheduler_total_steps"], 30000)
            self.assertEqual(
                {key: item["config"][key] for key in expected[str(item["arm"])]},
                expected[str(item["arm"])],
            )

    def test_full_stage_rejects_more_than_one_arm(self) -> None:
        root = Path(__file__).resolve().parents[1]

        with self.assertRaisesRegex(ValueError, "exactly one arm"):
            build_plan(root, "full", ["complex_nmse", "compressed_complex"], DEFAULT_SEEDS)

    def test_full_stage_requires_exact_three_locked_seeds(self) -> None:
        root = Path(__file__).resolve().parents[1]

        with self.assertRaisesRegex(ValueError, "exactly seeds"):
            build_plan(root, "full", ["complex_nmse"], [17, 29])

    def test_resolved_training_yaml_excludes_runtime_only_project_root(self) -> None:
        from src.config import load_train_config

        root = Path(__file__).resolve().parents[1]
        item = build_plan(root, "screen", ["control"], [17])[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resolved_config.yaml"
            _write_flat_yaml(path, item["config"])

            loaded = load_train_config(path, project_root=root)

            self.assertEqual(loaded.seed, 17)
            self.assertEqual(loaded.max_steps, 1500)

    def test_completed_run_is_skipped_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            summary = run_dir / "run_summary.json"
            original = json.dumps({"status": "completed", "global_step": 1500})
            summary.write_text(original, encoding="utf-8")

            action = classify_run(run_dir, max_steps=1500, resume=False)

            self.assertEqual(action["status"], "already-complete")
            self.assertEqual(summary.read_text(encoding="utf-8"), original)

    def test_completed_reuse_rejects_tampered_resolved_yaml(self) -> None:
        root = Path(__file__).resolve().parents[1]
        item = build_plan(root, "screen", ["control"], [17])[0]
        with tempfile.TemporaryDirectory() as tmp:
            item = dict(item, run_dir=Path(tmp) / "run")
            run_dir = item["run_dir"]
            run_dir.mkdir()
            (run_dir / "run_summary.json").write_text(
                json.dumps({"status": "completed", "global_step": 1500}), encoding="utf-8"
            )
            (run_dir / "resolved_config.json").write_text(
                json.dumps(item["config"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (run_dir / "resolved_config.yaml").write_text("seed: 29\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "resolved YAML|evidence"):
                from scripts.run_gain_ablation import execute_run

                execute_run(root, item, resume=False)

    def test_incomplete_run_requires_resume_and_local_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "run_summary.json").write_text(
                json.dumps({"status": "failed", "global_step": 500}), encoding="utf-8"
            )
            checkpoint = run_dir / "step_500.pth"
            checkpoint.write_bytes(b"checkpoint")

            with self.assertRaisesRegex(FileExistsError, "--resume"):
                classify_run(run_dir, max_steps=1500, resume=False)

            action = classify_run(run_dir, max_steps=1500, resume=True)
            self.assertEqual(action, {"status": "resume", "checkpoint": checkpoint})

    def test_resume_never_restarts_an_incomplete_directory_without_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "stdout.log").write_text("interrupted", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "checkpoint"):
                classify_run(run_dir, max_steps=1500, resume=True)

    def test_malformed_trainer_summary_becomes_failed_evidence(self) -> None:
        root = Path(__file__).resolve().parents[1]
        item = build_plan(root, "screen", ["control"], [17])[0]
        with tempfile.TemporaryDirectory() as tmp:
            item = dict(item, run_dir=Path(tmp) / "run")

            def fake_run(command: list[str], _root: Path):
                item["run_dir"].joinpath("run_summary.json").write_text("{bad", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "trained", ""), 0.01

            with patch("scripts.run_gain_ablation._run", side_effect=fake_run):
                from scripts.run_gain_ablation import execute_run

                result = execute_run(root, item, resume=False)

            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["phase"], "train-evidence")
            persisted = json.loads(item["run_dir"].joinpath("run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["status"], "failed")

    def test_malformed_evaluation_becomes_failed_run_and_aggregate_report(self) -> None:
        source_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(
                source_root / "configs/ablations/gain_calibration",
                root / "configs/ablations/gain_calibration",
            )

            def fake_run(command: list[str], _root: Path):
                if any(value.replace("\\", "/").endswith("scripts/train.py") for value in command):
                    config_path = Path(command[command.index("--config") + 1])
                    from src.config import load_train_config

                    checkpoint_dir = Path(load_train_config(config_path, project_root=root).checkpoint_dir)
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint = checkpoint_dir / "best.pth"
                    checkpoint.write_bytes(b"mock")
                    (checkpoint_dir / "run_summary.json").write_text(
                        json.dumps({"status": "completed", "global_step": 1500, "best_path": str(checkpoint)}),
                        encoding="utf-8",
                    )
                else:
                    output = Path(command[command.index("--output") + 1])
                    output.write_text("{bad", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "mocked", ""), 0.01

            from scripts.run_gain_ablation import run_experiment

            with patch("scripts.run_gain_ablation._run", side_effect=fake_run):
                report = run_experiment(
                    root,
                    stage="screen",
                    arms=["control"],
                    seeds=[17],
                    resume=False,
                )

            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["runs"][0]["status"], "failed")
            persisted = json.loads(
                (root / "reports/generated/gain_calibration_screen.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted["runs"][0]["status"], "failed")

    def test_screen_promotion_requires_paired_three_seed_gates(self) -> None:
        runs = []
        for seed in DEFAULT_SEEDS:
            runs.append(self._completed("control", seed, 1.0, 2.0, 0.80, 0.5, -6.0))
            runs.append(self._completed("complex_nmse", seed, 2.2, 1.8, 0.798, 0.8, -1.0))
            runs.append(self._completed("complex_nmse_sisdr", seed, 2.4, 1.0, 0.79, 0.9, -0.5))
            runs.append(self._completed("compressed_complex", seed, 1.5, 1.9, 0.799, -0.1, -0.2))

        summary = summarize_screen(runs)

        self.assertEqual(summary["promoted_arm"], "complex_nmse")
        rejected = {item["arm"]: item["rejection_reasons"] for item in summary["arms"]}
        self.assertIn("control-is-baseline", rejected["control"])
        self.assertTrue(any("si_sdri_delta" in reason for reason in rejected["complex_nmse_sisdr"]))
        self.assertTrue(any("positive gain" in reason for reason in rejected["compressed_complex"]))

    def test_screen_never_promotes_without_exact_completed_matrix(self) -> None:
        runs = [
            self._completed(arm, seed, 2.5, 2.0, 0.8, 0.9, -0.5)
            for arm in DEFAULT_ARMS
            for seed in DEFAULT_SEEDS
            if not (arm == "control" and seed == 43)
        ]

        summary = summarize_screen(runs)

        self.assertIsNone(summary["promoted_arm"])
        self.assertIn("exact 4x3 completed matrix required", summary["promotion_rejection_reasons"])

    def test_screen_rejects_boolean_or_extra_promotion_metrics(self) -> None:
        runs = [
            self._completed(arm, seed, 2.5, 2.0, 0.8, 0.9, -0.5)
            for arm in DEFAULT_ARMS
            for seed in DEFAULT_SEEDS
        ]
        candidate = next(row for row in runs if row["arm"] == "complex_nmse" and row["seed"] == 17)
        candidate["validation_metrics"]["snri"] = True
        candidate["validation_metrics"]["unexpected"] = 1.0

        summary = summarize_screen(runs)

        self.assertIsNone(summary["promoted_arm"])
        row = next(item for item in summary["arms"] if item["arm"] == "complex_nmse")
        self.assertIn("missing or non-finite validation metrics", row["rejection_reasons"])

    def test_validation_metric_parser_rejects_bool_and_nonfinite(self) -> None:
        from scripts.run_gain_ablation import _evaluation_metrics

        valid = {
            "enhanced": {
                "snr_improvement_mean": 2.0,
                "si_sdr_improvement_mean": 1.0,
                "stoi": {"mean": 0.8},
                "projection_gain": {"median": 0.9},
                "gain_error_db": {"median": -1.0},
            }
        }
        for invalid in (True, "2.0", float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                payload = json.loads(json.dumps(valid))
                payload["enhanced"]["snr_improvement_mean"] = invalid
                with self.assertRaises(FloatingPointError):
                    _evaluation_metrics(payload)

    def test_mocked_full_flow_produces_verifier_consumable_schema(self) -> None:
        import torch

        from scripts.verify import gain_calibration_artifact_audit
        from src.checkpoint import file_sha256
        from src.config import load_train_config
        from src.model import MobileDeepFilterNet, MobileDeepFilterNetConfig
        from src.splits import manifest_fingerprint, slice_fingerprint

        source_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(
                source_root / "configs/ablations/gain_calibration",
                root / "configs/ablations/gain_calibration",
            )
            shutil.copy2(source_root / "configs/train.yaml", root / "configs/train.yaml")
            manifest_root = root / "manifests/v2"
            manifest_root.mkdir(parents=True)
            for split, count in (("train", 10), ("val", 500), ("test", 1000)):
                (manifest_root / f"fold_0_{split}.jsonl").write_text(
                    "".join(json.dumps({"id": index}) + "\n" for index in range(count)),
                    encoding="utf-8",
                )
            test_manifest = manifest_root / "fold_0_test.jsonl"
            val_manifest = manifest_root / "fold_0_val.jsonl"
            test_hash = manifest_fingerprint(test_manifest)
            val_hash = manifest_fingerprint(val_manifest)
            (manifest_root / "fold_0_summary.json").write_text(
                json.dumps({"fingerprints": {"val": val_hash, "test": test_hash}}), encoding="utf-8"
            )
            screen_path = root / "reports/generated/gain_calibration_screen.json"
            screen_path.parent.mkdir(parents=True)
            screen_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "stage": "screen",
                        "status": "completed",
                        "promoted_arm": "complex_nmse",
                        "runs": [
                            self._completed(
                                arm,
                                seed,
                                1.0 if arm == "control" else (2.2 if arm == "complex_nmse" else 1.2),
                                2.0 if arm == "control" else (1.8 if arm == "complex_nmse" else 1.0),
                                0.8 if arm == "control" else (0.798 if arm == "complex_nmse" else 0.79),
                                0.5 if arm == "control" else (0.8 if arm == "complex_nmse" else -0.1),
                                -6.0 if arm == "control" else -1.0,
                            )
                            for arm in DEFAULT_ARMS
                            for seed in DEFAULT_SEEDS
                        ],
                    }
                ),
                encoding="utf-8",
            )
            model = MobileDeepFilterNet(
                MobileDeepFilterNetConfig(enc_channels=2, num_encoder_blocks=1, gru_hidden=2, k_tap=1)
            )
            baseline = root / "checkpoints/best.pth"
            baseline.parent.mkdir(parents=True)
            torch.save(
                {
                    "schema_version": 2,
                    "model": model.state_dict(),
                    "model_cfg": model.cfg.__dict__,
                    "audio_cfg": {"sr": 16000, "n_fft": 320, "hop": 160, "freq_bins": 161},
                    "config": {},
                },
                baseline,
            )
            calls: list[list[str]] = []
            first_test_evaluation_seen = False

            def fake_run(command: list[str], _root: Path):
                nonlocal first_test_evaluation_seen
                calls.append(command)
                if any(value.replace("\\", "/").endswith("scripts/train.py") for value in command):
                    config_path = Path(command[command.index("--config") + 1])
                    cfg = load_train_config(config_path, project_root=root).to_dict()
                    checkpoint_dir = Path(cfg["checkpoint_dir"])
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint = checkpoint_dir / "best.pth"
                    loss_keys = (
                        "loss", "alpha_loss", "beta_si_sdr", "loss_eps",
                        "sisdr_warmup_start", "sisdr_warmup_end",
                        "compression_exponent", "compression_complex_weight",
                    )
                    torch.save(
                        {
                            "schema_version": 2,
                            "model": model.state_dict(),
                            "model_cfg": model.cfg.__dict__,
                            "audio_cfg": {"sr": 16000, "n_fft": 320, "hop": 160, "freq_bins": 161},
                            "config": cfg,
                            "global_step": 30000,
                            "runtime": {"device": "cuda", "gpu": "mock-gpu"},
                            "experiment_id": cfg["experiment_id"],
                            "loss": cfg["loss"],
                            "loss_config": {key: cfg[key] for key in loss_keys},
                            "manifest_fingerprints": {"test": test_hash},
                        },
                        checkpoint,
                    )
                    (checkpoint_dir / "run_summary.json").write_text(
                        json.dumps(
                            {
                                "status": "completed",
                                "global_step": 30000,
                                "best_path": str(checkpoint),
                            }
                        ),
                        encoding="utf-8",
                    )
                else:
                    output = Path(command[command.index("--output") + 1])
                    checkpoint = Path(command[command.index("--checkpoint") + 1])
                    manifest = Path(command[command.index("--manifest") + 1])
                    offset = int(command[command.index("--offset") + 1])
                    count = int(command[command.index("--max-items") + 1])
                    config_path = Path(command[command.index("--config") + 1])
                    if manifest.name.endswith("_test.jsonl") and not first_test_evaluation_seen:
                        first_test_evaluation_seen = True
                        selection_path = root / "reports/generated/gain_calibration_selection.json"
                        self.assertTrue(selection_path.is_file(), "selection receipt must precede test evaluation")
                        frozen = json.loads(selection_path.read_text(encoding="utf-8"))
                        self.assertEqual(frozen["selected_seed"], 17)
                        self.assertEqual(len(frozen["selection_contract_sha256"]), 64)
                    config_json = config_path.with_name("resolved_config.json")
                    config_hash = file_sha256(config_json) if config_json.is_file() else None
                    from tests.test_verify import TestVerificationGate

                    TestVerificationGate._write_gain_evaluation(
                        output,
                        checkpoint_hash=file_sha256(checkpoint),
                        config_hash=config_hash,
                        manifest_hash=manifest_fingerprint(manifest),
                        slice_hash=slice_fingerprint(manifest, offset, count),
                        offset=offset,
                        candidate=checkpoint != baseline,
                    )
                    payload = json.loads(output.read_text(encoding="utf-8"))
                    payload["metadata"]["manifest"] = str(manifest)
                    payload["metadata"]["evaluation_profile"] = (
                        command[command.index("--profile") + 1] if "--profile" in command else "full"
                    )
                    output.write_text(json.dumps(payload), encoding="utf-8")
                    if manifest.name.endswith("_val.jsonl"):
                        self.assertEqual(manifest.resolve(), val_manifest.resolve())
                        self.assertEqual(offset, 0)
                        self.assertEqual(count, 500)
                        payload = json.loads(output.read_text(encoding="utf-8"))
                        seed = load_train_config(config_path, project_root=root).seed
                        payload["enhanced"]["snr_improvement_mean"] = 3.0 - seed / 1000.0
                        output.write_text(json.dumps(payload), encoding="utf-8")
                return subprocess.CompletedProcess(command, 0, "mocked", ""), 0.01

            from scripts.run_gain_ablation import run_experiment

            with patch("scripts.run_gain_ablation._run", side_effect=fake_run):
                report = run_experiment(
                    root,
                    stage="full",
                    arms=["complex_nmse"],
                    seeds=list(DEFAULT_SEEDS),
                    resume=False,
                )

            self.assertEqual(report["selected_arm"], "complex_nmse")
            self.assertEqual(report["selected_seed"], 17)
            self.assertTrue(first_test_evaluation_seen)
            self.assertEqual(len(report["runs"]), 3)
            self.assertEqual(len(calls), 14)
            self.assertEqual(report["screen_report"]["sha256"], file_sha256(screen_path))
            for row in report["runs"]:
                self.assertEqual(set(row["evaluations"]), {"comparison", "audit"})
                for reference in row["evaluations"].values():
                    self.assertEqual(file_sha256(root / reference["path"]), reference["sha256"])

            calls_after_first = len(calls)
            with patch("scripts.run_gain_ablation._run", side_effect=fake_run):
                resumed = run_experiment(
                    root,
                    stage="full",
                    arms=["complex_nmse"],
                    seeds=list(DEFAULT_SEEDS),
                    resume=True,
                )
            self.assertEqual(resumed["status"], "completed")
            self.assertTrue(all(row["execution_status"] == "already-complete" for row in resumed["runs"]))
            self.assertEqual(len(calls) - calls_after_first, 8)

            selected_hash = resumed["selected_checkpoint"]["sha256"]
            export_root = root / "checkpoints/gain_calibration"
            (export_root / "export.ts").write_bytes(b"mock-torchscript")
            (export_root / "export.onnx").write_bytes(b"mock-onnx")
            (export_root / "export.json").write_text(
                json.dumps({"source_checkpoint_sha256": selected_hash}), encoding="utf-8"
            )
            generated = root / "reports/generated"
            (generated / "gain_calibration_export_parity.json").write_text(
                json.dumps(
                    {
                        "status": "pass",
                        "steps": 20,
                        "checkpoint_sha256": selected_hash,
                        "torchscript_sha256": file_sha256(export_root / "export.ts"),
                        "onnx_sha256": file_sha256(export_root / "export.onnx"),
                    }
                ),
                encoding="utf-8",
            )
            latency = {"n": 500, "mean_ms": 1.0, "p95_ms": 2.0, "max_ms": 3.0, "realtime_factor": 5.0}
            (generated / "gain_calibration_benchmark.json").write_text(
                json.dumps(
                    {
                        "checkpoint_sha256": selected_hash,
                        "device": "cuda",
                        "gpu": "mock-gpu",
                        "iterations": 500,
                        "neural_core": latency,
                        "streaming_end_to_end": latency,
                    }
                ),
                encoding="utf-8",
            )
            scripted = MagicMock()
            scripted.eval.return_value = scripted
            with (
                patch("torch.jit.load", return_value=scripted),
                patch("onnxruntime.InferenceSession"),
                patch("scripts.verify.export_parity_audit", return_value={"status": "pass", "steps": 20}),
            ):
                audit = gain_calibration_artifact_audit(root)
            self.assertEqual(audit["status"], "pass", audit)

            selected_path = root / "checkpoints/gain_calibration/best.pth"
            selected_bytes = selected_path.read_bytes()
            selected_path.write_bytes(b"collision")
            with patch("scripts.run_gain_ablation._run", side_effect=fake_run):
                collision = run_experiment(
                    root,
                    stage="full",
                    arms=["complex_nmse"],
                    seeds=list(DEFAULT_SEEDS),
                    resume=True,
                )
            self.assertEqual(collision["status"], "failed")
            self.assertIn("Selected checkpoint collision", collision["error"])
            persisted_collision = json.loads(
                (root / "reports/generated/gain_calibration_full.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted_collision["status"], "failed")
            selected_path.write_bytes(selected_bytes)
            with patch("scripts.run_gain_ablation._run", side_effect=fake_run):
                run_experiment(
                    root,
                    stage="full",
                    arms=["complex_nmse"],
                    seeds=list(DEFAULT_SEEDS),
                    resume=True,
                )

            def failing_nonselected_eval(command: list[str], command_root: Path):
                completed = fake_run(command, command_root)
                if "--output" in command:
                    output = Path(command[command.index("--output") + 1])
                    if "seed_29" in output.as_posix() and output.name == "comparison_evaluation.json":
                        payload = json.loads(output.read_text(encoding="utf-8"))
                        payload["enhanced"]["snr_improvement_mean"] = -0.1
                        output.write_text(json.dumps(payload), encoding="utf-8")
                return completed

            with patch("scripts.run_gain_ablation._run", side_effect=failing_nonselected_eval):
                rejected = run_experiment(
                    root,
                    stage="full",
                    arms=["complex_nmse"],
                    seeds=list(DEFAULT_SEEDS),
                    resume=True,
                )
            self.assertEqual(rejected["status"], "rejected")
            self.assertEqual(rejected["selected_seed"], 17)
            self.assertFalse(rejected["production_eligible"])
            failed_seed = next(row for row in rejected["runs"] if row["seed"] == 29)
            self.assertEqual(failed_seed["status"], "rejected")
            self.assertIn("comparison: snri_positive", failed_seed["rejection_reasons"])

            def malformed_post_eval(command: list[str], command_root: Path):
                completed = fake_run(command, command_root)
                if "--output" in command:
                    output = Path(command[command.index("--output") + 1])
                    if "seed_29" in output.as_posix() and output.name == "audit_evaluation.json":
                        output.write_text("{bad", encoding="utf-8")
                return completed

            with patch("scripts.run_gain_ablation._run", side_effect=malformed_post_eval):
                failed = run_experiment(
                    root,
                    stage="full",
                    arms=["complex_nmse"],
                    seeds=list(DEFAULT_SEEDS),
                    resume=True,
                )
            self.assertEqual(failed["status"], "failed")
            failed_seed = next(row for row in failed["runs"] if row["seed"] == 29)
            self.assertEqual(failed_seed["status"], "failed")
            self.assertIn("Malformed evaluation JSON", failed_seed["error"])

    def test_validation_binding_rejects_manifest_offset_count_and_slice_drift(self) -> None:
        from scripts.run_gain_ablation import _validate_validation_binding
        from src.splits import manifest_fingerprint, slice_fingerprint

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifests/v2/fold_0_val.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                "".join(json.dumps({"id": index}) + "\n" for index in range(500)),
                encoding="utf-8",
            )
            expected = {
                "manifest": str(manifest),
                "manifest_fingerprint": manifest_fingerprint(manifest),
                "slice_offset": 0,
                "slice_count": 500,
                "slice_fingerprint": slice_fingerprint(manifest, 0, 500),
                "max_items": 500,
                "evaluation_profile": "screen",
            }
            _validate_validation_binding(root, {"metadata": dict(expected)})

            mutations = {
                "manifest": ("manifest_fingerprint", "0" * 64),
                "offset": ("slice_offset", 1),
                "count": ("slice_count", 499),
                "slice": ("slice_fingerprint", "f" * 64),
                "profile": ("evaluation_profile", "full"),
            }
            for name, (key, value) in mutations.items():
                with self.subTest(name=name):
                    metadata = dict(expected)
                    metadata[key] = value
                    with self.assertRaisesRegex(ValueError, "validation manifest binding"):
                        _validate_validation_binding(root, {"metadata": metadata})

    @staticmethod
    def _completed(
        arm: str,
        seed: int,
        snri: float,
        si_sdri: float,
        stoi: float,
        gain: float,
        gain_error_db: float,
    ) -> dict[str, object]:
        return {
            "status": "completed",
            "arm": arm,
            "seed": seed,
            "validation_metrics": {
                "snri": snri,
                "si_sdri": si_sdri,
                "stoi": stoi,
                "projection_gain_median": gain,
                "gain_error_db_median": gain_error_db,
            },
        }


if __name__ == "__main__":
    unittest.main()
