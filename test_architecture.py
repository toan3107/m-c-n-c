from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import run_experiments
import summarize_experiments
from summarize_experiments import architecture_contrasts
from model import ARCHITECTURE_MODELS, FeatureChangeConditioner, build_model
from train import compute_loss


torch.set_num_threads(2)


class FeatureChangeTests(unittest.TestCase):
    def test_constant_features_and_first_observation_are_unchanged(self):
        features = torch.randn(2, 1, 8, 4, 4).expand(-1, 3, -1, -1, -1)
        for mode in ("change", "rate"):
            conditioner = FeatureChangeConditioner(8, mode, 30.)
            self.assertTrue(torch.equal(conditioner(features, torch.tensor([[0., 5., 17.]]).repeat(2, 1)), features))
            self.assertTrue(torch.equal(conditioner(features[:, :1], torch.zeros(2, 1)), features[:, :1]))

    def test_change_and_rate_have_identical_capacity_and_match_at_30_days(self):
        change = FeatureChangeConditioner(8, "change", 30.)
        rate = FeatureChangeConditioner(8, "rate", 30.)
        rate.load_state_dict(change.state_dict(), strict=True)
        features = torch.randn(2, 3, 8, 4, 4)
        gaps = torch.tensor([[0., 30., 30.]]).repeat(2, 1)
        self.assertTrue(torch.equal(change(features, gaps), rate(features, gaps)))
        gaps[:, 1:] *= 2
        self.assertFalse(torch.allclose(change(features, gaps), rate(features, gaps)))
        self.assertTrue(torch.equal(rate(features, gaps)[:, 0], features[:, 0]))

    def test_invalid_observation_intervals_are_rejected(self):
        conditioner = FeatureChangeConditioner(8, "rate", 30.)
        features = torch.randn(1, 3, 8, 4, 4)
        for value in (0., -1., float("nan")):
            with self.subTest(gap=value), self.assertRaisesRegex(ValueError, "finite and positive"):
                conditioner(features, torch.tensor([[0., value, 5.]]))


class ArchitectureTests(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(42)
        return [torch.randn(1, 3, 3, 32, 32, requires_grad=True),
                torch.tensor([[0., 5., 17.]]), torch.tensor([10.]),
                torch.randn(1, 3, 2), torch.randn(1, 2)]

    def test_all_new_architectures_backpropagate_from_the_same_loss(self):
        for name in ARCHITECTURE_MODELS:
            with self.subTest(model=name):
                inputs = self.inputs()
                model = build_model({"model": name, "base_channels": 4})
                mask, sdf = model(*inputs)
                target = torch.zeros_like(mask)
                valid = torch.ones_like(target)
                loss, _ = compute_loss(mask, sdf, target, target, valid, valid, .1)
                loss.backward()
                self.assertEqual(mask.shape, (1, 1, 32, 32))
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                self.assertGreater(float(model.temporal.time_a.weight.grad.abs().sum()), 0)
                self.assertGreater(float(model.sdf_head.weight.grad.abs().sum()), 0)
                for index in range(3):
                    self.assertGreater(float(inputs[0].grad[:, index].abs().sum()), 0)
                if model.change_conditioner is not None:
                    self.assertGreater(float(model.change_conditioner.projection.weight.grad.abs().sum()), 0)
                if model.multiscale:
                    self.assertGreater(float(model.fine_temporal.time_a.weight.grad.abs().sum()), 0)
                    self.assertGreater(float(model.fine_to_bottleneck.weight.grad.abs().sum()), 0)
                    if model.fine_change_conditioner is not None:
                        self.assertGreater(float(model.fine_change_conditioner.projection.weight.grad.abs().sum()), 0)

    def test_sdf_head_never_drives_future_mask_inference(self):
        for name in ARCHITECTURE_MODELS:
            with self.subTest(model=name):
                inputs = self.inputs()
                model = build_model({"model": name, "base_channels": 4}).eval()
                with torch.no_grad():
                    mask, sdf = model(*inputs)
                    model.sdf_head.bias.add_(3)
                    other_mask, other_sdf = model(*inputs)
                self.assertTrue(torch.equal(mask, other_mask))
                self.assertFalse(torch.equal(sdf, other_sdf))

    def test_all_architectures_use_horizon_and_restore_checkpoints(self):
        for name in ARCHITECTURE_MODELS:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as folder:
                inputs = self.inputs()
                config = {"model": name, "base_channels": 4, "time_scale_days": 30.}
                model = build_model(config).eval()
                path = Path(folder) / "checkpoint.pt"
                torch.save({"config": config, "model": model.state_dict()}, path)
                checkpoint = torch.load(path, weights_only=False)
                restored = build_model(checkpoint["config"]).eval()
                restored.load_state_dict(checkpoint["model"], strict=True)
                with torch.no_grad():
                    first, _ = model(*inputs)
                    same, _ = restored(*inputs)
                    inputs[2] *= 5
                    different, _ = restored(*inputs)
                self.assertTrue(torch.equal(first, same))
                self.assertFalse(torch.allclose(first, different))

    def test_fine_history_control_is_exactly_parameter_matched(self):
        inputs = self.inputs()
        full = build_model({"model": "spatial_cfc_multiscale", "base_channels": 4}).eval()
        control = build_model({"model": "spatial_cfc_multiscale_lastfine", "base_channels": 4}).eval()
        control.load_state_dict(full.state_dict(), strict=True)
        self.assertEqual(sum(p.numel() for p in full.parameters()), sum(p.numel() for p in control.parameters()))
        calls = []
        for model, expected in ((full, 4), (control, 2)):
            elapsed = []
            hook = model.fine_temporal.register_forward_pre_hook(lambda module, args: elapsed.append(float(args[2][0])))
            with torch.no_grad():
                calls.append(model(*inputs)[0])
            hook.remove()
            self.assertEqual(len(elapsed), expected)
            self.assertEqual(elapsed[0], 0.)
        self.assertFalse(torch.allclose(*calls))

    def test_rate_path_is_differentiable_in_observation_intervals(self):
        inputs = self.inputs()
        inputs[1].requires_grad_(True)
        model = build_model({"model": "spatial_cfc_multiscale_rate", "base_channels": 4})
        mask, _ = model(*inputs)
        mask.square().mean().backward()
        self.assertTrue(torch.isfinite(inputs[1].grad).all())
        self.assertGreater(float(inputs[1].grad[:, 1:].abs().sum()), 0)

    def test_existing_backbone_keys_are_preserved(self):
        original = build_model({"model": "spatial_cfc", "base_channels": 4})
        variant = build_model({"model": "spatial_cfc_multiscale_rate", "base_channels": 4})
        self.assertTrue(set(original.state_dict()).issubset(variant.state_dict()))
        self.assertEqual(len(original.state_dict()), 101)
        self.assertEqual(sum(p.numel() for p in original.parameters()), 251790)


class ArchitectureProfileTests(unittest.TestCase):
    def test_full_profile_has_13_runs_one_seed_and_no_sdf_weight_sweep(self):
        output = io.StringIO()
        arguments = ["run_experiments.py", "--stage", "confirm", "--config", "configs/method_arch_v3.json", "--dry-run"]
        with patch("sys.argv", arguments), contextlib.redirect_stdout(output):
            run_experiments.main()
        text = output.getvalue()
        settings = json.JSONDecoder().raw_decode(text)[0]
        self.assertEqual(len(settings["experiments"]), 13)
        self.assertEqual(settings["seeds"], [42])
        self.assertEqual(settings["epochs"], 120)
        self.assertEqual(settings["patience"], 20)
        self.assertEqual(settings["image_size"], 512)
        self.assertEqual(settings["training_splits"], ["train"])
        self.assertEqual(settings["lr_schedule"], "warmup_cosine")
        self.assertNotIn("--split test", text)
        self.assertEqual(text.count("--split val"), 13)
        self.assertNotIn("cfc_sdf_001", settings["experiments"])
        self.assertNotIn("cfc_sdf_003", settings["experiments"])
        for name in ("cfc_sdf", "cfc_no_gap", "cfc_no_calendar", *(name for name in settings["experiments"] if name.endswith("_sdf") and name != "cfc_no_sdf")):
            self.assertEqual(run_experiments.EXPERIMENTS[name][1], .1)


class ArchitectureContrastTests(unittest.TestCase):
    def test_differences_pair_seeds_without_inventing_sd_or_z_comparisons(self):
        def row(seed, iou, boundary, area):
            return dict(seed=seed, iou=iou, boundary_f1_20m=boundary, area_mae_km2=area)
        groups = {
            "cfc_sdf": [row(42, .8, .7, .3), row(43, .9, .8, .2)],
            "cfc_change_sdf": [row(42, .81, .72, .28)],
        }
        contrast = architecture_contrasts(groups)[0]
        self.assertEqual(contrast["paired_seeds"], 1)
        self.assertAlmostEqual(contrast["iou_difference_mean"], .01)
        self.assertAlmostEqual(contrast["area_mae_km2_difference_mean"], -.02)
        self.assertIsNone(contrast["iou_difference_sd"])
        self.assertFalse(any("z_mae" in key for key in contrast))
        self.assertEqual(architecture_contrasts({}), [])

    def test_complete_13_arm_summary_writes_contrasts_and_readable_plot(self):
        profile = json.loads(Path("configs/method_arch_v3.json").read_text())
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "experiment_plan.json").write_text(json.dumps(profile))
            metrics = dict(iou=.8, dice=.9, boundary_f1_20m=.7, area_mae_km2=.2,
                           area_relative_mae=.05, z_mae_m=None, z_rmse_m=None)
            for name in profile["experiments"]:
                run = root / name / "seed_42" / "val_evaluation"
                run.mkdir(parents=True)
                data = dict(seed=42, split="val", samples=1, parameters=100,
                            lambda_sdf=run_experiments.EXPERIMENTS[name][1], z_valid_samples=0,
                            macro_reservoir=metrics, per_reservoir={"synthetic_fixture": metrics})
                (run / "metrics.json").write_text(json.dumps(data))
            with patch("sys.argv", ["summarize_experiments.py", "--stage-dir", folder]):
                with contextlib.redirect_stdout(io.StringIO()):
                    summarize_experiments.main()
            with (root / "comparison.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 13)
            with (root / "architecture_contrasts.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 7)
            self.assertTrue(json.loads((root / "aggregation_status.json").read_text())["complete"])
            self.assertGreater((root / "comparison.png").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
