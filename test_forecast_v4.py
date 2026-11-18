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

import evaluate
import run_experiments
import train
import test_forecast
import summarize_experiments
from metrics import score_mask, summarize_rows
from model import ResidualForecastHead, build_model
from physics import PhysicalMeter
from selection import checkpoint_selection


torch.set_num_threads(2)


class ResidualHeadTests(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(42)
        return [torch.randn(1, 3, 3, 32, 32, requires_grad=True),
                torch.tensor([[0., 5., 17.]]), torch.tensor([10.]),
                torch.randn(1, 3, 2), torch.randn(1, 2)]

    def test_reference_outputs_and_state_keys_remain_unchanged(self):
        torch.manual_seed(42)
        model = build_model({"model": "spatial_cfc", "base_channels": 4}).eval()
        dynamic = torch.randn(1, 3, 32, 2, 2)
        gaps, horizon = torch.tensor([[0., 5., 17.]]), torch.tensor([10.])
        context, target = torch.randn(1, 3, 2), torch.randn(1, 2)
        with torch.no_grad():
            expected = torch.zeros_like(dynamic[:, 0])
            for index in range(3):
                event = torch.cat((context[:, index], torch.ones(1, 1)), dim=1)
                observation = dynamic[:, index] + model.event_embedding(event)[:, :, None, None]
                expected = model.temporal(observation, expected, gaps[:, index] / 30.)
            event = torch.cat((target, torch.zeros(1, 1)), dim=1)
            query = model.future_query + model.event_embedding(event)[:, :, None, None]
            expected = model.temporal(query.expand_as(expected), expected, horizon / 30.)
            actual = model._evolve(dynamic, gaps, horizon, context, target, cell=model.temporal,
                                   event_embedding=model.event_embedding, future_query=model.future_query)
        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(len(model.state_dict()), 101)

    def test_residual_heads_have_identical_weights_and_differ_only_by_envelope(self):
        free = ResidualForecastHead(8, False)
        anchor = ResidualForecastHead(8, True)
        anchor.load_state_dict(free.state_dict(), strict=True)
        hidden, query = torch.randn(2, 8, 4, 4), torch.randn(2, 8, 4, 4)
        elapsed = torch.tensor([.5, 2.])
        self.assertEqual(sum(p.numel() for p in free.parameters()), sum(p.numel() for p in anchor.parameters()))
        expected = hidden + (-torch.expm1(-elapsed[:, None, None, None])) * (free(query, hidden, elapsed) - hidden)
        self.assertTrue(torch.allclose(anchor(query, hidden, elapsed), expected))

    def test_zero_horizon_is_exact_identity_only_for_anchored_head(self):
        hidden, query = torch.randn(2, 8, 4, 4), torch.randn(2, 8, 4, 4)
        for anchor in (False, True):
            head = ResidualForecastHead(8, anchor)
            self.assertEqual(torch.equal(head(query, hidden, torch.zeros(2)), hidden), anchor)

    def test_invalid_elapsed_times_are_rejected(self):
        head = ResidualForecastHead(8, True)
        hidden = torch.randn(1, 8, 4, 4)
        for value in (-1., float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                head(hidden, hidden, torch.tensor([value]))

    def test_variants_backpropagate_to_history_and_restore_without_sdf_feedback(self):
        for name in ("spatial_cfc_residual", "spatial_cfc_residual_anchor"):
            with self.subTest(model=name):
                inputs = self.inputs()
                config = {"model": name, "base_channels": 4}
                model = build_model(config)
                mask, sdf = model(*inputs)
                valid, target = torch.ones_like(mask), torch.zeros_like(mask)
                loss, _ = train.compute_loss(mask, sdf, target, target, valid, valid, .1)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                self.assertGreater(float(model.forecast_head.increment[0].weight.grad.abs().sum()), 0.)
                self.assertGreater(float(model.temporal.time_a.weight.grad.abs().sum()), 0.)
                for step in range(3):
                    self.assertGreater(float(inputs[0].grad[:, step].abs().sum()), 0.)
                restored = build_model(config).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                model.eval()
                with torch.no_grad():
                    first = model(*inputs)[0]
                    self.assertTrue(torch.equal(first, restored(*inputs)[0]))
                    restored.sdf_head.bias.add_(10.)
                    self.assertTrue(torch.equal(first, restored(*inputs)[0]))
                    inputs[2] *= 5
                    self.assertFalse(torch.allclose(first, restored(*inputs)[0]))

    def test_only_observations_enter_cfc_in_new_variants(self):
        for name, calls in (("spatial_cfc", 4), ("spatial_cfc_residual", 3), ("spatial_cfc_residual_anchor", 3)):
            model = build_model({"model": name, "base_channels": 4})
            seen = []
            hook = model.temporal.register_forward_pre_hook(lambda module, args: seen.append(args[2]))
            model(*self.inputs())
            hook.remove()
            self.assertEqual(len(seen), calls)


class SelectionTests(unittest.TestCase):
    def metrics(self, failures=0, z_mae=2., area=.1):
        return {"physical": {"fixed_target_cohort": {
            "target_samples": 10, "valid_predictions": 10 - failures,
            "invalid_predictions": failures, "coverage": (10 - failures) / 10,
            "z_mae_m": z_mae if not failures else None,
            "z_rmse_m": z_mae if not failures else None,
            "area_mae_fraction_of_fz_span": area,
        }}}

    def score(self, *args, **kwargs):
        return checkpoint_selection(self.metrics(*args, **kwargs), "z_mae_support_first")

    def test_complete_coverage_beats_low_partial_error(self):
        self.assertLess(self.score(z_mae=10.)["key"], self.score(failures=1, area=.00001)["key"])
        self.assertLess(self.score(z_mae=1.)["key"], self.score(z_mae=2.)["key"])

    def test_equal_failure_ties_use_all_eligible_area_not_conditional_z(self):
        first, second = self.score(failures=2, area=.1), self.score(failures=2, area=.2)
        self.assertLess(first["key"], second["key"])
        self.assertEqual(first["criterion"], "eligible_area_mae_fraction_of_fz_span")
        self.assertIsNone(first["z_mae_m"])

    def test_zero_eligible_targets_cannot_select_a_z_checkpoint(self):
        metrics = self.metrics()
        metrics["physical"]["fixed_target_cohort"]["target_samples"] = 0
        with self.assertRaisesRegex(ValueError, "supported primary"):
            checkpoint_selection(metrics, "z_mae_support_first")

    def test_fixed_target_cohort_counts_invalid_predictions_and_averages_lakes(self):
        meter = PhysicalMeter()
        meter.stats["a"].update(level_evaluation="primary", z_target_samples=1, z_samples=1,
                                z_abs_error_sum=4., z_sq_error_sum=16., eligible_area_scaled_error_sum=.1)
        meter.stats["b"].update(level_evaluation="primary", z_target_samples=9, z_samples=9,
                                z_abs_error_sum=18., z_sq_error_sum=36., eligible_area_scaled_error_sum=.9,
                                target_out_of_support=2)
        cohort = meter.summarize()["fixed_target_cohort"]
        self.assertEqual(cohort["z_mae_m"], 3.)
        self.assertEqual(cohort["target_samples"], 10)
        self.assertEqual(cohort["target_out_of_support"], 2)
        meter.stats["b"].update(z_invalid_predictions=1, z_samples=8)
        cohort = meter.summarize()["fixed_target_cohort"]
        self.assertIsNone(cohort["z_mae_m"])
        self.assertEqual(cohort["coverage"], .9)
        self.assertEqual(cohort["target_samples"], 10)

    def test_profile_contains_only_three_fixed_loss_runs_without_test(self):
        output = io.StringIO()
        args = ["run_experiments.py", "--stage", "confirm", "--config", "configs/method_forecast_v4.json", "--dry-run"]
        with patch("sys.argv", args), contextlib.redirect_stdout(output):
            run_experiments.main()
        text = output.getvalue()
        profile = json.JSONDecoder().raw_decode(text)[0]
        self.assertEqual(len(profile["experiments"]), 3)
        self.assertEqual(profile["selection_metric"], "z_mae_support_first")
        self.assertEqual(profile["seeds"], [42])
        self.assertEqual(profile["image_size"], 512)
        self.assertEqual(profile["epochs"], 120)
        self.assertEqual(text.count("--lambda-sdf 0.1"), 3)
        self.assertEqual(text.count("--selection-metric z_mae_support_first"), 3)
        self.assertNotIn("--split test", text)

    def test_horizon_summaries_also_require_complete_z_coverage(self):
        import numpy as np
        from physics import FZCurve
        curve = FZCurve(np.array([.01, 1.]), np.array([10., 20.]))
        target = np.ones((20, 20), dtype=bool)
        good = score_mask(target, target, 100., curve)
        bad = score_mask(np.zeros_like(target), target, 100., curve)
        rows = [dict(reservoir="a", **good), dict(reservoir="a", **bad)]
        summary = summarize_rows(rows, strict_z=True)
        self.assertIsNone(summary["macro_reservoir"]["z_mae_m"])
        self.assertEqual(summary["fixed_target_cohort"]["coverage"], .5)
        self.assertEqual(summary["conditional_z_metrics"]["z_mae_m"], 0.)

    def test_forecast_comparison_keeps_failed_arm_instead_of_reporting_partial_z(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            profile = json.loads(Path("configs/method_forecast_v4.json").read_text())
            (root / "experiment_plan.json").write_text(json.dumps(profile))
            base = dict(iou=.8, dice=.9, boundary_f1_20m=.7, area_mae_km2=.1,
                        area_rmse_km2=.2, area_relative_mae=.1, z_mae_m=1., z_rmse_m=2.)
            for index, name in enumerate(profile["experiments"]):
                metrics = dict(base)
                invalid = int(index == 1)
                if invalid:
                    metrics.update(z_mae_m=None, z_rmse_m=None)
                cohort = dict(target_samples=2, invalid_predictions=invalid, coverage=1 - invalid / 2)
                lake = dict(z_target_samples=2, z_invalid_predictions=invalid,
                            level_evaluation="primary", z_mae_m=1., z_rmse_m=2.)
                data = dict(seed=42, split="val", samples=2, parameters=100, lambda_sdf=.1,
                            z_valid_samples=2-invalid, macro_reservoir=metrics,
                            per_reservoir={"fixture": base}, checkpoint_selection={},
                            physical=dict(fixed_target_cohort=cohort, per_reservoir={"fixture": lake}))
                destination = root / name / "seed_42/val_evaluation"
                destination.mkdir(parents=True)
                (destination / "metrics.json").write_text(json.dumps(data))
            with patch("sys.argv", ["summarize_experiments.py", "--stage-dir", folder]):
                with contextlib.redirect_stdout(io.StringIO()):
                    summarize_experiments.main()
            with (root / "comparison.csv").open(newline="") as handle:
                rows = {row["experiment"]: row for row in csv.DictReader(handle)}
            failed = rows["cfc_residual_sdf"]
            self.assertEqual(failed["z_mae_m_mean"], "")
            self.assertEqual(failed["z_coverage_mean"], "0.5")
            self.assertEqual(len(rows), 3)
            self.assertGreater((root / "comparison.png").stat().st_size, 1000)


class ForecastTrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        test_forecast.ForecastTests.setUp(self)
        with self.manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows[-1]["split"] = "val"
        with self.manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def command(self, output):
        return ["train.py", "--splits", str(self.manifest), "--output-dir", str(output),
                "--model", "spatial_cfc_residual_anchor", "--selection-metric", "z_mae_support_first",
                "--device", "cpu", "--image-size", "32", "--base-channels", "2", "--threads", "2",
                "--epochs", "3", "--no-balance-reservoirs"]

    def test_selection_and_resume_match_uninterrupted_training_and_native_evaluation(self):
        full, resumed = self.root / "full", self.root / "resumed"
        with patch("sys.argv", self.command(full)), contextlib.redirect_stdout(io.StringIO()):
            train.main()
        real_save = torch.save

        def interrupt(data, path, *args, **kwargs):
            real_save(data, path, *args, **kwargs)
            if Path(path).name == "last.pt" and data["epoch"] == 1:
                raise InterruptedError("Controlled checkpoint interruption")

        with patch("sys.argv", self.command(resumed)), patch("train.torch.save", side_effect=interrupt):
            with self.assertRaises(InterruptedError), contextlib.redirect_stdout(io.StringIO()):
                train.main()
        with patch("sys.argv", self.command(resumed) + ["--resume", str(resumed / "last.pt")]):
            with contextlib.redirect_stdout(io.StringIO()):
                train.main()
        a = torch.load(full / "last.pt", weights_only=False)
        b = torch.load(resumed / "last.pt", weights_only=False)
        self.assertTrue(all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"]))
        self.assertEqual(a["best_selection"], b["best_selection"])
        self.assertTrue((full / "forecast_curves.png").is_file())
        checkpoint = torch.load(full / "best.pt", weights_only=False)
        self.assertEqual(checkpoint["selection"], a["best_selection"])
        args = ["evaluate.py", "--checkpoint", str(full / "best.pt"), "--splits", str(self.manifest),
                "--split", "val", "--output-dir", str(self.root / "evaluation"), "--device", "cpu"]
        with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
            evaluate.main()
        result = json.loads((self.root / "evaluation/metrics.json").read_text())
        self.assertEqual(result["checkpoint_selection"], checkpoint["selection"])
        args = self.command(resumed) + ["--resume", str(resumed / "last.pt"), "--selection-metric", "iou"]
        with patch("sys.argv", args), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "selection_metric"):
                train.main()


if __name__ == "__main__":
    unittest.main()
