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
import test_forecast
import train
from model import build_model


class ProfileTests(unittest.TestCase):
    def dry_run(self, *extra):
        output = io.StringIO()
        with patch("sys.argv", ["run_experiments.py", *extra, "--dry-run"]), contextlib.redirect_stdout(output):
            run_experiments.main()
        return output.getvalue()

    def test_refit_has_no_validation_and_includes_both_source_splits(self):
        output = self.dry_run("--stage", "refit", "--config", "configs/method_v2.json")
        settings = json.JSONDecoder().raw_decode(output)[0]
        self.assertEqual(settings["epochs"], 120)
        self.assertEqual(settings["seeds"], [42])
        self.assertEqual(settings["training_splits"], ["train", "val"])
        self.assertEqual(settings["checkpoint"], "last.pt")
        self.assertIn("--no-validation --train-splits train val", output)
        self.assertNotIn("--split val", output)
        _, end = json.JSONDecoder().raw_decode(output)
        self.assertNotIn("evaluate.py", output[end:])

    def test_refit_test_is_opt_in_and_selects_last_not_best(self):
        with self.assertRaisesRegex(ValueError, "locked"):
            self.dry_run("--stage", "refit-test")
        output = self.dry_run("--stage", "refit-test", "--confirm-final-test")
        self.assertIn("last.pt", output)
        self.assertNotIn("best.pt", output)
        _, end = json.JSONDecoder().raw_decode(output)
        self.assertNotIn("train.py", output[end:])

    def test_cli_overrides_profile_without_silent_seed_change(self):
        output = self.dry_run("--stage", "confirm", "--config", "configs/method_v2.json",
                              "--epochs", "90", "--base-channels", "32")
        settings = json.JSONDecoder().raw_decode(output)[0]
        self.assertEqual(settings["epochs"], 90)
        self.assertEqual(settings["base_channels"], 32)
        self.assertEqual(settings["lr_schedule"], "warmup_cosine")
        self.assertEqual(settings["seeds"], [42])

    def test_profile_overfit_remains_one_method(self):
        output = self.dry_run("--stage", "overfit", "--config", "configs/method_v2.json")
        settings = json.JSONDecoder().raw_decode(output)[0]
        self.assertEqual(settings["experiments"], ["cfc_sdf"])

    def test_test_unlock_cannot_bypass_missing_refit_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            arguments = ["run_experiments.py", "--stage", "refit-test", "--confirm-final-test", "--output-root", folder]
            with patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "frozen refit plan"):
                    run_experiments.main()

    def test_refit_test_checks_frozen_settings_and_completion(self):
        with tempfile.TemporaryDirectory() as folder:
            common = ["--experiments", "cfc_sdf", "--epochs", "2", "--device", "cpu", "--output-root", folder]
            with patch("sys.argv", ["run_experiments.py", "--stage", "refit", *common]):
                with patch("run_experiments.execute"), contextlib.redirect_stdout(io.StringIO()):
                    run_experiments.main()
            summary_path = Path(folder) / "refit/cfc_sdf/seed_42/summary.json"
            summary_path.write_text(json.dumps({"no_validation": True, "last_epoch": 1}))
            arguments = ["run_experiments.py", "--stage", "refit-test", "--confirm-final-test", *common]
            with patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "complete its declared"):
                    run_experiments.main()
            summary_path.write_text(json.dumps({"no_validation": True, "last_epoch": 2}))
            with patch("sys.argv", arguments + ["--context-length", "4"]), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "differ from training"):
                    run_experiments.main()
            with patch("sys.argv", arguments), patch("run_experiments.execute") as execute:
                with contextlib.redirect_stdout(io.StringIO()):
                    run_experiments.main()
            evaluation = execute.call_args_list[0].args[0]
            self.assertEqual(Path(evaluation[2]).name, "evaluate.py")
            self.assertEqual(Path(evaluation[evaluation.index("--checkpoint") + 1]).name, "last.pt")


class IsolatedTimeTests(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(42)
        return [torch.randn(1, 3, 3, 32, 32), torch.tensor([[0., 5., 17.]]),
                torch.tensor([10.]), torch.randn(1, 3, 2), torch.randn(1, 2)]

    def test_no_gap_preserves_calendar_effect(self):
        model = build_model({"model": "spatial_cfc_nogap", "base_channels": 4}).eval()
        inputs = self.inputs()
        with torch.no_grad():
            first, _ = model(*inputs)
            inputs[1] *= 5
            inputs[2] *= 4
            same, _ = model(*inputs)
            inputs[3] += 1
            inputs[4] -= 1
            changed, _ = model(*inputs)
        self.assertTrue(torch.equal(first, same))
        self.assertFalse(torch.allclose(first, changed))

    def test_no_calendar_preserves_horizon_effect(self):
        model = build_model({"model": "spatial_cfc_nocalendar", "base_channels": 4}).eval()
        inputs = self.inputs()
        with torch.no_grad():
            first, _ = model(*inputs)
            inputs[3] += 1
            inputs[4] -= 1
            same, _ = model(*inputs)
            inputs[2] *= 4
            changed, _ = model(*inputs)
        self.assertTrue(torch.equal(first, same))
        self.assertFalse(torch.allclose(first, changed))


class RefitTrainingTests(unittest.TestCase):
    def setUp(self):
        test_forecast.ForecastTests.setUp(self)
        with self.manifest.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[-1]["split"] = "val"
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def command(self, output):
        return ["train.py", "--splits", str(self.manifest), "--output-dir", str(output),
                "--device", "cpu", "--image-size", "32", "--base-channels", "2",
                "--batch-size", "2", "--threads", "2", "--no-balance-reservoirs",
                "--no-validation", "--train-splits", "train", "val", "--epochs", "4",
                "--lr-schedule", "warmup_cosine", "--warmup-epochs", "2"]

    def test_fixed_refit_and_scheduler_resume_are_exact(self):
        full, resumed = self.root / "full", self.root / "resumed"
        with patch("sys.argv", self.command(full)), contextlib.redirect_stdout(io.StringIO()):
            train.main()
        real_save = torch.save

        def interrupt_after_first_save(data, path, *args, **kwargs):
            real_save(data, path, *args, **kwargs)
            if Path(path).name == "last.pt" and data["epoch"] == 1:
                raise InterruptedError("Simulated process interruption after checkpoint")

        with patch("sys.argv", self.command(resumed)), patch("train.torch.save", side_effect=interrupt_after_first_save):
            with self.assertRaises(InterruptedError), contextlib.redirect_stdout(io.StringIO()):
                train.main()
        with patch("sys.argv", self.command(resumed) + ["--resume", str(resumed / "last.pt")]), contextlib.redirect_stdout(io.StringIO()):
            train.main()
        a = torch.load(full / "last.pt", weights_only=False)
        b = torch.load(resumed / "last.pt", weights_only=False)
        self.assertTrue(all(torch.equal(a["model"][key], b["model"][key]) for key in a["model"]))
        self.assertEqual([r["lr"] for r in a["history"]], [r["lr"] for r in b["history"]])
        self.assertAlmostEqual(a["history"][0]["lr"], .00015)
        self.assertAlmostEqual(a["history"][-1]["lr"], .000003)
        self.assertFalse((full / "best.pt").exists())
        self.assertFalse((full / "val_windows.csv").exists())
        self.assertTrue((full / "loss_curves.png").is_file())
        self.assertNotIn("val_loss", a["history"][0])
        summary = json.loads((full / "summary.json").read_text())
        self.assertTrue(summary["no_validation"])
        self.assertEqual(summary["selected_checkpoint"], "last.pt")
        self.assertIsNone(summary["best_val_iou"])
        self.assertEqual(summary["last_epoch"], 4)
        with (full / "train_windows.csv").open(encoding="utf-8", newline="") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_refit_validation_reporting_is_rejected(self):
        output = self.root / "run"
        with patch("sys.argv", self.command(output)), contextlib.redirect_stdout(io.StringIO()):
            train.main()
        arguments = ["evaluate.py", "--checkpoint", str(output / "last.pt"),
                     "--split", "val", "--output-dir", str(self.root / "invalid"), "--device", "cpu"]
        with patch("sys.argv", arguments), self.assertRaisesRegex(ValueError, "cannot be reported"):
            evaluate.main()

    def test_scheduled_budget_change_is_not_a_valid_resume(self):
        output = self.root / "run"
        with patch("sys.argv", self.command(output)), contextlib.redirect_stdout(io.StringIO()):
            train.main()
        arguments = self.command(output) + ["--epochs", "5", "--resume", str(output / "last.pt")]
        with patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "original fixed epoch budget"):
                train.main()

    def test_target_test_split_cannot_train_even_without_validation(self):
        arguments = self.command(self.root / "bad") + ["--train-splits", "train", "test"]
        with patch("sys.argv", arguments), self.assertRaisesRegex(ValueError, "remain separate"):
            train.main()


if __name__ == "__main__":
    unittest.main()
