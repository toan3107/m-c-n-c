from __future__ import annotations

import contextlib
import io
import json
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

import test_forecast
from dataset import ReservoirSequenceDataset
from metrics import score_mask, summarize_rows
from model import MODEL_NAMES, build_model
from physics import FZCurve
import run_experiments
import gpu_preflight


torch.set_num_threads(2)


class TemporalComparisonTests(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(42)
        return [torch.randn(2, 3, 3, 32, 32), torch.tensor([[0., 5., 17.]]).repeat(2, 1),
                torch.full((2,), 10.), torch.randn(2, 3, 2), torch.randn(2, 2)]

    def test_every_comparator_trains_and_has_expected_shape(self):
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = build_model({"model": name, "base_channels": 4})
                mask, sdf = model(*self.inputs())
                self.assertEqual(tuple(mask.shape), (2, 1, 32, 32))
                loss = mask.square().mean() + .1 * sdf.square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_time_agnostic_controls_ignore_all_timestamp_cues(self):
        for name in ("last_frame", "segmenter_persistence", "convgru", "spatial_cfc_notime"):
            with self.subTest(model=name):
                model = build_model({"model": name, "base_channels": 4}).eval()
                inputs = self.inputs()
                with torch.no_grad():
                    first, _ = model(*inputs)
                    inputs[1] *= 3
                    inputs[2] *= 10
                    inputs[3] += 5
                    inputs[4] -= 3
                    second, _ = model(*inputs)
                self.assertTrue(torch.equal(first, second))

    def test_time_aware_comparators_depend_on_horizon(self):
        for name in ("dt_convgru", "spatial_cfc"):
            model = build_model({"model": name, "base_channels": 4}).eval()
            inputs = self.inputs()
            with torch.no_grad():
                first, _ = model(*inputs)
                inputs[2] *= 10
                second, _ = model(*inputs)
            self.assertFalse(torch.allclose(first, second))

    def test_last_frame_baseline_cannot_use_earlier_images(self):
        model = build_model({"model": "last_frame", "base_channels": 4}).eval()
        inputs = self.inputs()
        with torch.no_grad():
            first, _ = model(*inputs)
            inputs[0][:, :-1] += 100
            second, _ = model(*inputs)
        self.assertTrue(torch.equal(first, second))

    def test_final_test_requires_explicit_unlock(self):
        with patch("sys.argv", ["run_experiments.py", "--stage", "final-test"]):
            with self.assertRaisesRegex(ValueError, "locked"):
                run_experiments.main()

    def test_pilot_dry_run_contains_no_final_test_evaluation(self):
        output = io.StringIO()
        with patch("sys.argv", ["run_experiments.py", "--stage", "pilot", "--dry-run"]):
            with contextlib.redirect_stdout(output):
                run_experiments.main()
        self.assertNotIn("--split test", output.getvalue())
        self.assertIn("--split val", output.getvalue())


class PersistenceDataTests(unittest.TestCase):
    setUp = test_forecast.ForecastTests.setUp

    def test_persistence_training_uses_only_last_context_mask_as_label(self):
        dataset = ReservoirSequenceDataset(self.manifest, {"train"}, image_size=32,
                                            supervision="last_context_mask")
        target = dataset.samples[0].target
        real_open = Image.open
        def guarded(path, *args, **kwargs):
            if Path(path).resolve() in (target.mask_path, target.image_path):
                raise AssertionError("Future observation used in persistence training")
            return real_open(path, *args, **kwargs)
        with patch("dataset.Image.open", side_effect=guarded):
            item = dataset[0]
        self.assertEqual(item["target_mask_path"], str(dataset.samples[0].context[-1].mask_path))
        self.assertAlmostEqual(item["target_area_km2"], 384 * 100 / 1e6)


class NativeMetricTests(unittest.TestCase):
    def setUp(self):
        self.curve = FZCurve(np.array([0., 1.]), np.array([10., 20.]))
        self.mask = np.zeros((32, 32), dtype=bool)
        self.mask[8:24, 8:24] = True

    def test_identical_masks_score_one(self):
        row = score_mask(self.mask, self.mask, 100., self.curve)
        self.assertEqual(row["iou"], 1)
        self.assertEqual(row["boundary_f1_20m"], 1)
        self.assertEqual(row["area_abs_error_km2"], 0)

    def test_empty_prediction_is_not_a_good_boundary_forecast(self):
        row = score_mask(np.zeros_like(self.mask), self.mask, 100., self.curve)
        self.assertEqual(row["iou"], 0)
        self.assertEqual(row["boundary_f1_20m"], 0)

    def test_macro_does_not_weight_reservoirs_by_window_count(self):
        good = score_mask(self.mask, self.mask, 100., self.curve)
        bad = score_mask(np.zeros_like(self.mask), self.mask, 100., self.curve)
        rows = [dict(reservoir="a", **bad)] + [dict(reservoir="b", **good) for _ in range(9)]
        self.assertEqual(summarize_rows(rows)["macro_reservoir"]["iou"], .5)


class PreflightTests(unittest.TestCase):
    def test_cuda_unavailable_fails_without_silent_cpu_fallback(self):
        with patch("sys.argv", ["gpu_preflight.py", "--device", "cuda"]), patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA PyTorch"):
                gpu_preflight.main()

    def test_cpu_probe_is_marked_cpu_and_package_capture_uses_utf8(self):
        with tempfile.TemporaryDirectory() as folder:
            args = ["gpu_preflight.py", "--device", "cpu", "--image-size", "32",
                    "--base-channels", "4", "--batch-size", "1", "--output-dir", folder]
            with patch("sys.argv", args), patch("gpu_preflight.subprocess.run", return_value=SimpleNamespace(stdout="torch==test\n")) as command:
                with contextlib.redirect_stdout(io.StringIO()):
                    gpu_preflight.main()
            report = json.loads((Path(folder) / "environment.json").read_text(encoding="utf-8"))
            self.assertFalse(report["gpu_verified"])
            self.assertEqual(command.call_args.kwargs["env"]["PYTHONIOENCODING"], "utf-8")


if __name__ == "__main__":
    unittest.main()
