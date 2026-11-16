from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from dataset import (ReservoirSequenceDataset, context_timing, geometry_for,
                     letterbox_field, make_windows, read_samples, signed_distance)
from model import CfCMaskForecaster, SpatialCfCCell
from physics import FZCurve, PhysicalMeter
from train import compute_loss


torch.set_num_threads(2)


class ForecastTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "images").mkdir()
        (self.root / "masks").mkdir()
        (self.root / "curve.csv").write_text("F_km2,Z_m\n0,10\n2,20\n", encoding="utf-8")
        records = []
        for index, day in enumerate((1, 3, 10, 21, 25)):
            name = f"2020-01-{day:02d}.png"
            image = np.random.default_rng(index).integers(0, 255, (32, 48, 3), dtype=np.uint8)
            mask = np.zeros((32, 48), dtype=np.uint8)
            mask[8:24, 10 + index:38 - index] = 255
            Image.fromarray(image).save(self.root / "images" / name)
            Image.fromarray(mask).save(self.root / "masks" / name)
            records.append(dict(split="train", reservoir="fixture", acquisition_date=name[:-4],
                                image_path=f"images/{name}", mask_path=f"masks/{name}",
                                conversion_table="curve.csv", input_contract="preprocessed_enhanced_rgb_v1",
                                pixel_area_m2=100, geometry_status="fixed_grid", level_evaluation="primary"))
        self.manifest = self.root / "splits.csv"
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)

    def test_relative_paths_and_irregular_windows(self) -> None:
        samples = read_samples(self.manifest, {"train"})
        windows = make_windows(samples, 3)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0].horizon_days, 11)
        self.assertTrue(samples[0].image_path.is_absolute())
        self.assertEqual(len(make_windows(samples, 3, max_horizon_days=5)), 1)

    def test_target_image_is_never_opened(self) -> None:
        dataset = ReservoirSequenceDataset(self.manifest, {"train"}, image_size=32)
        forbidden = dataset.samples[0].target.image_path
        real_open = Image.open
        def guarded(path, *args, **kwargs):
            if Path(path).resolve() == forbidden:
                raise AssertionError("Target image was read")
            return real_open(path, *args, **kwargs)
        with patch("dataset.Image.open", side_effect=guarded):
            item = dataset[0]
        self.assertEqual(tuple(item["context_images"].shape), (3, 3, 32, 32))
        self.assertEqual(item["delta_days"].tolist(), [0, 2, 7])
        self.assertEqual(float(item["horizon_days"]), 11)
        self.assertEqual(tuple(item["mask"].shape), (1, 32, 32))

    def test_nonchronological_validation_is_rejected(self) -> None:
        text = self.manifest.read_text(encoding="utf-8")
        self.manifest.write_text(text.replace("train,fixture", "val,fixture", 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "precede validation"):
            read_samples(self.manifest, {"train"})

    def test_test_reservoir_cannot_also_be_used_for_training(self) -> None:
        text = self.manifest.read_text(encoding="utf-8")
        self.manifest.write_text(text.replace("train,fixture", "test,fixture", 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Test reservoir"):
            read_samples(self.manifest, {"train"})

    def test_padding_and_sdf_sign(self) -> None:
        mask = np.zeros((20, 40), dtype=np.float32)
        mask[5:15, 10:30] = 1
        sdf = signed_distance(mask, 5)
        self.assertLess(sdf[10, 20], 0)
        self.assertGreater(sdf[0, 0], 0)
        geometry = geometry_for(40, 20, 32)
        valid = letterbox_field(np.ones_like(mask), 32, geometry, nearest=True)
        self.assertTrue(np.all(valid[:8] == 0))
        self.assertTrue(np.all(valid[8:24] == 1))

    def test_invalid_timestamp_order(self) -> None:
        with self.assertRaises(ValueError):
            context_timing([date(2020, 1, 2), date(2020, 1, 1)], date(2020, 1, 3))

    def test_fz_never_extrapolates(self) -> None:
        curve = FZCurve.read(self.root / "curve.csv")
        self.assertEqual(curve.interpolate_no_extrapolation(1), 15)
        self.assertIsNone(curve.interpolate_no_extrapolation(3))

    def test_degenerate_fz_curve_is_rejected(self) -> None:
        path = self.root / "invalid_curve.csv"
        path.write_text("F_km2,Z_m\n1,10\n1,20\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            FZCurve.read(path)

    def test_area_macro_is_not_dropped_when_z_is_out_of_support(self) -> None:
        meter = PhysicalMeter()
        meter.stats["fixture"].update(level_evaluation="primary", samples=1,
                                      area_samples=1, area_abs_error_sum=2,
                                      area_sq_error_sum=4, prediction_out_of_support=1)
        macro = meter.summarize()["macro_primary_reservoirs"]
        self.assertEqual(macro["area_mae_km2"], 2)
        self.assertIsNone(macro["z_mae_m"])


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = CfCMaskForecaster(base_channels=4)
        self.context = torch.randn(1, 3, 3, 32, 32, requires_grad=True)
        self.delta = torch.tensor([[0., 5., 17.]])
        self.horizon = torch.tensor([10.])
        self.seasons = torch.tensor([[[0., 1.], [.1, .9], [.2, .8]]])
        self.target_season = torch.tensor([[.3, .7]])

    def forward_model(self):
        return self.model(self.context, self.delta, self.horizon, self.seasons, self.target_season)

    def test_shapes_and_gradients_reach_past_images_and_cfc(self) -> None:
        mask_logits, sdf_logits = self.forward_model()
        self.assertEqual(tuple(mask_logits.shape), (1, 1, 32, 32))
        target = torch.zeros_like(mask_logits)
        valid = torch.ones_like(target)
        loss, _ = compute_loss(mask_logits, sdf_logits, target, target, valid, valid, .1)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for index in range(3):
            self.assertGreater(float(self.context.grad[:, index].abs().sum()), 0)
        self.assertGreater(float(self.model.temporal.time_a.weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.sdf_head.weight.grad.abs().sum()), 0)

    def test_sdf_head_does_not_drive_mask_at_inference(self) -> None:
        self.model.eval()
        with torch.no_grad():
            first_mask, first_sdf = self.forward_model()
            self.model.sdf_head.bias.add_(2)
            second_mask, second_sdf = self.forward_model()
        self.assertTrue(torch.equal(first_mask, second_mask))
        self.assertFalse(torch.equal(first_sdf, second_sdf))

    def test_forecast_depends_on_horizon(self) -> None:
        self.model.eval()
        with torch.no_grad():
            first, _ = self.forward_model()
            self.horizon = torch.tensor([90.])
            second, _ = self.forward_model()
        self.assertFalse(torch.allclose(first, second))

    def test_negative_time_is_rejected(self) -> None:
        cell = SpatialCfCCell(4)
        with self.assertRaises(ValueError):
            cell(torch.zeros(1, 4, 4, 4), torch.zeros(1, 4, 4, 4), torch.tensor([-1.]))

    def test_no_sdf_ablation_removes_sdf_head_gradient(self) -> None:
        mask_logits, sdf_logits = self.forward_model()
        target = torch.zeros_like(mask_logits)
        valid = torch.ones_like(target)
        loss, _ = compute_loss(mask_logits, sdf_logits, target, target, valid, valid, 0.)
        loss.backward()
        self.assertEqual(float(self.model.sdf_head.weight.grad.abs().sum()), 0)
        self.assertGreater(float(self.model.segmentation_head.weight.grad.abs().sum()), 0)

    def test_empty_sdf_validity_does_not_create_nan(self) -> None:
        mask_logits, sdf_logits = self.forward_model()
        target = torch.zeros_like(mask_logits)
        valid = torch.ones_like(target)
        loss, components = compute_loss(mask_logits, sdf_logits, target, target, valid, target, .1)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(components["sdf_loss"], 0)


if __name__ == "__main__":
    unittest.main()
