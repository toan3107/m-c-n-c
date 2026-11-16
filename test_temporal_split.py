from __future__ import annotations

import contextlib
import csv
import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from dataset import ReservoirSequenceDataset, load_windows, read_samples
from prepare_temporal_split import split_records
import run_experiments
import test_forecast


class ChronologicalDataTests(unittest.TestCase):
    setUp = test_forecast.ForecastTests.setUp

    def make_temporal_fixture(self):
        with self.manifest.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields, rows = reader.fieldnames, list(reader)
        rows[-1]["split"] = "val"
        with self.manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def test_validation_keeps_context_before_the_split_boundary(self):
        self.make_temporal_fixture()
        train = load_windows(self.manifest, {"train"})
        val = load_windows(self.manifest, {"val"})
        self.assertEqual(len(train), 1)
        self.assertEqual(len(val), 1)
        self.assertEqual(val[0].target.acquisition_date.isoformat(), "2020-01-25")
        self.assertTrue(all(sample.split == "train" for sample in val[0].context))
        self.assertTrue(all(sample.split == "train" for sample in train[0].context))
        self.assertLess(train[0].target.acquisition_date, val[0].target.acquisition_date)
        self.assertEqual(len(load_windows(self.manifest, {"val"}, max_horizon_days=5)), 1)

    def test_training_never_opens_validation_image_or_label(self):
        self.make_temporal_fixture()
        dataset = ReservoirSequenceDataset(self.manifest, {"train"}, image_size=32)
        val = read_samples(self.manifest, {"val"})[0]
        forbidden = {val.image_path, val.mask_path}
        real_open = Image.open
        def guarded(path, *args, **kwargs):
            if Path(path).resolve() in forbidden:
                raise AssertionError("Validation asset used for training")
            return real_open(path, *args, **kwargs)
        with patch("dataset.Image.open", side_effect=guarded):
            dataset[0]

    def test_validation_opens_only_context_rgb_and_target_label(self):
        self.make_temporal_fixture()
        dataset = ReservoirSequenceDataset(self.manifest, {"val"}, image_size=32)
        window = dataset.samples[0]
        forbidden = {sample.mask_path for sample in window.context} | {window.target.image_path}
        real_open = Image.open
        def guarded(path, *args, **kwargs):
            if Path(path).resolve() in forbidden:
                raise AssertionError("Context label or future RGB used for validation")
            return real_open(path, *args, **kwargs)
        with patch("dataset.Image.open", side_effect=guarded):
            item = dataset[0]
        self.assertEqual(item["context_dates"], "2020-01-03,2020-01-10,2020-01-21")


class SplitPolicyTests(unittest.TestCase):
    def records(self):
        return [dict(reservoir=reservoir, acquisition_date=f"2020-01-{day:02d}", split="test")
                for reservoir in ("source", "heldout") for day in range(1, 11)]

    def test_last_twenty_percent_and_heldout_lake(self):
        rows = split_records(self.records(), test_reservoirs={"heldout"})
        source = [row for row in rows if row["reservoir"] == "source"]
        heldout = [row for row in rows if row["reservoir"] == "heldout"]
        self.assertEqual([row["split"] for row in source], ["train"] * 8 + ["val"] * 2)
        self.assertTrue(all(row["split"] == "test" for row in heldout))
        self.assertEqual(rows, split_records(rows, test_reservoirs={"heldout"}))

    def test_bad_fraction_and_missing_test_reservoir_are_rejected(self):
        for fraction in (0, 1, -0.2):
            with self.subTest(fraction=fraction), self.assertRaises(ValueError):
                split_records(self.records(), fraction, {"heldout"})
        with self.assertRaisesRegex(ValueError, "must exist"):
            split_records(self.records(), test_reservoirs={"absent"})

    def test_one_seed_is_the_default_for_confirmation(self):
        output = io.StringIO()
        with patch("sys.argv", ["run_experiments.py", "--stage", "confirm", "--dry-run"]):
            with contextlib.redirect_stdout(output):
                run_experiments.main()
        settings = json.JSONDecoder().raw_decode(output.getvalue())[0]
        self.assertEqual(settings["seeds"], [42])
        self.assertEqual(settings["epochs"], 80)
        self.assertNotIn("--seed 43", output.getvalue())
        self.assertNotIn("--split test", output.getvalue())


if __name__ == "__main__":
    unittest.main()
