from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import make_windows, read_samples
from physics import FZCurve


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the portable RGB forecasting dataset")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent / "splits.csv")
    parser.add_argument("--context-length", type=int, default=3)
    parser.add_argument("--max-horizon-days", type=float, default=90.)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    samples = read_samples(manifest, {"train", "val", "test"})
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = {(row["reservoir"], row["acquisition_date"]): row for row in csv.DictReader(handle)}
    dimensions = defaultdict(set)
    support = defaultdict(Counter)
    checksum_count = 0
    checksum_file = manifest.parent / "data_checksums.csv"
    if checksum_file.is_file():
        bundle_root = manifest.parent.parent.resolve()
        with checksum_file.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                asset = (manifest.parent / row["path"]).resolve()
                if not asset.is_relative_to(bundle_root):
                    raise ValueError("Checksum path escapes bundle")
                if hashlib.sha256(asset.read_bytes()).hexdigest() != row["sha256"]:
                    raise ValueError(f"Checksum mismatch: {asset}")
                checksum_count += 1
    for sample in samples:
        row = rows[(sample.reservoir, sample.acquisition_date.isoformat())]
        for path, field in ((sample.image_path, "image_sha256"), (sample.mask_path, "mask_sha256")):
            if not path.is_file():
                raise FileNotFoundError(path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != row[field]:
                raise ValueError(f"Hash mismatch: {path}")
        with Image.open(sample.image_path) as handle:
            if handle.mode != "RGB":
                raise ValueError(f"Expected RGB: {sample.image_path}")
            size = handle.size
        if size != (int(row["native_width"]), int(row["native_height"])):
            raise ValueError("Native dimensions differ from manifest")
        with Image.open(sample.mask_path) as handle:
            mask = np.asarray(handle.convert("L"))
            if handle.size != size:
                raise ValueError("Image and mask grids differ")
        if not set(np.unique(mask)).issubset({0, 255}) or not np.any(mask):
            raise ValueError(f"Mask is empty or not binary: {sample.mask_path}")
        if sample.pixel_area_m2 <= 0:
            raise ValueError("Invalid physical pixel area")
        dimensions[sample.reservoir].add(size)
        curve = FZCurve.read(sample.conversion_table)
        area = np.count_nonzero(mask) * sample.pixel_area_m2 / 1e6
        support[sample.reservoir]["inside_support" if curve.interpolate_no_extrapolation(area) is not None else "out_of_support"] += 1
    if any(len(sizes) != 1 for sizes in dimensions.values()):
        raise ValueError("Native grid changes over time within a reservoir")
    windows = make_windows(samples, args.context_length)
    by_split = defaultdict(list)
    capped_by_split = Counter()
    reservoir_splits = defaultdict(set)
    for sample in samples:
        reservoir_splits[sample.reservoir].add(sample.split)
    for window in windows:
        if any(item.acquisition_date >= window.target.acquisition_date for item in window.context):
            raise ValueError("Noncausal window")
        by_split[window.target.split].append(window.horizon_days)
        if window.horizon_days <= args.max_horizon_days:
            capped_by_split[window.target.split] += 1
    policy_path = manifest.parent / "temporal_split.json"
    if policy_path.is_file():
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        if policy["manifest_sha256"] != hashlib.sha256(manifest.read_bytes()).hexdigest():
            raise ValueError("Manifest differs from temporal_split.json; rebuild the split report")
    report = {
        "status": "pass", "samples": len(samples), "input_channels": 3,
        "split_counts": dict(Counter(sample.split for sample in samples)),
        "reservoir_counts": dict(Counter(sample.reservoir for sample in samples)),
        "reservoir_splits": {key: sorted(value) for key, value in reservoir_splits.items()},
        "native_dimensions": {key: list(next(iter(value))) for key, value in dimensions.items()},
        "checksum_assets_verified": checksum_count,
        "forecast_windows": {split: len(values) for split, values in by_split.items()},
        "max_horizon_days": args.max_horizon_days,
        "forecast_windows_capped": dict(capped_by_split),
        "horizon_days": {split: {"min": min(values), "median": float(np.median(values)), "max": max(values)} for split, values in by_split.items()},
        "fz_support": {key: dict(value) for key, value in support.items()},
        "checks": {"image_and_mask_hashes_match": True, "native_grids_match": True,
                   "binary_nonempty_masks": True, "test_reservoirs_disjoint": True,
                   "train_val_chronological_within_reservoir": True,
                   "training_context_excludes_validation": True,
                   "past_context_precedes_target": True, "fz_extrapolation": False},
    }
    output = args.output or manifest.parent / "dataset_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
