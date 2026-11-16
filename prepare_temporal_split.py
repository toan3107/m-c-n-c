from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from dataset import load_windows


def split_records(rows: list[dict], validation_fraction: float = 0.2,
                  test_reservoirs: set[str] | None = None, context_length: int = 3) -> list[dict]:
    if not 0 < validation_fraction < 1:
        raise ValueError("Validation fraction must be between zero and one")
    if context_length < 1:
        raise ValueError("Context length must be positive")
    test_reservoirs = {"el_vado", "riano"} if test_reservoirs is None else set(test_reservoirs)
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        day = date.fromisoformat(row["acquisition_date"])
        key = (row["reservoir"], day)
        if key in seen:
            raise ValueError(f"Duplicate reservoir/date: {key}")
        seen.add(key)
        groups[row["reservoir"]].append(dict(row))
    if not test_reservoirs or not test_reservoirs.issubset(groups):
        raise ValueError("Every declared test reservoir must exist in the manifest")
    if test_reservoirs == set(groups):
        raise ValueError("At least one development reservoir is required")
    result = []
    for reservoir, group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: row["acquisition_date"])
        validation_count = math.ceil(len(ordered) * validation_fraction)
        boundary = len(ordered) - validation_count
        if reservoir not in test_reservoirs and boundary <= context_length:
            raise ValueError(f"Not enough early observations for training: {reservoir}")
        for index, row in enumerate(ordered):
            row["split"] = "test" if reservoir in test_reservoirs else "train" if index < boundary else "val"
            result.append(row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible chronological train/val split within development lakes")
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent / "splits.csv")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-reservoirs", nargs="+", default=["el_vado", "riano"])
    parser.add_argument("--context-length", type=int, default=3)
    parser.add_argument("--max-horizon-days", type=float, default=90.)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    rows = split_records(rows, args.validation_fraction, set(args.test_reservoirs), args.context_length)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    by_reservoir = defaultdict(list)
    for row in rows:
        by_reservoir[row["reservoir"]].append(row)
    all_windows = load_windows(manifest, {"train", "val", "test"}, args.context_length)
    capped = [window for window in all_windows if window.horizon_days <= args.max_horizon_days]
    details = {}
    for reservoir, group in sorted(by_reservoir.items()):
        by_split = defaultdict(list)
        for row in group:
            by_split[row["split"]].append(row["acquisition_date"])
        details[reservoir] = {
            "images": {split: len(days) for split, days in sorted(by_split.items())},
            "dates": {split: {"first": min(days), "last": max(days)} for split, days in sorted(by_split.items())},
            "windows_all": dict(Counter(window.target.split for window in all_windows if window.reservoir == reservoir)),
            "windows_capped": dict(Counter(window.target.split for window in capped if window.reservoir == reservoir)),
        }
    report = {
        "protocol": "six_development_lakes_chronological_validation_two_unseen_test_lakes_v1",
        "validation_fraction": args.validation_fraction,
        "split_unit": "acquisition_date_within_each_development_reservoir",
        "rounding": "ceil(validation_fraction * number_of_acquisitions)",
        "boundary_selected_before_horizon_filter": True,
        "validation_mode": "rolling_origin_with_previously_observed_rgb_only",
        "validation_may_use_earlier_training_rgb": True,
        "validation_masks_used_for_training": False,
        "global_calendar_cutoff": False,
        "context_length": args.context_length,
        "max_horizon_days": args.max_horizon_days,
        "context_span_capped": False,
        "test_reservoirs": sorted(set(args.test_reservoirs)),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "image_split_counts": dict(Counter(row["split"] for row in rows)),
        "window_counts_all": dict(Counter(window.target.split for window in all_windows)),
        "window_counts_capped": dict(Counter(window.target.split for window in capped)),
        "per_reservoir": details,
    }
    path = manifest.parent / "temporal_split.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
