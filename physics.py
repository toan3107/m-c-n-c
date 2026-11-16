from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FZCurve:
    area_km2: np.ndarray
    level_m: np.ndarray

    @classmethod
    def read(cls, path: str | Path) -> "FZCurve":
        rows: list[tuple[float, float]] = []
        with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append((float(row["F_km2"]), float(row["Z_m"])))
        if len(rows) < 2:
            raise ValueError(f"F-Z curve needs at least two rows: {path}")

        grouped: dict[float, list[float]] = defaultdict(list)
        for area, level in rows:
            grouped[area].append(level)
        areas = np.array(sorted(grouped), dtype=np.float64)
        levels = np.array([np.mean(grouped[area]) for area in areas], dtype=np.float64)
        if len(areas) < 2 or not np.isfinite(areas).all() or not np.isfinite(levels).all():
            raise ValueError(f"F-Z curve needs distinct finite area values: {path}")
        if np.any(np.diff(levels) < -1e-8):
            raise ValueError(f"F-Z curve is not monotone: {path}")
        return cls(area_km2=areas, level_m=levels)

    def interpolate_no_extrapolation(self, area_km2: float) -> float | None:
        if not math.isfinite(area_km2):
            return None
        if area_km2 < self.area_km2[0] or area_km2 > self.area_km2[-1]:
            return None
        return float(np.interp(area_km2, self.area_km2, self.level_m))


def batch_number(batch: dict[str, object], key: str, index: int) -> float:
    value = batch[key]
    if isinstance(value, torch.Tensor):
        return float(value[index].item())
    if isinstance(value, (list, tuple)):
        return float(value[index])
    return float(value)


class PhysicalMeter:
    """Area and mask-derived F-Z metrics evaluated on the native raster grid."""

    def __init__(self) -> None:
        self.curves: dict[str, FZCurve] = {}
        self.stats: dict[str, dict[str, float | str]] = defaultdict(
            lambda: {
                "level_evaluation": "disabled",
                "samples": 0.0,
                "area_samples": 0.0,
                "area_abs_error_sum": 0.0,
                "area_sq_error_sum": 0.0,
                "z_samples": 0.0,
                "z_abs_error_sum": 0.0,
                "z_sq_error_sum": 0.0,
                "target_out_of_support": 0.0,
                "prediction_out_of_support": 0.0,
            }
        )

    def curve(self, path: str) -> FZCurve:
        if path not in self.curves:
            self.curves[path] = FZCurve.read(path)
        return self.curves[path]

    def update(self, logits: torch.Tensor, batch: dict[str, object]) -> None:
        probabilities = torch.sigmoid(logits.detach())
        reservoirs = list(batch["reservoir"])
        evaluation_modes = list(batch["level_evaluation"])
        curve_paths = list(batch["conversion_table"])

        for index, reservoir in enumerate(reservoirs):
            stats = self.stats[reservoir]
            mode = str(evaluation_modes[index])
            stats["level_evaluation"] = mode
            stats["samples"] = float(stats["samples"]) + 1.0
            pixel_area = batch_number(batch, "pixel_area_m2", index)
            target_area = batch_number(batch, "target_area_km2", index)
            if not math.isfinite(pixel_area) or not math.isfinite(target_area):
                continue

            left = int(batch_number(batch, "letterbox_left", index))
            top = int(batch_number(batch, "letterbox_top", index))
            resized_width = int(batch_number(batch, "resized_width", index))
            resized_height = int(batch_number(batch, "resized_height", index))
            native_width = int(batch_number(batch, "native_width", index))
            native_height = int(batch_number(batch, "native_height", index))
            content = probabilities[
                index : index + 1,
                :,
                top : top + resized_height,
                left : left + resized_width,
            ]
            native_probability = F.interpolate(
                content,
                size=(native_height, native_width),
                mode="bilinear",
                align_corners=False,
            )
            predicted_pixels = int((native_probability >= 0.5).sum().item())
            predicted_area = predicted_pixels * pixel_area / 1e6
            area_error = predicted_area - target_area
            stats["area_samples"] = float(stats["area_samples"]) + 1.0
            stats["area_abs_error_sum"] = float(stats["area_abs_error_sum"]) + abs(area_error)
            stats["area_sq_error_sum"] = float(stats["area_sq_error_sum"]) + area_error**2

            if mode == "disabled" or not curve_paths[index]:
                continue
            curve = self.curve(str(curve_paths[index]))
            target_z = curve.interpolate_no_extrapolation(target_area)
            predicted_z = curve.interpolate_no_extrapolation(predicted_area)
            if target_z is None:
                stats["target_out_of_support"] = (
                    float(stats["target_out_of_support"]) + 1.0
                )
            if predicted_z is None:
                stats["prediction_out_of_support"] = (
                    float(stats["prediction_out_of_support"]) + 1.0
                )
            if target_z is None or predicted_z is None:
                continue
            z_error = predicted_z - target_z
            stats["z_samples"] = float(stats["z_samples"]) + 1.0
            stats["z_abs_error_sum"] = float(stats["z_abs_error_sum"]) + abs(z_error)
            stats["z_sq_error_sum"] = float(stats["z_sq_error_sum"]) + z_error**2

    @staticmethod
    def finalize(stats: dict[str, float | str]) -> dict[str, float | str | int | None]:
        area_count = int(float(stats["area_samples"]))
        z_count = int(float(stats["z_samples"]))
        return {
            "level_evaluation": str(stats["level_evaluation"]),
            "samples": int(float(stats["samples"])),
            "area_samples": area_count,
            "area_mae_km2": (
                float(stats["area_abs_error_sum"]) / area_count if area_count else None
            ),
            "area_rmse_km2": (
                math.sqrt(float(stats["area_sq_error_sum"]) / area_count)
                if area_count
                else None
            ),
            "z_samples": z_count,
            "z_mae_m": float(stats["z_abs_error_sum"]) / z_count if z_count else None,
            "z_rmse_m": (
                math.sqrt(float(stats["z_sq_error_sum"]) / z_count) if z_count else None
            ),
            "target_out_of_support": int(float(stats["target_out_of_support"])),
            "prediction_out_of_support": int(float(stats["prediction_out_of_support"])),
        }

    def summarize(self) -> dict[str, object]:
        per_reservoir = {
            reservoir: self.finalize(stats)
            for reservoir, stats in sorted(self.stats.items())
        }
        primary = [
            values
            for values in per_reservoir.values()
            if values["level_evaluation"] == "primary"
        ]
        macro_primary = {}
        for metric in ("area_mae_km2", "area_rmse_km2", "z_mae_m", "z_rmse_m"):
            values = [float(item[metric]) for item in primary if item[metric] is not None]
            macro_primary[metric] = float(np.mean(values)) if values else None
        return {
            "per_reservoir": per_reservoir,
            "macro_primary_reservoirs": macro_primary,
            "note": (
                "Z is derived from predicted and target mask areas. Curves are never "
                "extrapolated; diagnostic and disabled reservoirs are excluded from the "
                "primary macro average."
            ),
        }
