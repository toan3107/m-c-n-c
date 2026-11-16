from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from physics import FZCurve


def shoreline(mask: np.ndarray) -> np.ndarray:
    edge = mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=1)
    # The raster crop boundary is not an observed shoreline.
    edge[[0, -1], :] = False
    edge[:, [0, -1]] = False
    return edge


def boundary_f1(prediction: np.ndarray, target: np.ndarray, tolerance_px: float) -> float:
    p, t = shoreline(prediction), shoreline(target)
    if not np.any(p) or not np.any(t):
        return float(not np.any(p) and not np.any(t))
    precision = float(np.mean(distance_transform_edt(~t)[p] <= tolerance_px))
    recall = float(np.mean(distance_transform_edt(~p)[t] <= tolerance_px))
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.


def score_mask(prediction: np.ndarray, target: np.ndarray, pixel_area_m2: float,
               curve: FZCurve, tolerance_m: float = 20.) -> dict:
    if prediction.shape != target.shape:
        raise ValueError("Physical metrics require matching native grids")
    if not math.isfinite(pixel_area_m2) or pixel_area_m2 <= 0:
        raise ValueError("Invalid native pixel area")
    p, t = prediction.astype(bool), target.astype(bool)
    intersection = int(np.count_nonzero(p & t))
    union = int(np.count_nonzero(p | t))
    pn, tn = int(np.count_nonzero(p)), int(np.count_nonzero(t))
    predicted_area, target_area = pn * pixel_area_m2 / 1e6, tn * pixel_area_m2 / 1e6
    predicted_z = curve.interpolate_no_extrapolation(predicted_area)
    target_z = curve.interpolate_no_extrapolation(target_area)
    error_z = None if predicted_z is None or target_z is None else predicted_z - target_z
    return {
        "iou": intersection / union if union else 1.,
        "dice": 2 * intersection / (pn + tn) if pn + tn else 1.,
        "boundary_f1_20m": boundary_f1(p, t, tolerance_m / math.sqrt(pixel_area_m2)),
        "predicted_area_km2": predicted_area, "target_area_km2": target_area,
        "area_abs_error_km2": abs(predicted_area - target_area),
        "area_sq_error_km4": (predicted_area - target_area) ** 2,
        "area_relative_error": abs(predicted_area - target_area) / target_area if target_area else None,
        "predicted_z_m": predicted_z, "target_z_m": target_z,
        "z_abs_error_m": abs(error_z) if error_z is not None else None,
        "z_sq_error_m2": error_z ** 2 if error_z is not None else None,
        "target_out_of_support": int(target_z is None),
        "prediction_out_of_support": int(predicted_z is None),
    }


def mean_present(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot summarize zero forecast windows")
    by_reservoir = {}
    for reservoir in sorted({row["reservoir"] for row in rows}):
        group = [row for row in rows if row["reservoir"] == reservoir]
        z_mse = mean_present(group, "z_sq_error_m2")
        area_mse = mean_present(group, "area_sq_error_km4")
        by_reservoir[reservoir] = {
            "samples": len(group), "iou": mean_present(group, "iou"), "dice": mean_present(group, "dice"),
            "boundary_f1_20m": mean_present(group, "boundary_f1_20m"),
            "area_mae_km2": mean_present(group, "area_abs_error_km2"),
            "area_rmse_km2": math.sqrt(area_mse) if area_mse is not None else None,
            "area_relative_mae": mean_present(group, "area_relative_error"),
            "z_mae_m": mean_present(group, "z_abs_error_m"),
            "z_rmse_m": math.sqrt(z_mse) if z_mse is not None else None,
            "z_samples": sum(row["z_abs_error_m"] is not None for row in group),
            "target_out_of_support": sum(row["target_out_of_support"] for row in group),
            "prediction_out_of_support": sum(row["prediction_out_of_support"] for row in group),
        }
    keys = ("iou", "dice", "boundary_f1_20m", "area_mae_km2", "area_rmse_km2", "area_relative_mae", "z_mae_m", "z_rmse_m")
    return {"samples": len(rows), "per_reservoir": by_reservoir,
            "macro_reservoir": {key: mean_present(list(by_reservoir.values()), key) for key in keys},
            "z_valid_samples": sum(row["z_abs_error_m"] is not None for row in rows),
            "note": "Z is mask-derived via fixed F-Z, not independently gauge-validated. Z errors require coverage; native IoU/Dice are window means, then reservoir means."}


def horizon_bin(days: float) -> str:
    return "01-30d" if days <= 30 else "31-60d" if days <= 60 else "61-90d" if days <= 90 else "over-90d"
