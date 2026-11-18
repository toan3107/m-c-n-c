from __future__ import annotations

import math


SELECTION_METRICS = ("iou", "z_mae_support_first")


def checkpoint_selection(metrics: dict, mode: str) -> dict:
    if mode == "iou":
        value = float(metrics["macro_reservoir"]["iou"])
        result = {"metric": mode, "key": [-value], "value": value, "criterion": "macro_iou"}
    elif mode == "z_mae_support_first":
        cohort = metrics["physical"]["fixed_target_cohort"]
        if not cohort["target_samples"]:
            raise ValueError("Z checkpoint selection requires supported primary validation targets")
        failures = int(cohort["invalid_predictions"])
        value = cohort["area_mae_fraction_of_fz_span"] if failures else cohort["z_mae_m"]
        result = {
            "metric": mode, "key": [failures, float(value)], "value": float(value),
            "criterion": "eligible_area_mae_fraction_of_fz_span" if failures else "macro_z_mae_m",
            **cohort,
        }
    else:
        raise ValueError(f"Unknown checkpoint selection metric: {mode}")
    if not all(math.isfinite(value) for value in result["key"]):
        raise ValueError("Non-finite checkpoint selection score")
    return result
