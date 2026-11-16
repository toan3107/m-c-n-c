from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize all prespecified seeds without selecting the best seed")
    parser.add_argument("--stage-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.stage_dir.resolve()
    groups, details, lake_values = defaultdict(list), [], defaultdict(list)
    for path in sorted(root.glob("*/seed_*/*_evaluation/metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("smoke_only"):
            continue
        name = path.parents[2].name
        row = {"experiment": name, "seed": data["seed"], "split": data["split"],
               "samples": data["samples"], "parameters": data.get("parameters"),
               "lambda_sdf": data["lambda_sdf"], "z_valid_samples": data["z_valid_samples"],
               **data["macro_reservoir"]}
        details.append(row)
        groups[name].append(row)
        for reservoir, values in data["per_reservoir"].items():
            lake_values[(name, reservoir)].append(values)
    if not details:
        raise ValueError("No completed validation/test evaluations found")
    plan = json.loads((root / "experiment_plan.json").read_text(encoding="utf-8"))
    missing = [(name, seed) for name in plan["experiments"] for seed in plan["seeds"]
               if not any(row["experiment"] == name and row["seed"] == seed for row in details)]
    metrics = ("iou", "dice", "boundary_f1_20m", "area_mae_km2", "area_relative_mae", "z_mae_m", "z_rmse_m")
    summary = []
    for name, runs in sorted(groups.items()):
        row = {"experiment": name, "seeds_completed": len(runs), "parameters": runs[0]["parameters"]}
        for metric in metrics:
            values = [float(run[metric]) for run in runs if run[metric] is not None]
            row[f"{metric}_mean"] = float(np.mean(values)) if values else None
            row[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(row)
    write_csv(root / "all_seed_metrics.csv", details)
    write_csv(root / "comparison.csv", summary)
    by_lake = []
    for (name, lake), runs in sorted(lake_values.items()):
        row = {"experiment": name, "reservoir": lake, "seeds_completed": len(runs)}
        for metric in metrics:
            values = [float(run[metric]) for run in runs if run[metric] is not None]
            row[f"{metric}_mean"] = float(np.mean(values)) if values else None
        by_lake.append(row)
    write_csv(root / "per_reservoir_comparison.csv", by_lake)
    differences = []
    proposed = {row["seed"]: row for row in groups.get("cfc_sdf", [])}
    for name, runs in sorted(groups.items()):
        if name == "cfc_sdf":
            continue
        common = [run for run in runs if run["seed"] in proposed]
        deltas = [float(proposed[run["seed"]]["iou"]) - float(run["iou"]) for run in common]
        if deltas:
            differences.append({"contrast": f"cfc_sdf_minus_{name}", "paired_seeds": len(deltas),
                                "macro_iou_difference_mean": float(np.mean(deltas)),
                                "macro_iou_difference_sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None})
    write_csv(root / "paired_seed_differences.csv", differences)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    colors = ["#64748b", "#167d9a", "#c47b25", "#32885a", "#b55968", "#6565a5", "#ba4c22"]
    for axis, metric, title in zip(axes, ("iou", "boundary_f1_20m", "area_mae_km2"),
                                   ("Future-mask native IoU", "Boundary F1 (20 m)", "Area MAE (km2)")):
        x = np.arange(len(summary))
        y = [row[f"{metric}_mean"] for row in summary]
        sd = [row[f"{metric}_sd"] or 0. for row in summary]
        axis.bar(x, y, yerr=sd, color=[colors[i % len(colors)] for i in x], capsize=3)
        axis.set_xticks(x, [row["experiment"] for row in summary], rotation=50, ha="right", fontsize=9)
        axis.set_title(title)
        axis.grid(axis="y", alpha=.2)
    single_seed = all(len(runs) == 1 for runs in groups.values())
    fig.suptitle("Single-seed method check; run-to-run variability not estimated" if single_seed else
                 "Mean across completed seeds; sample SD where available, not confidence intervals")
    fig.savefig(root / "comparison.png", dpi=160)
    plt.close(fig)
    status = {"complete": not missing, "missing_experiment_seed_pairs": missing,
              "runs_completed": len(details), "single_seed_only": single_seed,
              "scientific_note": "SD is unavailable for a single seed. Overlapping windows are not independent replicates. Seed SD is optimization variability, not evidence of global generalization. Z errors require coverage and are mask-derived, not independent gauge validation."}
    (root / "aggregation_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
