from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image

from dataset import resolve_asset


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded CPU test of training, resume and inference, not a performance experiment")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    code_dir = Path(__file__).resolve().parent
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "mini_run").exists():
        raise FileExistsError("Use a fresh integration output directory")
    manifest = code_dir / "splits.csv"
    with manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, all_rows = reader.fieldnames, list(reader)
    # Diagnostic fixture: one train target, then one validation target with train RGB history.
    rows = sorted([dict(row) for row in all_rows if row["reservoir"] == "ankhe"],
                  key=lambda row: row["acquisition_date"])[:5]
    if len(rows) != 5:
        raise ValueError("Integration fixture needs five dates")
    for index, row in enumerate(rows):
        row["split"] = "train" if index < 4 else "val"
        for key in ("image_path", "mask_path", "conversion_table"):
            row[key] = str(resolve_asset(row[key], manifest))
    mini_manifest = output / "mini_splits.csv"
    with mini_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    run = output / "mini_run"
    command = [sys.executable, str(code_dir / "train.py"), "--splits", str(mini_manifest),
               "--output-dir", str(run), "--device", "cpu", "--image-size", "64",
               "--base-channels", "4", "--batch-size", "1", "--threads", "2",
               "--no-balance-reservoirs"]
    subprocess.run(command + ["--epochs", "1"], check=True)
    subprocess.run(command + ["--epochs", "2", "--resume", str(run / "last.pt")], check=True)
    checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != 2 or [row["epoch"] for row in checkpoint["history"]] != [1, 2]:
        raise AssertionError("Resume did not preserve epoch history")
    for key in ("optimizer", "scaler", "python_rng", "numpy_rng", "torch_rng", "loader_rng"):
        if key not in checkpoint:
            raise AssertionError(f"Missing resume state: {key}")
    for name in ("best.pt", "last.pt", "history.csv", "loss_curves.png", "summary.json"):
        if not (run / name).is_file():
            raise AssertionError(f"Missing training artifact: {name}")
    forecast = output / "forecast"
    subprocess.run([sys.executable, str(code_dir / "predict.py"), "--checkpoint", str(run / "best.pt"),
                    "--context", str(code_dir / "example_context.json"), "--target-date", "2020-03-18",
                    "--conversion-table", str(code_dir / "bang_quy_doi" / "ankhe_fz.csv"),
                    "--pixel-area-m2", "100", "--output-dir", str(forecast), "--device", "cpu", "--threads", "2"], check=True)
    with Image.open(forecast / "predicted_mask.png") as handle:
        if handle.size != (1401, 1001):
            raise AssertionError("Inference mask did not return to the native grid")
    prediction = json.loads((forecast / "prediction.json").read_text(encoding="utf-8"))
    if prediction["horizon_days"] != 5 or prediction["target_date"] in prediction["context_dates"]:
        raise AssertionError("Inference timing is not causal")
    report = {"status": "pass", "purpose": "integration_only_not_scientific_results",
              "train_windows_per_epoch": 1, "validation_windows_per_epoch": 1,
              "epochs_including_resume": 2, "checkpoint_resume_verified": True,
              "history_and_loss_plot_created": True, "past_only_inference_verified": True,
              "validation_uses_earlier_training_rgb": True,
              "native_prediction_size": [1401, 1001], "final_test_evaluated": False}
    (output / "integration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
