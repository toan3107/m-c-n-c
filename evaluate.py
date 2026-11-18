from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from dataset import ReservoirSequenceDataset
from metrics import horizon_bin, score_mask, summarize_rows
from model import build_model
from physics import FZCurve, PhysicalMeter, batch_number
from selection import checkpoint_selection


def main() -> None:
    parser = argparse.ArgumentParser(description="Native-grid future-mask evaluation; final test is opt-in")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--splits", type=Path, default=Path(__file__).resolve().parent / "splits.csv")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save-masks", action="store_true")
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_final_test:
        raise ValueError("Final test is locked; pass --confirm-final-test only after model selection")
    torch.set_num_threads(max(1, args.threads))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("smoke_only") and args.split == "test":
        raise ValueError("A smoke checkpoint must not score the final test")
    config = checkpoint["config"]
    strict_z = config.get("selection_metric", "iou") == "z_mae_support_first"
    if args.split == "val" and config.get("no_validation"):
        raise ValueError("Refit used source validation labels for training; they cannot be reported as held-out validation")
    if args.split == "test" and config.get("no_validation") and checkpoint.get("epoch") != config["epochs"]:
        raise ValueError("Fixed-budget refit is incomplete; final test remains locked")
    if config.get("overfit_windows") and args.split == "test":
        raise ValueError("Overfit diagnostics must not score the final test")
    manifest_hash = hashlib.sha256(args.splits.read_bytes()).hexdigest()
    if config.get("manifest_sha256", manifest_hash) != manifest_hash:
        raise ValueError("Checkpoint and evaluation manifest differ")
    dataset = ReservoirSequenceDataset(args.splits, {args.split}, image_size=config["image_size"],
                                       context_length=config["context_length"], max_horizon_days=config["max_horizon_days"],
                                       sdf_truncation_fraction=config["sdf_truncation_fraction"],
                                       sdf_border_margin_fraction=config["sdf_border_margin_fraction"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        (output / "masks").mkdir(exist_ok=True)
    rows, curves = [], {}
    physical_meter = PhysicalMeter()
    with torch.inference_mode():
        for batch in loader:
            # Float32 inference is shared across all comparators, independent of training AMP.
            logits, _ = model(batch["context_images"].to(device), batch["delta_days"].to(device),
                              batch["horizon_days"].to(device), batch["context_season"].to(device),
                              batch["target_season"].to(device))
            physical_meter.update(logits, batch)
            for index, reservoir in enumerate(batch["reservoir"]):
                left, top, rw, rh, nw, nh = [int(batch_number(batch, key, index)) for key in
                    ("letterbox_left", "letterbox_top", "resized_width", "resized_height", "native_width", "native_height")]
                probability = F.interpolate(torch.sigmoid(logits[index:index+1, :, top:top+rh, left:left+rw]).float(),
                                            size=(nh, nw), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
                prediction = probability >= .5
                with Image.open(batch["target_mask_path"][index]) as handle:
                    target = np.asarray(handle.convert("L")) >= 128
                curve_path = batch["conversion_table"][index]
                if curve_path not in curves:
                    curves[curve_path] = FZCurve.read(curve_path)
                days = float(batch["horizon_days"][index])
                row = {"reservoir": reservoir, "context_dates": batch["context_dates"][index],
                       "target_date": batch["target_date"][index], "horizon_days": days,
                       "level_evaluation": batch["level_evaluation"][index],
                       "horizon_bin": horizon_bin(days),
                       **score_mask(prediction, target, batch_number(batch, "pixel_area_m2", index), curves[curve_path])}
                rows.append(row)
                if args.save_masks:
                    Image.fromarray(prediction.astype(np.uint8) * 255).save(output / "masks" / f"{reservoir}__{row['target_date']}.png")
    with (output / "per_window.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_rows(rows, strict_z=strict_z)
    summary["physical"] = physical_meter.summarize()
    if strict_z:
        cohort = summary["physical"]["fixed_target_cohort"]
        if cohort["target_samples"]:
            summary["checkpoint_selection"] = checkpoint_selection(summary, config["selection_metric"])
        summary["note"] += " Primary Z MAE/RMSE require full coverage of the fixed supported-target cohort; partial errors are diagnostic only."
    summary.update({"split": args.split, "model": config["model"], "seed": config["seed"],
                    "lambda_sdf": config["lambda_sdf"], "parameters": config.get("parameters"),
                    "checkpoint_epoch": checkpoint.get("epoch"), "manifest_sha256": manifest_hash,
                    "smoke_only": bool(checkpoint.get("smoke_only", False)),
                    "horizon_days": {"min": min(row["horizon_days"] for row in rows),
                                     "max": max(row["horizon_days"] for row in rows)},
                    "by_horizon": {name: summarize_rows([row for row in rows if row["horizon_bin"] == name], strict_z=strict_z)
                                   for name in sorted({row["horizon_bin"] for row in rows})}})
    (output / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"split": args.split, "samples": len(rows), "macro_reservoir": summary["macro_reservoir"]}, indent=2))


if __name__ == "__main__":
    main()
