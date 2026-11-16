from __future__ import annotations

import argparse
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import context_timing, geometry_for, letterbox_rgb, resolve_asset
from model import MODEL_NAMES, build_model
from physics import FZCurve


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast a future mask using only past RGB images")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--context", type=Path, required=True, help="JSON list of image_path/date records")
    parser.add_argument("--target-date", required=True, type=date.fromisoformat)
    parser.add_argument("--conversion-table", type=Path, required=True)
    parser.add_argument("--pixel-area-m2", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/prediction"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not math.isfinite(args.pixel_area_m2) or args.pixel_area_m2 <= 0:
        raise ValueError("--pixel-area-m2 must describe the native grid, not the resized input")
    torch.set_num_threads(max(1, args.threads))
    resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(resolved_device if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    # Only load checkpoints you trust: training checkpoints can contain Python objects.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config.get("model") not in MODEL_NAMES or config.get("input_channels") != 3:
        raise ValueError("This predictor requires a 3-channel forecasting checkpoint")
    context_file = args.context.resolve()
    records = json.loads(context_file.read_text(encoding="utf-8"))
    if len(records) != config["context_length"]:
        raise ValueError(f"Checkpoint requires {config['context_length']} past images")
    dates = [date.fromisoformat(item["date"]) for item in records]
    delta, horizon, context_season, target_season = context_timing(dates, args.target_date)
    geometry = None
    images, paths = [], []
    for record in records:
        path = resolve_asset(record["image_path"], context_file)
        with Image.open(path) as handle:
            if handle.mode != "RGB":
                raise ValueError(f"Expected preprocessed RGB: {path}")
            image = np.asarray(handle)
        if geometry is None:
            geometry = geometry_for(image.shape[1], image.shape[0], config["image_size"])
        images.append(letterbox_rgb(image, config["image_size"], geometry))
        paths.append(str(path))
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    context = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).unsqueeze(0).to(device)
    with torch.inference_mode():
        mask_logits, sdf_logits = model(context, delta[None].to(device), horizon[None].to(device),
                                        context_season[None].to(device), target_season[None].to(device))
        top, left = int(geometry["letterbox_top"]), int(geometry["letterbox_left"])
        rw, rh = int(geometry["resized_width"]), int(geometry["resized_height"])
        native_size = (int(geometry["native_height"]), int(geometry["native_width"]))

        def native(field: torch.Tensor) -> np.ndarray:
            content = field[..., top:top + rh, left:left + rw]
            return F.interpolate(content, size=native_size, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()

        probability = native(torch.sigmoid(mask_logits))
        auxiliary_sdf = native(torch.tanh(sdf_logits))
    if not np.isfinite(probability).all() or not np.isfinite(auxiliary_sdf).all():
        raise FloatingPointError("Checkpoint produced non-finite predictions")
    mask = probability >= 0.5
    water_pixels = int(np.count_nonzero(mask))
    area = water_pixels * args.pixel_area_m2 / 1e6
    curve = FZCurve.read(args.conversion_table)
    level = curve.interpolate_no_extrapolation(area)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(output / "predicted_mask.png")
    Image.fromarray(np.round(probability * 255).astype(np.uint8)).save(output / "predicted_probability.png")
    np.save(output / "auxiliary_sdf_normalized.npy", auxiliary_sdf)
    result = {
        "checkpoint": str(args.checkpoint.resolve()), "smoke_only": bool(checkpoint.get("smoke_only", False)),
        "context_images": paths, "context_dates": [item.isoformat() for item in dates],
        "target_date": args.target_date.isoformat(), "horizon_days": float(horizon),
        "native_width": native_size[1], "native_height": native_size[0],
        "pixel_area_m2": args.pixel_area_m2, "water_pixels": water_pixels,
        "F_km2": area, "Z_m": level,
        "fz_status": "inside_support" if level is not None else "out_of_support_no_extrapolation",
        "fz_area_support_km2": [float(curve.area_km2[0]), float(curve.area_km2[-1])],
        "conversion_table": str(args.conversion_table.resolve()),
        "note": "F-Z is a fixed postprocessing conversion; SDF is auxiliary and does not generate the mask.",
    }
    (output / "prediction.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
