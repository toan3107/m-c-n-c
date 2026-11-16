from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from dataset import load_windows
from model import build_model
from train import compute_loss, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Check environment and a representative training step, not only nvidia-smi")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--max-horizon-days", type=float, default=90.)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/preflight"))
    args = parser.parse_args()
    if args.device == "cuda" and (not torch.cuda.is_available() or torch.version.cuda is None):
        raise RuntimeError("CUDA PyTorch is unavailable. Run setup_3090.ps1; nvidia-smi alone is not a PyTorch CUDA test.")
    torch.set_num_threads(2)
    seed_everything(42)
    device = torch.device(args.device)
    report = {"python_executable": sys.executable, "python": platform.python_version(),
              "platform": platform.platform(), "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
              "device": str(device), "gpu_verified": device.type == "cuda",
              "image_size": args.image_size, "batch_size": args.batch_size, "base_channels": args.base_channels}
    if device.type == "cuda":
        report.update(gpu=torch.cuda.get_device_name(), compute_capability=list(torch.cuda.get_device_capability()),
                      total_vram_gib=torch.cuda.get_device_properties(0).total_memory / 1024**3)
        torch.cuda.reset_peak_memory_stats()
    model = build_model({"model": "spatial_cfc", "base_channels": args.base_channels}).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda", init_scale=1024.)
    b, s = args.batch_size, args.image_size
    images = torch.rand(b, 3, 3, s, s, device=device)
    delta = torch.tensor([[0., 15., 30.]], device=device).repeat(b, 1)
    horizon = torch.full((b,), 30., device=device)
    context_season = torch.zeros(b, 3, 2, device=device)
    target_season = torch.zeros(b, 2, device=device)
    started = time.monotonic()
    with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
        logits, sdf = model(images, delta, horizon, context_season, target_season)
        mask = torch.zeros_like(logits)
        mask[..., s // 4:3 * s // 4, s // 4:3 * s // 4] = 1
        valid = torch.ones_like(logits)
        loss, _ = compute_loss(logits, sdf, mask, torch.zeros_like(sdf), valid, valid, .1)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if not torch.isfinite(loss) or any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise FloatingPointError("Representative training step produced non-finite loss/gradients")
    scaler.step(optimizer)
    scaler.update()
    if device.type == "cuda":
        torch.cuda.synchronize()
        report["peak_allocated_vram_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    report.update(status="pass", forward_backward_optimizer_step=True,
                  probe_seconds=time.monotonic() - started, probe_loss=float(loss.detach()))
    manifest = Path(__file__).resolve().parent / "splits.csv"
    report["development_windows"] = {}
    for split in ("train", "val"):
        windows = load_windows(manifest, {split}, 3, args.max_horizon_days)
        report["development_windows"][split] = {
            "count": len(windows), "reservoirs": sorted({w.reservoir for w in windows}),
            "target_gap_days": [min(w.horizon_days for w in windows), max(w.horizon_days for w in windows)],
            "max_context_span_days": max((w.context[-1].acquisition_date - w.context[0].acquisition_date).days for w in windows)}
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze_env = os.environ.copy()
    freeze_env.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True,
                            text=True, encoding="utf-8", env=freeze_env, check=True)
    (output / "environment_freeze.txt").write_text(freeze.stdout, encoding="utf-8")
    (output / "environment.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
