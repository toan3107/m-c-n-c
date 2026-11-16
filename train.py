from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import ReservoirSequenceDataset
from model import MODEL_NAMES, build_model
from physics import PhysicalMeter
from plot_history import plot_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spatial CfC future-mask forecasting with auxiliary SDF loss")
    parser.add_argument("--splits", type=Path, default=Path(__file__).resolve().parent / "splits.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/cfc_sdf_l01"))
    parser.add_argument("--train-splits", nargs="+", default=["train"])
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--model", choices=MODEL_NAMES, default="spatial_cfc")
    parser.add_argument("--overfit-windows", type=int, help="Diagnostic only: cap train/val windows and disable augmentation")
    parser.add_argument("--context-length", type=int, default=3)
    parser.add_argument("--max-horizon-days", type=float)
    parser.add_argument("--time-scale-days", type=float, default=30.0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-sdf", type=float, default=0.1)
    parser.add_argument("--sdf-truncation-fraction", type=float, default=0.05)
    parser.add_argument("--sdf-border-margin-fraction", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--balance-reservoirs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sample reservoirs uniformly instead of sampling images uniformly.",
    )
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return (values * valid).sum() / valid.sum().clamp_min(1.0)


def compute_loss(
    seg_logits: torch.Tensor,
    sdf_logits: torch.Tensor,
    mask: torch.Tensor,
    sdf_target: torch.Tensor,
    valid: torch.Tensor,
    sdf_valid: torch.Tensor,
    lambda_sdf: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    bce = masked_mean(F.binary_cross_entropy_with_logits(seg_logits, mask, reduction="none"), valid)
    probability = torch.sigmoid(seg_logits)
    dims = (1, 2, 3)
    intersection = (probability * mask * valid).sum(dim=dims)
    denominator = ((probability + mask) * valid).sum(dim=dims)
    dice_loss = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()

    sdf_prediction = torch.tanh(sdf_logits)
    sdf_error = F.smooth_l1_loss(sdf_prediction, sdf_target, reduction="none")
    # Give boundary pixels more influence while retaining a small global shape signal.
    boundary_weight = 0.1 + torch.exp(-torch.abs(sdf_target) / 0.25)
    sdf_loss = masked_mean(sdf_error * boundary_weight, sdf_valid) / masked_mean(
        boundary_weight, sdf_valid
    ).clamp_min(1e-6)

    total = bce + dice_loss + lambda_sdf * sdf_loss
    values = {
        "loss": float(total.detach()),
        "bce": float(bce.detach()),
        "dice_loss": float(dice_loss.detach()),
        "sdf_loss": float(sdf_loss.detach()),
    }
    return total, values


class ConfusionMeter:
    def __init__(self) -> None:
        self.counts: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(4, dtype=np.float64))

    def update(
        self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, reservoirs: list[str]
    ) -> None:
        prediction = torch.sigmoid(logits) >= 0.5
        target_bool = target >= 0.5
        valid_bool = valid >= 0.5
        for i, reservoir in enumerate(reservoirs):
            p = prediction[i] & valid_bool[i]
            t = target_bool[i] & valid_bool[i]
            v = valid_bool[i]
            tp = (p & t).sum().item()
            fp = (p & ~t & v).sum().item()
            fn = (~p & t & v).sum().item()
            tn = (~p & ~t & v).sum().item()
            self.counts[reservoir] += np.array([tp, fp, fn, tn], dtype=np.float64)

    @staticmethod
    def metrics(counts: np.ndarray) -> dict[str, float]:
        tp, fp, fn, _ = counts
        eps = 1e-8
        return {
            "iou": float(tp / (tp + fp + fn + eps)),
            "dice": float(2 * tp / (2 * tp + fp + fn + eps)),
            "precision": float(tp / (tp + fp + eps)),
            "recall": float(tp / (tp + fn + eps)),
        }

    def summarize(self) -> dict[str, object]:
        overall = np.sum(list(self.counts.values()), axis=0)
        per_reservoir = {
            reservoir: self.metrics(counts)
            for reservoir, counts in sorted(self.counts.items())
        }
        macro = {
            key: float(np.mean([value[key] for value in per_reservoir.values()]))
            for key in ("iou", "dice", "precision", "recall")
        }
        return {
            "micro": self.metrics(overall),
            "macro_reservoir": macro,
            "per_reservoir": per_reservoir,
        }


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_sdf: float,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    smoke_test: bool = False,
) -> tuple[dict[str, float], dict[str, object]]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = defaultdict(float)
    sample_count = 0
    meter = ConfusionMeter()
    physical_meter = None if training else PhysicalMeter()

    for batch_index, batch in enumerate(loader):
        context = batch["context_images"].to(device)
        mask = batch["mask"].to(device)
        sdf = batch["sdf"].to(device)
        valid = batch["valid"].to(device)
        sdf_valid = batch["sdf_valid"].to(device)
        batch_size = context.shape[0]

        if training:
            optimizer.zero_grad(set_to_none=True)
        amp_enabled = device.type == "cuda"
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            seg_logits, sdf_logits = model(
                context,
                batch["delta_days"].to(device),
                batch["horizon_days"].to(device),
                batch["context_season"].to(device),
                batch["target_season"].to(device),
            )
            loss, components = compute_loss(
                seg_logits,
                sdf_logits,
                mask,
                sdf,
                valid,
                sdf_valid,
                lambda_sdf=lambda_sdf,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss; check inputs and training configuration")
        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

        for key, value in components.items():
            totals[key] += value * batch_size
        sample_count += batch_size
        meter.update(seg_logits.detach(), mask, valid, list(batch["reservoir"]))
        if physical_meter is not None:
            physical_meter.update(seg_logits, batch)
        if smoke_test and batch_index == 0:
            break

    averages = {key: value / sample_count for key, value in totals.items()}
    metrics = meter.summarize()
    if physical_meter is not None:
        metrics["physical"] = physical_meter.summarize()
    return averages, metrics


def make_loader(
    split_csv: Path,
    splits: set[str],
    args: argparse.Namespace,
    augment: bool,
    shuffle: bool,
) -> DataLoader:
    dataset = ReservoirSequenceDataset(
        split_csv,
        splits=splits,
        image_size=args.image_size,
        context_length=args.context_length,
        max_horizon_days=args.max_horizon_days,
        sdf_truncation_fraction=args.sdf_truncation_fraction,
        sdf_border_margin_fraction=args.sdf_border_margin_fraction,
        augment=augment and not args.overfit_windows,
        supervision="last_context_mask" if augment and args.model == "segmenter_persistence" else "future_mask",
    )
    if args.overfit_windows is not None:
        if args.overfit_windows < 1:
            raise ValueError("--overfit-windows must be positive")
        dataset.samples = dataset.samples[:args.overfit_windows]
    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    if shuffle and args.balance_reservoirs and not args.overfit_windows:
        counts = Counter(sample.reservoir for sample in dataset.samples)
        weights = torch.as_tensor(
            [1.0 / counts[sample.reservoir] for sample in dataset.samples],
            dtype=torch.double,
        )
        sampler = WeightedRandomSampler(
            weights,
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(max(1, args.threads))
    seed_everything(args.seed)
    if args.lambda_sdf < 0:
        raise ValueError("--lambda-sdf cannot be negative")
    if args.image_size % 16 != 0:
        raise ValueError("--image-size must be divisible by 16")
    if args.val_split in args.train_splits or args.test_split in args.train_splits:
        raise ValueError("Training, validation and final-test splits must remain separate")
    if args.val_split == args.test_split:
        raise ValueError("Validation cannot use the final-test split")
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update({"device_resolved": str(device), "input_channels": 3,
                   "manifest_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
                   "torch_version": str(torch.__version__), "cuda_runtime": torch.version.cuda})
    config = {key: str(value) if isinstance(value, Path) else value for key, value in config.items()}
    if not args.resume and (output_dir / "last.pt").exists():
        raise FileExistsError("Run already has a checkpoint; use --resume or a new output directory")

    train_loader = make_loader(
        args.splits, set(args.train_splits), args, augment=True, shuffle=True
    )
    val_loader = make_loader(
        args.splits, {args.val_split}, args, augment=False, shuffle=False
    )

    for name, loader in (("train", train_loader), ("val", val_loader)):
        with (output_dir / f"{name}_windows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["reservoir", "context_dates", "target_date", "horizon_days"])
            writer.writeheader()
            for window in loader.dataset.samples:
                writer.writerow({"reservoir": window.reservoir,
                                 "context_dates": ",".join(item.acquisition_date.isoformat() for item in window.context),
                                 "target_date": window.target.acquisition_date.isoformat(),
                                 "horizon_days": window.horizon_days})
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda", init_scale=1024.)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    config["parameters"] = parameter_count
    print(f"Model: {args.model}")
    print(f"Device: {device} | parameters: {parameter_count:,}")
    print(
        f"Train: {len(train_loader.dataset)} | val: {len(val_loader.dataset)} | "
        f"lambda_sdf: {args.lambda_sdf}"
    )

    history: list[dict[str, float | int]] = []
    best_iou, best_epoch, stale_epochs, start_epoch = -1.0, 0, 0, 1
    if args.resume:
        if args.resume.resolve().parent != output_dir:
            raise ValueError("Resume must use the same output directory as last.pt")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        for key in ("model", "manifest_sha256", "base_channels", "context_length", "image_size", "time_scale_days", "lambda_sdf", "max_horizon_days", "seed", "balance_reservoirs", "num_workers", "overfit_windows"):
            if checkpoint["config"][key] != config[key]:
                raise ValueError(f"Resume configuration differs: {key}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        history = checkpoint["history"]
        best_iou, best_epoch = checkpoint["best_iou"], checkpoint["best_epoch"]
        stale_epochs, start_epoch = checkpoint["stale_epochs"], checkpoint["epoch"] + 1
        random.setstate(checkpoint["python_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if device.type == "cuda" and checkpoint.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in checkpoint["cuda_rng"]])
        if checkpoint.get("loader_rng") is not None:
            train_loader.generator.set_state(checkpoint["loader_rng"].cpu())
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    if args.smoke_test:
        train_loss, train_metrics = run_epoch(
            model, train_loader, device, args.lambda_sdf, optimizer, scaler, smoke_test=True
        )
        with torch.no_grad():
            val_loss, val_metrics = run_epoch(
                model, val_loader, device, args.lambda_sdf, None, None, smoke_test=True
            )
        print(json.dumps({"train": train_loss, "train_metrics": train_metrics}, indent=2))
        print(json.dumps({"val": val_loss, "val_metrics": val_metrics}, indent=2))
        smoke_report = {"model": args.model, "input_channels": 3,
                        "train_windows": len(train_loader.dataset), "val_windows": len(val_loader.dataset),
                        "lambda_sdf": args.lambda_sdf, "train": train_loss, "val": val_loss,
                        "final_test_evaluated": False}
        (output_dir / "smoke_report.json").write_text(json.dumps(smoke_report, indent=2), encoding="utf-8")
        torch.save({"model": model.state_dict(), "config": config, "smoke_only": True}, output_dir / "smoke.pt")
        print("Smoke test passed.")
        return

    history_path = output_dir / "history.csv"

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        train_loss, train_metrics = run_epoch(
            model, train_loader, device, args.lambda_sdf, optimizer, scaler
        )
        with torch.no_grad():
            val_loss, val_metrics = run_epoch(
                model, val_loader, device, args.lambda_sdf, None, None
            )
        val_iou = float(val_metrics["macro_reservoir"]["iou"])
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss["loss"],
            "train_iou": float(train_metrics["micro"]["iou"]),
            "val_loss": val_loss["loss"],
            "val_iou": val_iou,
            "val_micro_iou": float(val_metrics["micro"]["iou"]),
            "val_dice": float(val_metrics["micro"]["dice"]),
            "seconds": time.time() - started,
        }
        for prefix, losses in (("train", train_loss), ("val", val_loss)):
            row.update({f"{prefix}_{key}": losses[key] for key in ("bce", "dice_loss", "sdf_loss")})
        row["lambda_sdf"] = args.lambda_sdf
        history.append(row)
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(
            f"Epoch {epoch:03d} | train loss {row['train_loss']:.4f} "
            f"IoU {row['train_iou']:.4f} | val loss {row['val_loss']:.4f} "
            f"IoU {row['val_iou']:.4f} Dice {row['val_dice']:.4f}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "config": config,
                },
                output_dir / "best.pt",
            )
            (output_dir / "best_val_metrics.json").write_text(
                json.dumps(val_metrics, indent=2), encoding="utf-8"
            )
        else:
            stale_epochs += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "config": config,
                    "history": history, "best_iou": best_iou, "best_epoch": best_epoch,
                    "stale_epochs": stale_epochs, "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                    "loader_rng": train_loader.generator.get_state()}, output_dir / "last.pt")
        if stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    if history:
        plot_history(history_path, output_dir / "loss_curves.png")

    summary: dict[str, object] = {"best_epoch": best_epoch, "best_val_iou": best_iou,
                                "overfit_diagnostic_only": bool(args.overfit_windows)}
    if args.evaluate_test:
        if not (output_dir / "best.pt").is_file():
            raise FileNotFoundError("Test evaluation requires best.pt in the output directory")
        checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        test_loader = make_loader(
            args.splits, {args.test_split}, args, augment=False, shuffle=False
        )
        with torch.no_grad():
            test_loss, test_metrics = run_epoch(
                model, test_loader, device, args.lambda_sdf, None, None
            )
        summary["test_loss"] = test_loss
        summary["test_metrics"] = test_metrics
        (output_dir / "test_metrics.json").write_text(
            json.dumps(test_metrics, indent=2), encoding="utf-8"
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
