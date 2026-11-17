from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


EXPERIMENTS = {
    "rgb_persistence": ("segmenter_persistence", 0.),
    "last_frame": ("last_frame", 0.),
    "convgru": ("convgru", 0.),
    "dt_convgru": ("dt_convgru", 0.),
    "cfc_notime": ("spatial_cfc_notime", 0.),
    "cfc_no_sdf": ("spatial_cfc", 0.),
    "cfc_sdf": ("spatial_cfc", .1),
    "cfc_sdf_001": ("spatial_cfc", .01),
    "cfc_sdf_003": ("spatial_cfc", .03),
    "cfc_no_gap": ("spatial_cfc_nogap", .1),
    "cfc_no_calendar": ("spatial_cfc_nocalendar", .1),
    "cfc_change_sdf": ("spatial_cfc_change", .1),
    "cfc_rate_sdf": ("spatial_cfc_rate", .1),
    "cfc_multiscale_sdf": ("spatial_cfc_multiscale", .1),
    "cfc_multiscale_rate_sdf": ("spatial_cfc_multiscale_rate", .1),
    "cfc_multiscale_lastfine_sdf": ("spatial_cfc_multiscale_lastfine", .1),
}
MAIN_EXPERIMENTS = list(EXPERIMENTS)[:7]
DEFAULTS = dict(image_size=512, base_channels=16, batch_size=2, context_length=3,
                max_horizon_days=90., lr=.0003, weight_decay=.0001,
                lr_schedule="constant", warmup_epochs=0, min_lr_ratio=.01)


def apply_profile(args: argparse.Namespace) -> None:
    profile = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    allowed = set(DEFAULTS) | {"epochs", "patience", "seeds", "experiments"}
    if set(profile) - allowed:
        raise ValueError(f"Unknown config fields: {sorted(set(profile) - allowed)}")
    for key in allowed:
        if getattr(args, key) is None:
            setattr(args, key, profile.get(key, DEFAULTS.get(key)))
    if args.experiments and any(name not in EXPERIMENTS for name in args.experiments):
        raise ValueError("Profile contains an unknown experiment")


def execute(command: list[str], log_path: Path) -> None:
    environment = os.environ.copy()
    environment.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    print("RUN: " + subprocess.list2cmdline(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n" + datetime.now(timezone.utc).isoformat() + "\n")
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace", env=environment) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
    if code:
        raise RuntimeError(f"Command failed (exit {code}); stopped without retry. See {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential, validation-first RTX 3090 method experiments")
    parser.add_argument("--stage", choices=["smoke", "overfit", "pilot", "confirm", "final-test", "refit", "refit-test"], required=True)
    parser.add_argument("--config", type=Path, help="Explicit JSON profile; CLI values override it")
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--base-channels", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--max-horizon-days", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--lr-schedule", choices=["constant", "warmup_cosine"])
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--min-lr-ratio", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    explicit_experiments = args.experiments
    apply_profile(args)
    test_stage = args.stage in ("final-test", "refit-test")
    refit = args.stage in ("refit", "refit-test")
    if test_stage and not args.confirm_final_test:
        raise ValueError("Final test is locked until you explicitly pass --confirm-final-test")
    if not test_stage and args.confirm_final_test:
        raise ValueError("--confirm-final-test is only valid with a locked test stage")
    code = Path(__file__).resolve().parent
    names = args.experiments or (["cfc_sdf"] if args.stage == "overfit" else MAIN_EXPERIMENTS)
    if args.stage == "overfit" and explicit_experiments is None:
        names = ["cfc_sdf"]
    seeds = args.seeds or [42]
    epochs = args.epochs if args.epochs is not None else (120 if args.stage == "overfit" or refit else 80 if args.stage in ("confirm", "final-test") else 20)
    if epochs < 1 or args.batch_size < 1 or args.context_length < 1 or args.base_channels < 1:
        raise ValueError("Epochs, batch size, context length and base channels must be positive")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use a nonempty list of distinct seeds")
    patience = epochs + 1 if args.stage == "overfit" or refit else args.patience or (15 if args.stage in ("confirm", "final-test") else 8)
    output_root = args.output_root.resolve()
    stage_root = output_root / args.stage
    settings = {"experiments": names, "seeds": seeds, "epochs": epochs, "patience": patience,
                "image_size": args.image_size, "base_channels": args.base_channels,
                "batch_size": args.batch_size, "context_length": args.context_length, "max_horizon_days": args.max_horizon_days,
                "lr": args.lr, "weight_decay": args.weight_decay,
                "lr_schedule": args.lr_schedule, "warmup_epochs": args.warmup_epochs,
                "min_lr_ratio": args.min_lr_ratio, "no_validation": refit,
                "training_splits": ["train", "val"] if refit else ["train"],
                "checkpoint": "last.pt" if refit else "best.pt",
                "device": args.device, "stage": args.stage,
                "manifest_sha256": hashlib.sha256((code / "splits.csv").read_bytes()).hexdigest(),
                "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(code.glob("*.py"))}}
    print(json.dumps(settings, indent=2), flush=True)
    commands = []
    for name in names:
        model, weight = EXPERIMENTS[name]
        for seed in seeds:
            training_stage = "refit" if refit else "confirm" if test_stage else args.stage
            run = output_root / training_stage / name / f"seed_{seed}"
            if not test_stage:
                train = [sys.executable, "-u", str(code / "train.py"), "--model", model,
                         "--lambda-sdf", str(weight), "--seed", str(seed), "--epochs", str(epochs),
                         "--patience", str(patience), "--image-size", str(args.image_size),
                         "--base-channels", str(args.base_channels), "--batch-size", str(args.batch_size),
                         "--context-length", str(args.context_length), "--max-horizon-days", str(args.max_horizon_days),
                         "--lr", str(args.lr), "--weight-decay", str(args.weight_decay),
                         "--lr-schedule", args.lr_schedule, "--warmup-epochs", str(args.warmup_epochs),
                         "--min-lr-ratio", str(args.min_lr_ratio),
                         "--num-workers", "0", "--device", args.device, "--output-dir", str(run)]
                if args.stage == "smoke":
                    train.append("--smoke-test")
                if args.stage == "overfit":
                    train += ["--overfit-windows", "2"]
                if refit:
                    train += ["--no-validation", "--train-splits", "train", "val"]
                commands.append((name, seed, run, train))
            if args.stage in ("pilot", "confirm") or test_stage:
                split = "test" if test_stage else "val"
                destination = stage_root / name / f"seed_{seed}" / f"{split}_evaluation"
                evaluation = [sys.executable, "-u", str(code / "evaluate.py"), "--checkpoint", str(run / settings["checkpoint"]),
                              "--split", split, "--device", args.device, "--output-dir", str(destination)]
                if split == "test":
                    evaluation.append("--confirm-final-test")
                commands.append((name, seed, destination, evaluation))
    if args.dry_run:
        for _, _, _, command in commands:
            print(subprocess.list2cmdline(command))
        return
    if args.stage == "refit-test":
        training_plan_path = output_root / "refit" / "experiment_plan.json"
        if not training_plan_path.is_file():
            raise ValueError("Refit-test requires a completed, frozen refit plan")
        training_plan = json.loads(training_plan_path.read_text(encoding="utf-8"))
        for key in settings.keys() - {"stage", "device"}:
            if training_plan.get(key) != settings[key]:
                raise ValueError(f"Refit-test settings/code differ from training: {key}")
        for name in names:
            for seed in seeds:
                summary_path = output_root / "refit" / name / f"seed_{seed}" / "summary.json"
                if not summary_path.is_file():
                    raise ValueError(f"Refit is not complete: {name}, seed {seed}")
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if not summary.get("no_validation") or summary.get("last_epoch") != epochs:
                    raise ValueError("Refit must complete its declared fixed budget before testing")
    stage_root.mkdir(parents=True, exist_ok=True)
    plan_path = stage_root / "experiment_plan.json"
    if plan_path.is_file():
        if json.loads(plan_path.read_text(encoding="utf-8")) != settings:
            raise ValueError("Stage settings/code changed; use a new --output-root to preserve comparability")
    else:
        plan_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    for name, seed, destination, command in commands:
        destination.mkdir(parents=True, exist_ok=True)
        is_training = Path(command[2]).name == "train.py"
        marker = destination / ("smoke_report.json" if args.stage == "smoke" else "summary.json" if is_training else "metrics.json")
        if marker.is_file():
            print(f"SKIP complete: {name}, seed {seed}, {destination.name}", flush=True)
            continue
        if is_training and (destination / "last.pt").is_file():
            if not args.resume:
                raise FileExistsError(f"Partial run: {destination}; rerun the same stage with --resume")
            command += ["--resume", str(destination / "last.pt")]
        execute(command, destination / "console.log")
    if args.stage in ("pilot", "confirm") or test_stage:
        execute([sys.executable, str(code / "summarize_experiments.py"), "--stage-dir", str(stage_root)], stage_root / "summary.log")
    print(f"STAGE COMPLETE: {args.stage}. Outputs: {stage_root}", flush=True)


if __name__ == "__main__":
    main()
