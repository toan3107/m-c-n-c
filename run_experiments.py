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
}
MAIN_EXPERIMENTS = list(EXPERIMENTS)[:7]


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
    parser.add_argument("--stage", choices=["smoke", "overfit", "pilot", "confirm", "final-test"], required=True)
    parser.add_argument("--experiments", nargs="+", choices=list(EXPERIMENTS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-horizon-days", type=float, default=90.)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--confirm-final-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.stage == "final-test" and not args.confirm_final_test:
        raise ValueError("Final test is locked until you explicitly pass --confirm-final-test")
    if args.stage != "final-test" and args.confirm_final_test:
        raise ValueError("--confirm-final-test is only valid with --stage final-test")
    code = Path(__file__).resolve().parent
    names = args.experiments or (["cfc_sdf"] if args.stage == "overfit" else MAIN_EXPERIMENTS)
    seeds = args.seeds or [42]
    epochs = args.epochs or (120 if args.stage == "overfit" else 80 if args.stage in ("confirm", "final-test") else 20)
    patience = epochs + 1 if args.stage == "overfit" else 15 if args.stage in ("confirm", "final-test") else 8
    output_root = args.output_root.resolve()
    stage_root = output_root / args.stage
    settings = {"experiments": names, "seeds": seeds, "epochs": epochs, "patience": patience,
                "image_size": args.image_size, "base_channels": args.base_channels,
                "batch_size": args.batch_size, "context_length": 3, "max_horizon_days": args.max_horizon_days,
                "lr": .0003, "device": args.device, "stage": args.stage,
                "manifest_sha256": hashlib.sha256((code / "splits.csv").read_bytes()).hexdigest(),
                "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(code.glob("*.py"))}}
    print(json.dumps(settings, indent=2), flush=True)
    commands = []
    for name in names:
        model, weight = EXPERIMENTS[name]
        for seed in seeds:
            training_stage = "confirm" if args.stage == "final-test" else args.stage
            run = output_root / training_stage / name / f"seed_{seed}"
            if args.stage != "final-test":
                train = [sys.executable, "-u", str(code / "train.py"), "--model", model,
                         "--lambda-sdf", str(weight), "--seed", str(seed), "--epochs", str(epochs),
                         "--patience", str(patience), "--image-size", str(args.image_size),
                         "--base-channels", str(args.base_channels), "--batch-size", str(args.batch_size),
                         "--context-length", "3", "--max-horizon-days", str(args.max_horizon_days),
                         "--num-workers", "0", "--device", args.device, "--output-dir", str(run)]
                if args.stage == "smoke":
                    train.append("--smoke-test")
                if args.stage == "overfit":
                    train += ["--overfit-windows", "2"]
                commands.append((name, seed, run, train))
            if args.stage in ("pilot", "confirm", "final-test"):
                split = "test" if args.stage == "final-test" else "val"
                destination = stage_root / name / f"seed_{seed}" / f"{split}_evaluation"
                evaluation = [sys.executable, "-u", str(code / "evaluate.py"), "--checkpoint", str(run / "best.pt"),
                              "--split", split, "--device", args.device, "--output-dir", str(destination)]
                if split == "test":
                    evaluation.append("--confirm-final-test")
                commands.append((name, seed, destination, evaluation))
    if args.dry_run:
        for _, _, _, command in commands:
            print(subprocess.list2cmdline(command))
        return
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
    if args.stage in ("pilot", "confirm", "final-test"):
        execute([sys.executable, str(code / "summarize_experiments.py"), "--stage-dir", str(stage_root)], stage_root / "summary.log")
    print(f"STAGE COMPLETE: {args.stage}. Outputs: {stage_root}", flush=True)


if __name__ == "__main__":
    main()
