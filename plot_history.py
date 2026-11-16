from __future__ import annotations

import argparse
import csv
from pathlib import Path


def plot_history(history_path: str | Path, output: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(history_path).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return
    values = lambda key: [float(row[key]) for row in rows]
    epochs = values("epoch")
    has_lr = "lr" in rows[0] and all(row.get("lr") for row in rows)
    count = 4 if has_lr else 3
    fig, axes = plt.subplots(1, count, figsize=(4 * count, 3.5), constrained_layout=True)
    for prefix, style in (("train", "-"), ("val", "--")):
        if f"{prefix}_loss" not in rows[0]:
            continue
        axes[0].plot(epochs, values(f"{prefix}_loss"), style, label=prefix)
        axes[1].plot(epochs, [float(row[f"{prefix}_sdf_loss"]) * float(row["lambda_sdf"]) for row in rows], style, label=f"{prefix} weighted SDF")
    axes[2].plot(epochs, values("train_iou"), label="training micro IoU")
    if "val_iou" in rows[0]:
        axes[2].plot(epochs, values("val_iou"), label="validation reservoir-macro IoU")
    if has_lr:
        axes[3].plot(epochs, values("lr"), label="optimizer learning rate")
    for axis, title in zip(axes, ("Total loss", "Auxiliary SDF contribution", "Future-mask IoU", "Learning rate")):
        axis.set(title=title, xlabel="Epoch")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    parser.add_argument("--output", type=Path, default=Path("loss_curves.png"))
    args = parser.parse_args()
    plot_history(args.history, args.output)
