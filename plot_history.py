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
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for prefix, style in (("train", "-"), ("val", "--")):
        axes[0].plot(epochs, values(f"{prefix}_loss"), style, label=prefix)
        axes[1].plot(epochs, [float(row[f"{prefix}_sdf_loss"]) * float(row["lambda_sdf"]) for row in rows], style, label=f"{prefix} weighted SDF")
    axes[2].plot(epochs, values("val_iou"), label="validation reservoir-macro IoU")
    for axis, title in zip(axes, ("Total loss", "Auxiliary SDF contribution", "Future-mask IoU")):
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
