import numpy as np
import matplotlib.pyplot as plt


import argparse

_a = argparse.ArgumentParser(description="Plot truth/background/analysis/error fields of one run_cycle frame.")
_a.add_argument("--frame-file", default="outputs/cycle/frame_0020.npz")
_a.add_argument("--out", default=None, help="default: <frame-file>_fields.png")
_args = _a.parse_args()
FILE = _args.frame_file
OUT = _args.out or FILE.replace(".npz", "_fields.png")


def show_triplet(ax_row, truth, background, analysis, title, cmap="RdBu_r"):
    vmin = min(truth.min(), background.min(), analysis.min())
    vmax = max(truth.max(), background.max(), analysis.max())

    im0 = ax_row[0].imshow(truth, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax_row[0].set_title(f"Truth: {title}")
    ax_row[0].axis("off")

    im1 = ax_row[1].imshow(background, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax_row[1].set_title(f"Background: {title}")
    ax_row[1].axis("off")

    im2 = ax_row[2].imshow(analysis, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax_row[2].set_title(f"Analysis: {title}")
    ax_row[2].axis("off")

    return im2


def show_error(ax, pred, truth, title):
    err = np.abs(pred - truth)
    im = ax.imshow(err, origin="lower", cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.axis("off")
    return im


def main():
    data = np.load(FILE)

    truth = data["truth"]          # (H,W,3)
    bg    = data["background"]     # (H,W,3) or empty at frame 0
    ana   = data["analysis"]       # (H,W,3)

    # 通道定义
    # 0: b
    # 1: u_x
    # 2: u_z

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))

    # Row 1: buoyancy b
    im = show_triplet(
        axes[0, :3],
        truth[..., 0],
        bg[..., 0],
        ana[..., 0],
        "b",
        cmap="RdBu_r"
    )
    cbar = plt.colorbar(im, ax=axes[0, :3], fraction=0.02, pad=0.02)
    cbar.set_label("b")

    im_err = show_error(
        axes[0, 3],
        ana[..., 0],
        truth[..., 0],
        "|Analysis - Truth| : b"
    )
    cbar = plt.colorbar(im_err, ax=axes[0, 3], fraction=0.046, pad=0.04)
    cbar.set_label("abs error")

    # Row 2: u_x
    im = show_triplet(
        axes[1, :3],
        truth[..., 1],
        bg[..., 1],
        ana[..., 1],
        "u_x",
        cmap="RdBu_r"
    )
    cbar = plt.colorbar(im, ax=axes[1, :3], fraction=0.02, pad=0.02)
    cbar.set_label("u_x")

    im_err = show_error(
        axes[1, 3],
        ana[..., 1],
        truth[..., 1],
        "|Analysis - Truth| : u_x"
    )
    cbar = plt.colorbar(im_err, ax=axes[1, 3], fraction=0.046, pad=0.04)
    cbar.set_label("abs error")

    # Row 3: u_z
    im = show_triplet(
        axes[2, :3],
        truth[..., 2],
        bg[..., 2],
        ana[..., 2],
        "u_z",
        cmap="RdBu_r"
    )
    cbar = plt.colorbar(im, ax=axes[2, :3], fraction=0.02, pad=0.02)
    cbar.set_label("u_z")

    im_err = show_error(
        axes[2, 3],
        ana[..., 2],
        truth[..., 2],
        "|Analysis - Truth| : u_z"
    )
    cbar = plt.colorbar(im_err, ax=axes[2, 3], fraction=0.046, pad=0.04)
    cbar.set_label("abs error")

    plt.tight_layout()
    plt.savefig(OUT, dpi=300)

    print("saved:", OUT)


if __name__ == "__main__":
    main()