import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

"""Plot truth / background / analysis / error for selected cycle frames of run_cycle output."""
import argparse

_a = argparse.ArgumentParser(description=__doc__)
_a.add_argument('--data-dir', default='outputs/cycle')
_a.add_argument('--frames', type=int, nargs='+', default=[1, 5, 10, 15, 20])
_a.add_argument('--channel', type=int, choices=[0, 1, 2], default=0, help='0: b, 1: u_x, 2: u_z')
_a.add_argument('--error-max', type=float, default=0.08)
_args = _a.parse_args()

DATA_DIR = Path(_args.data_dir)
FRAMES = _args.frames
CHANNEL = _args.channel
ERROR_MAX = _args.error_max
CHANNEL_NAME = {0: "Buoyancy b", 1: "Velocity u_x", 2: "Velocity u_z"}
SAVE_PATH = DATA_DIR / f"sparse_DA_{CHANNEL_NAME[CHANNEL].replace(' ','_')}.png"


# =====================================================
# Load data
# =====================================================

truths = []
backgrounds = []
analyses = []
masks = []


for frame in FRAMES:

    file = DATA_DIR / f"frame_{frame:04d}.npz"

    data = np.load(file)


    truth = data["truth"][..., CHANNEL]

    background = data["background"][..., CHANNEL]

    analysis = data["analysis"][..., CHANNEL]


    mask = data["mask"]


    # mask shape:
    # (H,W) or (H,W,3)

    if mask.ndim == 3:
        mask = mask[...,0]


    truths.append(truth)
    backgrounds.append(background)
    analyses.append(analysis)
    masks.append(mask)



# =====================================================
# Check fixed observation
# =====================================================

for i in range(1,len(masks)):

    if not np.array_equal(
        masks[0],
        masks[i]
    ):

        raise RuntimeError(
            "Observation mask changes between frames!"
        )


mask = masks[0]


print(
    "Fixed mask verified"
)

print(
    "Observation fraction:",
    mask.mean()
)

print(
    "Observation points:",
    mask.sum()
)



# =====================================================
# Colormap
# =====================================================

field_cmap = plt.cm.RdBu_r.copy()

# missing region
field_cmap.set_bad(
    color="white"
)


error_cmap = plt.cm.magma.copy()



# =====================================================
# Color range
# =====================================================

vmin = min(
    x.min()
    for x in truths
)

vmax = max(
    x.max()
    for x in truths
)



# =====================================================
# Figure
# =====================================================

fig, axes = plt.subplots(
    5,
    len(FRAMES),

    figsize=(18,13)
)



row_titles = [

    "Sparse observations\n"
    "1% observed\n"
    "99% missing",

    "Ground truth",

    "Background forecast",

    "DPS reconstruction",

    "Absolute error"

]



field_img = None
error_img = None



# =====================================================
# Plot
# =====================================================

for col,frame in enumerate(FRAMES):


    truth = truths[col]

    bg = backgrounds[col]

    ana = analyses[col]


    # ---------------------------------
    # Row 1: sparse observation
    # ---------------------------------

    ax = axes[0,col]


    # IMPORTANT:
    # only observed values
    obs_field = np.where(
        mask,
        truth,
        np.nan
    )


    field_img = ax.imshow(

        obs_field.T,

        origin="lower",

        cmap=field_cmap,

        aspect="auto",

        interpolation="nearest",

        vmin=vmin,

        vmax=vmax

    )


    # black observation markers

    z,x = np.where(
        mask.T
    )


# observation value markers
    obs_values = truth[mask]


    ax.scatter(
    x,
    z,
    s=35,
    c=obs_values,
    cmap=field_cmap,
    vmin=vmin,
    vmax=vmax,
    edgecolors="black",
    linewidths=0.6
    )


    ax.set_title(
        f"t={frame}",
        fontsize=12,
        fontweight="bold"
    )


    ax.axis("off")



    # ---------------------------------
    # Truth
    # ---------------------------------

    ax = axes[1,col]


    ax.imshow(

        truth.T,

        origin="lower",

        cmap=field_cmap,

        aspect="auto",

        interpolation="bilinear",

        vmin=vmin,

        vmax=vmax

    )


    ax.axis("off")



    # ---------------------------------
    # Background
    # ---------------------------------

    ax = axes[2,col]


    bg_rmse = np.sqrt(
        np.mean(
            (bg-truth)**2
        )
    )


    ax.imshow(

        bg.T,

        origin="lower",

        cmap=field_cmap,

        aspect="auto",

        interpolation="bilinear",

        vmin=vmin,

        vmax=vmax

    )


    ax.text(

        0.5,

        -0.12,

        f"RMSE={bg_rmse:.4f}",

        transform=ax.transAxes,

        ha="center",

        fontsize=10

    )


    ax.axis("off")



    # ---------------------------------
    # DPS analysis
    # ---------------------------------

    ax = axes[3,col]


    ana_rmse = np.sqrt(
        np.mean(
            (ana-truth)**2
        )
    )


    ax.imshow(

        ana.T,

        origin="lower",

        cmap=field_cmap,

        aspect="auto",

        interpolation="bilinear",

        vmin=vmin,

        vmax=vmax

    )


    ax.text(

        0.5,

        -0.12,

        f"RMSE={ana_rmse:.4f}",

        transform=ax.transAxes,

        ha="center",

        fontsize=10

    )


    ax.axis("off")



    # ---------------------------------
    # Error
    # ---------------------------------

    ax = axes[4,col]


    error = np.abs(
        ana-truth
    )


    error_img = ax.imshow(

        error.T,

        origin="lower",

        cmap=error_cmap,

        aspect="auto",

        interpolation="bilinear",

        vmin=0,

        vmax=ERROR_MAX

    )


    ax.axis("off")



# =====================================================
# Labels
# =====================================================

for i,title in enumerate(row_titles):

    axes[i,0].text(

        -0.35,

        0.5,

        title,

        transform=axes[i,0].transAxes,

        rotation=90,

        va="center",

        ha="center",

        fontsize=12,

        fontweight="bold"

    )



# =====================================================
# Colorbars
# =====================================================


cax1 = fig.add_axes(
    [0.92,0.58,0.015,0.22]
)


fig.colorbar(
    field_img,
    cax=cax1
)



cax2 = fig.add_axes(
    [0.92,0.16,0.015,0.22]
)


fig.colorbar(
    error_img,
    cax=cax2
)



# =====================================================
# Title
# =====================================================

fig.suptitle(

    "RB Diffusion-DA Sparse Observation Assimilation\n"
    "Fixed 1% Observations (99% Missing)\n"
    f"{CHANNEL_NAME[CHANNEL]}",

    fontsize=18,

    fontweight="bold"

)



plt.tight_layout(
    rect=[0,0,0.9,0.93]
)



plt.savefig(

    SAVE_PATH,

    dpi=400,

    bbox_inches="tight"

)
print(
    "saved:",
    SAVE_PATH
)