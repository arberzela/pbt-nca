"""
Fitness plot with embedded GIF frames along the curve.

Reads the W&B-exported CSV in this directory and plots mean fitness
over meta-iterations.  For each .gif file (whose filename encodes the
meta-iteration), one frame is extracted and placed as an inset image
on the plot at the corresponding x position, connected to the curve
by a thin leader line.
"""

import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

# ── configuration ────────────────────────────────────────────────────────────

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
CSV_PATH = sorted(SCRIPT_DIR.glob("wandb_export_*.csv"))[0]  # first match
GIF_DIR = SCRIPT_DIR
OUT_PATH = SCRIPT_DIR / "fitness_with_frames_7.png"

SMOOTH_WINDOW = 10
INSET_ZOOM = 0.65          # size of each embedded frame
LINE_COLOR = "#0077BB"     # Okabe-Ito blue
FILL_COLOR = "#0077BB"
BG_COLOR = "#fafafa"


# ── helpers ──────────────────────────────────────────────────────────────────

def smooth(arr, w):
    return pd.Series(arr).rolling(w, center=True, min_periods=1).mean().values


def load_csv(path):
    """Return (steps, mean, lo, hi) arrays from the W&B CSV."""
    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    df = df.rename(columns={df.columns[0]: "Step"})
    df["Step"] = df["Step"].astype(str).str.strip('"').astype(int)

    # identify the base mean column (not MIN/MAX)
    mean_col = [c for c in df.columns
                if "fitness/mean" in c
                and "__MIN" not in c
                and "__MAX" not in c][0]
    min_col = mean_col + "__MIN"
    max_col = mean_col + "__MAX"

    steps = df["Step"].values
    mean = df[mean_col].astype(float).values
    lo = df[min_col].astype(float).values if min_col in df.columns else mean
    hi = df[max_col].astype(float).values if max_col in df.columns else mean
    return steps, mean, lo, hi


def load_gif_frames(gif_dir):
    """Return sorted list of (iteration, PIL.Image) from .gif files."""
    frames = []
    for i, p in enumerate(sorted(gif_dir.glob("*.gif"))):
        print(i, p.name)
        try:
            iteration = int(p.stem)
        except ValueError:
            continue
        img = Image.open(p)
        # count total frames and pick one near the end (~75%)
        n_frames = 0
        try:
            while True:
                n_frames += 1
                img.seek(n_frames)
        except EOFError:
            pass
        if i == 3:
            target = max(0, int(n_frames * 0.65) - 1)
        elif i == 4:
            target = max(0, int(n_frames * 0.90) - 1)
        elif i == 5:
            target = max(0, int(n_frames * 0.95) - 1)
        elif i == 6:
            target = max(0, int(n_frames * 0.98) - 1)
        elif i == 8:
            target = max(0, int(n_frames * 0.98) - 1)
        elif i == 9:
            target = max(0, int(n_frames * 0.7) - 1)
        elif i == 10:
            target = max(0, int(n_frames * 0.85) - 1)
        elif i == 11:
            target = max(0, int(n_frames * 0.48) - 1)
        elif i == 12:
            target = max(0, int(n_frames * 0.95) - 1)
        else:
            target = max(0, int(n_frames * 0.65) - 1)
        img.seek(target)
        frames.append((iteration, img.copy().convert("RGBA")))
    frames.sort(key=lambda t: t[0])
    return frames


# ── main plot ────────────────────────────────────────────────────────────────

def make_plot():
    steps, mean, lo, hi = load_csv(CSV_PATH)
    gif_frames = load_gif_frames(GIF_DIR)

    sm_mean = smooth(mean, SMOOTH_WINDOW)
    sm_lo = smooth(lo, SMOOTH_WINDOW)
    sm_hi = smooth(hi, SMOOTH_WINDOW)

    # ── global style ─────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 20,
        "axes.linewidth": 1.5,
    })

    fig, ax = plt.subplots(figsize=(16, 7)) # 17, 6.5
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")

    # ── fitness curve ────────────────────────────────────────────────────────
    ax.plot(steps, sm_mean, color=LINE_COLOR, linewidth=3.4, alpha=0.95,
            label="Mean Fitness")
    ax.fill_between(steps, sm_lo, sm_hi,
                    color=FILL_COLOR, alpha=0.12, label="Min / Max band")

    # ── embed GIF frames ─────────────────────────────────────────────────────
    # Alternate frames above and below the curve
    y_data_range = sm_mean.max() - sm_mean.min()

    ABOVE_OFFSET = 0.65   # distance above curve point (in y_data_range units)
    BELOW_OFFSET = 0.75   # distance below curve point

    for idx, (iteration, pil_img) in enumerate(gif_frames):
        #print(idx, iteration)
        # find the y value on the smoothed curve at this iteration
        if iteration < steps[0] or iteration > steps[-1]:
            continue
        step_idx = np.searchsorted(steps, iteration)
        step_idx = min(step_idx, len(sm_mean) - 1)
        y_val = sm_mean[step_idx]

        # alternate above / below
        above = (idx % 2 == 0)
        if above:
            if idx == 0:
                y_img = y_val + (ABOVE_OFFSET + 0.25) * y_data_range
            else:
                y_img = y_val + ABOVE_OFFSET * y_data_range
        else:
            if idx == 1:
                y_img = y_val - (BELOW_OFFSET + 0.05) * y_data_range
            else:
                y_img = y_val - BELOW_OFFSET * y_data_range

        # draw leader line from curve point to image centre
        ax.annotate("",
                    xy=(iteration, y_val),
                    xytext=(iteration, y_img),
                    arrowprops=dict(arrowstyle="-",
                                   color="#888888",
                                   linewidth=0.9,
                                   linestyle="--"))

        # small dot on the curve
        ax.plot(iteration, y_val, "o", color=LINE_COLOR,
                markersize=5, zorder=5)

        # place image
        img_arr = np.array(pil_img)
        im = OffsetImage(img_arr, zoom=INSET_ZOOM)
        im.image.axes = ax
        ab = AnnotationBbox(
            im, (iteration, y_img),
            frameon=True,
            bboxprops=dict(edgecolor="#555555", linewidth=1.0,
                           facecolor="white", boxstyle="round,pad=0.1"),
            zorder=4,
        )
        ax.add_artist(ab)

        # iteration label: placed on the far side from the curve
        if above:
            ax.text(iteration, y_img + 0.10 * y_data_range,
                    f"iter {iteration}", ha="center", va="bottom",
                    fontsize=7.5, color="#444444", fontstyle="italic")
        else:
            ax.text(iteration, y_img - 0.10 * y_data_range,
                    f"iter {iteration}", ha="center", va="top",
                    fontsize=7.5, color="#444444", fontstyle="italic")

    # ── expand y-axis so all frames are fully visible ────────────────────────
    y_bottom = sm_mean.min() - (BELOW_OFFSET + 0.55) * y_data_range
    y_top = sm_mean.max() + (ABOVE_OFFSET + 0.55) * y_data_range
    ax.set_ylim(0.1, 0.91)

    # ── spines & ticks ───────────────────────────────────────────────────────
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#1a1a2e")
    ax.spines["bottom"].set_color("#bbbbbb")

    ax.tick_params(axis="y", colors="#1a1a2e", labelsize=12, length=4)
    ax.tick_params(axis="x", colors="#555555", labelsize=12, length=4)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=10))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.margins(x=0.02)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6,
                  color="#cccccc", alpha=0.8)

    # ── labels & title ───────────────────────────────────────────────────────
    ax.set_xlabel("Meta-iteration", fontsize=16, color="#444444", labelpad=8, fontweight="semibold")
    ax.set_ylabel("Mean Fitness", fontsize=16, color="#1a1a2e",
                  labelpad=10, fontweight="semibold")

    #ax.legend(loc="lower right", fontsize=10, framealpha=0.9,
    #          edgecolor="#cccccc", fancybox=False)

    ax.set_title(
        f"rolling mean smoothed (window = {SMOOTH_WINDOW})",
        fontsize=12, color="#888888", pad=6, loc="left",
    )
    fig.suptitle(
        "PBT-NCA · Fitness over Meta-iterations with Emergent Dynamics",
        fontsize=16, fontweight="bold", color="#1a1a2e", y=0.97,
    )

    # ── save ─────────────────────────────────────────────────────────────────
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        out = OUT_PATH.with_suffix(suffix)
        fig.savefig(out, dpi=200, bbox_inches="tight",
                    transparent=True)
        print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    make_plot()
