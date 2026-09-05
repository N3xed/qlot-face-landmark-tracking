"""Generate the hidden-state-carry comparison figure for the paper (C2, A0 vs A1).

This script produces the quantitative companion to the qualitative per-clip
error plots in the eval notebooks. It contrasts the A0 (QLOT-final) reference
with A1 (coordinates-only carry, hidden state reset to zero at every frame),
using only the saved ``results_*.pkl`` files -- no model or dataset needed.

Two panels are produced side by side:

1. Per-frame normalized landmark error, averaged over all 150 WFLW-V test
   clips and over the three seeds, with a ±1 std band over clips. A0 uses the
   full recurrent state; A1 carries only the previous coordinates. The plot
   makes the mechanism visible: both start identically at frame 0 (four
   refinement iterations), but A1's error jumps at frame 1 (one iteration, no
   hidden state) and never recovers.
2. Allan deviation NAD(m) for m = 1..60 on log axes, mean over seeds, for A0
   and A1, scaled by the mean face size (sqrt(w*h) of the ground-truth
   landmark bounding box) to give the jitter in pixels. This shows the jitter
   difference across all averaging timescales, complementing the per-frame view.
   Additionally, the NAD curves of A2 (local only) and A6 (no temporal GNLL)
   are overlaid as dashed, semi-transparent lines for context; they are not
   part of the C2 comparison. They also appear in panel 1 as dashed mean
   curves without error bands.

Both metrics use the same normalization as the eval notebooks / ``calc_nmf`` /
``navar_pos_all``: errors are divided by the geometric mean of the
ground-truth bounding-box width and height (sqrt(w*h)) per frame.

Usage: ``python results/fig_hidden_state_carry.py [out_path]``
Default output: ``results/fig_hidden_state_carry.pdf`` (and .png).
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

RESULTS_DIR = Path(__file__).parent
SEEDS = ("s1", "s2", "s3")
CONFIGS = {"QLOT-final": "a4-v2-mean-only-wmr", "A1 (no hidden state)": "a1-v2-coordinates-only-carry"}
# Additional ablations shown as dashed, semi-transparent curves (no error
# bands, no slope annotations); they are excluded from the C2 comparison.
NAD_EXTRA_CONFIGS = {
    "A2 (local only)": "a2-v2-per-landmark-gru",
    "A6 (no temporal GNLL)": "a6-v2-spatial-gnll",
}


def load(name: str, seed: str) -> dict:
    with open(RESULTS_DIR / f"results_{name}-{seed}.pkl", "rb") as f:
        return pickle.load(f)


def per_frame_nme(d: dict) -> np.ndarray:
    """Mean bbox-size-normalized NME per frame, over clips. Returns (120,) in %."""
    return d["wflw_v"]["nmes_size"].mean(axis=0) * 100.0


def per_clip_per_frame_nme(d: dict) -> np.ndarray:
    """Per-clip per-frame NME. Returns (clips, 120) in %."""
    return d["wflw_v"]["nmes_size"] * 100.0


def nad_curve(d: dict) -> np.ndarray:
    """NAD(m) = sqrt(mean over landmarks of navar_pos[m-1]). Returns (60,)."""
    navar = d["wflw_v"]["navar_pos"]  # (60, videos, landmarks)
    return np.sqrt(navar.mean(axis=-1)).mean(axis=-1)


def mean_face_size(d: dict) -> float:
    """Mean per-frame sqrt(w*h) of the ground-truth landmark bounding box, in px."""
    labels = d["wflw_v"]["labels"]  # (clips, frames, landmarks, 2)
    bbox_min = labels.min(axis=2)  # (clips, frames, 2)
    bbox_max = labels.max(axis=2)  # (clips, frames, 2)
    sizes = np.sqrt(np.prod(bbox_max - bbox_min, axis=-1))  # (clips, frames)
    return float(sizes.mean())

def nad_relative_errors(m_vals: int, total_vals: int, num_clips: int, num_seeds: int) -> np.ndarray:
    """
    Compute relative error based on Eq. 26 of [1] and mean over clips and seeds.
    Returns (m_vals,).
    
    Args:
        m_vals: Number of averaging timescales (e.g., 60 for m=1 to 60).
        total_vals: Number of data points in a clip.
        num_clips: Number of clips (e.g., 150 for WFLW-V test set).
        num_seeds: Number of seeds (e.g., 3).
    Returns:
        Relative error array of shape (m_vals,); entry i is the relative error of
        NAD(m = i + 1). Assumes the num_clips * num_seeds averaged sequences are
        independent (seeds share the same test clips, so the true band is slightly
        wider).

    References:
        [1] N. El-Sheimy, H. Hou, and X. Niu, “Analysis and Modeling of Inertial Sensors Using Allan Variance,” IEEE Trans. Instrum. Meas., vol. 57, no. 1, pp. 140–149, Jan. 2008, doi: 10.1109/TIM.2007.908635.
    """
    # n is the number of data points in a cluster/block (see [1]); here n = m = 1..m_vals,
    # ordered ascending so entry i is the relative error of NAD(m = i + 1), matching the
    # indexing of navar_pos_all / nad_curve.
    n = np.arange(1, m_vals + 1)
    rel_err = 1.0 / np.sqrt(2 * (total_vals / n - 1) * num_clips * num_seeds)
    return rel_err

def main() -> None:
    out_base = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR / "fig_hidden_state_carry"

    # Collect per-seed series for each config.
    frame_nme: dict[str, list[np.ndarray]] = {k: [] for k in CONFIGS}  # per seed, (120,)
    clip_frame_nme: dict[str, list[np.ndarray]] = {k: [] for k in CONFIGS}  # per seed, (clips, 120)
    nad: dict[str, list[np.ndarray]] = {k: [] for k in CONFIGS}  # per seed, (60,)
    frame_nme_extra: dict[str, list[np.ndarray]] = {k: [] for k in NAD_EXTRA_CONFIGS}
    nad_extra: dict[str, list[np.ndarray]] = {k: [] for k in NAD_EXTRA_CONFIGS}
    face_sizes: list[float] = []
    for label, name in CONFIGS.items():
        for seed in SEEDS:
            d = load(name, seed)
            frame_nme[label].append(per_frame_nme(d))
            clip_frame_nme[label].append(per_clip_per_frame_nme(d))
            nad[label].append(nad_curve(d))
            face_sizes.append(mean_face_size(d))
    for label, name in NAD_EXTRA_CONFIGS.items():
        for seed in SEEDS:
            d = load(name, seed)
            frame_nme_extra[label].append(per_frame_nme(d))
            nad_extra[label].append(nad_curve(d))
            face_sizes.append(mean_face_size(d))
    # Identical for all configs/seeds (same WFLW-V test clips); average to be safe.
    face_size = float(np.mean(face_sizes))

    fig, (ax_gap, ax_nad) = plt.subplots(1, 2, figsize=(13, 4.5))
    colors = {"QLOT-final": "tab:blue", "A1 (no hidden state)": "tab:red"}
    extra_colors = {"A2 (local only)": "tab:orange", "A6 (no temporal GNLL)": "tab:green"}
    frames = np.arange(120)
    m_vals = np.arange(1, 61)

    # --- Panel 1: per-frame NME over the clip, mean over seeds, ±1 std over clips ---
    for label in CONFIGS:
        mean_curve = np.mean(frame_nme[label], axis=0)  # (120,), mean over seeds
        # std over all clips of all seeds (seed-pooled), for the band
        pooled = np.concatenate(clip_frame_nme[label], axis=0)  # (3*clips, 120)
        std_curve = pooled.std(axis=0)
        c = colors[label]
        ax_gap.plot(frames, mean_curve, color=c, label=label, linewidth=1.8)
        ax_gap.fill_between(frames, mean_curve - std_curve, mean_curve + std_curve, color=c, alpha=0.15)
    for label in NAD_EXTRA_CONFIGS:
        mean_curve = np.mean(frame_nme_extra[label], axis=0)  # (120,), mean over seeds
        ax_gap.plot(
            frames,
            mean_curve,
            color=extra_colors[label],
            label=label,
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
        )
    ax_gap.axvline(0.5, color="gray", linestyle=":", linewidth=1)
    ax_gap.annotate(
        "Frame 0: 4 iterations\nFrames ≥1: 1 iteration",
        xy=(1, ax_gap.get_ylim()[1]),
        xytext=(4, ax_gap.get_ylim()[1] * 0.97),
        fontsize=8,
        color="gray",
        va="top",
    )
    ax_gap.set_xlabel("Frame Index")
    ax_gap.set_ylabel("NME (% face size)")
    ax_gap.set_title("Per-frame Tracking Error (150 WFLW-V clips, 3 seeds)")
    ax_gap.legend(framealpha=0.5, loc="upper right")
    ax_gap.grid(alpha=0.3)

    # --- Panel 2: NAD(m) on log-log ---
    num_clips = clip_frame_nme[next(iter(CONFIGS))][0].shape[0]
    rel_err = nad_relative_errors(
        m_vals=len(m_vals), total_vals=frames.size, num_clips=num_clips, num_seeds=len(SEEDS)
    )
    mean_nads: dict[str, np.ndarray] = {}
    for label in CONFIGS:
        mean_nad = np.mean(nad[label], axis=0) * face_size  # (60,), px, mean over seeds
        mean_nads[label] = mean_nad
        c = colors[label]
        ax_nad.loglog(m_vals, mean_nad, color=c, label=label, linewidth=1.8)
        ax_nad.fill_between(
            m_vals, mean_nad * (1.0 - rel_err), mean_nad * (1.0 + rel_err), color=c, alpha=0.15
        )

    # Dashed, semi-transparent NAD curves for the additional ablations (no
    # error bands, no slope annotations -- context only).
    for label in NAD_EXTRA_CONFIGS:
        mean_nad = np.mean(nad_extra[label], axis=0) * face_size  # (60,), px
        ax_nad.loglog(
            m_vals,
            mean_nad,
            color=extra_colors[label],
            label=label,
            linestyle="--",
            linewidth=1.5,
            alpha=0.6,
        )

    ax_nad.annotate(
        f"mean face size: {face_size:.0f} px",
        xy=(0.03, 0.97),
        xycoords="axes fraction",
        fontsize=9,
        color="gray",
        va="top",
    )
    ax_nad.set_xlabel("Averaging Time $m$ (frames)")
    ax_nad.set_ylabel(f"NAD($m$) $\\times$ face size (px)")
    ax_nad.set_title(f"Normalized Allan Deviation (150 WFLW-V clips, 3 seeds)")
    ax_nad.legend(framealpha=0.5)
    ax_nad.grid(alpha=0.3, which="both")
    # Custom ticks, formatted as plain numbers -- integers on the x-axis --
    # instead of powers of 10. y: 0.5/1/2 major, the rest labeled minor ticks.
    ax_nad.set_xticks([1, 3, 6, 10, 30, 60])
    ax_nad.set_yticks([0.5, 1.0, 2.0])
    ax_nad.set_yticks([0.6, 0.7, 0.8, 0.9, 1.2, 1.4, 1.6, 1.8], minor=True)
    ax_nad.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax_nad.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax_nad.yaxis.set_minor_formatter(FormatStrFormatter("%g"))
    ax_nad.tick_params(axis="y", which="minor", labelsize=8)

    # Finalize the layout BEFORE measuring the data->pixel transform:
    # tight_layout resizes/moves the axes, so measuring earlier would compute
    # stale on-screen angles for the slope annotations.
    fig.tight_layout()
    fig.canvas.draw()

    # Annotate the log-log slope of each curve before and after its maximum:
    # fit a line to (log m, log NAD) over the segment, print the exponent, and
    # rotate the text by the segment's on-screen angle, obtained by mapping the
    # fitted line's endpoints through the log-scaled axes transform.
    transform = ax_nad.transData

    def screen_angle(m_lo: float, m_hi: float, slope: float, intercept: float) -> float:
        """Angle (degrees) of a log-log fitted line as rendered on the axes."""
        x0, x1 = m_lo, m_hi
        y0 = 10 ** (slope * np.log10(x0) + intercept)
        y1 = 10 ** (slope * np.log10(x1) + intercept)
        (px0, py0), (px1, py1) = transform.transform([(x0, y0), (x1, y1)])
        return float(np.degrees(np.arctan2(py1 - py0, px1 - px0)))

    # Slope segments per config: (m_start, m_end, label_side) fit ranges, one
    # before and one after each curve's maximum. The label sits on the
    # geometric middle of the segment, nudged perpendicular to the fitted line
    # to the given side of the line ("above" or "below").
    slope_segments = {
        "QLOT-final": [(1, 9, "below"), (30, 55, "below")],
        "A1 (no hidden state)": [(1, 4, "above"), (25, 60, "below")],
    }
    for label, segments in slope_segments.items():
        curve = mean_nads[label]
        c = colors[label]
        i_max = int(np.argmax(curve))
        for m_lo, m_hi, side in segments:
            sel = (m_vals >= m_lo) & (m_vals <= m_hi)
            slope, intercept = np.polyfit(np.log10(m_vals[sel]), np.log10(curve[sel]), 1)
            # Text anchor: geometric middle of the segment on the fitted line.
            m_mid = float(np.sqrt(m_lo * m_hi))
            y_mid = 10 ** (slope * np.log10(m_mid) + intercept)
            angle = screen_angle(m_lo, m_hi, slope, intercept)
            # Nudge the label off the curve along the line's normal direction.
            # The normal's screen-space y-component is positive for any
            # non-vertical line, so a positive offset places the label above
            # the line and a negative offset below it.
            offset_px = 10 if side == "above" else -10
            theta = np.radians(angle)
            (px, py) = transform.transform((m_mid, y_mid))
            (px_t, py_t) = (px - np.sin(theta) * offset_px, py + np.cos(theta) * offset_px)
            m_txt, y_txt = transform.inverted().transform((px_t, py_t))
            ax_nad.annotate(
                f"slope$\\approx${"−" if slope < 0 else "+"}{abs(slope):.2f}",
                xy=(m_mid, y_mid),
                xytext=(m_txt, y_txt),
                fontsize=8,
                color=c,
                ha="center",
                va="center",
                rotation=angle,
                rotation_mode="anchor",
            )
            ax_nad.plot([m_lo, m_hi],
                [10 ** (slope * np.log10(m_lo) + intercept), 10 ** (slope * np.log10(m_hi) + intercept)],
                color=c, linestyle=":", linewidth=1.5, alpha=0.7
            )
        # Mark the maximum as a point annotated with its m value; the text is
        # offset a few points above the marker (in points, not data units).
        ax_nad.plot(m_vals[i_max], curve[i_max], marker="o", markersize=6, color=c,
                    markerfacecolor="white", markeredgewidth=1.5, zorder=5)
        ax_nad.annotate(
            f"$m_c={m_vals[i_max]}$",
            xy=(m_vals[i_max], curve[i_max]),
            xytext=(0, 2.5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=c,
            fontweight="bold",
        )

    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out_base.with_suffix('.pdf')} and .png")

    # Also print the headline numbers used in the caption.
    for label in CONFIGS:
        gap = np.mean(frame_nme[label], axis=0)
        print(
            f"{label}: frame0 NME={gap[0]:.3f}%, frames1+ mean NME={gap[1:].mean():.3f}%, "
            f"NAD(1)={np.mean(nad[label], axis=0)[0] * face_size:.4f} px"
        )


if __name__ == "__main__":
    main()
