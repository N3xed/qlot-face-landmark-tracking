"""Delta-error autocorrelation figure for the NAD supplement (sec-supp-nad).

The supplement's delta-error ACF representation (eq. @eq-acf) writes NAD^2(m) as
a linear functional of the lagged delta-error covariances gamma_u(h). This figure
measures that ACF directly from the saved WFLW-V predictions, providing an
independent, model-level check of the NAD reading:

  - QLOT-final (recurrent state) shows a *negative* lag-1 delta-error
    correlation (u_l anti-correlated with u_{l-1}): each frame's update corrects
    part of the previous frame's error. This is the mechanism that damps the NAD
    rise to ~+0.22 and bounds the error walk.
  - A1 (hidden state reset each frame) shows a near-zero / slightly positive
    lag-1 correlation, then a delayed negative region peaking around lag 4-10:
    correction engages only once the accumulated drift is large enough to be
    visible in a single frame.

The delta error is u_l = e_l - e_{l-1}, where e_l is the face-size-normalized
landmark error (same normalization as navar_pos_all / the NAD panel). The ACF is
the Pearson autocorrelation of u, computed per clip per channel (landmark x/y),
then averaged over channels, clips, and seeds. Lag 0 is 1 by construction.

Usage: ``python results/fig_delta_error_acf.py [out_path]``
Default output: ``results/fig_delta_error_acf.pdf`` (and .png).
"""

import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent
SEEDS = ("s1", "s2", "s3")
CONFIGS = {"QLOT-final": "a4-v2-mean-only-wmr", "A1 (no hidden state)": "a1-v2-coordinates-only-carry"}
MAX_LAG = 30


def load(name: str, seed: str) -> dict:
    with open(RESULTS_DIR / f"results_{name}-{seed}.pkl", "rb") as f:
        return pickle.load(f)


def normalized_errors(d: dict) -> np.ndarray:
    """Face-size-normalized landmark error. Returns (clips, frames, lmk, 2)."""
    preds = d["wflw_v"]["preds"]  # (clips, 120, 98, 2)
    labels = d["wflw_v"]["labels"]  # (clips, 120, 98, 2)
    bmin = labels.min(axis=2)
    bmax = labels.max(axis=2)
    size = np.sqrt(np.prod(bmax - bmin, axis=-1))  # (clips, frames)
    return (preds - labels) / size[..., None, None]


def delta_error_acf(d: dict) -> np.ndarray:
    """Delta-error ACF for lags 0..MAX_LAG. Returns (MAX_LAG+1,).

    The delta error u_l = e_l - e_{l-1} is normalized to unit variance per clip
    and per channel (landmark x/y), so lag-0 equals 1 and the result is a true
    correlation. Averaged over channels, clips.
    """
    e = normalized_errors(d)
    u = np.diff(e, axis=1)  # (clips, frames-1, lmk, 2)
    clips, t = u.shape[0], u.shape[1]
    uf = u.reshape(clips, t, -1)  # (clips, t, lmk*2)
    sd = uf.std(axis=1, keepdims=True) + 1e-12
    un = uf / sd  # unit variance per clip/channel
    acf = np.empty(MAX_LAG + 1)
    for h in range(MAX_LAG + 1):
        acf[h] = float(np.mean(un[:, : t - h, :] * un[:, h:, :]))
    return acf


def main() -> None:
    out_base = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR / "fig_delta_error_acf"

    # Per-seed ACF series for each config.
    acf: dict[str, list[np.ndarray]] = {k: [] for k in CONFIGS}
    for label, name in CONFIGS.items():
        for seed in SEEDS:
            acf[label].append(delta_error_acf(load(name, seed)))

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    colors = {"QLOT-final": "tab:blue", "A1 (no hidden state)": "tab:red"}
    lags = np.arange(MAX_LAG + 1)

    for label in CONFIGS:
        mean_acf = np.mean(acf[label], axis=0)  # (MAX_LAG+1,), mean over seeds
        # spread across seeds for the band
        std_acf = np.std(acf[label], axis=0)
        c = colors[label]
        ax.plot(lags, mean_acf, color=c, label=label, linewidth=1.8, marker="o", markersize=3.5)
        ax.fill_between(lags, mean_acf - std_acf, mean_acf + std_acf, color=c, alpha=0.15)

    ax.axhline(0.0, color="gray", linewidth=0.8, zorder=0)
    ax.set_xlabel("Lag $h$ (frames)")
    ax.set_ylabel(r"$\gamma_u(h) \,/\, \gamma_u(0)$")
    ax.set_title("Delta-Error Autocorrelation (150 WFLW-V clips, 3 seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, MAX_LAG)

    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out_base.with_suffix('.pdf')} and .png")

    # Headline numbers for the caption / prose.
    for label in CONFIGS:
        mean_acf = np.mean(acf[label], axis=0)
        hmin = int(np.argmin(mean_acf[1:]) + 1)
        print(
            f"{label}: lag1={mean_acf[1]:+.3f}, lag2={mean_acf[2]:+.3f}, "
            f"min={mean_acf[hmin]:+.3f} at lag {hmin}"
        )


if __name__ == "__main__":
    main()
