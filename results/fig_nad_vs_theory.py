"""Generate the NAD-vs-theory figure for the supplement (sec-supp-nad).

Overlays the measured WFLW-V NAD curves of A1 (coordinates-only carry) and
QLOT-final (mean-only WMR) with the canonical error processes derived in the
supplement's NAD analysis, in face-size-normalized units (x10^3, matching the
NAD columns of tab:ablations):

- A1 vs. free random walk: NAD(m) = NAD(1) * sqrt((2m^2+1)/(3m)). The NAD(1)
  anchor is tuned (factor 1.07) for best log-space agreement over the rising
  branch. A1's rising slope (+0.42 over m in [1,4]) matches the finite-range
  free-walk benchmark (+0.43 over [1,10]). A bounded walk does NOT describe
  A1's rise: the best-fit AR(1) (tau_c = 8.5, anchored at the true NAD(1);
  grid fit, log-space RMSE over m in [1,60]) damps the rise to +0.27 and
  peaks too late (m_c = 16 vs. measured 10), even though it fits the full
  range better than the free walk (RMSE 0.037 vs 0.46) simply because A1's
  curve turns over. A1 integrates freely, then corrects late -- a delayed,
  not continuous, restoring force.
- QLOT-final vs. bounded walk: single-timescale AR(1) (eq-ar1) with
  tau_c = 11 frames (rho = exp(-1/tau_c)) and, additionally, a two-timescale
  mixture gamma(h) = sigma^2 [w e^{-h/tau_a} + (1-w) e^{-h/tau_b}] with
  tau_a = 4.5, tau_b = 20, w = 0.4. The single AR(1) matches the damped rise
  (+0.25 benchmark vs +0.22 measured) but decays too steeply past the peak
  (-0.22 vs -0.13 measured); the mixture reproduces rise, peak (m_c = 22) and
  the flat tail (-0.12), consistent with the cross-clip spread of correlation
  times reported in the supplement. Both QLOT-final theory curves are anchored
  at the measured NAD(1).

Error bands are the analytical per-clip relative error of El-Sheimy et al.
(2008), Eq. 26, scaled by sqrt(150 clips * 3 seeds) -- identical to
fig_hidden_state_carry.py.

Usage: ``python results/fig_nad_vs_theory.py [out_path]``
Default output: ``results/fig_nad_vs_theory.pdf`` (and .png).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, NullFormatter

from fig_hidden_state_carry import (
    SEEDS,
    load,
    nad_curve,
    nad_relative_errors,
)

RESULTS_DIR = Path(__file__).parent
CONFIGS = {"QLOT-final": "a4-v2-mean-only-wmr", "A1 (no hidden state)": "a1-v2-coordinates-only-carry"}

# Parameters stated in the supplement (sec-supp-nad, tab-nad-fits).
TAU_C = 11.0  # AR(1) error correlation time of QLOT-final, frames
RHO = float(np.exp(-1.0 / TAU_C))
# Two-timescale mixture fitted to QLOT-final's normalized NAD shape
# (grid fit, log-space RMSE over m in [1, 60]).
MIX_TAU_A, MIX_TAU_B, MIX_W = 4.5, 20.0, 0.4
# AR(1) bounded walk fitted to A1's NAD curve (true NAD(1) anchor, tau_c by
# grid fit, log-space RMSE over m in [1, 60]).
A1_TAU_C = 8.5
A1_RHO = float(np.exp(-1.0 / A1_TAU_C))
# NAD(1) anchor scale factors tuned for best log-space agreement with the
# measured curves over their rising branches. QLOT-final uses the true NAD(1).
ANCHOR_FACTOR = {"QLOT-final": 1.0, "A1 (no hidden state)": 1.07}
RISE_FIT_RANGE = {"QLOT-final": (1, 9), "A1 (no hidden state)": (1, 4)}  # empirical fit intervals
BENCH_RANGE = {"QLOT-final": (1, 22), "A1 (no hidden state)": (1, 10)}  # theory benchmark intervals
FALL_FIT_RANGE = {"QLOT-final": (30, 55), "A1 (no hidden state)": (25, 60)}


def nad_free_walk(m: np.ndarray, nad1: float) -> np.ndarray:
    """Exact free-random-walk NAD curve through NAD(1) (supplement, finite-range benchmark)."""
    return nad1 * np.sqrt((2.0 * m**2 + 1.0) / (3.0 * m))


def _lag_sum(m: np.ndarray, rho: float) -> np.ndarray:
    """Analytic sum_{h=1}^{m-1} (m-h) rho^h, valid for real m >= 1."""
    return m * rho * (1.0 - rho ** (m - 1.0)) / (1.0 - rho) - rho * (
        1.0 - m * rho ** (m - 1.0) + (m - 1.0) * rho**m
    ) / (1.0 - rho) ** 2


def nad_ar1(m: np.ndarray, rho: float, sigma_d2: float) -> np.ndarray:
    """Exact AR(1) bounded-walk NAD curve (supplement, eq-ar1)."""
    s2 = sigma_d2 / (1.0 - rho**2)
    V = s2 * (m + 2.0 * _lag_sum(m, rho))
    C = s2 * rho * (1.0 - rho**m) ** 2 / (1.0 - rho) ** 2
    return np.sqrt((V - C) / m**2)


def _nad2_from_acf(m: int, gamma: np.ndarray) -> float:
    """NAD^2(m) = (V(m) - C(m)) / m^2 from a lag array gamma(0..2m-1)."""
    i = np.arange(m)
    lag_v = np.abs(i[:, None] - i[None, :])
    lag_c = np.arange(m, 2 * m)[None, :] - i[:, None]
    return float((gamma[lag_v].sum() - gamma[lag_c].sum()) / m**2)


def nad_mixed(m: np.ndarray, tau_a: float, tau_b: float, w: float, sigma2: float) -> np.ndarray:
    """Bounded walk with two-timescale correlation gamma(h) = sigma2 [w e^{-h/tau_a} + (1-w) e^{-h/tau_b}]."""
    out = []
    for mi in m:
        m_i = max(1, int(round(mi)))
        h = np.arange(0, 2 * m_i)
        gamma = sigma2 * (w * np.exp(-h / tau_a) + (1.0 - w) * np.exp(-h / tau_b))
        out.append(_nad2_from_acf(m_i, gamma))
    return np.sqrt(np.array(out))


def fit_slope(m_vals: np.ndarray, curve: np.ndarray, m_lo: int, m_hi: int) -> float:
    """Least-squares slope of log10(NAD) vs log10(m) over m in [m_lo, m_hi]."""
    sel = (m_vals >= m_lo) & (m_vals <= m_hi)
    return float(np.polyfit(np.log10(m_vals[sel]), np.log10(curve[sel]), 1)[0])


def bench_slope(theory_fn, m_lo: int, m_hi: int) -> float:
    """Finite-range benchmark slope of a theory curve, on the integer NAD grid."""
    m_int = np.arange(m_lo, m_hi + 1, dtype=float)
    return float(np.polyfit(np.log10(m_int), np.log10(theory_fn(m_int)), 1)[0])


def main() -> None:
    out_base = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR / "fig_nad_vs_theory"

    nad: dict[str, list[np.ndarray]] = {k: [] for k in CONFIGS}
    for label, name in CONFIGS.items():
        for seed in SEEDS:
            nad[label].append(nad_curve(load(name, seed)))

    m_vals = np.arange(1, 61)
    m_dense = np.logspace(0, np.log10(60), 400)
    # Face-size-normalized units, scaled by 10^3 (as in tab:ablations).
    mean_nads = {label: np.mean(nad[label], axis=0) * 1e3 for label in CONFIGS}  # (60,)

    # Theory curves, each anchored at the corresponding (tuned) NAD(1).
    anchor1 = {label: float(mean_nads[label][0]) * ANCHOR_FACTOR[label] for label in CONFIGS}
    # NAD(1)^2 = gamma(0) - gamma(1) determines the mixture level sigma2.
    rho_a, rho_b = np.exp(-1.0 / MIX_TAU_A), np.exp(-1.0 / MIX_TAU_B)
    sigma2_mix = anchor1["QLOT-final"] ** 2 / (MIX_W * (1.0 - rho_a) + (1.0 - MIX_W) * (1.0 - rho_b))
    theory_fn = {
        "A1 (no hidden state)": lambda m: nad_free_walk(m, anchor1["A1 (no hidden state)"]),
        "A1 (ar1)": lambda m: nad_ar1(m, A1_RHO, float(mean_nads["A1 (no hidden state)"][0]) ** 2 * (1.0 + A1_RHO)),
        "QLOT-final": lambda m: nad_ar1(m, RHO, anchor1["QLOT-final"] ** 2 * (1.0 + RHO)),
        "QLOT-final (mix)": lambda m: nad_mixed(m, MIX_TAU_A, MIX_TAU_B, MIX_W, sigma2_mix),
    }
    theory = {label: theory_fn[label](m_dense) for label in theory_fn}

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    colors = {"QLOT-final": "tab:blue", "A1 (no hidden state)": "tab:red"}
    rel_err = nad_relative_errors(m_vals=len(m_vals), total_vals=120, num_clips=150, num_seeds=len(SEEDS))

    for label in CONFIGS:
        c = colors[label]
        curve = mean_nads[label]
        ax.loglog(m_vals, curve, color=c, label=label, linewidth=1.8)
        ax.fill_between(
            m_vals, curve * (1.0 - rel_err), curve * (1.0 + rel_err), color=c, alpha=0.15
        )
        # Mark the measured maximum.
        i_max = int(np.argmax(curve))
        ax.plot(
            m_vals[i_max], curve[i_max], marker="o", markersize=6, color=c,
            markerfacecolor="white", markeredgewidth=1.5, zorder=5,
        )
        ax.annotate(
            f"$m_c={m_vals[i_max]}$",
            xy=(m_vals[i_max], curve[i_max]),
            xytext=(0, 1.25), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, color=c, fontweight="bold",
        )

    theory_styles = {
        "A1 (no hidden state)": ("--", 0.75, r"free walk: $\mathrm{NAD}(1)\sqrt{(2m^2+1)/3m}$"),
        "A1 (ar1)": (":", 0.9, rf"AR(1) bounded walk fit, $\tau_c={A1_TAU_C:g}$"),
        "QLOT-final": ("--", 0.75, rf"AR(1) bounded walk, $\tau_c={TAU_C:.0f}$"),
        "QLOT-final (mix)": (":", 0.9, "two-AR(1) mix: " rf"$0.4\,e^{{-h/{MIX_TAU_A:g}}}+0.6\,e^{{-h/{MIX_TAU_B:g}}}$"),
    }
    for label, (ls, alpha, leg) in theory_styles.items():
        c = colors[{"A1 (ar1)": "A1 (no hidden state)"}.get(label, label.replace(" (mix)", ""))]
        ax.loglog(m_dense, theory[label], color=c, linestyle=ls, linewidth=1.4, alpha=alpha, label=leg)

    # Slope annotations: empirical fit vs. theory benchmark (supplement, tab-nad-fits).
    for label in CONFIGS:
        m_lo, m_hi = RISE_FIT_RANGE[label]
        b_lo, b_hi = BENCH_RANGE[label]
        fit = fit_slope(m_vals, mean_nads[label], m_lo, m_hi)
        bench = bench_slope(theory_fn[label], b_lo, b_hi)
        m_mid = float(np.sqrt(m_lo * m_hi))
        y_mid = float(np.interp(np.log10(m_mid), np.log10(m_vals), np.log10(mean_nads[label])))
        ax.annotate(
            f"$+{fit:.2f}$ fit\n$+{bench:.2f}$ {'free walk' if label.startswith('A1') else 'AR(1)'}",
            xy=(m_mid, 10**y_mid), xytext=(0, -13), textcoords="offset points",
            fontsize=8, color=colors[label], ha="center", va="top",
        )
        f_lo, f_hi = FALL_FIT_RANGE[label]
        fall = fit_slope(m_vals, mean_nads[label], f_lo, f_hi)
        print(
            f"{label}: rise fit [{m_lo},{m_hi}] = {fit:+.3f}, theory benchmark [{b_lo},{b_hi}] = {bench:+.3f}, "
            f"fall fit [{f_lo},{f_hi}] = {fall:+.3f}"
        )

    # Mixture summary for QLOT-final.
    mix_rise = bench_slope(theory_fn["QLOT-final (mix)"], *RISE_FIT_RANGE["QLOT-final"])
    mix_fall = bench_slope(theory_fn["QLOT-final (mix)"], *FALL_FIT_RANGE["QLOT-final"])
    mix_mc = int(np.arange(1, 61)[np.argmax(theory_fn["QLOT-final (mix)"](np.arange(1, 61, dtype=float)))])
    print(f"QLOT-final mix: rise [{RISE_FIT_RANGE['QLOT-final'][0]},{RISE_FIT_RANGE['QLOT-final'][1]}] = {mix_rise:+.3f}, "
          f"fall [{FALL_FIT_RANGE['QLOT-final'][0]},{FALL_FIT_RANGE['QLOT-final'][1]}] = {mix_fall:+.3f}, m_c = {mix_mc}")

    # Fitted AR(1) for A1: report the rise damping mismatch (supplement: the
    # continuous restoring force is not supported by A1's data).
    a1_ar1 = theory_fn["A1 (ar1)"](m_vals.astype(float))
    a1_meas = mean_nads["A1 (no hidden state)"]
    rmse_ar1 = float(np.sqrt(np.mean((np.log10(a1_ar1) - np.log10(a1_meas)) ** 2)))
    rmse_fw = float(np.sqrt(np.mean((np.log10(theory_fn["A1 (no hidden state)"](m_vals.astype(float))) - np.log10(a1_meas)) ** 2)))
    print(f"A1 AR(1) fit (tau_c={A1_TAU_C}): rise [1,4] = {bench_slope(theory_fn['A1 (ar1)'], 1, 4):+.3f} "
          f"(measured +0.42), fall [25,60] = {bench_slope(theory_fn['A1 (ar1)'], 25, 60):+.3f}, "
          f"m_c = {int(m_vals[np.argmax(a1_ar1)])} (measured 10), full-range log-RMSE = {rmse_ar1:.4f} "
          f"(free walk {rmse_fw:.4f})")

    ax.set_xlabel("Averaging Time $m$ (frames)")
    ax.set_ylabel("NAD($m$) $\\times 10^3$")
    ax.set_title("NAD vs. Canonical Error Processes (150 WFLW-V clips, 3 seeds)")
    ax.legend(framealpha=0.5, fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax.set_xticks([1, 3, 6, 10, 30, 60])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax.set_yticks([3, 5, 10, 20, 30])
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    # ax.yaxis.set_minor_formatter(FormatStrFormatter("%g"))

    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {out_base.with_suffix('.pdf')} and .png")

    for label in CONFIGS:
        print(f"{label}: NAD(1)={mean_nads[label][0]:.3f}e-3 (anchor {anchor1[label]:.3f}e-3), "
              f"measured m_c={m_vals[int(np.argmax(mean_nads[label]))]}")


if __name__ == "__main__":
    main()
