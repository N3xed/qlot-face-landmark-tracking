"""Ablation test-result analysis for the QLOT ablation study (A0-A7).

WHAT THIS SCRIPT IS FOR
-----------------------
Recomputes every quantitative result of the ablation study directly from the raw
per-run pickles (``results_<config>-{s1,s2,s3}.pkl``, produced by the eval
notebooks) and prints both a full statistical summary and ready-to-paste,
paper-facing tables. It is the single source of truth for the numbers that appear
in the paper's tables:

  * ``supplement.typ`` -> ``tab:ablations`` (the 8-config x 9-metric table),
  * ``supplement.typ`` -> ``tab:checkpoint-selection`` (A1 vs A1-nme).

Companion script: ``results/verify_claims.py`` covers the *prose* numeric claims
(main.typ Results + supplement body text) using the same metric math; this script
covers the *tables*. Keep the two in sync.

INPUTS
------
``results_<config>-<seed>.pkl`` for the 8 configs in ``CONFIGS`` x 3 seeds
(s1, s2, s3), each containing per-image NME arrays for the static sets and, for
WFLW-V, raw ``preds`` / ``labels`` (clips x frames x landmarks x 2), per-frame
``nmes_size``, and the ``is_hard`` clip flag. The ``A1-nme`` config holds the same
three A1 training runs but with checkpoints re-selected purely by validation NME
(no stability-aware secondary criterion); comparing it to the stability-aware A1
checkpoints substantiates the paper's "checkpoint selection matters" claim.

METRIC DEFINITIONS / HOW VALUES ARE COMPUTED
--------------------------------------------
Normalization conventions (user-specified; differ from the earlier scripts):
  * WFLW and 300-W/IBUG static NME: inter-ocular (iod) normalization.
  * FaceSynthetics static NME: face-size normalization sqrt(w*h) of the
    ground-truth landmark bounding box ("size").
  * WFLW-V metrics (V-NME, NMF, NAD): face-size normalization.

Per-frame / per-clip math:
  * Static NME (%): per-image mean over landmarks, x100.
  * V-NME (%): per-frame ``nmes_size`` averaged over frames -> per-clip scalar.
  * NMF: RMS over landmarks, then over frames, of the frame-to-frame error
    increment normalized by face size (relative to a 256x256 standard area,
    x100), following ``calc_nmf`` in ``src/utils/torch/misc.py``. Per clip;
    easy/hard splits use the official ``is_hard`` flag (75 easy / 75 hard clips).
  * NAD(m): position-based normalized Allan deviation at block size m, following
    ``navar_pos_all`` semantics (overlapping block averages of the size-normalized
    error, successive-block differences, /2), RMS over frames and landmarks. Per
    clip; reported at m in {1, 4, 16} and scaled by 1e3 in the tables.

Aggregation (must match the paper's metric definitions): NMF and NAD are RMS over
landmarks and frames, then **mean over clips**. Config-level values are the mean
over the 3 seeds (+/- sample std, ddof=1).

STATISTICS
----------
Significance convention (matches ``bootstrap_ablations.py``): paired bootstrap,
``N_BOOT`` = 10,000 resamples over videos (temporal metrics) or images (static
NME), computed within seed and pooled across the 3 seeds. A delta is marked
significant ("*" / dagger) iff the 95% percentile CI excludes zero. The RNG is
seeded (123) so results are reproducible up to bootstrap noise; note that very
borderline CIs (e.g. A6 on 300-W) can flip significance across RNG draws.

OUTPUT (stdout, six numbered sections)
--------------------------------------
  1. Config-level metrics, mean +/- std over seeds, for all 8 metrics.
  2. Per-config delta vs A0 with pooled 95% bootstrap CIs, significance stars, and
     per-seed sign pattern.
  3. Checkpoint-selection: A1 (stability-aware) vs A1-nme (NME-only), paired, plus
     the per-frame V-NME gap trajectory and frame-0 indistinguishability check.
  4. Paper table ``tab:ablations`` (WFLW/300-W/Synth/V-NME in %, NMF easy/hard,
     NAD(m) x 1e3; dagger = significant vs A0).
  5. Paper table ``tab:checkpoint-selection`` (dagger = significant between the
     two selection rules).
  6. Prose spot values used by ``verify_claims.py`` (overall/easy/hard NMF per
     config and the key percent deltas).

Usage: ``python results/ablation_analysis.py``
"""

import pickle
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent
SEEDS = ("s1", "s2", "s3")
N_BOOT = 10_000
RNG = np.random.default_rng(123)

CONFIGS = {
    "A0": "baseline2",
    "A1": "a1-v2-coordinates-only-carry",
    "A1-nme": "a1-v2-coordinates-only-carry-nme",
    "A2": "a2-v2-per-landmark-gru",
    "A3": "a3-v2-gru-cell",
    "A4": "a4-v2-mean-only-wmr",
    "A5": "a5-v2-l2-coordinate-only",
    "A6": "a6-v2-spatial-gnll",
    "A7": "a7-v2-geometric-fourier",
}


def load(config_key: str, seed: str) -> dict:
    with open(RESULTS_DIR / f"results_{CONFIGS[config_key]}-{seed}.pkl", "rb") as f:
        return pickle.load(f)


RUNS = {k: {s: load(k, s) for s in SEEDS} for k in CONFIGS}
BASE = RUNS["A0"]

# ==============================================================================
# Metric helpers
# ==============================================================================


def per_clip_nmf(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """NMF per clip, following calc_nmf in src/utils/torch/misc.py."""
    err = preds - labels  # (C, T, L, 2)
    bbox = labels.max(axis=2) - labels.min(axis=2)  # (C, T, 2)
    d_s = np.sqrt(bbox.prod(axis=-1) / (256 * 256))[:, 1:]  # (C, T-1)
    err_norm = np.sqrt(np.square(np.diff(err, axis=1)).sum(axis=-1))  # (C, T-1, L)
    nmf_n = np.sqrt(np.square(100.0 * err_norm / d_s[..., None]).mean(axis=-1))
    return np.sqrt(np.square(nmf_n).mean(axis=-1))  # (C,)


def per_clip_nad(preds: np.ndarray, labels: np.ndarray, m: int) -> np.ndarray:
    """Position NAD per clip (pooled over landmarks), navar_pos_all semantics."""
    bbox = labels.max(axis=2) - labels.min(axis=2)
    size = np.sqrt(bbox.prod(axis=-1))
    errors = (preds - labels) / size[..., None, None]
    cs = np.concatenate(
        [np.zeros((errors.shape[0], 1, *errors.shape[2:])), np.cumsum(errors, axis=1)], axis=1
    )
    blocks = (cs[:, m:] - cs[:, :-m]) / m
    diffs = blocks[:, m:] - blocks[:, :-m]
    return np.sqrt(np.square(diffs).sum(axis=-1).mean(axis=1).mean(axis=-1) / 2)  # (C,)


# --- per-image / per-clip metric extractors ---------------------------------
# Normalization convention (this re-analysis):
#   WFLW:   nme_iod   (inter-ocular)
#   300-W:  nme_iod   (inter-ocular)
#   Synth:  nme_s     (face size)
#   WFLW-V: nmes_size (face size)


def wflw_nme(d: dict) -> np.ndarray:
    return d["wflw"]["full"]["nme_iod"].mean(-1) * 100.0


def ibug_full_nme(d: dict) -> np.ndarray:
    return (
        np.concatenate(
            [d["ibug"]["common"]["nme_iod"].mean(-1), d["ibug"]["challenging"]["nme_iod"].mean(-1)]
        )
        * 100.0
    )


def synth_nme(d: dict) -> np.ndarray:
    return d["face_synth"]["test"]["nme_s"].mean(-1) * 100.0


def clip_nme(v: dict) -> np.ndarray:
    return v["nmes_size"].mean(axis=1) * 100.0


def nmf_of(v: dict) -> np.ndarray:
    return per_clip_nmf(v["preds"], v["labels"])


def _hard_mask(v: dict) -> np.ndarray:
    return v["is_hard"].astype(bool)


def nmf_easy_of(v: dict) -> np.ndarray:
    """Per-clip NMF restricted to the easy subset (75 easy clips)."""
    return per_clip_nmf(v["preds"], v["labels"])[~_hard_mask(v)]


def nmf_hard_of(v: dict) -> np.ndarray:
    """Per-clip NMF restricted to the hard subset (75 hard clips)."""
    return per_clip_nmf(v["preds"], v["labels"])[_hard_mask(v)]


def nad_of(v: dict, m: int) -> np.ndarray:
    return per_clip_nad(v["preds"], v["labels"], m)


def boot_ci(diffs: np.ndarray) -> tuple[float, float, float]:
    n = len(diffs)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    b = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(diffs.mean()), float(lo), float(hi)


def is_sig(ci: tuple[float, float, float]) -> bool:
    _, lo, hi = ci
    return lo > 0 or hi < 0


def pooled_video_delta(ab_key: str, metric_fn) -> tuple[float, float, float]:
    diffs = []
    for s in SEEDS:
        diffs.append(metric_fn(RUNS[ab_key][s]["wflw_v"]) - metric_fn(BASE[s]["wflw_v"]))
    return boot_ci(np.concatenate(diffs))


def pooled_image_delta(ab_key: str, metric_fn) -> tuple[float, float, float]:
    diffs = []
    for s in SEEDS:
        diffs.append(metric_fn(RUNS[ab_key][s]) - metric_fn(BASE[s]))
    return boot_ci(np.concatenate(diffs))


def pooled_pair_delta(key_a: str, key_b: str, metric_fn, video: bool) -> tuple[float, float, float]:
    """Paired delta between two arbitrary configs (same seeds), pooled."""
    diffs = []
    for s in SEEDS:
        if video:
            diffs.append(metric_fn(RUNS[key_a][s]["wflw_v"]) - metric_fn(RUNS[key_b][s]["wflw_v"]))
        else:
            diffs.append(metric_fn(RUNS[key_a][s]) - metric_fn(RUNS[key_b][s]))
    return boot_ci(np.concatenate(diffs))


def mean_over_seeds(key: str, fn, video: bool = False) -> float:
    vals = []
    for s in SEEDS:
        d = RUNS[key][s]["wflw_v"] if video else RUNS[key][s]
        vals.append(fn(d).mean() if isinstance(fn(d), np.ndarray) else fn(d))
    return float(np.mean(vals))


# ==============================================================================
# 1. Config-level summary table (mean ± std over seeds)
# ==============================================================================

print("=" * 130)
print("1. CONFIG-LEVEL METRICS (mean ± std over 3 seeds)")
print("   Static NME: WFLW/300-W = inter-ocular; FaceSynthetics = face-size; WFLW-V metrics = face-size")
print("=" * 130)

summary_metrics = [
    ("WFLW", lambda d: wflw_nme(d), False, "{:.3f}"),
    ("300W", lambda d: ibug_full_nme(d), False, "{:.3f}"),
    ("Synth", lambda d: synth_nme(d), False, "{:.3f}"),
    ("V-NME", lambda v: clip_nme(v), True, "{:.3f}"),
    ("NMF", lambda v: nmf_of(v), True, "{:.2f}"),
    ("NAD1", lambda v: nad_of(v, 1), True, "{:.5f}"),
    ("NAD4", lambda v: nad_of(v, 4), True, "{:.5f}"),
    ("NAD16", lambda v: nad_of(v, 16), True, "{:.5f}"),
]

header = f"{'config':<10}" + "".join(f"{name:>18}" for name, _, _, _ in summary_metrics)
print(header)
print("-" * len(header))
config_summary: dict[str, dict[str, float]] = {}
for key in CONFIGS:
    row = f"{key:<10}"
    config_summary[key] = {}
    for name, fn, video, _fmt in summary_metrics:
        vals = []
        for s in SEEDS:
            d = RUNS[key][s]["wflw_v"] if video else RUNS[key][s]
            vals.append(float(fn(d).mean()))
        vals = np.array(vals)
        config_summary[key][name] = float(vals.mean())
        row += f"{vals.mean():>10.4f}±{vals.std(ddof=1):<7.4f}"
    print(row)

# selected checkpoint steps (to show stability-aware vs NME-only selection)
print("\nSelected checkpoint steps:")
for key in CONFIGS:
    steps = [RUNS[key][s]["step"] for s in SEEDS]
    print(f"  {key:<10} steps={steps}")

# ==============================================================================
# 2. Deltas vs A0 with pooled paired-bootstrap CIs
# ==============================================================================

print()
print("=" * 130)
print("2. DELTA vs A0 (pooled within-seed paired bootstrap, 10k resamples; * = 95% CI excludes 0)")
print("=" * 130)

delta_metrics = [
    ("WFLW", wflw_nme, False),
    ("300W", ibug_full_nme, False),
    ("Synth", synth_nme, False),
    ("V-NME", clip_nme, True),
    ("NMF", nmf_of, True),
    ("NAD1", lambda v: nad_of(v, 1), True),
    ("NAD4", lambda v: nad_of(v, 4), True),
    ("NAD16", lambda v: nad_of(v, 16), True),
]

header = f"{'abl':<6}{'metric':<8}{'delta':>10}{'%':>9}   {'95% CI':<28}{'sig':<4}{'per-seed signs'}"
print(header)
print("-" * 110)
delta_results: dict[tuple[str, str], tuple] = {}
for key in ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]:
    for name, fn, video in delta_metrics:
        if video:
            ci = pooled_video_delta(key, fn)
            signs = [
                np.sign(fn(RUNS[key][s]["wflw_v"]).mean() - fn(BASE[s]["wflw_v"]).mean())
                for s in SEEDS
            ]
        else:
            ci = pooled_image_delta(key, fn)
            signs = [np.sign(fn(RUNS[key][s]).mean() - fn(BASE[s]).mean()) for s in SEEDS]
        rel = ci[0] / config_summary["A0"][name] * 100.0 # type: ignore
        delta_results[(key, name)] = (ci, rel, signs)
        sig = "*" if is_sig(ci) else " "
        sign_str = "/".join("+" if x > 0 else "-" if x < 0 else "0" for x in signs)
        print(
            f"{key:<6}{name:<8}{ci[0]:>+10.4f}{rel:>+8.1f}%   "
            f"[{ci[1]:>+.5f}, {ci[2]:>+.5f}]{sig:<4}{sign_str}"
        )
    print()

# ==============================================================================
# 3. Checkpoint-selection ablation: A1 (stability-aware) vs A1-nme (NME-only)
# ==============================================================================

print()
print("=" * 130)
print("3. CHECKPOINT SELECTION: A1 stability-aware vs A1-nme (NME-only selection), paired")
print("=" * 130)

sel_metrics = [
    ("WFLW", wflw_nme, False),
    ("300W", ibug_full_nme, False),
    ("V-NME", clip_nme, True),
    ("NMF", nmf_of, True),
    ("NAD1", lambda v: nad_of(v, 1), True),
    ("NAD4", lambda v: nad_of(v, 4), True),
    ("NAD16", lambda v: nad_of(v, 16), True),
]
print(f"{'metric':<8}{'A1':>10}{'A1-nme':>10}{'delta':>10}   {'95% CI':<28}{'sig':<4}")
print("-" * 80)
for name, fn, video in sel_metrics:
    ci = pooled_pair_delta("A1", "A1-nme", fn, video)
    a = config_summary["A1"][name]
    b = mean_over_seeds("A1-nme", fn, video)
    sig = "*" if is_sig(ci) else " "
    print(f"{name:<8}{a:>10.4f}{b:>10.4f}{ci[0]:>+10.4f}   [{ci[1]:>+.5f}, {ci[2]:>+.5f}]{sig:<4}")

# Per-frame NME gap A1 vs A0, and A1-nme vs A0 (does the instability show?)
print("\nPer-frame V-NME gap vs A0 (mean over seeds), selected frames:")
print(f"{'frame':>6}{'A1-A0':>10}{'A1nme-A0':>10}")
for f in (0, 1, 2, 4, 8, 30, 60, 119):
    g1 = np.mean(
        [
            RUNS["A1"][s]["wflw_v"]["nmes_size"][:, f].mean() - BASE[s]["wflw_v"]["nmes_size"][:, f].mean()
            for s in SEEDS
        ]
    ) * 100
    g2 = np.mean(
        [
            RUNS["A1-nme"][s]["wflw_v"]["nmes_size"][:, f].mean()
            - BASE[s]["wflw_v"]["nmes_size"][:, f].mean()
            for s in SEEDS
        ]
    ) * 100
    print(f"{f:>6}{g1:>+10.4f}{g2:>+10.4f}")

# frame-0 paired diffs (both vs A0) should be n.s.
for key in ("A1", "A1-nme"):
    d = np.concatenate(
        [
            RUNS[key][s]["wflw_v"]["nmes_size"][:, 0] * 100
            - BASE[s]["wflw_v"]["nmes_size"][:, 0] * 100
            for s in SEEDS
        ]
    )
    ci = boot_ci(d)
    print(f"frame-0 {key} vs A0: diff={ci[0]:+.4f} [{ci[1]:+.5f}, {ci[2]:+.5f}] sig={is_sig(ci)}")

# Does A1-nme look fine on NME (i.e. instability invisible under NME-primary lens)?
print("\nNMF of A1-nme vs A0 (instability still present in the NME-selected checkpoints?):")
ci = pooled_video_delta("A1-nme", nmf_of)
print(f"  A1-nme - A0 NMF: {ci[0]:+.3f} [{ci[1]:+.3f}, {ci[2]:+.3f}] sig={is_sig(ci)}")
ci = pooled_video_delta("A1-nme", clip_nme)
print(f"  A1-nme - A0 V-NME: {ci[0]:+.4f} [{ci[1]:+.4f}, {ci[2]:+.4f}] sig={is_sig(ci)}")

# ==============================================================================
# 4. Paper-facing tables (supplement.typ tab:ablations / tab:checkpoint-selection)
#    Numbers are mean over 3 seeds; dagger = significant delta vs the reference
#    (A0 for the main table, A1 for the checkpoint-selection table).
# ==============================================================================

print()
print("=" * 130)
print("4. PAPER TABLE: ablations (supplement tab:ablations)")
print("   WFLW/300W/Synth/V-NME in %, NMF easy/hard, NAD(m) x 1e3; dagger = sig vs A0")
print("=" * 130)

paper_metrics = [
    ("WFLW", lambda d: wflw_nme(d), False, "{:.3f}"),
    ("300W", lambda d: ibug_full_nme(d), False, "{:.3f}"),
    ("Synth", lambda d: synth_nme(d), False, "{:.3f}"),
    ("V-NME", lambda v: clip_nme(v), True, "{:.3f}"),
    ("NMF_e", lambda v: nmf_easy_of(v), True, "{:.2f}"),
    ("NMF_h", lambda v: nmf_hard_of(v), True, "{:.2f}"),
    ("NAD1", lambda v: nad_of(v, 1), True, "x1e3:{:.2f}"),
    ("NAD4", lambda v: nad_of(v, 4), True, "x1e3:{:.2f}"),
    ("NAD16", lambda v: nad_of(v, 16), True, "x1e3:{:.2f}"),
]


def paper_cell(key: str, name: str, fn, video: bool, fmt: str) -> str:
    """Mean-over-seeds value, formatted; dagger appended if delta vs A0 is sig."""
    vals = []
    for s in SEEDS:
        d = RUNS[key][s]["wflw_v"] if video else RUNS[key][s]
        vals.append(float(fn(d).mean()))
    mean = float(np.mean(vals))
    scale = 1e3 if fmt.startswith("x1e3") else 1.0
    val_str = f"{mean * scale:.2f}" if scale != 1.0 else fmt.format(mean)
    if key == "A0":
        return val_str
    if video:
        ci = pooled_video_delta(key, fn)
    else:
        ci = pooled_image_delta(key, fn)
    return val_str + ("*" if is_sig(ci) else " ")


col_keys = ["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"]
hdr = f"{'Config':<22}" + "".join(f"{n:>9}" for n, _, _, _ in paper_metrics)
print(hdr)
print("-" * len(hdr))
for key in col_keys:
    row = f"{key:<22}"
    for name, fn, video, fmt in paper_metrics:
        row += f"{paper_cell(key, name, fn, video, fmt):>9}"
    print(row)
print("(dagger = paired-bootstrap 95% CI excludes 0; best temporal result per column in A0/A4 rows)")

# ---- Checkpoint-selection table (A1 vs A1-nme), supplement tab:checkpoint-selection
print()
print("=" * 130)
print("5. PAPER TABLE: checkpoint selection (supplement tab:checkpoint-selection)")
print("   dagger = significant paired delta between the two selection rules")
print("=" * 130)

sel_paper_metrics = [
    ("WFLW", lambda d: wflw_nme(d), False, "{:.3f}"),
    ("V-NME", lambda v: clip_nme(v), True, "{:.3f}"),
    ("NMF_e", lambda v: nmf_easy_of(v), True, "{:.2f}"),
    ("NMF_h", lambda v: nmf_hard_of(v), True, "{:.2f}"),
    ("NAD4", lambda v: nad_of(v, 4), True, "x1e3:{:.2f}"),
    ("NAD16", lambda v: nad_of(v, 16), True, "x1e3:{:.2f}"),
]
print(f"{'A1 checkpoints':<22}" + "".join(f"{n:>9}" for n, _, _, _ in sel_paper_metrics))
print("-" * 90)
for key, label in (("A1", "stability-aware (A1)"), ("A1-nme", "NME-only (A1-nme)")):
    row = f"{label:<22}"
    for name, fn, video, fmt in sel_paper_metrics:
        vals = []
        for s in SEEDS:
            d = RUNS[key][s]["wflw_v"] if video else RUNS[key][s]
            vals.append(float(fn(d).mean()))
        mean = float(np.mean(vals))
        scale = 1e3 if fmt.startswith("x1e3") else 1.0
        val_str = f"{mean * scale:.2f}" if scale != 1.0 else fmt.format(mean)
        dagger = ""
        if key == "A1-nme":
            ci = pooled_pair_delta("A1", "A1-nme", fn, video)
            dagger = "*" if is_sig(ci) else " "
        row += f"{val_str + dagger:>9}"
    print(row)

# ---- Text-claim spot values used in prose (main.typ / supplement.typ)
print()
print("=" * 130)
print("6. PROSE SPOT VALUES (for text claims)")
print("=" * 130)


def agg(key: str, fn, video: bool = True) -> float:
    return mean_over_seeds(key, fn, video)


def rel(key: str, name: str, fn, video: bool = True) -> float:
    return (agg(key, fn, video) - agg("A0", fn, video)) / agg("A0", fn, video) * 100.0


print(f"A0   NMF overall/easy/hard: {agg('A0', nmf_of):.2f} / {agg('A0', nmf_easy_of):.2f} / {agg('A0', nmf_hard_of):.2f}")
print(f"A1   NMF overall/easy/hard: {agg('A1', nmf_of):.2f} / {agg('A1', nmf_easy_of):.2f} / {agg('A1', nmf_hard_of):.2f}")
print(f"A1-nme NMF overall/easy/hard: {agg('A1-nme', nmf_of):.2f} / {agg('A1-nme', nmf_easy_of):.2f} / {agg('A1-nme', nmf_hard_of):.2f}")
print(f"A4   NMF overall/easy/hard: {agg('A4', nmf_of):.2f} / {agg('A4', nmf_easy_of):.2f} / {agg('A4', nmf_hard_of):.2f}")
print(f"A5   NMF overall/easy/hard: {agg('A5', nmf_of):.2f} / {agg('A5', nmf_easy_of):.2f} / {agg('A5', nmf_hard_of):.2f}")
print(f"A6   NMF overall/easy/hard: {agg('A6', nmf_of):.2f} / {agg('A6', nmf_easy_of):.2f} / {agg('A6', nmf_hard_of):.2f}")
print()
for key in ("A1", "A2", "A4", "A5", "A6", "A7"):
    print(
        f"{key}: NMF {rel(key, 'NMF', nmf_of):+.1f}%  "
        f"V-NME {rel(key, 'V-NME', clip_nme):+.1f}%  "
        f"WFLW {rel(key, 'WFLW', wflw_nme, video=False):+.1f}%  "
        f"NAD1 {rel(key, 'NAD1', lambda v: nad_of(v, 1)):+.1f}%  "
        f"NAD4 {rel(key, 'NAD4', lambda v: nad_of(v, 4)):+.1f}%  "
        f"NAD16 {rel(key, 'NAD16', lambda v: nad_of(v, 16)):+.1f}%"
    )
# A1-nme vs A1 relative (checkpoint selection), easy/hard/overall NMF
for lbl, fn in (("overall", nmf_of), ("easy", nmf_easy_of), ("hard", nmf_hard_of)):
    r = (agg("A1-nme", fn) - agg("A1", fn)) / agg("A1", fn) * 100.0
    print(f"A1-nme vs A1 NMF ({lbl}): {r:+.2f}%  ({agg('A1', fn):.2f} -> {agg('A1-nme', fn):.2f})")
r = (agg("A1-nme", clip_nme) - agg("A1", clip_nme)) / agg("A1", clip_nme) * 100.0
print(f"A1-nme vs A1 V-NME: {r:+.2f}%  ({agg('A1', clip_nme):.3f} -> {agg('A1-nme', clip_nme):.3f})")
