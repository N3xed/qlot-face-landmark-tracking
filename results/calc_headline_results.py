"""Headline results for QLOT-final (= ablation run A4: ``a4-v2-mean-only-wmr``, 3 seeds).

WHAT THIS SCRIPT IS FOR
-----------------------
Computes the paper-facing headline numbers for the final model from the raw
evaluation pickles produced by the eval notebooks
(``eval_a4-v2-mean-only-wmr-{s1,s2,s3}.ipynb``), aggregated as mean over seeds
(matching ``ablation_analysis.py`` and the paper's ``tab:sota``):

  * 300-W NMEs (%, inter-ocular / IOD normalized) for all test splits:
    common, challenging, full (common+challenging), indoor, outdoor, and the
    combined 300-W private test set (indoor+outdoor).
  * WFLW NMEs (%, IOD normalized) for all test splits:
    full test set plus the six attribute subsets (large-pose, expression,
    illumination, make-up, occlusion, blur).
  * FaceSynthetics NME (%, face-size normalized) for our test split.
  * WFLW-V video metrics (all face-size normalized):
      - V-NME (%; combined and split into easy / hard videos),
      - NMF (easy videos, hard videos, and combined),
      - E_temporal (combined videos; see below).

QLOT-final is the locked reference/headline configuration (A4, mean-only WMR).
NOTE: ``main.typ``'s ``tab:sota`` row (300-W 3.49 / WFLW 3.93 / V-NME 1.07 /
NMF 108.1) reports the A4 *mean over the 3 seeds* -- exactly what this script
computes by default. A single run (e.g. the deployed A4s2 ONNX export,
``model-a4s2-*.onnx``) differs slightly (3.52 / 3.93 / 1.06 / 108.3); pass a
single ``--pkl`` for that.

INPUTS
------
One or more ``results_<name>.pkl`` files (default: the three A4 seeds), each
with per-image results for the static sets (``wflw`` / ``ibug`` /
``face_synth``: per-split ``nme_iod`` and ``nme_s`` per-landmark arrays) and,
for WFLW-V (``wflw_v``), raw ``preds`` / ``labels`` (clips x frames x
landmarks x 2), per-frame ``nmes_size``, and the official ``is_hard`` clip
flag (75 easy / 75 hard clips).

AGGREGATION
-----------
Every metric is computed per run (mean over its images / clips) and then
averaged over runs (mean +/- sample std, ddof=1), matching
``ablation_analysis.py``. For a single run only the plain value is shown.
(With equal numbers of equal-length clips/images per seed, the seed mean
coincides with a pooled mean over all runs -- verified for V-NME.)

METRIC DEFINITIONS
------------------
  * Static NME (%): per-image mean over landmarks of the stored normalized
    error (``nme_iod`` = inter-ocular, ``nme_s`` = face size sqrt(w*h) of the
    GT landmark bounding box), averaged over images, x100. Combined splits are
    pooled per image (image-count weighted).
  * V-NME (%): per-frame ``nmes_size`` averaged over frames -> per-clip scalar,
    then mean over clips (all clips, or the easy/hard subsets given by the
    official ``is_hard`` flag), x100.
  * NMF: RMS over landmarks, then over frames, of the frame-to-frame error
    increment normalized by face size (relative to a 256x256 standard area,
    x100), following ``calc_nmf`` in ``src/utils/torch/misc.py``. Per clip;
    reported as the mean over clips (paper convention: RMS over landmarks and
    frames, mean over clips). Easy/hard splits use the official ``is_hard``
    flag.
  * E_temporal (dimensionless; positive = predicted landmarks move faster than
    GT, negative = slower): temporal normalized mean error of Chandran et al.,
    comparing frame-to-frame *velocity magnitudes* of prediction and ground
    truth, ignoring absolute positional error:

        E = (1 / (N * T)) * sum_t sum_k
                ( ||p_k^{t+1} - p_k^t||_2 - ||g_k^{t+1} - g_k^t||_2 )
                / ||g_k^{t+1} - g_k^t||_2

    NOTE: computed *signed*, exactly as the typeset formula reads (no absolute
    value around the velocity-magnitude difference). Signs cancel where the
    prediction over- vs under-shoots GT motion: on WFLW-V the signed value is
    ~-0.001 (nearly balanced), while the absolute-difference variant is ~0.36.
    Either way, the per-transition 1/||g^t+1 - g^t|| normalization makes the
    metric's scale dataset-dependent (GT-velocity distribution), so it is not
    comparable across datasets without that caveat.

    Computed in pixel space per clip (mean over its N * (T-1) transitions),
    then averaged over clips (all WFLW-V clips have equal length, so this is
    identical to a pooled mean). Transitions whose GT velocity magnitude is
    <= EPS are excluded (the masked fraction is printed; ~0 on WFLW-V).

OUTPUT (stdout, six numbered sections)
--------------------------------------
  1. 300-W IOD NME per split (mean +/- std over seeds).
  2. WFLW IOD NME per split.
  3. FaceSynthetics face-size NME (test split).
  4. WFLW-V: V-NME easy/hard/combined, NMF easy/hard/combined, E_temporal
     combined.
  5. Cross-checks against the scalar values stored in the pkl by the eval
     notebook (``nme_size``, ``nmf``), per seed.
  6. ``tab:sota``-ready summary row (2-decimal formatting as in main.typ).

Usage: ``python results/calc_headline_results.py [--pkl PATH [PATH ...]]``
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).parent
DEFAULT_PKS = [RESULTS_DIR / f"results_a4-v2-mean-only-wmr-{s}.pkl" for s in ("s1", "s2", "s3")]

# Guard for the E_temporal denominator (GT velocity magnitude, pixel units).
EPS = 1e-8

WFLW_SPLIT_ORDER = ("full", "largepose", "expression", "illumination", "makeup", "occlusion", "blur")

STANDARD_AREA = 256 * 256  # NMF reference area, see calc_nmf in src/utils/torch/misc.py


def load_run(pkl_path: Path) -> dict:
    """Load an evaluation-results pickle produced by an eval notebook."""
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def static_nme_percent(split: dict, key: str) -> float:
    """Mean static NME (%) of one split from the stored per-landmark array.

    Args:
        split: Per-split result dict holding a ``(num_images, nlandmarks)``
            normalized-error array and ``num_images``.
        key: ``"nme_iod"`` (inter-ocular) or ``"nme_s"`` (face size).
    """
    assert split[key].shape[0] == split["num_images"]
    return float(split[key].mean(-1).mean() * 100.0)


def pooled_static_nme_percent(splits: list[dict], key: str) -> float:
    """Image-count-weighted static NME (%) over several splits (pooled per image)."""
    per_image = np.concatenate([s[key].mean(-1) for s in splits])
    return float(per_image.mean() * 100.0)


def per_clip_nmf(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """NMF per clip, following calc_nmf in src/utils/torch/misc.py.

    Args:
        preds: Predicted landmarks, shape (clips, frames, landmarks, 2).
        labels: Ground-truth landmarks, same shape as ``preds``.

    Returns:
        Per-clip NMF values, shape (clips,). RMS over landmarks and frames of
        the size-normalized frame-to-frame error increment (x100).
    """
    err = preds - labels  # (C, T, L, 2)
    bbox = labels.max(axis=2) - labels.min(axis=2)  # (C, T, 2)
    d_s = np.sqrt(bbox.prod(axis=-1) / STANDARD_AREA)[:, 1:]  # (C, T-1)
    err_norm = np.sqrt(np.square(np.diff(err, axis=1)).sum(axis=-1))  # (C, T-1, L)
    nmf_n = np.sqrt(np.square(100.0 * err_norm / d_s[..., None]).mean(axis=-1))
    return np.sqrt(np.square(nmf_n).mean(axis=-1))  # (C,)


def per_clip_e_temporal(preds: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float]:
    """Temporal normalized mean error (Chandran et al.) per clip.

    E_temporal compares the frame-to-frame velocity *magnitudes* of prediction
    and ground truth, relative to the GT velocity magnitude; absolute
    positional error cancels out. The velocity-magnitude difference is kept
    *signed* (no absolute value), exactly as the typeset formula reads, so a
    positive value means the predicted landmarks moved faster than GT overall.

    Args:
        preds: Predicted landmarks, shape (clips, frames, landmarks, 2).
        labels: Ground-truth landmarks, same shape as ``preds``.

    Returns:
        Tuple of (per-clip E_temporal values of shape (clips,), fraction of
        frame transitions excluded because the GT velocity magnitude was
        <= EPS).
    """
    v_pred = np.linalg.norm(np.diff(preds, axis=1), axis=-1)  # (C, T-1, L)
    v_gt = np.linalg.norm(np.diff(labels, axis=1), axis=-1)  # (C, T-1, L)
    valid = v_gt > EPS
    terms = (v_pred - v_gt) / np.maximum(v_gt, EPS)
    per_clip = (terms * valid).sum(axis=(1, 2)) / valid.sum(axis=(1, 2))  # (C,)
    return per_clip, float(1.0 - valid.mean())


def run_metrics(d: dict) -> dict[str, float]:
    """Compute every headline metric for one evaluation run."""
    ibug, wflw, synth, v = d["ibug"], d["wflw"], d["face_synth"]["test"], d["wflw_v"]

    preds, labels = np.asarray(v["preds"]), np.asarray(v["labels"])
    assert preds.shape == labels.shape and preds.ndim == 4
    is_hard = v["is_hard"].astype(bool)
    assert is_hard.sum() == 75 and (~is_hard).sum() == 75, (
        f"expected the official 75 easy / 75 hard clip split, got {(~is_hard).sum()}/{is_hard.sum()}"
    )
    nmf = per_clip_nmf(preds, labels)
    e_temporal_clips, masked_frac = per_clip_e_temporal(preds, labels)
    clip_nme = np.asarray(v["nmes_size"]).mean(axis=1) * 100.0  # (C,) per-clip

    m: dict[str, float] = {
        "300w_common": static_nme_percent(ibug["common"], "nme_iod"),
        "300w_challenging": static_nme_percent(ibug["challenging"], "nme_iod"),
        "300w_full": pooled_static_nme_percent([ibug["common"], ibug["challenging"]], "nme_iod"),
        "300w_indoor": static_nme_percent(ibug["indoor"], "nme_iod"),
        "300w_outdoor": static_nme_percent(ibug["outdoor"], "nme_iod"),
        "300w_private": pooled_static_nme_percent([ibug["indoor"], ibug["outdoor"]], "nme_iod"),
        "synth_test": static_nme_percent(synth, "nme_s"),
        "v_nme": float(clip_nme.mean()),
        "v_nme_easy": float(clip_nme[~is_hard].mean()),
        "v_nme_hard": float(clip_nme[is_hard].mean()),
        "nmf_easy": float(nmf[~is_hard].mean()),
        "nmf_hard": float(nmf[is_hard].mean()),
        "nmf_all": float(nmf.mean()),
        "e_temporal": float(e_temporal_clips.mean()),  # equal-length clips => pooled mean
        "masked_frac": masked_frac,
    }
    for key in WFLW_SPLIT_ORDER:
        m[f"wflw_{key}"] = static_nme_percent(wflw[key], "nme_iod")
    for key in sorted(wflw):  # future-proof: any split not in WFLW_SPLIT_ORDER
        m.setdefault(f"wflw_{key}", static_nme_percent(wflw[key], "nme_iod"))
    return m


def format_mean_std(values: list[float], n_runs: int) -> str:
    """Mean over runs, with +/- sample std (ddof=1) appended when n_runs > 1."""
    mean = float(np.mean(values))
    if n_runs == 1:
        return f"{mean:8.4f}"
    return f"{mean:8.4f} +/- {float(np.std(values, ddof=1)):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--pkl",
        type=Path,
        nargs="+",
        default=DEFAULT_PKS,
        help="Evaluation-results pickle(s) to aggregate over "
        f"(default: the three A4 seeds {', '.join(p.name for p in DEFAULT_PKS)}).",
    )
    args = parser.parse_args()

    runs = [(p, load_run(p)) for p in args.pkl]
    per_run = [(p, d, run_metrics(d)) for p, d in runs]
    n_runs = len(per_run)

    names = [d.get("name", p.stem.removeprefix("results_")) for p, d, _ in per_run]
    steps = [d.get("step", "?") for _, d, _ in per_run]
    print("=" * 100)
    print(f"HEADLINE RESULTS: {', '.join(names)} (steps {', '.join(map(str, steps))}) [QLOT-final]")
    print(f"Aggregation: {'single run' if n_runs == 1 else f'mean +/- std (ddof=1) over {n_runs} runs'}")
    print("=" * 100)

    # ------------------------------------------------------------------
    # 1. 300-W (iBUG protocol), NME % inter-ocular
    # ------------------------------------------------------------------
    print("\n1. 300-W NME (%, IOD normalized)")
    print("-" * 100)
    rows_300w: list[tuple[str, str, list[str]]] = [
        ("common (LFPW+HELEN test)", "300w_common", ["common"]),
        ("challenging (IBUG)", "300w_challenging", ["challenging"]),
        ("full (common + challenging)", "300w_full", ["common", "challenging"]),
        ("indoor (300-W private test)", "300w_indoor", ["indoor"]),
        ("outdoor (300-W private test)", "300w_outdoor", ["outdoor"]),
        ("private test (indoor + outdoor)", "300w_private", ["indoor", "outdoor"]),
    ]
    for label, key, splits in rows_300w:
        n_images = sum(per_run[0][1]["ibug"][s]["num_images"] for s in splits)
        print(f"  {label:<34} ({n_images:>5} imgs)  {format_mean_std([m[key] for _, _, m in per_run], n_runs)}")

    # ------------------------------------------------------------------
    # 2. WFLW, NME % inter-ocular
    # ------------------------------------------------------------------
    print("\n2. WFLW NME (%, IOD normalized)")
    print("-" * 100)
    wflw = per_run[0][1]["wflw"]
    split_keys = [k for k in WFLW_SPLIT_ORDER if k in wflw]
    split_keys += sorted(k for k in wflw if k not in split_keys)  # future-proof
    for key in split_keys:
        split = wflw[key]
        vals = [m[f"wflw_{key}"] for _, _, m in per_run]
        print(f"  {split['dataset_name']:<34} ({split['num_images']:>5} imgs)  {format_mean_std(vals, n_runs)}")

    # ------------------------------------------------------------------
    # 3. FaceSynthetics, NME % face-size
    # ------------------------------------------------------------------
    print("\n3. FaceSynthetics NME (%, face-size normalized)")
    print("-" * 100)
    synth = per_run[0][1]["face_synth"]["test"]
    label = f"test ({synth['dataset_name']})"
    print(f"  {label:<34} ({synth['num_images']:>5} imgs)  {format_mean_std([m['synth_test'] for _, _, m in per_run], n_runs)}")

    # ------------------------------------------------------------------
    # 4. WFLW-V video metrics (face-size normalized)
    # ------------------------------------------------------------------
    print("\n4. WFLW-V (NME % / NMF face-size normalized; E_temporal dimensionless)")
    print("-" * 100)
    v0 = per_run[0][1]["wflw_v"]
    n_clips, clip_len = v0["preds"].shape[:2]
    n_hard = int(v0["is_hard"].astype(bool).sum())
    print(f"  ({n_clips} clips x {clip_len} frames per run; {n_clips - n_hard} easy / {n_hard} hard)")
    rows_v: list[tuple[str, str]] = [
        (f"V-NME easy   ({n_clips - n_hard} clips)", "v_nme_easy"),
        (f"V-NME hard   ({n_hard} clips)", "v_nme_hard"),
        (f"V-NME combined ({n_clips} clips)", "v_nme"),
        (f"NMF easy   ({n_clips - n_hard} clips)", "nmf_easy"),
        (f"NMF hard   ({n_hard} clips)", "nmf_hard"),
        (f"NMF combined ({n_clips} clips)", "nmf_all"),
        (f"E_temporal combined ({n_clips} clips)", "e_temporal"),
    ]
    for label, key in rows_v:
        print(f"  {label:<34}          {format_mean_std([m[key] for _, _, m in per_run], n_runs)}")
    masked = max(m["masked_frac"] for _, _, m in per_run)
    print(f"  (masked GT-static E_temporal transitions: {masked:.2e})")

    # ------------------------------------------------------------------
    # 5. Cross-checks vs scalar values stored by the eval notebook
    # ------------------------------------------------------------------
    print("\n5. Cross-checks vs values stored in the pkl")
    print("-" * 100)
    for (p, d, m), name in zip(per_run, names):
        v = d["wflw_v"]
        line = f"  {name:<40}"
        if "nme_size" in v:
            line += f" V-NME delta {m['v_nme'] - v['nme_size']:+.2e}"
        if "nmf" in v:
            line += f" NMF delta {m['nmf_all'] - v['nmf']:+.2e}"
        print(line)

    # ------------------------------------------------------------------
    # 6. tab:sota-ready row (2-decimal formatting as in main.typ)
    # ------------------------------------------------------------------
    print("\n6. Paper-row summary (tab:sota formatting, mean over runs)")
    print("-" * 100)
    mean = {k: float(np.mean([m[k] for _, _, m in per_run])) for k in per_run[0][2]}
    print(f"  300-W {mean['300w_full']:.2f} | WFLW {mean['wflw_full']:.2f} | Synth {mean['synth_test']:.2f} | "
          f"V-NME {mean['v_nme']:.2f} | V-NMF {mean['nmf_all']:.1f} | E_temporal {mean['e_temporal']:.4f}")


if __name__ == "__main__":
    main()
