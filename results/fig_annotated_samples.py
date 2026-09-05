"""Generate the annotated-samples qualitative figure for the paper.

Shows a grid of test samples annotated with ground-truth labels (green) and
model predictions (red, with 95% covariance ellipses when the checkpoint
predicts covariances). One row per dataset: 300-W, WFLW, FaceSynthetics and
WFLW-V, with a centered title above each row and a figure-level legend next to
the first row title (green dot = Labels, red dot = Predictions).

The per-panel annotation logic mirrors ``ImgEvalResult.display()`` in
``src/eval.py`` (``draw_keypoints`` with the green/red instance pair plus
per-panel NME text), but predictions are read from a saved ``results_*.pkl``
file, so no model or GPU is needed -- only the datasets (for the images) and
the pickle. The 300-W row pools the "common" and "challenging" test subsets
(i.e. the full 300-W test set); the WFLW-V row shows individual frames from
distinct test clips.

Usage:
    python results/fig_annotated_samples.py --dataset-dir /path/to/datasets [out_path]

Default output: ``results/fig_annotated_samples.pdf`` (and .png).
The selected indices / (clip, frame) pairs are printed so a good draw can be
reproduced via ``--seed`` (or by switching ``--mode``).
"""

import argparse
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

RESULTS_DIR = Path(__file__).parent
SRC_DIR = RESULTS_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.torch.viz import draw_keypoints
from utils.datasets.image import WFLW_V_Frames
from data import Datasets

DEFAULT_RESULTS = RESULTS_DIR / "results_a4-v2-mean-only-wmr-s1.pkl"

SelectionMode = Literal["random", "worst", "best", "median", "percentile"]

# Image rows of the figure: display name -> [(results pkl group, pkl split key, Datasets attribute), ...].
# 300-W is the union of the "common" and "challenging" test subsets.
IMAGE_ROWS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("300-W", [("ibug", "common", "ibug_test_common"), ("ibug", "challenging", "ibug_test_challenging")]),
    ("WFLW", [("wflw", "full", "wflw_test_full")]),
    ("FaceSynthetics", [("face_synth", "test", "face_synth_test")]),
]
VIDEO_ROW_NAME = "WFLW-V"


@dataclass
class Panel:
    """A single annotated sample to draw."""

    img: torch.Tensor  # (3, H, W), float in [0, 1]
    labels: torch.Tensor  # (num_landmarks, 2)
    xy: torch.Tensor  # (num_landmarks, 2)
    cov: torch.Tensor  # (num_landmarks, 3) raw LowRankCov2D params (log_sigma_x, log_sigma_y, atanh(rho))
    texts: list[str]  # Per-panel text annotations (e.g. NME values)


def _order(scores: np.ndarray, mode: SelectionMode, rng: np.random.Generator) -> np.ndarray:
    """Return indices into ``scores`` ordered according to the selection mode."""
    if mode == "random":
        return rng.permutation(len(scores))
    if mode == "worst":
        return np.argsort(-scores)
    if mode == "best":
        return np.argsort(scores)
    if mode == "median":
        return np.argsort(np.abs(scores - np.median(scores)))
    raise ValueError(f"Invalid selection mode: {mode}")


def select_indices(scores: np.ndarray, n: int, mode: SelectionMode, rng: np.random.Generator) -> list[int]:
    """Pick ``n`` indices into ``scores`` according to the selection mode."""
    if mode == "percentile":
        # Samples at equally spaced positions of the sorted NME_size distribution, returned in
        # ascending NME order (best on the left, worst on the right). Using positions
        # ``round(q * (N - 1))`` makes the center panel exactly the median for odd n.
        idx = np.argsort(scores)
        return idx[np.unique(np.round(np.linspace(0.25, 0.75, n) * (len(scores) - 1)).astype(int))].tolist()
    return _order(scores, mode, rng)[: min(n, len(scores))].tolist()


def select_frames(nmes: np.ndarray, n: int, mode: SelectionMode, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Pick ``n`` (clip, frame) pairs from per-frame NMEs, at most one frame per clip."""
    if mode == "percentile":
        # For each equally spaced NME_size quantile, take the frame achieving it, in ascending NME
        # order (center = median for odd n). Since each quantile resolves to a different frame, the
        # clips they come from are distinct as well, keeping the one-frame-per-clip property.
        flat = nmes.flatten()
        order = np.argsort(flat)
        idx = order[np.unique(np.round(np.linspace(0.0, 1.0, n) * (flat.size - 1)).astype(int))]
        return [divmod(int(i), nmes.shape[1]) for i in idx]
    num_clips, num_frames = nmes.shape
    assert n <= num_clips, f"Cannot pick {n} frames from only {num_clips} clips (distinct clips required)"
    pairs: list[tuple[int, int]] = []
    used_clips: set[int] = set()
    for flat in _order(nmes.flatten(), mode, rng):
        clip_i, frame_i = divmod(int(flat), num_frames)
        if clip_i in used_clips:
            continue
        used_clips.add(clip_i)
        pairs.append((clip_i, frame_i))
        if len(pairs) == n:
            break
    return pairs


def image_row_panels(
    name: str,
    parts_spec: list[tuple[str, str, str]],
    results: dict,
    datasets: Datasets,
    n: int,
    mode: SelectionMode,
    rng: np.random.Generator,
) -> list[Panel]:
    """Build the panels for one image-dataset row from saved ``ImgEvalResult`` dicts."""
    parts = [(results[group][split], getattr(datasets, attr)) for group, split, attr in parts_spec]
    # Selection by size-normalized NME, averaged over landmarks (as in ImgEvalResult.display()).
    scores = np.concatenate([res["nme_s"].mean(axis=-1) for res, _ in parts]) * 100.0
    bounds = np.cumsum([int(res["num_images"]) for res, _ in parts])

    selected = select_indices(scores, n, mode, rng)
    print(f"{name}: selected image indices {selected} (of {int(bounds[-1])})")

    panels: list[Panel] = []
    for idx in selected:
        part_i = int(np.searchsorted(bounds, idx, side="right"))
        local_idx = idx - (int(bounds[part_i - 1]) if part_i > 0 else 0)
        res, dataset = parts[part_i]

        nme_iod = float(res["nme_iod"][local_idx].mean()) * 100.0
        nme_s = float(res["nme_s"][local_idx].mean()) * 100.0
        panels.append(
            Panel(
                img=dataset[local_idx]["image"].cpu(),
                labels=torch.from_numpy(res["labels"][local_idx]).float(),
                xy=torch.from_numpy(res["xy"][local_idx]).float(),
                cov=torch.from_numpy(res["cov"][local_idx]).float(),
                texts=[f"NME IOD: {nme_iod:.2f}%", f"NME Size: {nme_s:.2f}%"],
            )
        )
    return panels


def video_row_panels(
    results: dict,
    datasets: "Datasets",  # noqa: F821
    n: int,
    mode: SelectionMode,
    rng: np.random.Generator,
) -> list[Panel]:
    """Build the panels for the WFLW-V row: individual frames from distinct test clips."""
    wv = results["wflw_v"]
    dataset = datasets.wflw_v_test
    assert hasattr(dataset.dataset, "ordered_video_names"), "wflw_v_test must be a WFLW_V_Frames dataset"
    clip_names: list[str] = list(wv["clip_names"])

    assert isinstance(dataset.dataset, WFLW_V_Frames)
    assert list(dataset.dataset.ordered_video_names) == clip_names, (
        "WFLW-V clip ordering mismatch between the results file and the dataset; "
        "stored predictions would be assigned to the wrong frames."
    )

    nmes = np.asarray(wv["nmes_size"]) * 100.0  # (num_clips, num_frames), size-normalized, averaged over landmarks
    pairs = select_frames(nmes, n, mode, rng)
    print(f"{VIDEO_ROW_NAME}: selected (clip, frame) pairs {pairs}")

    panels: list[Panel] = []
    for clip_i, frame_i in pairs:
        sample = dataset[clip_i]
        img = sample["image"][frame_i].cpu()  # (3, H, W)
        del sample
        panels.append(
            Panel(
                img=img,
                labels=torch.from_numpy(wv["labels"][clip_i, frame_i]).float(),
                xy=torch.from_numpy(wv["preds"][clip_i, frame_i]).float(),
                cov=torch.from_numpy(wv["cov"][clip_i, frame_i]).float(),
                texts=[f"NME Size: {nmes[clip_i, frame_i]:.2f}%"],
            )
        )
    return panels


def draw_panel(
    ax: plt.Axes,
    panel: Panel,
    resolution: int | None,
    radius: float,
    width: int,
    text_fontsize: int = 7,
) -> None:
    """Draw one annotated sample, mirroring ``ImgEvalResult.display()`` (without the per-sample title)."""
    all_xy = [
        panel.labels,  # Ground Truth
        panel.xy,  # Predicted
    ]
    try:
        from model import LowRankCov2D

        cov = LowRankCov2D(panel.cov).as_cov2d_params().cpu()  # (num_landmarks, 3)
        all_cov: list[torch.Tensor] | None = [
            torch.zeros_like(cov),  # Ground Truth (not available)
            cov,  # Predicted
        ]
        # Suppress predicted keypoints with an all-zero covariance (no meaningful ellipse could be
        # drawn anyway). Not applied to the ground-truth instance: its covariance is an all-zero
        # placeholder, and visibility gates both the filled circle and the ellipse in
        # draw_keypoints(), so zeroing it would hide the ground-truth keypoints as well.
        visibility: list[torch.Tensor] | None = [torch.ones(len(cov), dtype=torch.bool), cov.amax(dim=-1) > 0]
    except Exception:
        all_cov = None
        visibility = None
    img = draw_keypoints(
        panel.img,
        all_xy,
        all_cov,
        colors=["green", "red"],
        probability_threshold=0.95,
        scale_to=resolution,
        radius=radius,
        width=width,
        visibility=visibility,
    )
    ax.imshow(img.permute(1, 2, 0).contiguous().numpy())
    ax.axis("off")
    handles: list = []
    for text in panel.texts:
        handles.extend(ax.plot([], [], " ", label=text))
    ax.legend(handles=handles, loc="upper right", handletextpad=0.0, handlelength=0, fontsize=text_fontsize)


def build_figure(
    rows: list[tuple[str, list[Panel]]],
    n_cols: int,
    resolution: int | None,
    radius: float,
    width: int,
) -> plt.Figure:
    """Assemble the full grid: one row per dataset, a centered title above each row."""
    n_rows = len(rows)
    panel_in = 3.0  # Side length of one (square) image panel in inches
    title_in = 0.28  # Height of the per-row title strip in inches
    fig = plt.figure(figsize=(panel_in * n_cols, (panel_in + title_in) * n_rows))
    height_ratios = [h for _ in range(n_rows) for h in (title_in, panel_in)]
    gs = fig.add_gridspec(
        2 * n_rows,
        n_cols,
        height_ratios=height_ratios,
        left=0.0,
        right=1.0,
        top=1.0,
        bottom=0.0,
        hspace=0.0,
        wspace=0.02,
    )
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(0.5, 0.4, rows[0][0], ha="center", va="center", fontsize=16)
    # Figure-level legend beside the first row title (to its right, as in ImgEvalResult.display()).
    legend_handles = [
        Line2D([], [], marker="o", linestyle="None", color="green", label="Labels"), # markersize=8, 
        Line2D([], [], marker="o", linestyle="None", color="red", label="Predictions"),
    ]
    fig.legend(handles=legend_handles, loc="upper right", ncols=2, bbox_to_anchor=(0.999, 1.005)) #textsize, , 

    for r, (name, panels) in enumerate(rows):
        if r > 0:  # First row title is already drawn above, next to the legend.
            title_ax = fig.add_subplot(gs[2 * r, :])
            title_ax.axis("off")
            title_ax.text(0.5, 0.4, name, ha="center", va="center", fontsize=16)
        for c in range(n_cols):
            ax = fig.add_subplot(gs[2 * r + 1, c])
            if c < len(panels):
                draw_panel(ax, panels[c], resolution=resolution, radius=radius, width=width)
            else:
                ax.axis("off")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the annotated-samples qualitative figure")
    parser.add_argument(
        "out_path", type=Path, nargs="?", default=RESULTS_DIR / "fig_annotated_samples", help="Output path without extension"
    )
    parser.add_argument(
        "--results", type=Path, default=DEFAULT_RESULTS, help="Path to the results_*.pkl file with saved predictions"
    )
    parser.add_argument(
        "--dataset-dir", type=Path, required=True, help="Path to the dataset directory (300W, WFLW, FaceSyntheticsSmall, WFLW_V)"
    )
    parser.add_argument("-n", "--num-images", type=int, default=3, help="Number of images per dataset row")
    parser.add_argument(
        "--mode",
        choices=["random", "worst", "best", "median", "percentile"],
        default="random",
        help="Sample selection mode (by NME_size); 'percentile' uses equally spaced NME_size quantiles (left = best, right = worst, center = median)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --mode random")
    parser.add_argument(
        "--resolution", type=int, default=800, help="Resolution the images are scaled to before drawing keypoints"
    )
    parser.add_argument("--radius", type=float, default=5, help="Keypoint radius (at the scaled resolution)")
    parser.add_argument("--width", type=int, default=4, help="Ellipse outline width (at the scaled resolution)")
    args = parser.parse_args()

    from data import Datasets

    with open(args.results, "rb") as f:
        results = pickle.load(f)
    print(f"Loaded results from {args.results} (run={results.get('name')}, step={results.get('step')})")

    datasets = Datasets(args.dataset_dir)
    rng = np.random.default_rng(args.seed)

    rows: list[tuple[str, list[Panel]]] = []
    for name, parts_spec in IMAGE_ROWS:
        rows.append((name, image_row_panels(name, parts_spec, results, datasets, args.num_images, args.mode, rng)))
    # rows.append((VIDEO_ROW_NAME, video_row_panels(results, datasets, args.num_images, args.mode, rng)))

    fig = build_figure(rows, args.num_images, resolution=args.resolution, radius=args.radius, width=args.width)
    out_base: Path = args.out_path
    fig.savefig(out_base.with_suffix(".pdf"), pad_inches=0, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=150, pad_inches=0, bbox_inches="tight")
    print(f"Saved {out_base.with_suffix('.pdf')} and {out_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
