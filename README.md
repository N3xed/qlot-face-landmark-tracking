# QLOT: Queried Learned Optimization for Face Landmark Tracking

QLOT treats face alignment as an iterative, query-conditioned learned optimization
process rather than single-shot inference. Inspired by RAFT, QLOT fuses per-landmark correlation features with temporal context carried
by a recurrent hidden state, exchanged across landmarks through a low-rank
Write-Mix-Read block. A composite objective of decoupled spatial and temporal Gaussian
negative log-likelihood terms supervises directional uncertainty and the temporal
derivatives of the error.

The result is of competitive static accuracy with state-of-the-art detectors while
substantially improving temporal stability, and staying
efficient enough for real-time mobile deployment (2M parameters and 0.65G MACs/frame).

Runtime (224×224 input, ONNX Runtime): ~2 ms/frame on an NVIDIA RTX 3080 (CUDA), ~8 ms
on CPU (1 frame, 1 iteration, 98 landmarks).

See the [Project Page](https://n3xed.github.io/qlot-face-landmark-tracking/) for more details and a local web
demo running on single-threaded WebAssembly.

## Repository layout

```
src/            Training, evaluation, demo, and ONNX export code
  model/        Model definition (feature extractor, detector, update predictor, covariance)
  kernels/      Fused Triton kernel for correlation sampling (only used for training)
  utils/        Dataset loaders, training utilities, visualization
  notebooks/    Evaluation and exploration notebooks
data/           Canonical landmark layouts and query-point presets (no dataset images)
docs/paper/     Paper and supplementary material source (Typst)
site/           Interactive browser demo (assets built by CI)
scripts/        Site asset build (site_build.py) and JS vendoring (site_vendor.sh)
results/        Figure-generation and claim-verification scripts
container/      Apptainer/Singularity definitions for the training environment
ABLATIONS.md    Pre-specified ablation matrix and results
```

## Setup

Requires Python ≥ 3.12. Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

The default environment targets CUDA 13 (`torch==2.13.0+cu130` via the PyTorch wheel
index; see `pyproject.toml`). Alternatively, use the container definitions in
`container/` for a fully pinned training environment.

## Pretrained models

Model weights are attached to the [latest GitHub release](https://github.com/N3xed/qlot-face-landmark-tracking/releases/latest)
(not committed to the repository):

| Asset | Format | Use |
|---|---|---|
| `qlot-final.pth` | PyTorch checkpoint | demo, evaluation, fine-tuning |
| `qlot-final.onnx` | ONNX | export / the browser demo |
| `qlot-paper.pdf`, `qlot-supplement.pdf` | PDF | paper and supplementary material |

## Usage

**Demo** (webcam or video file via OpenCV, interactive viewer):

```bash
uv run src/demo.py --model qlot-final.pth --camera /dev/video0 
```

**Evaluation** on datasets:

```bash
uv run src/eval.py qlot-final.pth --dataset-dir /path/to/datasets
```

Full evaluation results and models for all ablations and seeds are available at [Google Drive: qlot-results.tar.gz](https://drive.google.com/file/d/1L0ZQM9ahkXdIj0xQWegWRZX1IzGlc3e3/view?usp=drive_link).

Datasets (300-W, WFLW, WFLW-V, FaceSynthetics) are **not** included.
They are subject to their own licenses and must be obtained from their respective sources
(the download links contained in the Python sources are not guaranteed to work).
The canonical landmark layouts used for training and evaluation are provided in `data/`.

**ONNX export**:

```bash
uv run src/export_onnx.py qlot-final.pth
```

**Training**: `src/train.py` (see `uv run src/train.py --help`, requires the datasets
and a CUDA GPU with Triton support for the fused kernels).

## Interactive demo

A [self-contained browser demo](https://n3xed.github.io/qlot-face-landmark-tracking/) (ONNX Runtime Web + three.js) lives in `site/`. The
built site is published via GitHub Pages.

## Reproducing the paper

- Paper and supplement source: `docs/paper/` (Typst v0.15.1, compile `main.typ` and `supplement.typ`).
- Headline numbers: `results/calc_headline_results.py`, claim checks:
  `results/verify_claims.py`, figures: `results/fig_*.py` (may require datasets and/or
  ablation results).
- The ablation protocol is documented in `ABLATIONS.md`.

## License

Code and model weights are released under the [Apache License 2.0](LICENSE). The
preprint text and figures are available under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

`data/face_mesh.obj` is derived from the face model in
[ICT-FaceKit](https://github.com/USC-ICT/ICT-FaceKit), used under the MIT License (see [NOTICE](NOTICE) for the full attribution).

The models were trained on the 300-W, WFLW, WFLW-V, and FaceSynthetics datasets, each
subject to its own license and terms of use. Users intending commercial use should
review the terms of the underlying training datasets.
