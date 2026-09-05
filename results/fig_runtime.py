"""Generate the inference-runtime figure for the paper.

This script plots the average per-frame inference time as a function of the
number of query points (landmarks), using the measurements in
``results/performance.txt`` (ONNX Runtime, 224x224 input, batch size 1,
AMD Ryzen 7 3700X / NVIDIA RTX 3080).

Two panels are produced side by side:

1. GPU inference (left): CUDA execution provider, with and without CUDA graph
   capture. The runtime is essentially flat in the number of query points;
   graph capture roughly halves the (launch-overhead-dominated) time.
2. CPU inference (right): the runtime grows approximately linearly with the
   number of query points.

Usage: ``python results/fig_runtime.py [out_path]``
Default output: ``results/fig_runtime.pdf`` (and .png).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parent


def load_performance(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the benchmark table. Returns (n_queries, cpu_ms, cuda_ms, cuda_graph_ms)."""
    data = np.loadtxt(path, comments="#")
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def main() -> None:
    out_base = Path(sys.argv[1]) if len(sys.argv) > 1 else RESULTS_DIR / "fig_runtime"

    n_queries, cpu_ms, cuda_ms, cuda_graph_ms = load_performance(RESULTS_DIR / "performance.txt")

    # Exclude the CPU outlier at 29 query points (7.46 ms; breaks the otherwise
    # smooth linear trend -- a measurement artifact).
    cpu_keep = n_queries != 29

    fig, (ax_gpu, ax_cpu) = plt.subplots(1, 2, figsize=(13, 4.5))

    # --- Panel 1: GPU inference ---
    ax_gpu.plot(n_queries, cuda_ms, color="tab:blue", label="CUDA")
    ax_gpu.plot(n_queries, cuda_graph_ms, color="tab:orange", label="CUDA (graph capture)")
    ax_gpu.set_xlabel("Number of Query Points")
    ax_gpu.set_ylabel("Inference Time (ms)")
    ax_gpu.set_title("GPU (NVIDIA RTX 3080)")
    ax_gpu.grid(True, which="both", alpha=0.3)
    ax_gpu.legend()

    # --- Panel 2: CPU inference ---
    ax_cpu.plot(n_queries[cpu_keep], cpu_ms[cpu_keep], color="tab:green", label="CPU")
    ax_cpu.set_xlabel("Number of Query Points")
    ax_cpu.set_ylabel("Inference Time (ms)")
    ax_cpu.set_title("CPU (AMD Ryzen 7 3700X)")
    ax_cpu.grid(True, which="both", alpha=0.3)
    ax_cpu.legend()

    fig.tight_layout()
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved {out_base.with_suffix('.pdf')} and {out_base.with_suffix('.png')}")


if __name__ == "__main__":
    main()
