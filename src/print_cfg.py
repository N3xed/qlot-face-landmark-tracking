from __future__ import annotations

import argparse
from pathlib import Path
import math

import torch
import torch.nn.functional as F

from utils.torch.misc import load, Config
from model.detector import QLOT


def main():
    parser = argparse.ArgumentParser(description="Print the training config of a model")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        help="Optional checkpoint path to load before printing weights.",
    )
    parser.add_argument("--show-list", action="store_true", help="Show the list of best checkpoints with their NME and NMF values.")
    args = parser.parse_args()

    global_step, cfg = load(
        args.checkpoint,
        None,
        None,
        None,
    )

    print(f"Loaded config from {args.checkpoint} at global step {global_step}:")
    print(f"- global_step: {global_step}")
    print(f"- name: {cfg.name}/{cfg.run}")

    if (nme := cfg.nme) is not None:
        print(f"- nme: {nme:.6f}")
    if (nmf := cfg.nmf) is not None:
        print(f"- nmf: {nmf:.6f}")

    if args.show_list:
        best_checkpoints: dict[int, tuple[float, float]] = cfg.others.get("_best_checkpoints", {})
        best_checkpoints_list = sorted(best_checkpoints.items(), key=lambda x: (x[1][0], x[1][1], x[0]))

        if len(best_checkpoints_list) > 0:
            print("- best checkpoints:")

        for i, (step, (nme, nmf)) in enumerate(best_checkpoints_list):
            if i >= 20:
                break
            print(f"  - {i+1:02}: step={step}, NME={nme:.6f}, NMF={nmf:.6f}")

    
if __name__ == "__main__":
    main()