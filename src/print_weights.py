from __future__ import annotations

import argparse
from pathlib import Path
import math

import torch
import torch.nn.functional as F

from model.detector import QLOT


def _format_tensor(tensor: torch.Tensor, flatten = True) -> str:
    if flatten:
        values = [f"{value:.6f}" for value in tensor.detach().cpu().flatten().tolist()]
        return "[" + ", ".join(values) + "]"
    else:
        return f"{tensor.cpu()}".removeprefix("tensor(").removesuffix(")")

def _branch_weights(logits: torch.Tensor) -> torch.Tensor:
    return logits.softmax(dim=0) * logits.numel()


def _print_attn_temps(model: QLOT) -> None:
    attn = model.update_predictor.mixer
    
    print("Bypass Attention")
    # print(f"  gate-bias-raw {_format_tensor(attn.local_gate[-2].bias.detach(), flatten=False)}")  # type: ignore
    # print(f"  gate-bias {_format_tensor(attn.local_gate[-2].bias.sigmoid().detach(), flatten=False)}")  # type: ignore
    # print(f"  mix-gate-raw {_format_tensor(attn.mix_gate[0, :, 0, :].detach(), flatten=False)}")  # type: ignore
    # print(f"  mix-gate {_format_tensor(attn.mix_gate[0, :, 0, :].sigmoid().detach(), flatten=False)}")  # type: ignore
    print(f"  write-temp-raw {_format_tensor(attn.write_temperature[0, 0].detach(), False)}")
    print(f"  write-temp {_format_tensor((attn.write_temperature / 3.0).exp2()[0, 0].detach(), False)}")
    print(f"  weight-residual {_format_tensor(attn.residual_weights[0].detach())}")
    print(f"  weight-hidden {_format_tensor(attn.residual_weights[1].detach())}")
    print(f"  weight-attn {_format_tensor(attn.residual_weights[2].detach())}")
    print(f"  weight-mlp {_format_tensor(attn.residual_weights[3].detach())}")

def _print_feature_encoder_weights(model: QLOT) -> None:
    encoder = model.encoder

    print("Feature Encoder")
    for i, branch in enumerate(encoder.corr_feat_head_convs):
        print(f"  branch {i} softshrink temp {_format_tensor(branch[0].log_temp.detach().exp())}")  # type: ignore


def _load_model(checkpoint_path: Path | None) -> QLOT:
    model = QLOT(feature_extractor_pretrained=False)

    if checkpoint_path is None:
        return model.eval()

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
        if state_dict is None:
            raise ValueError(f"No model state dict found in {checkpoint_path}")
    else:
        state_dict = checkpoint
    state_dict = QLOT.translate_weights(state_dict)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"Warning: unexpected keys when loading checkpoint: {unexpected}")

    config = checkpoint.get("config")
    if config is not None:
        print(f"Loaded model config: {config}")
    print(f"Global step = {checkpoint.get("global_step", "unknown")}")

    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print learned fusion-weight parameters for the feature encoder and update predictor."
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        type=Path,
        help="Optional checkpoint path to load before printing weights.",
    )
    parser.add_argument("--list-parameters", action="store_true", help="List all model parameters and their shapes.")
    parser.add_argument("--print-summary", action="store_true", help="Print a summary of the model using torchinfo.")
    args = parser.parse_args()

    model = _load_model(args.checkpoint)

    if not args.list_parameters and not args.print_summary:
        _print_attn_temps(model)
        print()
        _print_feature_encoder_weights(model)

    if args.list_parameters:
        print()
        print("All Model Parameters:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.shape}")

    if args.print_summary:
        import torchinfo
        torchinfo.summary(
            model,
            input_size=[(1, 3, 224, 224), (1, 98, 3)],
            col_names=("input_size", "output_size", "num_params", "trainable"),
            row_settings=("var_names", "depth"),
            return_tensor_predictions=True,
            iterations=1,
        )


if __name__ == "__main__":
    main()