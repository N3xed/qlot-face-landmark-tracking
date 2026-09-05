from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import os
import sys
import torch.utils._pytree as pytree
import json

# When running from src/, local modules are available
from model.detector import QLOT
from model.utils import LandmarkPrediction, NUM_PREDS_PARAMS
from model.cov import LowRankCov2D
import argparse

try:
    from onnxruntime.transformers.float16 import convert_float_to_float16
except ImportError:
    convert_float_to_float16 = None

class OnnxWrapper(nn.Module):
    def __init__(
        self,
        model,
        num_queries: int,
        fixed_query_points: torch.Tensor | None = None,
        mask_queries: list[int] | None = None,
        dynamic_num_queries: bool = True,
    ):
        super().__init__()
        self.model = model
        self.num_queries = num_queries
        self.dynamic_num_queries = dynamic_num_queries
        if fixed_query_points is not None:
            # Bake the query points into the exported graph so the model no
            # longer needs (or accepts) them as an input.
            self.register_buffer("fixed_query_points", fixed_query_points, persistent=False)
        else:
            self.fixed_query_points = None

        if mask_queries:
            # Masked query indices are prevented from communicating through
            # the mixer's global basis slots (True = ignored).
            self.mask_queries = list(mask_queries)
            if not dynamic_num_queries:
                # With a fixed number of queries, bake the mask in as a
                # constant buffer so the graph has no dependency on the
                # query_points input for the mask's shape.
                landmarks_to_mask = torch.zeros(1, num_queries, dtype=torch.bool)
                landmarks_to_mask[:, self.mask_queries] = True
                self.register_buffer("landmarks_to_mask", landmarks_to_mask, persistent=False)
            else:
                # Built dynamically from query_points in forward() instead.
                self.landmarks_to_mask = None
        else:
            self.mask_queries = None
            self.landmarks_to_mask = None

    def forward(
        self,
        image,
        query_points,
        gating_cutoff,
        gating_radius,
        prefill_hidden_state,
        prefill_starting_landmarks
    ):
        if self.fixed_query_points is not None:
            # Ignore the input and use the baked-in query points instead.
            query_points = self.fixed_query_points

        landmarks_to_mask = self.landmarks_to_mask
        if self.mask_queries and self.dynamic_num_queries:
            # Dynamic query count: derive the mask from query_points so its
            # shape follows the dynamic dimension instead of specializing.
            landmarks_to_mask = torch.zeros_like(query_points[..., 0], dtype=torch.bool)
            landmarks_to_mask[:, self.mask_queries] = True

        init_pred = LandmarkPrediction.from_tensor(
            prefill_starting_landmarks,
            cov_type=LowRankCov2D
        )
        
        predictions = self.model(
            image=image,
            query_points=query_points,
            gating_radius=gating_radius,
            gating_cutoff=gating_cutoff,
            iterations=1,
            return_sequence=False,
            store_similarity_maps=False,
            prefill_hidden_state=prefill_hidden_state,
            prefill_starting_landmarks=init_pred,
            return_hidden_state=True,
            return_tensor_predictions=True, 
            detach_updates=False,
            force_cov_gating=True,
            use_naive_correlation=True,
            landmarks_to_mask=landmarks_to_mask,
        )
        
        pred, hidden_state = predictions
        return pred, hidden_state

def _convert_export_to_fp16(output_path: str, keep_io_fp32: bool = True) -> None:
    if convert_float_to_float16 is None:
        raise ImportError(
            "FP16 conversion requires onnxruntime.transformers.float16 to be installed"
        )

    import onnx

    print(f"Converting {output_path} to FP16...")
    model = onnx.load(output_path)
    model_fp16 = convert_float_to_float16(
        model,
        keep_io_types=keep_io_fp32,
        force_fp16_initializers=True,
    )
    onnx.save(model_fp16, output_path)
    print(f"FP16 conversion complete: {output_path}")


def _prune_unused_inputs(output_path: str, input_names: list[str]) -> None:
    """Remove graph inputs that are not consumed by any node.

    Inputs that the exported graph never uses (e.g. `query_points` after
    baking fixed query points in as a constant buffer) would otherwise remain
    listed as model inputs, forcing callers to keep providing them.
    """
    import onnx

    model = onnx.load(output_path)
    used = {inp for node in model.graph.node for inp in node.input}
    # Constant-foldable consumers (initializers) count as usage too.
    used |= {init.name for init in model.graph.initializer}
    keep = [inp for inp in model.graph.input if inp.name in used or inp.name not in input_names]
    removed = {inp.name for inp in model.graph.input} - {inp.name for inp in keep}
    if not removed:
        return
    del model.graph.input[:]
    model.graph.input.extend(keep)
    onnx.save(model, output_path)
    print(f"Pruned unused graph inputs: {sorted(removed)}")


def export_onnx(
    output_path,
    checkpoint_path=None,
    opset_version=18,
    fp16: bool = False,
    keep_io_fp32: bool = True,
    fixed_num_queries: int | None = None,
    fixed_queries: str | None = None,
    mask_queries: list[int] | None = None,
):
    print(f"Exporting model to {output_path} with opset {opset_version}...")
    
    print("Initializing model...")
    model = QLOT(
        feature_extractor_pretrained=checkpoint_path is None
    )
    
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            # Handle state dict (sometimes wrapped in 'state_dict' key or PL wrapper)
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            state_dict = QLOT.translate_weights(state_dict)
                
            # Clean up PL keys if needed (remove "model." prefix if present and not needed, 
            # as ContinuousLandmarkDetector expects keys without "model." if it wasn't wrapped)
            new_state_dict = {}
            msg_prefix = "model."
            for k, v in state_dict.items():
                if k.startswith(msg_prefix):
                    new_state_dict[k[len(msg_prefix):]] = v
                else:
                    new_state_dict[k] = v
            
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            print(f"Checkpoint loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            if len(missing) > 0:
                print(f"Missing keys (sample): {missing[:5]}")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint: {e}")
            import traceback
            traceback.print_exc()

    model.eval()
    # Disable gradients for all parameters to ensure graph is clean
    for param in model.parameters():
        param.requires_grad = False
    
    image_size = 224
    print(f"Baking grids for image size {image_size}x{image_size}...")
    model.bake_grids((image_size, image_size))

    # Dummy inputs for tracing (using B=1, Q=num_queries)
    batch_size = 1
    num_queries_dim = torch.export.Dim("num_queries", min=1, max=100000)
    gru_hidden_dim = 128

    device = torch.device("cpu")

    fixed_query_points = None
    is_dynamic_queries = fixed_queries is None and fixed_num_queries is None
    if fixed_queries is not None:
        fixed_queries_path = Path(fixed_queries)
        if fixed_queries_path.suffix == ".npy":
            points = np.load(fixed_queries)
            assert points.ndim == 2 and points.shape[1] == 3, (
                f"Fixed query points must have shape (N, 3), got {points.shape}"
            )
            fixed_query_points = torch.from_numpy(points).to(device=device, dtype=torch.float32).unsqueeze(0)
            num_queries = points.shape[0]
        elif fixed_queries_path.suffix == ".json":
            with open(fixed_queries, "r") as f:
                points = json.load(f)
            points = np.array(points, dtype=np.float32)
            assert points.ndim == 2 and points.shape[1] == 3, (
                f"Fixed query points must have shape (N, 3), got {points.shape}"
            )
            fixed_query_points = torch.from_numpy(points).to(device=device, dtype=torch.float32).unsqueeze(0)
            num_queries = points.shape[0]
        else:
            raise ValueError(f"Unsupported fixed queries file format: {fixed_queries_path.suffix}")

        print(f"Using {num_queries} fixed query points from {fixed_queries}")
    elif fixed_num_queries is not None:
        num_queries = fixed_num_queries
        print(f"Using a fixed number of queries: {num_queries}")
    else:
        num_queries = 98
    
    wrapper = OnnxWrapper(
        model,
        num_queries=num_queries,
        fixed_query_points=fixed_query_points,
        mask_queries=mask_queries,
        dynamic_num_queries=is_dynamic_queries,
    )
    wrapper.eval()

    if mask_queries:
        print(f"Masking {len(mask_queries)} query indices: {mask_queries}")
    
    print("Creating dummy inputs...")
    image = torch.randn(batch_size, 3, image_size, image_size, device=device)
    query_points = torch.randn(batch_size, num_queries, 3, device=device)
    gating_cutoff = torch.tensor(0.05, device=device)
    gating_radius = torch.tensor(1.0, device=device)
    prefill_hidden_state = torch.randn(batch_size, num_queries, gru_hidden_dim, device=device)
    prefill_starting_landmarks = torch.randn(
        batch_size,
        num_queries,
        NUM_PREDS_PARAMS,
        device=device,
    )
    
    input_names = [
        "image",
        "query_points",
        "gating_cutoff",
        "gating_radius",
        "prefill_hidden_state",
        "prefill_starting_landmarks"
    ]
    output_names = ["predictions", "hidden_state"]

    # Dynamic Axes configuration for correct dynamic shapes
    dynamic_axes = {
        "image": {},
        "query_points": {1: num_queries_dim},
        "gating_cutoff": {},
        "gating_radius": {},
        "prefill_hidden_state": {1: num_queries_dim},
        "prefill_starting_landmarks": {1: num_queries_dim},
    }

    # With fixed queries, all query-dependent shapes become static.
    if fixed_query_points is not None or fixed_num_queries is not None:
        for key in ("query_points", "prefill_hidden_state", "prefill_starting_landmarks"):
            dynamic_axes[key] = {}
        if fixed_query_points is not None:
            # The query_points input is unused in the exported graph and gets
            # pruned afterwards, but the dynamic_shapes dict must still cover
            # every positional argument of the wrapper's forward.
            dynamic_axes["query_points"] = {}
    
    print("Running export with dynamic axes...")
    try:
        torch.onnx.export(
            wrapper,
            (image, query_points, gating_cutoff, gating_radius, prefill_hidden_state, prefill_starting_landmarks),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_axes,
            dynamo=True,
            external_data=False,
            opset_version=opset_version,
            optimize=True,
            report=False
        )

        if fixed_query_points is not None:
            _prune_unused_inputs(output_path, input_names)

        if fp16:
            _convert_export_to_fp16(output_path, keep_io_fp32=keep_io_fp32)

        print(f"Success! Model exported to {output_path}")
    except Exception as e:
        print(f"Export failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="detector.onnx", help="Output path")
    parser.add_argument("--checkpoint", default=None, help="Path to trained model checkpoint")
    parser.add_argument(
        "--opset-version",
        type=int,
        default=18,
        help=(
            "Target ONNX opset version. Kept <20 by default: opset>=20 renames GridSample's "
            "mode attribute values (bilinear/nearest/bicubic -> linear/nearest/cubic), and "
            "onnxruntime 1.24.3's CUDA EP has no kernel for the renamed variant, forcing all "
            "GridSample ops (and cascading neighbors) onto CPU (16 Memcpy nodes instead of 1, "
            "~23%% slower measured on wrm15opt.onnx)."
        ),
    )
    parser.add_argument(
        "--fixed-num-queries",
        type=int,
        default=None,
        help="Number of queries to use for the exported model (default: dynamic)",
    )
    parser.add_argument(
        "--fixed-queries",
        type=str,
        default=None,
        help="Path to a .npy/.json file shape (N, 3) containing fixed query points to use for the exported model (default: dynamic)",
    )
    parser.add_argument(
        "--mask-queries",
        type=int,
        nargs="+",
        default=None,
        help="List of query indices to mask (prevent cross landmark communication) in the exported model (default: None)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Convert the exported ONNX model to FP16 while keeping FP32 I/O by default",
    )
    parser.add_argument(
        "--io-fp16",
        action="store_true",
        help="Convert ONNX model inputs and outputs to FP16 as well",
    )
    args = parser.parse_args()
    export_onnx(
        args.output,
        args.checkpoint,
        args.opset_version,
        fp16=args.fp16,
        keep_io_fp32=not args.io_fp16,
        fixed_num_queries=args.fixed_num_queries,
        fixed_queries=args.fixed_queries,
        mask_queries=args.mask_queries,
    )
