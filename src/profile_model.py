#!/usr/bin/env python3
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import torch
import torch.nn as nn

# Try importing dependencies
try:
    from fvcore.nn import FlopCountAnalysis
    from torchinfo import summary
except ImportError:
    print("Error: Missing dependencies. Please run:")
    print("uv pip install fvcore torchinfo")
    sys.exit(1)

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# Project imports
try:
    # Add src to path if running from root
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"

    # Ensure src is in path so we can import 'model' and 'utils' as top-level packages
    if str(src_dir) not in sys.path:
        sys.path.append(str(src_dir))

    from model import QLOT
    from utils.torch.misc import Config, load
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please run this script from the project root directory as a module:")
    print("  uv run python -m src.profile <model_path>")
    sys.exit(1)
    print(f"Error importing project modules: {e}")
    print("Please run this script from the project root directory.")
    sys.exit(1)


# --- Terminal Styling ---
class Style:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_box_header(title: str, width: int = 78):
    print(f"┌{'─' * (width - 2)}┐")
    print(f"│{title.center(width - 2)}│")
    print(f"├{'─' * (width - 2)}┤")


def print_section(title: str, width: int = 78):
    print(f"├{'─' * (width - 2)}┤")
    print(f"│ {Style.CYAN}{title.ljust(width - 4)}{Style.ENDC} │")
    print(f"├{'─' * (width - 2)}┤")


def print_row(label: str, value: str, width: int = 78, color: str = ""):
    label_len = len(label)
    val_len = len(value)
    # Ensure we don't have negative padding
    padding = max(0, width - 4 - label_len - val_len)

    # Construct the line without the end ANSI code first to measure length if needed,
    # but here we rely on padding.
    # We apply color to the value only.
    # The ENDC is applied after value.
    # Note: If value is empty, we still print spaces.

    line = f"│ {label}{' ' * padding}{color}{value}{Style.ENDC} │"
    print(line)


def print_row_3(col1: str, col2: str, col3: str, width: int = 78):
    # Split into rough thirds
    w1 = int(width * 0.2)
    w2 = int(width * 0.4)
    w3 = width - 4 - w1 - w2
    print(f"│ {col1.ljust(w1)}{col2.ljust(w2)}{col3.rjust(w3)} │")


def print_footer(width: int = 78):
    print(f"└{'─' * (width - 2)}┘")


def format_flops(flops: float) -> str:
    if flops >= 1e9:
        return f"{flops / 1e9:.2f} G"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} M"
    else:
        return f"{flops:.2f}"


def format_params(params: int) -> str:
    if params >= 1e6:
        return f"{params / 1e6:.2f} M"
    elif params >= 1e3:
        return f"{params / 1e3:.2f} K"
    else:
        return str(params)


def format_bytes(size: int) -> str:
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GB"
    elif size >= 1024**2:
        return f"{size / 1024**2:.2f} MB"
    elif size >= 1024:
        return f"{size / 1024:.2f} KB"
    else:
        return f"{size} B"


# --- Profilers ---


class PyTorchProfiler:
    def __init__(self, model_path: str, image_shape: Tuple[int, ...], num_queries: int, device: str = "cpu"):
        self.model_path = Path(model_path)
        self.image_shape = image_shape
        self.num_queries = num_queries
        self.device = torch.device(device)
        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        # Instantiate model
        # We assume standard initialization. If the checkpoint has config, we could use it,
        # but the prompt implies inferring from files/defaults.
        # ContinuousLandmarkDetector usually takes feature_extractor_pretrained argument.
        # We'll set it to False to avoid downloading weights during profiling if possible,
        # but strictly we should match the checkpoint.

        # Try to load checkpoint to peek at args if possible, or just default.
        model = QLOT(feature_extractor_pretrained=False)

        if self.model_path.exists():
            try:
                # Use project's load utility or standard torch load
                checkpoint = torch.load(self.model_path, map_location=self.device)

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if "model" in checkpoint:
                        state_dict = checkpoint["model"]
                    elif "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    else:
                        state_dict = checkpoint
                else:
                    state_dict = checkpoint

                # Clean up state dict (remove 'module.' prefix if DDP)
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

                # Flexible loading
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"{Style.WARNING}Warning: Could not load weights fully: {e}{Style.ENDC}")

        model.to(self.device)
        model.eval()
        return model

    def profile(self, print_summary: bool = False, iterations: int = 1) -> None:
        B, C, H, W = self.image_shape

        # Create dummy inputs
        images = torch.randn(B, C, H, W, device=self.device)
        queries = torch.randn(B, self.num_queries, 3, device=self.device)

        # 1. FLOPs Analysis using fvcore
        # We need to wrap the forward call because fvcore expects a module and args
        # The model signature is: forward(images, queries, iterations=3, ...)
        # We want to profile a standard forward pass.

        inputs = (images, queries)
        # Note: We fix iterations to a standard value (e.g. 3) or pass it in kwargs if supported.
        # FlopCountAnalysis supports args but kwargs support depends on version.
        # Easier to wrap in a lambda-like module or just pass defaults.

        class Wrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, img, q):
                # Return tensor predictions to satisfy fvcore/JIT
                return self.model(
                    img,
                    q,
                    iterations=iterations,
                    return_sequence=False,
                    return_tensor_predictions=True,
                    force_cov_gating=True,
                    use_naive_correlation=True,
                )

        wrapper = Wrapper(self.model)
        flops_counter = FlopCountAnalysis(wrapper, inputs)

        # Filter warnings from fvcore
        flops_counter.unsupported_ops_warnings(False)

        total_flops = flops_counter.total()  # This is actually MACs usually in fvcore
        flops_by_module = flops_counter.by_module()

        # 2. Parameter Analysis
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # 3. Component Breakdown
        # We define specific module paths to categorize FLOPs/Params

        components = {
            "Backbone": ["model.feature_extractor.backbone", "model.feature_extractor.global_proj"],
            "FPN": ["model.feature_extractor.projections", "model.feature_extractor.fusion_conv"],
            "Correlation": ["model.image_feature_correlator"],  # Includes query_encoder (we'll subtract later)
            "Update Predictor": ["model.update_predictor"],
            "Feature Encoder": ["model.encoder"],
            "Query Encoding": [
                "model.encoder.query_encoder",
                "model.image_feature_correlator.query_encoder",
                "model.image_feature_correlator.query_encoder_p",
            ],
        }

        comp_flops = {k: 0 for k in components}
        comp_params = {k: 0 for k in components}

        # Aggregate FLOPs
        for comp, prefixes in components.items():
            for prefix in prefixes:
                comp_flops[comp] += flops_by_module.get(prefix, 0)

        # Fix overlaps for FLOPs (Correlation includes Query Encoder)
        q_enc_flops = flops_by_module.get("model.image_feature_correlator.query_encoder", 0)
        if q_enc_flops > 0:
            comp_flops["Correlation"] = max(0, comp_flops["Correlation"] - q_enc_flops)
        q_enc_flops = flops_by_module.get("model.image_feature_correlator.query_encoder_p", 0)
        if q_enc_flops > 0:
            comp_flops["Correlation"] = max(0, comp_flops["Correlation"] - q_enc_flops)
        q_enc_flops_2 = flops_by_module.get("model.encoder.query_encoder", 0)
        if q_enc_flops_2 > 0:
            comp_flops["Feature Encoder"] = max(0, comp_flops["Feature Encoder"] - q_enc_flops_2)

        # Aggregate Params
        # Note: We need to use the same logic (finding modules by name)
        # But 'model.feature_extractor.backbone' corresponds to self.model.feature_extractor.backbone

        module_dict = dict(self.model.named_modules())
        # Add "model." prefix to keys to match our component list
        module_dict = {f"model.{k}" if k else "model": v for k, v in module_dict.items()}

        for comp, prefixes in components.items():
            for prefix in prefixes:
                if prefix in module_dict:
                    mod = module_dict[prefix]
                    # We use recurse=True to get all params in this module
                    comp_params[comp] += sum(p.numel() for p in mod.parameters())

        # Fix overlaps for Params
        if "model.image_feature_correlator.query_encoder" in module_dict:
            q_enc_params = sum(p.numel() for p in module_dict["model.image_feature_correlator.query_encoder"].parameters())
            comp_params["Correlation"] = max(0, comp_params["Correlation"] - q_enc_params)
        if "model.image_feature_correlator.query_encoder_p" in module_dict:
            q_enc_params = sum(p.numel() for p in module_dict["model.image_feature_correlator.query_encoder_p"].parameters())
            comp_params["Correlation"] = max(0, comp_params["Correlation"] - q_enc_params)
        if "model.encoder.query_encoder" in module_dict:
            q_enc_params_2 = sum(p.numel() for p in module_dict["model.encoder.query_encoder"].parameters())
            comp_params["Query Encoding"] = max(0, comp_params["Query Encoding"] - q_enc_params_2)

        # Fill in 'Other'
        known_flops = sum(comp_flops.values())
        comp_flops["Other"] = max(0, total_flops - known_flops)

        # For params, we might double count if we aren't careful, but here we sum disjoint sets + fixed overlap.
        # But wait, named_modules() includes parents and children.
        # If I sum backbone params, then sum feature_extractor params, I double count.
        # My components list is designed to be disjoint (Backbone, Projections, Correlation, Update, QueryEnc).
        # So summing them should be fine.

        known_params = sum(comp_params.values())
        comp_params["Other"] = max(0, total_params - known_params)

        # 4. Bandwidth Estimation
        # Sizes in bytes (FP32 = 4 bytes)
        input_size = int(np.prod(self.image_shape) * 4)
        # Backbone output: 128 channels, H/32, W/32 (approx for HGNet stride 32) -> 7x7 at 224
        # Based on ARCHITECTURE.md, feature extractor output is [B, 128, 56, 56] (stride 4)
        # and there's a pooling to global context.
        # Actually checking code:
        # feature_extractor outputs [B, 128, 56, 56]
        feat_size = int(B * 128 * (H // 4) * (W // 4) * 4)

        # Correlation: [B, Q, Heads, H_feat, W_feat]
        # Q=98, Heads=4, H=56, W=56
        corr_size = int(B * self.num_queries * 4 * (H // 4) * (W // 4) * 4)

        # GRU State: [B, Q, 128]
        gru_size = int(B * self.num_queries * 128 * 4)

        # Simple bandwidth model: Read Input + Write Feats + Read Feats + Write Corr + Read Corr + GRU RW
        # This is a lower bound estimate per frame
        est_bw_bytes = (
            input_size  # Read Image
            + feat_size * 2  # Write + Read Features
            + corr_size * 2  # Write + Read Correlation Maps (expensive!)
            + gru_size * 2 * 3  # GRU Read/Write per iteration (x3)
        )

        # --- Printing ---
        print_box_header("FACE LANDMARK DETECTOR PROFILE REPORT")
        print_row(
            "Source", f"{self.model_path.name} ({format_bytes(self.model_path.stat().st_size) if self.model_path.exists() else 'N/A'})"
        )
        print_row("Iterations", str(iterations))
        print_row("Type", "PyTorch (FP32)")
        print_row("Input Shape", f"{list(self.image_shape)}")
        print_row("Query Points", f"{self.num_queries} landmarks")

        print_section("COMPUTE COMPLEXITY")
        print_row("Total MACs", f"{format_flops(total_flops)}", color=Style.GREEN)
        print_row("Total FLOPS", f"{format_flops(total_flops * 2)} (approx)", color=Style.GREEN)

        print(f"│ {' ' * 74} │")
        print_row("Per Component", "")
        sorted_comps = sorted(comp_flops.items(), key=lambda x: x[1], reverse=True)
        for name, flop in sorted_comps:
            if flop > 0:
                pct = (flop / total_flops) * 100
                print_row(f"  {name}", f"{format_flops(flop)} ({pct:.1f}%)")

        print_section("PARAMETER COUNT")
        print_row("Total Parameters", f"{format_params(total_params)}", color=Style.GREEN)
        print_row("Model Size (FP32)", f"{format_bytes(int(total_params * 4))}")

        print(f"│ {' ' * 74} │")
        print_row("Per Component", "")
        sorted_params = sorted(comp_params.items(), key=lambda x: x[1], reverse=True)
        for name, param in sorted_params:
            if param > 0:
                pct = (param / total_params) * 100
                print_row(f"  {name}", f"{format_params(param)} ({pct:.1f}%)")
        print_footer()

        if print_summary:
            import torchinfo
            print(torchinfo.summary(
                self.model,
                input_size=[(1, 3, 224, 224), (1, 98, 3)],
                col_names=("input_size", "output_size", "num_params", "trainable"),
                row_settings=("var_names", "depth"),
                return_tensor_predictions=True,
                iterations=1,
            ))


class ONNXProfiler:
    OP_CATEGORIES = {
        "compute": {
            "Conv",
            "ConvInteger",
            "MatMul",
            "Gemm",
            "FusedConv",
            "BatchMatMul",
            "QLinearConv",
            "QLinearMatMul",
        },
        "normalization": {
            "LayerNormalization",
            "BatchNormalization",
            "InstanceNormalization",
            "GroupNormalization",
            "LpNormalization",
        },
        "activation": {
            "Relu",
            "Gelu",
            "FastGelu",
            "Sigmoid",
            "Tanh",
            "Swish",
            "Silu",
            "LeakyRelu",
            "HardSigmoid",
            "HardSwish",
            "Softmax",
            "LogSoftmax",
        },
        "reduction": {
            "ReduceMean",
            "ReduceSum",
            "ReduceMax",
            "ReduceMin",
            "ArgMax",
            "ArgMin",
            "GlobalAveragePool",
            "AveragePool",
            "MaxPool",
        },
        "layout": {
            "Transpose",
            "Reshape",
            "Flatten",
            "Squeeze",
            "Unsqueeze",
            "Expand",
            "Tile",
            "DepthToSpace",
            "SpaceToDepth",
        },
        "indexing": {
            "Gather",
            "GatherElements",
            "GatherND",
            "ScatterND",
            "Slice",
            "Pad",
        },
        "elementwise": {
            "Add",
            "Sub",
            "Mul",
            "Div",
            "Pow",
            "Neg",
            "Reciprocal",
            "Clip",
            "Where",
            "Equal",
            "Greater",
            "Less",
            "And",
            "Or",
            "Not",
            "Abs",
            "Sqrt",
            "Exp",
            "Log",
        },
        "shape": {
            "Shape",
            "Size",
            "Constant",
            "ConstantOfShape",
            "Cast",
            "Range",
            "NonZero",
            "Identity",
        },
        "join": {
            "Concat",
            "Split",
            "Compress",
        },
    }

    def __init__(
        self,
        model_path: str,
        warmup: int = 10,
        iterations: int = 100,
        num_queries: int = 98,
        detailed: bool = False,
        provider: str = "cpu",
        prefer_nhwc: bool = False,
        cuda_graph: bool = False,
        verbose: bool = False,
    ):
        if ort is None:
            raise ImportError("onnxruntime is not installed")
        self.model_path = model_path
        self.warmup = warmup
        self.iterations = iterations
        self.num_queries = num_queries
        self.detailed = detailed
        self.provider = provider.lower()
        self.prefer_nhwc = prefer_nhwc
        self.cuda_graph = cuda_graph
        self.providers = self._resolve_providers(self.provider, self.prefer_nhwc, self.cuda_graph)
        # Detailed per-op profiling uses plain session.run() (no IOBinding), which
        # cannot guarantee the fixed input/output addresses CUDA graph replay
        # requires. Use a separate provider list without enable_cuda_graph for that
        # session so it stays correct/meaningful; the main LATENCY TEST session
        # above keeps cuda_graph enabled.
        self.detail_providers = self._resolve_providers(self.provider, self.prefer_nhwc, cuda_graph=False)
        self.available_providers = ort.get_available_providers()

        # Load session
        self.sess_options = ort.SessionOptions()
        if verbose:
            self.sess_options.log_severity_level=1
        # self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self.sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # self.sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL # type: ignore
        # self.sess_options.optimized_model_filepath = "onnx_opt.onnx"
        # self.sess_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        self.session = ort.InferenceSession(self.model_path, self.sess_options, providers=self.providers)

    @staticmethod
    def _resolve_providers(provider: str, prefer_nhwc: bool = False, cuda_graph: bool = False) -> List[Any]:
        provider_aliases = {
            "cpu": "CPUExecutionProvider",
            "cuda": "CUDAExecutionProvider",
            "tensorrt": "TensorrtExecutionProvider",
        }
        available_providers = ort.get_available_providers()  # type: ignore
        if provider not in provider_aliases:
            valid = ", ".join(sorted(provider_aliases))
            raise ValueError(f"Unknown provider '{provider}'. Expected one of: {valid}")

        primary_provider = provider_aliases[provider]
        if primary_provider not in available_providers:
            available = ", ".join(available_providers) if available_providers else "none"
            raise ValueError(f"Requested provider '{primary_provider}' is not available. Available providers: {available}")

        providers: List[Any] = [primary_provider]
        if primary_provider == "CUDAExecutionProvider" and (prefer_nhwc or cuda_graph):
            options = {}
            if prefer_nhwc:
                options["prefer_nhwc"] = "1"
            if cuda_graph:
                options["enable_cuda_graph"] = "1"
            providers = [("CUDAExecutionProvider", options)]
        # if primary_provider != "CPUExecutionProvider" and "CPUExecutionProvider" in available_providers:
        #     providers.append("CPUExecutionProvider")
        return providers

    @classmethod
    def _categorize_op(cls, op_name: str) -> str:
        for category, ops in cls.OP_CATEGORIES.items():
            if op_name in ops:
                return category
        return "other"

    def _build_input_data(self) -> Tuple[Dict[str, np.ndarray], Optional[int]]:
        inputs = self.session.get_inputs()

        # Check inputs and determine shapes
        input_data = {}
        fixed_batch_size = None

        for inp in inputs:
            shape = []
            is_dynamic = False
            dtype = self._numpy_dtype_for_input(inp.type)
            for i, d in enumerate(inp.shape):
                if isinstance(d, str) or d is None or d < 0:
                    is_dynamic = True
                    dim_name = str(d).lower() if isinstance(d, str) else ""

                    if i == 0:
                        shape.append(1)
                    elif "quer" in dim_name:
                        shape.append(self.num_queries)
                    elif "batch" in dim_name:
                        shape.append(1)
                    else:
                        shape.append(224)
                else:
                    shape.append(d)
                    if i == 0:
                        if fixed_batch_size is not None and fixed_batch_size != d:
                            print(f"{Style.WARNING}Warning: Inconsistent fixed batch sizes?{Style.ENDC}")
                        fixed_batch_size = d

            if not shape:
                input_data[inp.name] = np.array(self._random_value(dtype), dtype=dtype)
            else:
                input_data[inp.name] = self._random_array(shape, dtype)
        return input_data, fixed_batch_size

    @staticmethod
    def _numpy_dtype_for_input(input_type: str) -> np.dtype:
        type_map = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(double)": np.float64,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(bool)": np.bool_,
        }
        return type_map.get(input_type, np.float32)

    @staticmethod
    def _random_value(dtype: np.dtype) -> Any:
        if np.issubdtype(dtype, np.floating):
            return np.random.randn()
        if np.issubdtype(dtype, np.integer):
            return np.random.randint(0, 10)
        if np.issubdtype(dtype, np.bool_):
            return True
        return np.random.randn()

    @classmethod
    def _random_array(cls, shape: List[int], dtype: np.dtype) -> np.ndarray:
        if np.issubdtype(dtype, np.floating):
            return np.random.randn(*shape).astype(dtype)
        if np.issubdtype(dtype, np.integer):
            return np.random.randint(0, 10, size=shape, dtype=dtype)
        if np.issubdtype(dtype, np.bool_):
            return np.ones(shape, dtype=dtype)
        return np.random.randn(*shape).astype(dtype)

    def _print_input_analysis(self, input_data: Dict[str, np.ndarray]) -> None:
        print_section("INPUT ANALYSIS")
        for inp in self.session.get_inputs():
            inferred = input_data[inp.name]
            shape_str = "Scalar (float32)" if inferred.shape == () else str(list(inferred.shape))
            dtype_str = str(inferred.dtype)
            if any(isinstance(d, str) or d is None or d < 0 for d in inp.shape):
                shape_str += f" (Dyn from {inp.shape})"
            print_row(inp.name, f"{shape_str} [{dtype_str}]")

    def _resize_batch(
        self, input_data: Dict[str, np.ndarray], batch_size: int, fixed_batch_size: Optional[int]
    ) -> Dict[str, np.ndarray]:
        current_inputs: Dict[str, np.ndarray] = {}
        for name, value in input_data.items():
            shape = list(value.shape)
            if len(shape) > 0 and shape[0] == 1 and fixed_batch_size is None:
                shape[0] = batch_size
                current_inputs[name] = self._random_array(shape, value.dtype)
            else:
                current_inputs[name] = value
        return current_inputs

    def _profile_detailed_graph(self, input_data: Dict[str, np.ndarray]) -> None:
        sess_options = ort.SessionOptions()  # type: ignore
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # type: ignore
        sess_options.enable_profiling = True
        sess_options.optimized_model_filepath = "onnx_opt.onnx"
        sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL # type: ignore
        sess_options.execution_order = ort.ExecutionOrder.PRIORITY_BASED  # type: ignore
        profiled_session = ort.InferenceSession(  # type: ignore
            self.model_path,
            sess_options,
            providers=self.detail_providers,
        )

        detailed_warmup_runs = max(1, self.warmup // 2)
        detailed_profile_runs = max(1, min(self.iterations, 10))
        for _ in range(detailed_warmup_runs):
            profiled_session.run(None, input_data)
        for _ in range(detailed_profile_runs):
            profiled_session.run(None, input_data)

        profile_path = profiled_session.end_profiling()

        with open(profile_path, "r", encoding="utf-8") as handle:
            profile_events = json.load(handle)

        node_events = []
        node_occurrences: Dict[str, int] = defaultdict(int)
        excluded_warmup_events = 0
        for event in profile_events:
            if event.get("cat") != "Node":
                continue
            duration_us = event.get("dur")
            if duration_us is None:
                continue

            args = event.get("args", {})
            op_name = args.get("op_name") or args.get("op") or "Unknown"
            provider = args.get("provider", "unknown")
            node_name = event.get("name") or args.get("node_name") or op_name
            occurrence = node_occurrences[node_name]
            node_occurrences[node_name] += 1
            if occurrence < detailed_warmup_runs:
                excluded_warmup_events += 1
                continue
            node_events.append(
                {
                    "node_name": node_name,
                    "op_name": op_name,
                    "provider": provider,
                    "duration_us": float(duration_us),
                    "category": self._categorize_op(op_name),
                }
            )

        total_duration_us = sum(event["duration_us"] for event in node_events)
        if total_duration_us <= 0:
            print_section("DETAILED GRAPH PROFILE")
            print_row("Status", "No node timing events captured", color=Style.WARNING)
            return

        by_node: Dict[str, float] = defaultdict(float)
        by_op: Dict[str, float] = defaultdict(float)
        by_category: Dict[str, float] = defaultdict(float)
        by_provider: Dict[str, float] = defaultdict(float)

        for event in node_events:
            by_node[event["node_name"]] += event["duration_us"]
            by_op[event["op_name"]] += event["duration_us"]
            by_category[event["category"]] += event["duration_us"]
            by_provider[event["provider"]] += event["duration_us"]

        def top_entries(values: Dict[str, float], limit: int = 50) -> List[Tuple[str, float]]:
            return sorted(values.items(), key=lambda item: item[1], reverse=True)[:limit]

        print_section("DETAILED GRAPH PROFILE")
        print_row("Profile File", profile_path)
        print_row("Warmup Runs Excluded", str(detailed_warmup_runs))
        print_row("Profiled Runs", str(detailed_profile_runs))
        print_row("Warmup Node Events Excluded", str(excluded_warmup_events))
        print_row("Node Events", str(len(node_events)))
        print_row("Accumulated Node Time", f"{total_duration_us / 1000.0:.2f} ms")

        print(f"│ {' ' * 74} │")
        print_row("Top Op Categories", "")
        for name, duration_us in top_entries(by_category, limit=8):
            pct = 100.0 * duration_us / total_duration_us
            print_row(f"  {name}", f"{duration_us / 1000.0:.2f} ms ({pct:.1f}%)")

        print(f"│ {' ' * 74} │")
        print_row("Top Op Types", "")
        for name, duration_us in top_entries(by_op):
            pct = 100.0 * duration_us / total_duration_us
            print_row(f"  {name}", f"{duration_us / 1000.0:.2f} ms ({pct:.1f}%)")

        print(f"│ {' ' * 74} │")
        print_row("Top Nodes", "")
        for name, duration_us in top_entries(by_node):
            pct = 100.0 * duration_us / total_duration_us
            print_row(f"  {name[:42]}", f"{duration_us / 1000.0:.2f} ms ({pct:.1f}%)")

        print(f"│ {' ' * 74} │")
        print_row("Execution Providers", "")
        for name, duration_us in top_entries(by_provider, limit=4):
            pct = 100.0 * duration_us / total_duration_us
            print_row(f"  {name}", f"{duration_us / 1000.0:.2f} ms ({pct:.1f}%)")

    @staticmethod
    def _provider_name(entry: Any) -> str:
        return entry[0] if isinstance(entry, tuple) else entry

    def _iobinding_device(self) -> Optional[str]:
        """
        Device type to use for IOBinding, or None to fall back to plain numpy
        `session.run()`. IOBinding keeps inputs/outputs resident on-device across
        calls, avoiding the implicit host<->device copy ORT otherwise performs on
        every `run()` call when the session isn't running on CPU.
        """
        primary = self._provider_name(self.providers[0])
        if primary == "CUDAExecutionProvider" or primary == "TensorrtExecutionProvider":
            return "cuda"
        return None

    def _run_timed_loop(self, current_inputs: Dict[str, np.ndarray]) -> float:
        """
        Run warmup + timed iterations for the given inputs, returning total
        elapsed wall time (seconds) for the timed portion. Uses IOBinding with
        device-resident buffers when supported by the requested provider so the
        timing reflects steady-state (already-on-device) inference rather than
        being dominated by per-call host<->device transfer overhead.
        """
        device_type = self._iobinding_device()

        if device_type is None:
            for _ in range(self.warmup):
                self.session.run(None, current_inputs)

            start = time.time()
            for _ in range(self.iterations):
                self.session.run(None, current_inputs)
            end = time.time()
            return end - start

        # Bind inputs/outputs once; reused (in-place) across every iteration so
        # no host<->device copy happens inside the timed loop.
        io_binding = self.session.io_binding()
        for inp in self.session.get_inputs():
            ort_value = ort.OrtValue.ortvalue_from_numpy(current_inputs[inp.name], device_type, 0)  # type: ignore
            io_binding.bind_ortvalue_input(inp.name, ort_value)
        for out in self.session.get_outputs():
            io_binding.bind_output(out.name, device_type, 0)

        remaining_warmup = self.warmup
        if self.cuda_graph:
            # CUDA graph replay requires every input/output to stay at a fixed
            # address across calls. The auto-allocating bind_output() above
            # doesn't guarantee that, so run once to let ORT materialize the
            # output buffers, then re-bind those exact OrtValues so their
            # addresses are pinned for every subsequent call. The first call
            # here also performs the CUDA graph capture itself; a second call
            # exercises the freshly-pinned addresses before we start replaying.
            self.session.run_with_iobinding(io_binding)
            output_names = [o.name for o in self.session.get_outputs()]
            for name, bound_output in zip(output_names, io_binding.get_outputs()):
                io_binding.bind_ortvalue_output(name, bound_output)
            self.session.run_with_iobinding(io_binding)
            remaining_warmup = max(0, self.warmup - 2)

        for _ in range(remaining_warmup):
            self.session.run_with_iobinding(io_binding)

        start = time.time_ns()
        for _ in range(self.iterations):
            self.session.run_with_iobinding(io_binding)
        end = time.time_ns()
        return (end - start) / 1.0e9  # Convert ns to seconds

    def profile(self):
        input_data, fixed_batch_size = self._build_input_data()

        provider_suffix = ""
        if self.prefer_nhwc:
            provider_suffix += " (NHWC)"
        if self.cuda_graph:
            provider_suffix += " (CUDA Graph)"

        print_box_header("ONNX BENCHMARK REPORT")
        print_row("Model", self.model_path)
        print_row("Requested Provider", self._provider_name(self.providers[0]) + provider_suffix)
        print_row("Session Providers", ", ".join(self.session.get_providers()))
        iobinding_device = self._iobinding_device()
        print_row("I/O Strategy", "IOBinding (device-resident)" if iobinding_device else "session.run (host numpy)")
        self._print_input_analysis(input_data)

        print_section("LATENCY TEST")
        print_row_3("Batch Size", "Latency (ms)", "Throughput (FPS)")

        batch_sizes = [1, 2, 4, 8, 16, 32]
        if fixed_batch_size is not None:
            print_row(f"Fixed Batch Size: {fixed_batch_size}", "", color=Style.WARNING)
            batch_sizes = [fixed_batch_size]

        for b in batch_sizes:
            current_inputs = self._resize_batch(input_data, b, fixed_batch_size)

            try:
                elapsed = self._run_timed_loop(current_inputs)

                avg_time = elapsed / self.iterations
                latency_ms = avg_time * 1000
                fps = b / avg_time

                print_row_3(str(b), f"{latency_ms:.2f} ms", f"{fps:.1f}")
            except Exception as e:
                print_row_3(str(b), "Error", str(e)[:20])

        if self.detailed:
            detailed_inputs = self._resize_batch(input_data, 1, fixed_batch_size)
            self._profile_detailed_graph(detailed_inputs)

        print_footer()


def main():
    parser = argparse.ArgumentParser(description="Profile Face Landmark Detector Model")
    parser.add_argument("models", type=str, nargs="+", help="Path(s) to .pth or .onnx model file(s)")
    parser.add_argument("--image-shape", type=int, nargs=4, default=[1, 3, 224, 224], help="Input image shape (B C H W)")
    parser.add_argument("--num-queries", type=int, default=98, help="Number of query points")
    parser.add_argument("--benchmark", action="store_true", help="Run inference benchmark (ONNX only)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--print-summary", action="store_true", help="Print detailed model summary (PyTorch only)")
    parser.add_argument("--detailed", action="store_true", help="Print detailed ONNX graph timing breakdown")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for ONNX Runtime")
    parser.add_argument(
        "--provider",
        choices=["cpu", "cuda", "tensorrt"],
        default="cpu",
        help="ONNX Runtime execution provider to benchmark",
    )
    parser.add_argument(
        "--nhwc",
        action="store_true",
        help="Prefer NHWC layout on the CUDA execution provider (prefer_nhwc=1)",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Enable CUDA graph capture on the CUDA execution provider (enable_cuda_graph=1)",
    )

    args = parser.parse_args()

    for model_path_str in args.models:
        model_path = Path(model_path_str)
        if not model_path.exists():
            print(f"Error: File {model_path} not found.")
            continue

        if model_path.suffix == ".onnx":
            if args.benchmark:
                profiler = ONNXProfiler(
                    str(model_path),
                    args.warmup,
                    args.iterations,
                    num_queries=args.num_queries,
                    detailed=args.detailed,
                    provider=args.provider,
                    prefer_nhwc=args.nhwc,
                    cuda_graph=args.cuda_graph,
                    verbose=args.verbose
                )
                profiler.profile()
            else:
                print("To benchmark ONNX model, use --benchmark flag.")
                profiler = ONNXProfiler(
                    str(model_path),
                    args.warmup,
                    args.iterations,
                    num_queries=args.num_queries,
                    detailed=args.detailed,
                    provider=args.provider,
                    prefer_nhwc=args.nhwc,
                    cuda_graph=args.cuda_graph,
                    verbose=args.verbose
                )
                profiler.profile()

        else:
            # Assume PyTorch
            profiler = PyTorchProfiler(str(model_path), tuple(args.image_shape), args.num_queries)
            profiler.profile(args.print_summary)

        print()  # Add spacing between reports


if __name__ == "__main__":
    main()
