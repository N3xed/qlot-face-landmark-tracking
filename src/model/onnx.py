import torch
import numpy as np
import onnxruntime as ort
from .utils import LandmarkPrediction, NUM_PREDS_PARAMS
from .cov import Cov


class _CudaGraphState:
    """
    Pinned IOBinding state for CUDA graph capture/replay on a dedicated
    ONNX Runtime session.

    ONNX Runtime's CUDA graph replay requires every bound input/output to keep
    the exact same memory address across calls, and the same tensor shapes
    throughout (see "Using CUDA Graphs" in the ONNX Runtime CUDA execution
    provider docs). ORT also documents a "multi-graph capture" mechanism using
    a `gpu_graph_id` to cache several captured graphs within a single session,
    which looked like the natural fit for one graph per `num_queries` value --
    but empirically (see `OnnxQLOT`) that corrupted results whenever the
    different graph ids used different tensor shapes, apparently because the
    session's single CUDA memory arena can reuse/overlap intermediate-buffer
    addresses across captures. Giving each shape its own session (own arena)
    avoids that entirely, which is why this class assumes it owns its
    session's one (default) captured graph exclusively.
    """

    def __init__(self, session: ort.InferenceSession, input_arrays: dict[str, np.ndarray]) -> None:
        self.io_binding = session.io_binding()
        self.input_ortvalues: dict[str, ort.OrtValue] = {}
        for name, arr in input_arrays.items():
            # ortvalue_from_numpy (host->device) tolerates non-contiguous arrays
            # via a strided copy, but we normalize to C-contiguous here anyway so
            # the pinned buffers match what `run()` will feed in later (see `run`).
            arr = np.ascontiguousarray(arr)
            ort_value = ort.OrtValue.ortvalue_from_numpy(arr, "cuda", 0)
            self.input_ortvalues[name] = ort_value
            self.io_binding.bind_ortvalue_input(name, ort_value)
        for out in session.get_outputs():
            self.io_binding.bind_output(out.name, "cuda", 0)

        # Prime + capture: the first call captures the CUDA graph and
        # materializes the auto-allocated output buffers at whatever address
        # ORT's arena happens to pick. Re-bind those exact OrtValues so the
        # address stays pinned for every future replay (required by CUDA
        # graphs). A second call exercises those now-fixed addresses before we
        # start replaying for real.
        session.run_with_iobinding(self.io_binding)
        output_names = [o.name for o in session.get_outputs()]
        for name, bound_output in zip(output_names, self.io_binding.get_outputs()):
            self.io_binding.bind_ortvalue_output(name, bound_output)
        session.run_with_iobinding(self.io_binding)

    def run(self, session: ort.InferenceSession, input_arrays: dict[str, np.ndarray]) -> list[np.ndarray]:
        # Refresh the pinned input buffers' contents in place (same address,
        # new data) instead of rebinding, then replay the captured graph.
        #
        # CRITICAL: every array MUST be C-contiguous. OrtValue.update_inplace
        # on the CUDA path does a raw cudaMemcpy of the destination tensor's
        # contiguous byte size starting at the source's data pointer -- it does
        # NOT honor numpy strides. Feeding a non-contiguous array (e.g. an
        # HWC->CHW permuted image tensor, which is the common case in demo.py)
        # silently copies the wrong memory layout, producing garbled inputs and
        # landmarks that no longer track the face. The CPU update_inplace path
        # happens to handle strides correctly, which is why this only breaks
        # when use_cuda_graph=True. np.ascontiguousarray is a no-op when the
        # array is already contiguous, so this is free in the steady state.
        for name, arr in input_arrays.items():
            arr = np.ascontiguousarray(arr)
            self.input_ortvalues[name].update_inplace(arr)
        session.run_with_iobinding(self.io_binding)
        return [o.numpy() for o in self.io_binding.get_outputs()]


class OnnxQLOT:
    def __init__(
        self,
        model_path: str,
        providers: list | None = None,
        use_cuda_graph: bool = False,
    ):
        """
        Wrapper for the exported ONNX model of ContinuousLandmarkDetector.
        matches the interface of the PyTorch model for inference.

        Args:
            model_path: Path to the exported ONNX model. Needed (rather than a
                pre-built session) because `use_cuda_graph=True` creates
                additional ONNX Runtime sessions on demand -- see below.
            providers: ONNX Runtime execution providers to use (as accepted by
                `onnxruntime.InferenceSession`'s `providers` argument).
                Defaults to CUDA (if available) falling back to CPU. Do not
                bake `enable_cuda_graph` into these yourself; pass
                `use_cuda_graph=True` instead.
            use_cuda_graph: Whether to use CUDA graph capture/replay for
                inference. Requires `providers` to include
                "CUDAExecutionProvider"; if CUDA isn't available/requested this
                is a no-op and the regular `session.run()` path is used.

                A separate ONNX Runtime session -- with its own CUDA memory
                arena and its own captured CUDA graph -- is created lazily per
                distinct `num_queries` value seen (i.e. once per landmark
                topology / query-point set), the first time it's used. This
                naturally happens on the first frame after a new query-point
                set is selected, since that's exactly when `num_queries`
                changes. Subsequent calls with a previously-seen `num_queries`
                reuse that session and replay its captured graph instead of
                re-capturing, which is both correct (a captured graph's
                structure only depends on shapes, not values) and fast.

                NOTE: one session (and its captured graph's device memory) is
                kept alive per distinct `num_queries` value for the lifetime of
                this object -- fine for the handful of fixed landmark
                topologies this project uses, but don't use this for workloads
                with many/unbounded distinct query-point counts.
        """
        self.model_path = model_path
        self.use_cuda_graph = use_cuda_graph

        if providers is None:
            available = ort.get_available_providers()
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available
                else ["CPUExecutionProvider"]
            )
        self._providers = providers
        self._cuda_available = any(self._provider_name(p) == "CUDAExecutionProvider" for p in providers)

        self._sess_options = ort.SessionOptions()
        self._sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # One session + captured graph per distinct num_queries value.
        self._graph_sessions: dict[int, ort.InferenceSession] = {}
        self._cuda_graph_states: dict[int, _CudaGraphState] = {}

        use_cuda_graph_sessions = self.use_cuda_graph and self._cuda_available
        # Plain (non-graph-capture) session, used whenever use_cuda_graph is
        # False or CUDA isn't available. Skipped when every call will go
        # through a per-num_queries graph session instead, to avoid loading
        # the model into (GPU) memory a redundant extra time.
        self.session = (
            None
            if use_cuda_graph_sessions
            else ort.InferenceSession(self.model_path, self._sess_options, providers=self._providers)
        )

        # Introspect the graph's inputs once: models exported with fixed query
        # points have no "query_points" input (the points are baked into the
        # graph), in which case __call__ ignores its query_points argument.
        meta_session = self.session
        if meta_session is None:
            # The CUDA-graph path creates sessions lazily, so probe the model
            # metadata with a throwaway CPU session instead.
            meta_session = ort.InferenceSession(self.model_path, self._sess_options, providers=["CPUExecutionProvider"])
        self._input_names = {i.name for i in meta_session.get_inputs()}
        self._supports_query_points = "query_points" in self._input_names
        # Number of queries baked into the graph, if statically known.
        self._fixed_num_queries = None if self._supports_query_points else self._infer_fixed_num_queries(meta_session)

        # Try loading init_hidden_state from graph initializers if present
        self.init_hidden_state: np.ndarray | None = None
        try:
            import onnx
            from onnx import numpy_helper
            onnx_model = onnx.load(self.model_path, load_external_data=False)
            for init in onnx_model.graph.initializer:
                if init.name == "init_hidden_state":
                    self.init_hidden_state = numpy_helper.to_array(init).astype(np.float32)
                    break
        except Exception:
            pass

        # Determine gru_hidden_dim from initial state or session inputs if available
        self.gru_hidden_dim = 128
        if self.init_hidden_state is not None:
            self.gru_hidden_dim = self.init_hidden_state.shape[-1]
        else:
            for inp in meta_session.get_inputs():
                if inp.name in ("hidden_state", "prefill_hidden_state") and len(inp.shape) >= 3 and isinstance(inp.shape[-1], int):
                    self.gru_hidden_dim = inp.shape[-1]
                    break

        if meta_session is not self.session:
            del meta_session

        # ONNX export is hardcoded to static batch size of 1,
        # note that the batch size cannot be easily made dynamic
        # due to ONNX Conv2D limitations.
        self.batch_size = 1

    @staticmethod
    def _provider_name(entry) -> str:
        return entry[0] if isinstance(entry, tuple) else entry

    @staticmethod
    def _infer_fixed_num_queries(session: ort.InferenceSession) -> int | None:
        """Determine the baked-in num_queries of a fixed-queries model from its static tensor shapes."""
        nodes = list(session.get_inputs()) + list(session.get_outputs())
        for node in nodes:
            if node.name in ("prefill_hidden_state", "prefill_starting_landmarks", "predictions", "hidden_state"):
                dim = node.shape[1]
                if isinstance(dim, int) and dim > 0:
                    return dim
        return None

    def _graph_providers(self) -> list:
        """Same providers as requested, with enable_cuda_graph=1 added to the CUDA entry."""
        result = []
        for p in self._providers:
            name = self._provider_name(p)
            if name == "CUDAExecutionProvider":
                options = dict(p[1]) if isinstance(p, tuple) else {}
                options["enable_cuda_graph"] = "1"
                result.append((name, options))
            else:
                result.append(p)
        return result

    def _get_graph_session(self, num_queries: int) -> ort.InferenceSession:
        session = self._graph_sessions.get(num_queries)
        if session is None:
            session = ort.InferenceSession(self.model_path, self._sess_options, providers=self._graph_providers())
            self._graph_sessions[num_queries] = session
        return session
    
    def __call__(
        self,
        image: torch.Tensor,
        query_points: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        predictions: LandmarkPrediction | torch.Tensor | None = None,
        iterations: int = 1, # Ignored (baked into exported model, usually 1)
        return_sequence: bool = False, # Ignored (exported model returns only final)
        return_hidden_state: bool = False,
        return_tensor_predictions: bool = False,
        detach_updates: bool = True, # Ignored
        landmarks_to_mask: torch.Tensor | None = None, # Ignored
        **kwargs,
    ):
        """
        Forward pass using ONNX Runtime.
        
        Note: Many arguments are ignored because they are static in the exported ONNX graph 
        (e.g. iterations, return_sequence, etc).
        
        Args:
            image: (1, 3, H, W) input RGB image tensor.
            query_points: (1, num_queries, 3) tensor of query points with (x, y, z) coordinates.
                Ignored if the loaded model was exported with fixed query points baked
                into the graph (i.e. it has no "query_points" input); the number of
                predicted landmarks is then determined by the model, not this argument.
            hidden_state: Either None, in which case it will be initialized using
                the model's learned initial hidden state (or zeros if unavailable),
                or a (1, num_queries, self.gru_hidden_dim) tensor from the previous frame.
            predictions: Either None, in which case it will be initialized with the query
                points (x, -y) and zero covariance/delta in normalized coordinates,
                or a LandmarkPrediction / tensor containing normalized predictions from
                the previous frame.
            iterations: Number of iterations to run on the same image. Ignored for ONNX
                model (usually 1).
            return_sequence: Whether to return the predictions of all iterations.
                Ignored for ONNX model (always returns final prediction).
            return_hidden_state: Whether to return the hidden state for the next frame.
            return_tensor_predictions: Whether to return raw tensor predictions instead of
                LandmarkPrediction objects. If True, returns a
                (1, num_queries, NUM_PREDS_PARAMS) tensor.
            detach_updates: Ignored for ONNX model.
            landmarks_to_mask: Ignored for ONNX model.
        """
        device = image.device
        batch_size, _, image_height, image_width = image.shape
        assert batch_size == self.batch_size

        # Note that this is also the number of landmarks the model will predict.
        if self._supports_query_points:
            _, num_queries, _ = query_points.shape
        else:
            # The query points are baked into the exported graph, so the
            # argument is ignored; use the model's fixed count if it is
            # statically known (fall back to the argument's count otherwise).
            num_queries = self._fixed_num_queries or query_points.shape[1]

        # --- 1. Prepare Inputs for ONNX ---
        onnx_inputs = self._build_onnx_inputs(
            image=image,
            query_points=query_points,
            hidden_state=hidden_state,
            predictions=predictions,
            batch_size=batch_size,
            num_queries=num_queries,
            image_height=image_height,
            image_width=image_width,
        )

        # --- 2. Run Inference ---
        if self.use_cuda_graph and self._cuda_available:
            # One dedicated session + captured graph per distinct num_queries
            # value. A cache miss here means either the very first call, or
            # the first call after a new query-point set (different landmark
            # count) was selected -- exactly when we want a (re-)capture.
            session = self._get_graph_session(num_queries)
            graph_state = self._cuda_graph_states.get(num_queries)
            if graph_state is None:
                graph_state = _CudaGraphState(session, onnx_inputs)
                self._cuda_graph_states[num_queries] = graph_state
            pred_np, hidden_np = graph_state.run(session, onnx_inputs)
        else:
            # Outputs: ["predictions", "hidden_state"]
            assert self.session is not None
            pred_np, hidden_np = self.session.run(["predictions", "hidden_state"], onnx_inputs)
        
        # --- 3. Process Outputs ---
        pred_tensor = torch.from_numpy(pred_np).to(device)
        hidden_tensor = torch.from_numpy(hidden_np).to(device)

        cov_xy_scale = (2.0 / image_width, 2.0 / image_height)
        
        outputs = []
        if return_tensor_predictions:
            final_pred = pred_tensor
        else:
            final_pred = LandmarkPrediction.from_tensor(pred_tensor, cov_type=Cov)
            final_pred.cov.xy_scale = cov_xy_scale
        
        outputs.append(final_pred)
        
        if return_hidden_state:
            outputs.append(hidden_tensor)
            
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def _build_onnx_inputs(
        self,
        image: torch.Tensor,
        query_points: torch.Tensor,
        hidden_state: torch.Tensor | None,
        predictions: LandmarkPrediction | torch.Tensor | None,
        batch_size: int,
        num_queries: int,
        image_height: int,
        image_width: int,
    ) -> dict[str, np.ndarray]:
        """Builds the numpy input dict expected by the exported ONNX graph."""
        # Image: (self.batch_size, 3, H, W)
        image_np = np.ascontiguousarray(image.detach().cpu().numpy())

        # Query Points: (self.batch_size, num_queries, 3). Skipped for models
        # exported with fixed (baked-in) query points.
        query_points_np = np.ascontiguousarray(query_points.detach().cpu().numpy()) if self._supports_query_points else None

        # Hidden State: (self.batch_size, num_queries, Hidden)
        if hidden_state is None:
            if self.init_hidden_state is not None:
                init_h = self.init_hidden_state.reshape(1, 1, -1)
                hidden_state_np = np.repeat(np.repeat(init_h, batch_size, axis=0), num_queries, axis=1)
            else:
                hidden_state_np = np.zeros(
                    (batch_size, num_queries, self.gru_hidden_dim),
                    dtype=np.float32
                )
        else:
            hidden_state_np = hidden_state.detach().cpu().to(torch.float32).numpy()
        hidden_state_np = np.ascontiguousarray(hidden_state_np, dtype=np.float32)

        # Predictions: (self.batch_size, num_queries, NUM_PREDS_PARAMS)
        if predictions is None:
            predictions_np = np.zeros((batch_size, num_queries, NUM_PREDS_PARAMS), dtype=np.float32)
            if query_points is not None:
                qp = query_points.detach().cpu().numpy()
                predictions_np[..., 0] = qp[..., 0]
                predictions_np[..., 1] = -qp[..., 1]
        elif isinstance(predictions, LandmarkPrediction):
            predictions_np = predictions.to_tensor().detach().cpu().to(torch.float32).numpy()
        elif isinstance(predictions, torch.Tensor):
            predictions_np = predictions.detach().cpu().to(torch.float32).numpy()
        else:
            predictions_np = np.asarray(predictions, dtype=np.float32)
        predictions_np = np.ascontiguousarray(predictions_np, dtype=np.float32)

        h_name = "prefill_hidden_state" if "prefill_hidden_state" in self._input_names else ("hidden_state" if "hidden_state" in self._input_names else "prefill_hidden_state")
        p_name = "prefill_starting_landmarks" if "prefill_starting_landmarks" in self._input_names else ("predictions" if "predictions" in self._input_names else "prefill_starting_landmarks")
        inputs = {
            "image": image_np,
            h_name: hidden_state_np,
            p_name: predictions_np,
        }
        if "gating_cutoff" in self._input_names:
            inputs["gating_cutoff"] = np.array([0.05], dtype=np.float32)
        if "gating_radius" in self._input_names:
            inputs["gating_radius"] = np.array([1.0], dtype=np.float32)
        if self._supports_query_points:
            assert query_points_np is not None
            inputs["query_points"] = query_points_np
        return inputs

    # Alias forward to __call__
    forward = __call__
