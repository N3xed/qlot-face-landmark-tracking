import torch
import numpy as np
import onnxruntime as ort
from .utils import LandmarkPrediction, NUM_PREDS_PARAMS
from .cov import LowRankCov2D


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
        if meta_session is not self.session:
            del meta_session

        # Hardcoded based on model architecture
        self.gru_hidden_dim = 128 
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
        gating_radius: float | torch.Tensor = 1.0,
        gating_cutoff: float | torch.Tensor = 0.05,
        iterations: int = 1, # Ignored (baked into exported model, usually 1)
        return_sequence: bool = False, # Ignored (exported model returns only final)
        store_similarity_maps: bool = False, # Ignored
        prefill_hidden_state: torch.Tensor | None = None,
        prefill_starting_landmarks: LandmarkPrediction | None = None,
        return_hidden_state: bool = False,
        return_tensor_predictions: bool = False,
        detach_updates: bool = True, # Ignored
        force_cov_gating=True, # Ignored (exported model always uses cov gating)
        use_naive_correlation=True, # Ignored (exported model always uses naive correlation)
        landmarks_to_mask: torch.Tensor | None = None, # Ignored (not supported in exported model
    ):
        """
        Forward pass using ONNX Runtime.
        
        Note: Many arguments are ignored because they are static in the exported ONNX graph 
        (e.g. iterations, return_sequence, gating_radius etc).
        
        Args:
            image: (1, 3, 224, 224) input RGB image tensor.
            query_points: (1, num_queries, 3) tensor of query points with (x, y, z) coordinates.
                Ignored if the loaded model was exported with fixed query points baked
                into the graph (i.e. it has no "query_points" input); the number of
                predicted landmarks is then determined by the model, not this argument.
            gating_radius: scalar  (model also allows per landmark radius, but scalar is
                baked into exported ONNX model currently) for gating radius. A distance in pixels
                where if the predicted standard deviation (sigma projected onto the
                landmark move direction) falls below this radius, it will start to dampen
                landmark position changes.
            gating_cutoff: scalar (model also allows per landmark cutoff, but scalar is
                baked into exported ONNX model currently) for gating cutoff. A value
                between 0 and 1, where if the predicted gating value (based on the
                predicted sigma and gating_radius, the weight in [0, 1]) is less than this
                cutoff, the landmark will stop moving completely.
            iterations: Number of iterations to run on the same image. Ignored for ONNX
                model (always 1).
            return_sequence: Whether to return the predictions of all iterations.
                Ignored for ONNX model (always returns final prediction).
            store_similarity_maps: Whether to store similarity maps.
                Not possible with ONNX model (baked into graph).
            prefill_hidden_state: Either None, in which case it will be initialized to
                zeros, or a (1, num_queries, self.gru_hidden_dim) tensor, the hidden state from the
                previous frame.
            prefill_starting_landmarks: Either None, in which case it will be initialized
                to zeros (in normalized coordinates this is the image center and sigma_x =
                sigma_y = 1), or a LandmarkPrediction object containing the predictions
                from the previous frame (or initial guess).
                Note that the model expects landmarks in pixel coordinates relative to the
                input image as inputs (normalization/unnormalization is handled within the
                model), this also applies to the covariance parameters.
            return_hidden_state: Whether to return the hidden state for the next frame.
                (Note that the ONNX model always returns the hidden state).
            return_tensor_predictions: Whether to return raw tensor predictions instead of
                LandmarkPrediction objects. If True, returns a
                (1, num_queries, NUM_PREDS_PARAMS) tensor.
                (Note that the ONNX model always returns tensor predictions, so this just controls
                whether this wrapper converts them to LandmarkPrediction or not).
            detach_updates: Whether to detach the predicted landmarks of the previous
                iteration from the computation graph. Ignored for ONNX model (only
                relevant for training, baked in as always True).
            force_cov_gating: Whether to force the use of covariance-based gating even
                if gating_radius is set to 0 (which normally disables it).
                Forced to True for ONNX model since the graph is exported with cov gating
                always on. Note that therefore setting gating_radius=0 potentially causes NaNs!
            use_naive_correlation: Whether to do the native full correlation for all
                feature maps (even though only a small subset of them are used for
                grid-sampling). Always True for the ONNX model since the optimized fused
                correlation+grid-sampling kernel is implemented in Triton and so cannot be exported.
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
            gating_radius=gating_radius,
            gating_cutoff=gating_cutoff,
            prefill_hidden_state=prefill_hidden_state,
            prefill_starting_landmarks=prefill_starting_landmarks,
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
        
        # Convert back to torch
        # The model will always return predictions in pixel coordinates (relative to the
        # input image size).
        # Shape: (self.batch_size, num_queries, NUM_PREDS_PARAMS), same format as
        # prefill_starting_landmarks input.
        pred_tensor = torch.from_numpy(pred_np).to(device)
        # Shape: (self.batch_size, num_queries, self.gru_hidden_dim), new hidden state for the next frame.
        hidden_tensor = torch.from_numpy(hidden_np).to(device)
        
        outputs = []
        
        # Prepare prediction output
        if return_tensor_predictions:
            final_pred = pred_tensor
        else:
            final_pred = LandmarkPrediction.from_tensor(pred_tensor, cov_type=LowRankCov2D)
        
        outputs.append(final_pred)
        
        # Handle return_hidden_state
        if return_hidden_state:
            outputs.append(hidden_tensor)
            
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def _build_onnx_inputs(
        self,
        image: torch.Tensor,
        query_points: torch.Tensor,
        gating_radius: float | torch.Tensor,
        gating_cutoff: float | torch.Tensor,
        prefill_hidden_state: torch.Tensor | None,
        prefill_starting_landmarks: LandmarkPrediction | None,
        batch_size: int,
        num_queries: int,
        image_height: int,
        image_width: int,
    ) -> dict[str, np.ndarray]:
        """Builds the numpy input dict expected by the exported ONNX graph."""
        # Image: (self.batch_size, 3, H, W)
        image_np = image.detach().cpu().numpy()

        # Query Points: (self.batch_size, num_queries, 3). Skipped for models
        # exported with fixed (baked-in) query points.
        query_points_np = query_points.detach().cpu().numpy() if self._supports_query_points else None

        # Gating Cutoff: (1,)
        if isinstance(gating_cutoff, (float, int)):
            gating_cutoff_np = np.array([gating_cutoff], dtype=np.float32)
        else:
            gating_cutoff_np = gating_cutoff.detach().cpu().numpy()
            if gating_cutoff_np.ndim == 0:
                gating_cutoff_np = gating_cutoff_np[None]

        # Gating Radius: (1,)
        if isinstance(gating_radius, (float, int)):
            gating_radius_np = np.array([gating_radius], dtype=np.float32)
        else:
            gating_radius_np = gating_radius.detach().cpu().numpy()
            if gating_radius_np.ndim == 0:
                gating_radius_np = gating_radius_np[None]

        # Prefill Hidden State: (self.batch_size, num_queries, Hidden)
        if prefill_hidden_state is None:
            prefill_hidden_state_np = np.zeros(
                (batch_size, num_queries, self.gru_hidden_dim),
                dtype=np.float32
            )
        else:
            prefill_hidden_state_np = prefill_hidden_state.detach().cpu().numpy()

        # Prefill Starting Landmarks: (self.batch_size, num_queries, NUM_PREDS_PARAMS)
        # where the last dimension is (x, y, log sigma_x, log sigma_y, rho_raw, dx, dy).
        #
        # rho = tanh(rho_raw) is the correlation coefficient in [-1, 1].
        # x, y, dx and dy are in pixel coordinates relative to the input image size.
        # The model uses a normalized [-1, 1] coordinate system and manages coordinate
        # conversion internally.
        if prefill_starting_landmarks is None:
            # If original model receives None, it inits to zeros (Normalized).
            # To achieve zeros (Normalized) via the normalize_coords path,
            # we must pass the unnormalized version of zeros.
            # Normalized zeros: x=0, y=0 (image center), sigma_x=1, sigma_y=1,
            # rho=0, dx=0, dy=0.

            # create normalized zeros
            norm_zeros = torch.zeros(
                batch_size, num_queries, NUM_PREDS_PARAMS,
                dtype=image.dtype, device=image.device
            )
            norm_pred = LandmarkPrediction.from_tensor(norm_zeros, cov_type=LowRankCov2D)

            # Unnormalize to get pixel coords tensor
            pixel_pred = norm_pred.unnormalize_coords(image_width, image_height)
            prefill_starting_landmarks_np = pixel_pred.to_tensor().detach().cpu().numpy()
        else:
            # Assumed to be in Pixel coordinates already
            prefill_starting_landmarks_np = prefill_starting_landmarks.to_tensor().detach().cpu().numpy()

        inputs = {
            "image": image_np,
            "gating_cutoff": gating_cutoff_np,
            "gating_radius": gating_radius_np,
            "prefill_hidden_state": prefill_hidden_state_np,
            "prefill_starting_landmarks": prefill_starting_landmarks_np,
        }
        # Models exported with fixed query points reject a "query_points" input.
        if self._supports_query_points:
            assert query_points_np is not None
            inputs["query_points"] = query_points_np
        return inputs

    # Alias forward to __call__
    forward = __call__
