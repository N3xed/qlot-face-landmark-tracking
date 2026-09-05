import torch
import torch.nn as nn
from typing import Any

from .feature_extractor import ImageFeatureExtractor, ImageFeatureCorrelator
from .update_predictor import UpdatePredictor, PhaseModulatedPE
from .lmk_features import LmkFeatureEncoder, lookup_grid, get_cached_grid, SampleGrid
from .cov import LowRankCov2D, CovGatedUpdate
from .utils import FlushStream, LandmarkPrediction, NUM_PREDS_PARAMS, preds_coords


class QLOT(nn.Module):
    """
    Queried Learned Optimization for Tracking (QLOT) model for facial landmark tracking.
    """

    queries_transform: torch.Tensor

    def __init__(
        self,
        feature_extractor_pretrained: bool | str = True,
        train_dropout_prob: float = 0.15,
    ):
        super().__init__()

        self.train_dropout_prob = train_dropout_prob

        self.num_channels = 96  # Channel dimension for feature maps

        self.context_dim = 96  # Dimension for global image features
        self.gru_hidden_dim = 128  # Dimension for GRU hidden state (per landmark)
        self.n_heads = [2, 3, 4]  # Number of correlation heads

        # Resolutions n of the correlation kernels where the kernel size is n times n.
        # 1 corresponds to a simple dot product, 0 means there is no linear projection
        # before the correlation operation.
        self.corr_resolutions = [2, 1, 1]

        self.sample_res = [5, 5, 5]  # Sample grid resolutions for correlation lookup

        self.max_log_stddev_clamp = 3.0  # Max log stddev for covariance clamping
        self.min_log_stddev_clamp = -10.0  # Min log stddev for covariance clamping
        self.point_clamp = 2.0  # Max absolute value for (x, y) coordinates

        self.feature_dim = 256  # Dimension of features per landmark

        # Number of correlation map channels is sum of (sample_res^2 * n_heads) for each correlation level.
        num_corr_features = sum(sample_res * sample_res * n_heads for sample_res, n_heads in zip(self.sample_res, self.n_heads))

        # Low rank mixer settings
        self.mixer_rank = 8  # Number of basis slots for low-rank mixing.
        self.mixer_heads = 4  # Number of parallel mixer heads.
        self.mixer_hidden_dim = 128  # Read/write routing MLPs hidden dim.
        self.mixer_value_dim = 256  # Total value dimension, split across heads.

        self.query_encoder = PhaseModulatedPE(in_dims=3, num_fourier_freq=16, num_phase_mod_freq=8, tau=0.03)

        self.query_pre_enc_dim = self.query_encoder.out_dims
        self.query_enc_dim = 96

        self.feature_extractor = ImageFeatureExtractor(
            pretrained=feature_extractor_pretrained,
            global_dim=self.context_dim,
            d_model=self.num_channels,
            n_heads=self.n_heads,
        )

        self.encoder = LmkFeatureEncoder(
            corr_feat_dim=num_corr_features,
            query_enc_hidden_dim=96,
            query_enc_dim=self.query_enc_dim,
            context_feature_dim=self.context_dim,
            inp_query_dim=self.query_pre_enc_dim,
            corr_heads=self.n_heads,
            corr_res=self.sample_res,
            out_feat_dim=self.feature_dim,
        )

        self.image_feature_correlator = ImageFeatureCorrelator(
            num_channels=self.num_channels,
            n_heads=self.n_heads,
            resolutions=self.corr_resolutions,
            n_feature_maps=self.feature_extractor.num_feature_maps,
            query_dim=self.query_pre_enc_dim,
        )

        self.update_predictor = UpdatePredictor(
            context_feat_dim=self.encoder.out_context_dim,
            corr_feat_dim=self.encoder.out_corr_feat_dim,
            hidden_state_dim=self.gru_hidden_dim,
            query_enc_dim=self.query_enc_dim,
            mixer_rank=self.mixer_rank,
            mixer_heads=self.mixer_heads,
            mixer_hidden_dim=self.mixer_hidden_dim,
            mixer_value_dim=self.mixer_value_dim,
            enable_spread=False,  # Disable dispersion-descriptor: The A4/QLOT-final variant of the paper.
        )

        # list of (batch_size, num_queries, n_heads, H, W))
        self.prev_similarity_maps: list[torch.Tensor] | None = None
        self.gated_update = CovGatedUpdate(method="scalar_tanh")

        # Flip y and z axes to match image coordinates
        self.register_buffer("queries_transform", torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32), persistent=False)

    def freeze_backbone(self, freeze: bool, all=False):
        self.feature_extractor.freeze_backbone(freeze)

    def flush_stream(self) -> None:
        """Merge stream-local module state after all worker streams have joined."""
        for module in self.modules():
            if isinstance(module, FlushStream):
                module.flush_stream()

    @staticmethod
    def filter_weights_backbone(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Filter the state dict to only include weights relevant for the backbone (feature extractor and correlator).
        """

        def is_relevant_key(k: str) -> bool:
            return k.startswith("feature_extractor.")  # or k.startswith("image_feature_correlator.")

        return {k: state_dict[k] for k in state_dict.keys() if is_relevant_key(k)}

    @staticmethod
    def translate_weights(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        result = {}
        for k, v in state_dict.items():
            if k.startswith("update_predictor.bypass_attention."):
                new_k = k.replace("bypass_attention", "mixer")
                result[new_k] = v
                continue
            result[k] = v
        return result

    def bake_grids(self, image_size: tuple[int, int]):
        """
        Precompute and register sampling grids for a fixed image size.
        This is useful for ONNX export to avoid dynamic grid generation.
        """
        H_in, W_in = image_size
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        # Access reductions for the active feature layers
        reductions = [self.feature_extractor.feature_map_reductions[i] for i in range(len(self.feature_extractor.feature_layers))]

        for i, (reduction, sample_res) in enumerate(zip(reductions, self.sample_res)):
            # Calculate feature map size
            H_feat = H_in // reduction
            W_feat = W_in // reduction

            # Get the correlation kernel size, this is needed since we use valid padding.
            # This reduces the correlation map size, so we need to scale the coordinates accordingly.
            corr_res = max(self.image_feature_correlator.resolutions[i], 1)

            # Generate and register the grid buffer
            sample_grid = get_cached_grid(sample_res, H_feat, W_feat, corr_res, device, dtype)
            self.register_buffer(f"static_grid_{i}", sample_grid.grid)
            self.register_buffer(f"static_scale_{i}", torch.as_tensor(sample_grid.scale, device=device, dtype=dtype))

    def get_grids(self, image_size: tuple[int, int], device: torch.device, dtype: torch.dtype) -> list[SampleGrid]:
        """
        Get the sampling grids for a fixed image size and cache them.
        Args:
            image_size: Tuple of (height, width) of the input image.
        Returns:
            List of SampleGrid objects for each feature level.
        """
        # Check if we have precomputed grids available
        sample_grids: list[SampleGrid] = []
        if hasattr(self, "static_grid_0"):
            grid_idx = 0
            while hasattr(self, f"static_grid_{grid_idx}"):
                g = getattr(self, f"static_grid_{grid_idx}")
                s = getattr(self, f"static_scale_{grid_idx}")
                sample_grids.append(SampleGrid(g, s))
                grid_idx += 1
        else:
            reductions = self.feature_extractor.feature_map_reductions
            image_height, image_width = image_size
            for lvl, (reduction, sample_res) in enumerate(zip(reductions, self.sample_res)):
                H_feat = image_height // reduction
                W_feat = image_width // reduction

                corr_res = max(self.image_feature_correlator.resolutions[lvl], 1)
                sample_grid = get_cached_grid(sample_res, H_feat, W_feat, corr_res, device, dtype)
                sample_grids.append(sample_grid)
        return sample_grids

    def forward(
        self,
        image: torch.Tensor,
        query_points: torch.Tensor,
        gating_radius: torch.Tensor | float = 0.0,
        gating_cutoff: torch.Tensor | float = 0.05,
        iterations=3,
        return_sequence=False,
        store_similarity_maps=False,
        return_similarity_maps=False,
        prefill_hidden_state: torch.Tensor | None = None,
        prefill_starting_landmarks: LandmarkPrediction | None = None,
        return_hidden_state=False,
        return_tensor_predictions=False,
        detach_updates=True,
        force_cov_gating=False,
        use_naive_correlation=False,
        landmarks_to_mask: torch.Tensor | None = None,
    ) -> Any:
        """
        The main forward pass.
        Args:
            image: Input image tensor, shape (batch_size, 3, H, W).
            query_points: 3D query points, shape (batch_size, num_queries, 3).
            gating_radius: Certainty radius in pixels below which (approx.) updates are
                           gated. 0.0 means no gating.
            gating_cutoff: Cutoff ratio between 0 and 1, where dx < cutoff * radius is set to zero.
            iterations: Number of refinement iterations.
            return_sequence: Whether to return the full sequence of predictions.
            store_similarity_maps: Whether to store the similarity maps for later use.
            return_similarity_maps: Whether to return the similarity maps as part of the output.
            prefill_hidden_state: Optional hidden state to prefill the GRU, shape (batch_size, num_queries, gru_hidden_dim).
            prefill_starting_landmarks: Optional starting landmarks to prefill the predictions in pixel coordinates.
            return_hidden_state: Whether to return the final hidden state.
            return_tensor_predictions: Whether to return predictions as tensors instead of LandmarkPrediction object.
            detach_updates: Whether to detach the predictions before each update.
            force_cov_gating: Whether to force covariance-based gating even if gating_radius is 0.0. Needed for ONNX export.
            use_naive_correlation: Whether to use naive Conv2D version instead of optimized triton kernel.
            landmarks_to_mask: Optional boolean tensor of shape (batch_size, num_queries) indicating which landmarks to mask (ignore) during updates. True = ignored.
        Returns:
            Depending on the flags, the predictions itself, or a list of predictions and
            enabled items in order of arguments.
        """
        batch_size, num_queries, _ = query_points.shape
        image_height, image_width = image.shape[2], image.shape[3]
        gating_radius_scale = min(image_width, image_height) * 0.5
        device = image.device

        # Keep export-friendly composition and avoid slice updates that lower to ScatterND.
        # Flip y and z axes to match image coordinates
        query_points = query_points * self.queries_transform

        # Extract image features
        image_feature_maps: list[torch.Tensor]  # Shape: list of (batch_size, d_model, H, W)
        global_features: torch.Tensor  # Shape: (batch_size, context_dim)
        image_feature_maps, global_features = self.feature_extractor(image)

        # Initialize hidden state
        if prefill_hidden_state is not None:
            # assert prefill_hidden_state.shape == (batch_size, num_queries, self.gru_hidden_dim), f"{prefill_hidden_state.shape=}"
            hidden_state = prefill_hidden_state.to(device)
        else:
            hidden_state = torch.zeros(
                batch_size,
                num_queries,
                self.gru_hidden_dim,
                device=device,
                requires_grad=self.training,  # Only require grad if we are training, to save memory during inference
            )

        query_pre_enc = self.query_encoder(query_points)  # Shape: (batch_size, num_queries, query_pre_enc_dim)
        query_encoding = self.encoder.forward_query_points(query_pre_enc)  # Shape: (batch_size, num_queries, query_enc_dim)
        global_features = global_features.unsqueeze(1)  # (batch_size, 1, context_dim)

        if store_similarity_maps or use_naive_correlation or return_similarity_maps:
            with torch.set_grad_enabled(use_naive_correlation):
                # Get similarity/correlation maps for query points
                similarity_maps = self.image_feature_correlator(
                    image_feature_maps, query_pre_enc
                )  # Shape: list of (batch_size, num_queries, n_heads, H, W)
            if store_similarity_maps:
                self.prev_similarity_maps = similarity_maps

        # Initial landmark prediction
        if prefill_starting_landmarks is None:
            # Initialize to center with unit variance,
            # since we are in normalized coordinates [-1, 1], center is 0.0
            # and variance 1 corresponds to exp(0)=1 in Cholesky params which is
            # (half_width, half_height) in pixel space.
            init_pred = torch.zeros(batch_size, num_queries, NUM_PREDS_PARAMS, device=device)
            preds_coords(init_pred)[...] = query_points[..., :2]  # Initialize (x, y) to query points
            if self.training:
                center = preds_coords(init_pred).mean(dim=1, keepdim=True)
                rel_xy = preds_coords(init_pred) - center

                scale = 1.0 + 0.5 * torch.randn(batch_size, 1, 1, device=device)
                translation = 0.4 * torch.randn(batch_size, 1, 2, device=device)

                point_jitter = 0.05 * torch.randn(batch_size, num_queries, 2, device=device)
                point_mask = (torch.rand(batch_size, num_queries, 1, device=device) < 0.2).float()

                preds_coords(init_pred)[...] = center + rel_xy * scale + translation + point_jitter * point_mask

            init_pred = LandmarkPrediction.from_tensor(init_pred, cov_type=LowRankCov2D)
        else:
            # assert prefill_starting_landmarks.mean.shape == (
            #     batch_size,
            #     num_queries,
            #     2,
            # ), f"{prefill_starting_landmarks.mean.shape=}"
            # assert prefill_starting_landmarks.cov.params.shape[:-1] == (
            #     batch_size,
            #     num_queries,
            # ), f"{prefill_starting_landmarks.cov.params.shape[:-1]=}"
            assert isinstance(prefill_starting_landmarks.cov, LowRankCov2D), f"{type(prefill_starting_landmarks.cov)=}"
            init_pred = prefill_starting_landmarks.normalize_coords(image_width, image_height)
            init_pred.mean = init_pred.mean.to(device)
            init_pred.cov = init_pred.cov.to(device)

        last_coords = init_pred.mean
        last_cov = init_pred.cov.params
        last_delta = init_pred.delta

        predictions: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for i in range(iterations):
            # Generate mask for training stability with subsets
            attn_mask = landmarks_to_mask
            if self.training and self.train_dropout_prob > 0.0:
                # Mask (Batch, NumQueries). True = Ignored.
                mask_probs = torch.rand((batch_size, num_queries), device=device)
                attn_mask = mask_probs < self.train_dropout_prob

                if num_queries in [70, 98]:
                    # Mask all pupil landmarks for WFLW/Face Synthetics landmarks.
                    attn_mask[:, -2:] = True

            if detach_updates:
                # Detach coordinates from last iteration like RAFT.
                last_coords = last_coords.detach()

            # Lookup local features around each query point
            if use_naive_correlation:
                corr_features = lookup_grid(
                    last_coords,  # Use (x, y) for lookup
                    self.sample_res,
                    similarity_maps,  # type: ignore
                    grids=self.get_grids((image_height, image_width), device, last_coords.dtype),
                )  # Shape: list of (batch_size, num_queries, n_heads, sample_res, sample_res)
            else:
                corr_features = self.image_feature_correlator.forward_fused_conv_sample(
                    image_features=image_feature_maps,
                    query_points=query_pre_enc,
                    grid_size=[(r, r) for r in self.sample_res],
                    last_pred_coords=last_coords,
                )  # Shape: list of (batch_size, num_queries, n_heads, sample_res, sample_res)

            context_feat, corr_feat = self.encoder(
                corr_features=corr_features,
                query_encoding=query_encoding,
                image_context_features=global_features,
                last_coords=last_coords,
                last_cov=last_cov,
                last_delta=last_delta,
            )  # Shape: (batch_size, num_queries, feature_dim)

            # Predict updates
            dx_dy, cov, hidden_state = self.update_predictor(
                context_feat=context_feat,
                corr_feat=corr_feat,
                hidden_state=hidden_state,
                query_encodings=query_encoding,
                mask=attn_mask,
            )  # updates shape: (batch_size, num_queries, 5)

            # Update predictions
            cov = LowRankCov2D(cov)
            if force_cov_gating or gating_radius > 0.0:
                # Assume isotropic scaling for radius
                dx_dy, k = self.gated_update(
                    dx_dy,
                    cov,
                    gating_radius / gating_radius_scale,
                    cutoff=gating_cutoff,
                )  # Shape: (batch_size, num_queries, 2)

            last_coords = last_coords + dx_dy
            last_cov = cov.params
            last_delta = dx_dy

            if return_sequence:
                predictions.append((last_coords, last_cov, last_delta))
            elif i == iterations - 1:  # Last iteration
                predictions.append((last_coords, last_cov, last_delta))

        if return_sequence:
            preds_out = (
                torch.stack([p[0] for p in predictions], dim=0),  # Shape: (batch_size, iterations, num_queries, 2)
                LowRankCov2D(torch.stack([p[1] for p in predictions], dim=0)),  # Shape: (batch_size, iterations, num_queries, 3)
                torch.stack([p[2] for p in predictions], dim=0),  # Shape: (batch_size, iterations, num_queries, 2)
            )
        else:
            preds_out = (
                predictions[0][0],  # Shape: (batch_size, num_queries, 2)
                LowRankCov2D(predictions[0][1]),  # Shape: (batch_size, num_queries, 3)
                predictions[0][2],  # Shape: (batch_size, num_queries, 2)
            )

        preds = LandmarkPrediction(*preds_out)

        # Clamp and unnormalize coordinates to pixel space
        preds = preds.unnormalize_coords_clamp(
            image_width,
            image_height,
            min=-self.point_clamp if not torch.compiler.is_exporting() else None,
            max=self.point_clamp if not torch.compiler.is_exporting() else None,
            log_min_std_dev=self.min_log_stddev_clamp if not torch.compiler.is_exporting() else None,
            log_max_std_dev=self.max_log_stddev_clamp if not torch.compiler.is_exporting() else None,
        )
        to_return: list[Any] = [preds if not return_tensor_predictions else preds.to_tensor()]

        if return_hidden_state:
            to_return.append(hidden_state)

        if return_similarity_maps:
            to_return.append(similarity_maps)  # type: ignore

        if len(to_return) == 1:
            return to_return[0]
        return tuple(to_return)
