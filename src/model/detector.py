import torch
import torch.nn as nn
from torch import Tensor
from typing import Any
import math
import torch.nn.functional as F
from dataclasses import dataclass

from .feature_extractor import ImageFeatureExtractor
from .update_predictor import PhaseModulatedPE, LowRankWriteReadMixer, HeadedLinear, GRU, logit
from .lmk_features import LocalPatchSampler
from .cov import Cov
from .utils import (
    FlushStream,
    LandmarkPrediction,
    NUM_PREDS_PARAMS,
    NUM_PREDS_COORDS,
    NUM_PREDS_COV_PARAMS,
    NUM_PREDS_DELTA,
    init,
)


@dataclass
class ImageFeatures:
    maps: list[Tensor]
    query_enc: Tensor
    context_feat: Tensor
    attn_mask: Tensor

    def as_frames(self, clip_len: int) -> "ImageFeatures":
        """
        Unflatten the batch dimension into (batch, clip_len, ...)
        """

        maps = [m.view(-1, clip_len, *m.shape[1:]) for m in self.maps]
        query_enc = self.query_enc.view(-1, clip_len, *self.query_enc.shape[1:])
        context_feat = self.context_feat.view(-1, clip_len, *self.context_feat.shape[1:])
        attn_mask = self.attn_mask.view(-1, clip_len, *self.attn_mask.shape[1:])
        return ImageFeatures(
            maps=maps,
            query_enc=query_enc,
            context_feat=context_feat,
            attn_mask=attn_mask,
        )

    def __getitem__(self, key) -> "ImageFeatures":
        return ImageFeatures(
            maps=[m[key] for m in self.maps],
            query_enc=self.query_enc[key],
            context_feat=self.context_feat[key],
            attn_mask=self.attn_mask[key],
        )


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
        self.log_std_dev_max = 10.0
        self.coord_min_max = (-2.0, 2.0)

        self.hidden_dim = 128
        self.sample_res = 5
        self.context_dim = 256
        self.query_enc_dim = 64

        # Query points
        self.query_encoder = PhaseModulatedPE(in_dims=3, num_fourier_freq=16, num_phase_mod_freq=8, tau=0.03)

        self.query_pre_enc_dim = self.query_encoder.out_dims
        self.query_mlp = nn.Sequential(
            nn.Linear(self.query_pre_enc_dim, 256), nn.SiLU(), nn.Linear(256, self.context_dim), nn.RMSNorm(self.context_dim)
        )
        init(self.query_mlp[0], nonlinearity="relu")
        init(self.query_mlp[2], nonlinearity="linear")
        self.query_proj = nn.Sequential(nn.Linear(self.query_pre_enc_dim, self.query_enc_dim), nn.SiLU())
        init(self.query_proj[0], nonlinearity="linear")

        # Flip y and z axes to match image coordinates
        self.register_buffer("queries_transform", torch.tensor([1.0, -1.0, -1.0], dtype=torch.float32), persistent=False)

        # Backbone
        self.feature_extractor = ImageFeatureExtractor(pretrained=feature_extractor_pretrained, stream_safe=False)

        # Global features
        self.global_feat_proj = nn.Sequential(
            nn.Linear(self.feature_extractor.global_channels, self.context_dim), nn.RMSNorm(self.context_dim)
        )
        init(self.global_feat_proj[0], nonlinearity="linear")

        # Hidden state init
        self.init_hidden_state = nn.Parameter(torch.randn(1, 1, self.hidden_dim) / math.sqrt(self.hidden_dim))

        # Last predictions encoding
        kin_dim = 64
        self.kin_encoder = nn.Sequential(nn.Linear(NUM_PREDS_PARAMS, kin_dim), nn.SiLU(), nn.Linear(kin_dim, self.context_dim))
        init(self.kin_encoder[0], nonlinearity="relu")
        init(self.kin_encoder[2], nonlinearity="linear")

        # Hidden state proj
        self.hidden_state_proj = nn.Linear(self.hidden_dim, self.context_dim)
        init(self.hidden_state_proj, nonlinearity="linear")

        self.context_fusion_weights = nn.Parameter(
            torch.tensor([0.1, 0.025, 0.01])  # global features  # kin features  # hidden state
        )

        # Local patch sampling
        self.sampler = LocalPatchSampler(
            map_channels=self.feature_extractor.map_channels, sample_res=5, context_dim=self.context_dim
        )
        total_vis_feat = sum(c + self.sampler.out_corr_dims for c in self.feature_extractor.map_channels)

        self.vision_feat_mlp = nn.Sequential(
            nn.Linear(total_vis_feat + self.context_dim, 256),
            nn.SiLU(),
        )
        init(self.vision_feat_mlp[0], nonlinearity="relu")

        self.mixer = LowRankWriteReadMixer(
            read_dim=self.query_enc_dim + self.hidden_dim,
            write_dim=self.context_dim,
            value_in_dim=256,
            value_dim=256,
            out_dim=256,
            rank=8,
            heads=4,
            hidden_dim=128,
            enable_spread=True,
        )

        self.gate_proj = nn.Linear(self.hidden_dim, 2 * self.hidden_dim)
        nn.init.kaiming_normal_(self.gate_proj.weight, nonlinearity="sigmoid")  # type: ignore
        with torch.no_grad():
            # Scale up, since we split into 2 gates.
            self.gate_proj.weight.mul_(math.sqrt(2.0))  # type: ignore

        # reset, update
        gate_bias: torch.Tensor = self.gate_proj.bias  # type: ignore
        # reset gate
        nn.init.constant_(gate_bias[: self.hidden_dim], 0.0)
        # update gate
        # Start with a lower update gate bias, so the hidden state is preserved at the start.
        nn.init.constant_(gate_bias[self.hidden_dim :], logit(0.4))

        self.hidden_norm = nn.RMSNorm(self.hidden_dim)
        self.gru = GRU(self.hidden_dim)

        self.pred_head = nn.Sequential(
            HeadedLinear(self.hidden_dim, 128, heads=2, head_dim=-3),
            nn.GLU(),
        )
        self.pred_head[0].init_glu()  # type: ignore

        self.coord_proj = nn.Linear(64, NUM_PREDS_COORDS)
        self.cov_proj = nn.Linear(64, NUM_PREDS_COV_PARAMS)
        nn.init.xavier_uniform_(self.coord_proj.weight, gain=0.1)  # type: ignore
        nn.init.constant_(self.coord_proj.bias, 0.0)  # type: ignore
        nn.init.xavier_uniform_(self.cov_proj.weight, gain=0.1)  # type: ignore
        nn.init.constant_(self.cov_proj.bias, 0.0)  # type: ignore

        with torch.no_grad():
            self.cov_proj.bias[0] = 0.2
            self.cov_proj.bias[1] = 0.05

    def freeze_backbone(self, freeze: bool, all=False):
        self.feature_extractor.freeze_backbone(freeze)

    def flush_stream(self) -> None:
        """Merge stream-local module state after all worker streams have joined."""
        for module in self.modules():
            if isinstance(module, FlushStream):
                module.flush_stream()

    def compile(self) -> None:
        compiled_forward_image = torch.compile(self.forward_image)
        compiled_forward_iteration = torch.compile(self._forward_iteration)

        def _mark_batch(t: Tensor, max_batch: int = 256) -> None:
            if t.shape[0] > 1:
                torch._dynamo.mark_dynamic(t, 0, min=2, max=max_batch)

        def _mark_batch_and_queries(t: Tensor, max_batch: int = 256, max_queries: int = 128) -> None:
            if t.shape[0] > 1:
                torch._dynamo.mark_dynamic(t, 0, min=2, max=max_batch)
            if t.shape[1] > 1:
                torch._dynamo.mark_dynamic(t, 1, min=2, max=max_queries)

        def forward_image(
            image: torch.Tensor,
            query_points: torch.Tensor,
            landmarks_to_mask: torch.Tensor | None = None,
        ) -> ImageFeatures:
            _mark_batch(image)
            _mark_batch_and_queries(query_points)
            if landmarks_to_mask is not None:
                _mark_batch_and_queries(landmarks_to_mask)
            return compiled_forward_image(
                image=image,
                query_points=query_points,
                landmarks_to_mask=landmarks_to_mask,
            )

        def _forward_iteration(
            last_hidden: Tensor,
            last_coords: Tensor,
            last_cov: Tensor,
            last_delta: Tensor,
            feat: ImageFeatures,
            detach_updates: bool,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            for m in feat.maps:
                _mark_batch(m)
            for t in (
                last_hidden,
                last_coords,
                last_cov,
                last_delta,
                feat.query_enc,
                feat.context_feat,
                feat.attn_mask,
            ):
                _mark_batch_and_queries(t)
            return compiled_forward_iteration(
                last_hidden=last_hidden,
                last_coords=last_coords,
                last_cov=last_cov,
                last_delta=last_delta,
                feat=feat,
                detach_updates=detach_updates,
            )

        self.forward_image = forward_image
        self._forward_iteration = _forward_iteration

    @staticmethod
    def filter_weights_backbone(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Filter the state dict to only include weights relevant for the backbone (feature extractor and correlator).
        """

        def is_relevant_key(k: str) -> bool:
            return k.startswith("feature_extractor.")  # or k.startswith("image_feature_correlator.")

        return {k: state_dict[k] for k in state_dict.keys() if is_relevant_key(k)}

    def new_hidden_state(self, batch_size: int, num_queries: int) -> torch.Tensor:
        return self.init_hidden_state.expand(batch_size, num_queries, self.hidden_dim).clone()

    def new_predictions(
        self, batch_size: int, num_queries: int, query_points: torch.Tensor, image_size: tuple[int, int]
    ) -> LandmarkPrediction:
        """
        Initialize new landmark predictions for a given batch.

        Args:
            batch_size (int): Number of samples in the batch.
            num_queries (int): Number of query points per sample.
            query_points (torch.Tensor): Initial query points of shape (batch_size, num_queries, 3).
            image_size (tuple[int, int]): Size of the image (height, width).

        Returns:
            LandmarkPrediction: Initialized landmark predictions.
        """

        device = self.init_hidden_state.device

        # Flip y and z axes to match image coordinates
        query_points = query_points * self.queries_transform

        # Initialize to center with unit variance,
        # since we are in normalized coordinates [-1, 1], center is 0.0
        # and variance 1 corresponds to exp(0)=1 in Cholesky params which is
        # (half_width, half_height) in pixel space.
        coords = query_points[..., :2].contiguous()
        if self.training:
            center = coords.mean(dim=1, keepdim=True)
            rel_xy = coords - center

            scale = 1.0 + 0.5 * torch.randn(batch_size, 1, 1, device=device)
            translation = 0.4 * torch.randn(batch_size, 1, 2, device=device)

            point_jitter = 0.05 * torch.randn(batch_size, num_queries, 2, device=device)
            point_mask = (torch.rand(batch_size, num_queries, 1, device=device) < 0.2).float()

            coords = center + rel_xy * scale + translation + point_jitter * point_mask

        cov = Cov(params=torch.zeros(batch_size, num_queries, NUM_PREDS_COV_PARAMS, device=device))
        # We are in normalized space, so we need to scale up the covariance (as it
        # assumes pixel space).
        cov.xy_scale = (2.0 / image_size[1], 2.0 / image_size[0])

        delta = torch.zeros(batch_size, num_queries, NUM_PREDS_DELTA, device=device)
        return LandmarkPrediction(mean=coords, cov=cov, delta=delta)

    def forward_image(
        self,
        image: Tensor,
        query_points: Tensor,
        landmarks_to_mask: Tensor | None = None,
    ) -> ImageFeatures:
        device = image.device
        batch_size, num_queries, _ = query_points.shape

        # Generate mask for training stability with subsets
        attn_mask = torch.zeros(batch_size, num_queries, dtype=torch.bool, device=device)
        if landmarks_to_mask is not None:
            attn_mask = attn_mask | landmarks_to_mask.bool()

        if self.training and self.train_dropout_prob > 0.0:
            # Mask (Batch, NumQueries). True = Ignored.
            mask_probs = torch.rand((batch_size, num_queries), device=device)
            attn_mask = attn_mask | (mask_probs < self.train_dropout_prob)

        # Flip y and z axes to match image coordinates
        query_points = query_points * self.queries_transform

        # Extract image features
        image_feature_maps: list[Tensor]  # Shape: list of (batch_size, C_i, H_i, W_i) for level i
        global_feat: Tensor  # Shape: (batch_size, global_feat_dim)
        image_feature_maps, global_feat = self.feature_extractor(image)

        query_enc: Tensor = self.query_encoder(query_points)  # Shape: (batch_size, num_queries, query_pre_enc_dim)
        context_feat: Tensor = self.query_mlp(query_enc)  # Shape: (batch_size, num_queries, context_dim)
        query_enc = self.query_proj(query_enc)  # Shape: (batch_size, num_queries, query_enc_dim)

        global_feat = self.context_fusion_weights[0] * self.global_feat_proj(global_feat)
        global_feat = global_feat.unsqueeze(1).expand(batch_size, num_queries, -1)  # (batch_size, num_queries, context_dim)

        context_feat = context_feat + global_feat

        return ImageFeatures(maps=image_feature_maps, query_enc=query_enc, context_feat=context_feat, attn_mask=attn_mask)

    def _forward_iteration(
        self,
        last_hidden: Tensor,
        last_coords: Tensor,
        last_cov: Tensor,
        last_delta: Tensor,
        feat: ImageFeatures,
        detach_updates: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if detach_updates:
            # Detach coordinates from last iteration like RAFT.
            last_coords = last_coords.detach()

        # Process context features
        kin_enc = self.kin_encoder(torch.cat([last_coords, Cov(last_cov).as_bounded_params(), last_delta * 10.0], dim=-1))
        context_feat = feat.context_feat + self.context_fusion_weights[1] * kin_enc

        # Fuse hidden state with context features
        normed_hidden_state = self.hidden_norm(last_hidden)  # (batch_size, num_queries, hidden_dim)
        context_feat = context_feat + self.context_fusion_weights[2] * self.hidden_state_proj(normed_hidden_state)

        # Lookup local features at each coord
        warped_features = self.sampler(
            coords=last_coords, feature_maps=feat.maps, context=context_feat
        )  # Shape: list of (batch_size, num_queries, C_i)
        vis_feat = torch.cat([f for t in warped_features for f in t], dim=-1)  # (batch_size, num_queries, total_map_feat)

        read_feat = torch.cat(
            [normed_hidden_state, feat.query_enc], dim=-1
        )  # (batch_size, num_queries, hidden_dim + query_enc_dim)

        map_feat = torch.cat([vis_feat, context_feat], dim=-1)
        map_feat = self.vision_feat_mlp(map_feat)

        # Mix features
        mixed = self.mixer(
            write_feat=context_feat,
            read_feat=read_feat,
            value_feat=map_feat,
            key_padding_mask=feat.attn_mask,
        )  # (batch_size, num_queries, hidden_dim)
        candidate_feat, gate_feat = mixed.chunk(2, dim=-1)  # (batch_size, num_queries, hidden_dim) each

        gate_feat: torch.Tensor = F.silu(gate_feat)
        gate_feat = self.gate_proj(gate_feat)  # (batch_size, num_queries, 2 * hidden_dim)
        gate_feat = gate_feat.sigmoid()
        reset, update = gate_feat.chunk(2, dim=-1)

        # Update hidden_state
        next_hidden: torch.Tensor = self.gru(
            prev_hidden=last_hidden,
            prev_hidden_candidate=normed_hidden_state,
            candidate=candidate_feat,
            update=update,
            reset=reset,
        )

        # Prediction head
        pred_feat: torch.Tensor = next_hidden.unsqueeze(-3)  # (batch_size, 1, num_queries, hidden_dim)
        pred_feat = self.pred_head(pred_feat)  # (batch_size, 2, num_queries, 64)
        coord_feat, cov_feat = pred_feat.unbind(dim=1)

        dx_dy: Tensor = self.coord_proj(coord_feat)  # Predict coordinate updates from coord_feat
        cov: Tensor = self.cov_proj(cov_feat)  # Predict covariance params

        # Update predictions
        cov_inst = Cov(cov).clamp(self.log_std_dev_max)

        next_coords = (last_coords + dx_dy).clamp(*self.coord_min_max)
        next_cov = cov_inst.params
        next_delta = dx_dy

        return next_coords, next_cov, next_delta, next_hidden

    def forward_refine(
        self,
        feat: ImageFeatures,
        hidden_state: Tensor,
        predictions: LandmarkPrediction,
        iterations: int = 3,
        return_sequence: bool = False,
        return_hidden_state: bool = True,
        return_tensor_predictions: bool = False,
        detach_updates: bool = True,
    ) -> Any:
        last_hidden = hidden_state
        last_coords = predictions.mean
        last_cov = predictions.cov.params
        last_delta = predictions.delta

        preds_out: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for i in range(iterations):
            last_coords, last_cov, last_delta, last_hidden = self._forward_iteration(
                last_hidden=last_hidden,
                last_coords=last_coords,
                last_cov=last_cov,
                last_delta=last_delta,
                feat=feat,
                detach_updates=detach_updates,
            )

            if return_sequence:
                preds_out.append((last_coords, last_cov, last_delta))
            elif i == iterations - 1:  # Last iteration
                preds_out.append((last_coords, last_cov, last_delta))

        if return_sequence:
            preds_out2 = (
                torch.stack([p[0] for p in preds_out], dim=0),  # Shape: (batch_size, iterations, num_queries, 2)
                Cov(
                    torch.stack([p[1] for p in preds_out], dim=0), xy_scale=predictions.cov.xy_scale
                ),  # Shape: (batch_size, iterations, num_queries, 4)
                torch.stack([p[2] for p in preds_out], dim=0),  # Shape: (batch_size, iterations, num_queries, 2)
            )
        else:
            preds_out2 = (
                preds_out[0][0],  # Shape: (batch_size, num_queries, 2)
                Cov(preds_out[0][1], xy_scale=predictions.cov.xy_scale),  # Shape: (batch_size, num_queries, 4)
                preds_out[0][2],  # Shape: (batch_size, num_queries, 2)
            )
        preds = LandmarkPrediction(*preds_out2)

        # Clamp coordinates
        preds.mean = preds.mean.clamp(*self.coord_min_max)
        preds.delta = preds.delta.clamp(*self.coord_min_max)

        to_return: list[Any] = [preds if not return_tensor_predictions else preds.to_tensor()]

        if return_hidden_state:
            to_return.append(last_hidden)

        if len(to_return) == 1:
            return to_return[0]
        return tuple(to_return)

    def forward(
        self,
        image: torch.Tensor,
        query_points: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        predictions: LandmarkPrediction | None = None,
        iterations=3,
        return_sequence=False,
        return_hidden_state=True,
        return_tensor_predictions=False,
        detach_updates=True,
        landmarks_to_mask: torch.Tensor | None = None,
    ) -> Any:
        """
        The main forward pass.
        Args:
            image: Input image tensor, shape (batch_size, 3, H, W).
            query_points: 3D query points, shape (batch_size, num_queries, 3).
            iterations: Number of refinement iterations.
            return_sequence: Whether to return the full sequence of predictions.
            hidden_state: Optional hidden state to prefill the GRU, shape (batch_size, num_queries, gru_hidden_dim).
            predictions: Optional starting landmarks to prefill the predictions in pixel coordinates.
            return_hidden_state: Whether to return the final hidden state.
            return_tensor_predictions: Whether to return predictions as tensors instead of LandmarkPrediction object.
            detach_updates: Whether to detach the predictions before each update.
            landmarks_to_mask: Optional boolean tensor of shape (batch_size, num_queries) indicating which landmarks to mask (ignore) during updates. True = ignored.
        Returns:
            Depending on the flags, the predictions itself, or a list of predictions and
            enabled items in order of arguments.
        """
        feat = self.forward_image(image=image, query_points=query_points, landmarks_to_mask=landmarks_to_mask)

        batch_size, num_queries, _ = query_points.shape
        image_size = (image.shape[-2], image.shape[-1])

        if hidden_state is None:
            hidden_state = self.new_hidden_state(batch_size, num_queries)
        if predictions is None:
            predictions = self.new_predictions(batch_size, num_queries, query_points, image_size)

        return self.forward_refine(
            feat=feat,
            hidden_state=hidden_state,
            predictions=predictions,
            iterations=iterations,
            return_sequence=return_sequence,
            return_hidden_state=return_hidden_state,
            return_tensor_predictions=return_tensor_predictions,
            detach_updates=detach_updates,
        )
