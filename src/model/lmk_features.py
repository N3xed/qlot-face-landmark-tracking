import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from .update_predictor import HeadedLinear
from .utils import NUM_PREDS_COORDS, NUM_PREDS_COV_PARAMS, NUM_PREDS_DELTA, init
import torch.nn.functional as F


class LocalPatchSampler(nn.Module):
    base_grid_x: torch.Tensor
    base_grid_y: torch.Tensor

    def __init__(self, map_channels: list[int], sample_res: int = 5, context_dim: int = 256, attn_heads: int = 4):
        super().__init__()

        self.map_channels = list(map_channels)
        self.sample_res = sample_res
        self.attn_heads = attn_heads
        self.head_ch_dims = [c // attn_heads for c in self.map_channels]
        self.attn_scales = [1.0 / math.sqrt(hd) for hd in self.head_ch_dims]

        t = torch.linspace(-1.0, 1.0, steps=sample_res, dtype=torch.float32) * (sample_res - 1.0)
        ys, xs = torch.meshgrid(t, t, indexing="ij")
        self.register_buffer("base_grid_x", xs.view(1, 1, sample_res, sample_res), persistent=False)
        self.register_buffer("base_grid_y", ys.view(1, 1, sample_res, sample_res), persistent=False)

        self.ctx_proj = nn.ModuleList([nn.Linear(context_dim, map_feat) for map_feat in map_channels])
        for c in self.ctx_proj:
            if isinstance(c, nn.Linear):
                init(c, nonlinearity="linear")

        chan_mult = 2
        kernel_size = 4
        self.corr_conv = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(attn_heads, chan_mult * attn_heads, kernel_size=kernel_size, padding=0), nn.SiLU())
                for _ in range(len(map_channels))
            ]
        )
        for c in self.corr_conv:
            assert isinstance(c, nn.Sequential)
            init(c[0], nonlinearity="relu")

        self.out_corr_dims = ((self.sample_res - kernel_size + 1) ** 2) * attn_heads * chan_mult

    def forward(
        self, feature_maps: list[torch.Tensor], coords: torch.Tensor, context: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Sample local patches from feature maps around the given coordinates,
        and apply depthwise convolutions to aggregate the local patches into feature vectors.

        Args:
            feature_maps: List of feature maps to sample from, each with shape (batch_size, C_i, H_i, W_i).
            coords: 2D query points in normalized [-1, 1] coordinates, shape (batch_size, num_queries, 2).

        Returns:
            List of tuples:
                - Sampled features, each with shape (batch_size, num_queries, C_i).
                - Logits corresponding to the attention weights, each with shape (batch_size, num_queries, res_h * res_w).
        """

        batch_size, num_queries, _ = coords.shape
        res = self.sample_res
        res_sq = res * res

        samples = self.lookup_grid(coords, feature_maps)

        out_samples = []
        for i, sample in enumerate(samples):
            head_ch_dim = self.head_ch_dims[i]
            scale = self.attn_scales[i]

            kernel: torch.Tensor = self.ctx_proj[i](context)  # (batch_size, num_queries, C_i)
            kernel = kernel.view(batch_size, num_queries, self.attn_heads, 1, head_ch_dim)

            sample = sample.view(batch_size, num_queries, self.attn_heads, head_ch_dim, res_sq)

            corr: torch.Tensor = kernel @ sample  # (batch_size, num_queries, H, 1, res_sq)
            corr = corr.squeeze(-2) * scale  # (batch_size, num_queries, H, res_sq)

            probs = corr.softmax(dim=-1)  # (batch_size, num_queries, H, res_sq)
            feat = sample @ probs[..., None]  # (batch_size, num_queries, H, C_i//H, 1)
            feat = feat.flatten(-3)  # (batch_size, num_queries, C_i)
            feat = F.rms_norm(feat, (self.map_channels[i],))

            corr = corr.view(batch_size * num_queries, self.attn_heads, res, res)
            corr = self.corr_conv[i](corr)  # (batch_size * num_queries, 8, res_h', res_w')
            corr = corr.view(batch_size, num_queries, self.out_corr_dims)

            out_samples.append((feat, corr))

        return out_samples

    def lookup_grid(
        self,
        query_points_xy: torch.Tensor,
        feature_maps: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """
        Lookup local patches around query points from multiple feature maps.

        Args:
            query_points_xy: 2D query points in normalized [-1, 1] coordinates, shape (batch_size, num_queries, 2).
            feature_maps: List of feature maps to sample from, each with shape (batch_size, C_i, H_i, W_i).
        Returns:
            Local features at each query point from all feature maps, list of shape
            (batch_size, num_queries, C_i, res_h, res_w).
        """
        batch_size, num_queries, _ = query_points_xy.shape
        sample_res = self.sample_res
        coords_center = query_points_xy.view(batch_size, num_queries, 1, 1, 2)

        # For each feature map, sample local features around each query point
        sampled_features = []
        for feat_map, C in zip(feature_maps, self.map_channels):
            H, W = feat_map.shape[-2], feat_map.shape[-1]
            base_grid = torch.stack([self.base_grid_x / W, self.base_grid_y / H], dim=-1)

            coords = coords_center + base_grid  # (batch_size, num_queries, sample_res, sample_res, 2)
            coords = coords.view(batch_size, num_queries * sample_res, sample_res, 2)

            feat = torch.nn.functional.grid_sample(
                feat_map,  # (B, C, H, W)
                coords,
                align_corners=False,
                mode="bilinear",
                padding_mode="border",
            )  # (B, C, num_queries * sample_res, sample_res)
            feat = feat.view(batch_size, C, num_queries, sample_res, sample_res)
            feat = feat.transpose(1, 2)  # (batch_size, num_queries, C, sample_res, sample_res)
            sampled_features.append(feat)
        return sampled_features
