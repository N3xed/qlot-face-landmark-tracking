import torch
import torch.nn as nn
import math
from dataclasses import dataclass
from .update_predictor import HeadedLinear
from .utils import NUM_PREDS_COORDS, NUM_PREDS_COV_PARAMS, NUM_PREDS_DELTA, init
import torch.nn.functional as F


@dataclass
class SampleGrid:
    grid: torch.Tensor


class LocalPatchSampler(nn.Module):

    _GRID_CACHE = {}

    def __init__(self, map_channels: list[int], sample_res: int = 3):
        super().__init__()

        self.sample_res = sample_res

        self.convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Flatten(0, 1),  # Flatten batch and query dimension
                    nn.Conv2d(in_channels=c, out_channels=c, kernel_size=sample_res, padding=0, groups=c)
                )
                for c in map_channels
            ]
        )

        with torch.no_grad():
            for c in self.convs:
                assert isinstance(c, nn.Sequential)
                conv = c[1]
                assert isinstance(conv, nn.Conv2d)
                init(conv, nonlinearity="linear")
                conv.weight.div_(2.0)
                conv.weight[:, :, sample_res // 2, sample_res // 2] = 1.0

    def forward(self, feature_maps: list[torch.Tensor], coords: torch.Tensor) -> list[torch.Tensor]:
        """
        Sample local patches from feature maps around the given coordinates,
        and apply depthwise convolutions to aggregate the local patches into feature vectors.

        Args:
            feature_maps: List of feature maps to sample from, each with shape (batch_size, C_i, H_i, W_i).
            coords: 2D query points in normalized [-1, 1] coordinates, shape (batch_size, num_queries, 2).

        Returns:
            List of sampled features, each with shape (batch_size, num_queries, C_i).
        """
        
        batch_size, num_queries, _ = coords.shape
        grids = [
            self.get_cached_grid(
                sample_res=self.sample_res,
                H=feat_map.shape[-2],
                W=feat_map.shape[-1],
                device=feat_map.device,
                dtype=feat_map.dtype,
            )
            for feat_map in feature_maps
        ]

        samples = self.lookup_grid(coords, feature_maps, grids)  # (batch_size, num_queries, C_i, res_h, res_w)
        samples = [
            conv(sample).view(batch_size, num_queries, -1) for conv, sample in zip(self.convs, samples)
        ]  # each (batch_size, num_queries, C_i)

        return samples

    @classmethod
    def get_cached_grid(
        cls,
        sample_res: int,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
        keep=10,
    ) -> SampleGrid:
        """
        Get a cached sampling grid for given sample resolution and feature map size.

        Args:
            sample_res: Resolution of the sample grid (e.g., 3 for 3x3 patches).
            H: Height of the feature map of the backbone.
            W: Width of the feature map of the backbone.
            device: Device to create the grid on.
            dtype: Data type of the grid.
            keep: Number of cached grids to keep.
        Returns:
            A sample grid object with grid of shape (sample_res, sample_res, 2) in normalized [-1, 1] coordinates.
        """

        # Assume same scale for (sample_res, H, W)
        key = (sample_res, H, W, device, dtype)
        if key in cls._GRID_CACHE:
            # Move to end to show it was recently used
            val = cls._GRID_CACHE.pop(key)
            cls._GRID_CACHE[key] = val
            return val

        assert H > 1 and W > 1, f"Feature map ({H}, {W}) invalid"

        t = torch.linspace(-1, 1, steps=sample_res, device=device, dtype=dtype)
        # Scale for align_corners=False (width = W)
        xs = t * (sample_res - 1.0) / W
        ys = t * (sample_res - 1.0) / H
        ys, xs = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack([xs, ys], dim=-1)  # (sample_res, sample_res, 2)

        val = SampleGrid(grid=base_grid)

        cls._GRID_CACHE[key] = val
        if len(cls._GRID_CACHE) > keep:
            first_key = next(iter(cls._GRID_CACHE))
            del cls._GRID_CACHE[first_key]

        return val

    @staticmethod
    def lookup_grid(
        query_points_xy: torch.Tensor,
        feature_maps: list[torch.Tensor],
        grids: list[SampleGrid],
    ) -> list[torch.Tensor]:
        """
        Lookup local patches around query points from multiple feature maps.

        Args:
            query_points_xy: 2D query points in normalized [-1, 1] coordinates, shape (batch_size, num_queries, 2).
            queried_feature_maps: List of feature maps to sample from, each with shape (batch_size, C_i, H_i, W_i).
            grids: List of SampleGrid objects corresponding to each feature map.
        Returns:
            Local features at each query point from all feature maps, list of shape
            (batch_size, num_queries, C_i, res_h, res_w).
        """
        batch_size, num_queries, _ = query_points_xy.shape

        # For each feature map, sample local features around each query point
        sampled_features = []
        for feat_map, grid in zip(feature_maps, grids):
            # assert feat_map.dim() == 5, f"{feat_map.shape=}"
            # assert feat_map.shape[:2] == (batch_size, num_queries), f"{feat_map.shape=}"
            _, C, _, _ = feat_map.shape

            base_grid = grid.grid  # (sample_res, sample_res, 2)
            res_h, res_w, _ = base_grid.shape

            base_grid = base_grid.view(1, 1, *base_grid.shape)  # (1, 1, sample_res, sample_res, 2)

            coords: torch.Tensor = query_points_xy.view(batch_size, num_queries, 1, 1, 2)
            coords = coords + base_grid  # (batch_size, num_queries, sample_res, sample_res, 2)
            coords = coords.view(batch_size, num_queries * res_h, res_w, 2)

            feat = torch.nn.functional.grid_sample(
                feat_map,  # (B, C, H, W)
                coords,
                align_corners=False,
                mode="bilinear",
                padding_mode="border",
            )  # (B, C, num_queries * res_h, res_w)
            feat = feat.view(batch_size, C, num_queries, res_h, res_w)
            feat = feat.transpose(1, 2)  # (batch_size, num_queries, C, res_h, res_w)
            sampled_features.append(feat)
        return sampled_features


class LearnedTempSoftShrink(nn.Module):
    """
    Symmetric soft-shrinkage activation with a learned temperature per channel.

    Computes ``f(x) = x * sigmoid((|x| - threshold) / temp)`` where ``temp`` is a
    per-channel learnable parameter (kept positive via ``exp(log_temp)``).

    Properties:
        - Large |x|: ``sigmoid -> 1`` so the output is ~linear pass-through (no
          saturation, unlike ``tanh``/``sigmoid``). Strong correlation responses
          keep their magnitude.
        - Small |x|: attenuated towards zero. With ``threshold > 0`` this becomes
          a smooth dead-zone whose width is controlled by ``threshold`` and whose
          sharpness is controlled by ``temp``.
        - Symmetric / odd: ``f(-x) = -f(x)``, so signed (anti-)correlation
          responses are treated consistently.
        - Non-zero gradient everywhere (unlike ``Tanhshrink`` whose gradient is
          ``tanh^2(x)`` and vanishes at the origin), which avoids stalling
          learning for poorly-localized queries whose correlation volume is
          near zero.

    The temperature is learned per channel (i.e. per correlation head). When
    instantiated once per feature level, this yields a learned temperature per
    head *and* per level.

    Args:
        num_channels: Number of channels (correlation heads) in the input.
        init_temp: Initial value of the per-channel temperature.
        threshold: Fixed shrinkage threshold (dead-zone half-width). ``0.0``
            gives pure temperature-gated soft attenuation with no hard dead-zone.
    """

    def __init__(self, num_channels: int, init_temp: float = 1.0, threshold: float = 0.0):
        super().__init__()
        self.threshold = float(threshold)
        self.num_channels = num_channels
        self.log_temp = nn.Parameter(torch.full((1, num_channels, 1, 1), math.log(init_temp)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W) where C == num_channels.
        """
        temp = self.log_temp.exp()
        return x * torch.sigmoid((x.abs() - self.threshold) / temp)

    def extra_repr(self) -> str:
        return (
            f"num_channels={self.log_temp.numel()}, "
            f"init_temp={math.exp(self.log_temp.mean().item()):.3f} (mean), "
            f"threshold={self.threshold}"
        )


class LmkFeatureEncoder(nn.Module):
    cov_scale: torch.Tensor

    def __init__(
        self,
        corr_feat_dim: int,
        context_feature_dim: int,
        corr_heads: list[int],
        corr_res: list[int],
        out_feat_dim: int = 196,
        inp_query_dim: int = 3,
        query_enc_hidden_dim: int = 64,
        query_enc_dim: int = 96,
    ):
        super().__init__()
        self.corr_feat_dim = corr_feat_dim
        self.context_feature_dim = context_feature_dim

        # Last prediction encoding
        self.preds_in_dim = max(NUM_PREDS_COV_PARAMS, NUM_PREDS_DELTA, NUM_PREDS_COORDS)
        self.pred_encoder = nn.Sequential(
            HeadedLinear(self.preds_in_dim, 32, heads=3, head_dim=-2, gain=nn.init.calculate_gain("relu")),
            nn.Flatten(-2),
            nn.SiLU(),
            nn.Linear(32 * 3, 96),
            nn.SiLU(),
        )
        init(self.pred_encoder[3], nonlinearity="relu")

        # Query points encoder
        self.context_fusion_dim = context_feature_dim + query_enc_dim + 96
        query_enc_hidden_dim = query_enc_hidden_dim
        self.query_encoder = nn.Sequential(
            nn.Linear(inp_query_dim, query_enc_hidden_dim),
            nn.SiLU(),
            nn.Linear(query_enc_hidden_dim, query_enc_dim),
            nn.SiLU(),
        )
        init(self.query_encoder[0], nonlinearity="relu")
        init(self.query_encoder[2], nonlinearity="relu")

        self.context_feat_mlp = nn.Sequential(
            nn.Linear(self.context_fusion_dim, out_feat_dim),
            nn.RMSNorm(out_feat_dim),
            nn.SiLU(),
        )
        init(self.query_encoder[0], nonlinearity="relu")

        self.context_feat_proj = nn.Linear(out_feat_dim, out_feat_dim)
        nn.init.xavier_normal_(self.context_feat_proj.weight, gain=0.1)
        nn.init.zeros_(self.context_feat_proj.bias)

        # Correlation feature encoding
        corr_channels = 128
        self.corr_hidden_dim = 96
        corr_convs = []
        for i, (n_heads, res) in enumerate(zip(corr_heads, corr_res)):
            spatial_out_dim = res - 3 + 1
            out_chan = int(math.ceil((corr_channels // (spatial_out_dim**2)) / float(n_heads)) * n_heads)
            flattened_dim = out_chan * spatial_out_dim**2
            print(f"corr_feat_conv_out_dim: {flattened_dim=}, {out_chan=}, {spatial_out_dim=}, {n_heads=}")

            branch = nn.Sequential(
                LearnedTempSoftShrink(num_channels=n_heads, init_temp=1.0, threshold=0.0),
                nn.Conv2d(in_channels=n_heads, out_channels=out_chan, kernel_size=3, groups=n_heads),
                nn.Flatten(start_dim=1),
                nn.SiLU(),
                nn.Linear(flattened_dim, self.corr_hidden_dim),
                nn.RMSNorm(self.corr_hidden_dim),
                nn.SiLU(),
            )
            init(branch[1], nonlinearity="relu")
            init(branch[4], nonlinearity="relu")
            corr_convs.append(branch)

        self.corr_feat_head_convs = nn.ModuleList(corr_convs)
        corr_feat_conv_out_dim = sum(self.corr_hidden_dim for _ in corr_heads)
        self.corr_feat_mlp = nn.Sequential(
            nn.Linear(corr_feat_conv_out_dim, out_feat_dim),
            nn.SiLU(),
            nn.Linear(out_feat_dim, out_feat_dim),
            nn.RMSNorm(out_feat_dim, elementwise_affine=False),
        )
        init(self.corr_feat_mlp[0], nonlinearity="relu")
        init(self.corr_feat_mlp[2], nonlinearity="linear")

        self.image_feature_proj = nn.Linear(self.corr_hidden_dim, 2 * context_feature_dim)
        nn.init.constant_(self.image_feature_proj.bias, 0.0)
        # Bias the FiLM scale (gamma) half towards 1.0 so image_context_features starts as an
        # identity pass-through instead of being scaled towards zero by the small Xavier weights.
        nn.init.constant_(self.image_feature_proj.bias[: self.context_feature_dim], 1.0)
        nn.init.xavier_normal_(self.image_feature_proj.weight, gain=0.1)

        self.out_corr_feat_dim = out_feat_dim
        self.out_context_dim = out_feat_dim
        self.n_levels = len(corr_heads)

        self.register_buffer("cov_scale", torch.tensor([1.0 / 6.0, 1.0 / 6.0, 1.0], dtype=torch.float32), persistent=False)

    def forward_query_points(self, query_points: torch.Tensor) -> torch.Tensor:
        """
        Encode query points.

        Args:
            query_points: Canonical 3D query points in the form of (x, y, z) with shape
            (num_queries, 3).
        Returns:
            Encoded query point features of shape (num_queries, query_enc_dim).
        """
        return self.query_encoder(query_points)

    def forward(
        self,
        corr_features: list[torch.Tensor],
        query_encoding: torch.Tensor,
        image_context_features: torch.Tensor,
        last_coords: torch.Tensor,
        last_cov: torch.Tensor,
        last_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode correlation features, context features, and last predictions into a unified feature space.

        Args:
            corr_features: Correlation features, list of shape (batch_size, num_queries, n_heads, sample_res, sample_res).
            query_encodings: Encoded query point features of shape (num_queries, query_enc_dim).
            context_features: Context features of shape (batch_size, num_queries, context_feature_dim).
            last_coords: Last prediction coords of shape (batch_size, num_queries, NUM_PREDS_COORDS).
            last_cov: Last prediction covariance of shape (batch_size, num_queries, NUM_PREDS_COV_PARAMS).
            last_delta: Last prediction delta of shape (batch_size, num_queries, NUM_PREDS_DELTA).
        Returns:
            Encoded features of shape (batch_size, num_queries, out_feature_dim).
        """
        batch_size = image_context_features.shape[0]

        # Encode last predictions (x, y, s11, s22, l21)
        cov = last_cov * self.cov_scale  # (B, num_queries, 3)
        if NUM_PREDS_COV_PARAMS < self.preds_in_dim:
            dims = self.preds_in_dim - NUM_PREDS_COV_PARAMS
            cov = F.pad(cov, (0, dims), value=0.0)

        delta = last_delta
        if NUM_PREDS_DELTA < self.preds_in_dim:
            dims = self.preds_in_dim - NUM_PREDS_DELTA
            delta = F.pad(delta, (0, dims), value=1.0)

        coords = last_coords
        if NUM_PREDS_COORDS < self.preds_in_dim:
            dims = self.preds_in_dim - NUM_PREDS_COORDS
            coords = F.pad(coords, (0, dims), value=-1.0)

        preds = torch.stack([cov, delta, coords], dim=-2)  # (B, num_queries, 3, preds_in_dim)
        pred_enc = self.pred_encoder(preds)  # (B, num_queries, pred_cov_dim)

        # Encode correlation features
        processed_corr_feats = []
        for i in range(self.n_levels):
            x = corr_features[i].flatten(0, 1)  # (B * num_queries, n_heads, sample_res, sample_res)
            x = self.corr_feat_head_convs[i](x)  # (B * num_queries, corr_hidden_dim)
            x = x.view(batch_size, -1, self.corr_hidden_dim)  # (B, num_queries, corr_hidden_dim)
            processed_corr_feats.append(x)

        corr_feat = torch.cat(processed_corr_feats, dim=-1)  # (B, num_queries, corr_feat_conv_out_dim)

        image_context_cond = self.image_feature_proj(processed_corr_feats[-1])  # (B, num_queries, 2 * context_feature_dim)

        # FiLM-based conditioning of image context features with highest-level correlation features
        image_context_features = (
            image_context_cond[..., : self.context_feature_dim]
        ) * image_context_features + image_context_cond[
            ..., self.context_feature_dim :
        ]  # (B, num_queries, context_feature_dim)

        context_feat = torch.cat(
            [
                pred_enc,
                query_encoding,
                image_context_features,
            ],
            dim=-1,
        )  # (B, num_queries, context_fusion_dim)

        context_feat = self.context_feat_mlp(context_feat)  # (B, num_queries, out_context_dim)
        corr_feat = self.corr_feat_mlp(corr_feat) + self.context_feat_proj(context_feat)  # (B, num_queries, out_corr_feat_dim)
        return context_feat, corr_feat
