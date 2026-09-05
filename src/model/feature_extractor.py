import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from kernels.fused_conv_sample import fused_conv_sample
from .utils import init, StreamSafeBatchNorm


def _replace_batchnorms(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            reference = child.weight if child.affine else child.running_mean
            replacement = StreamSafeBatchNorm(
                num_features=child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                track_running_stats=child.track_running_stats,
                device=reference.device if reference is not None else None,
                dtype=reference.dtype if reference is not None else None,
            )
            replacement.load_state_dict(child.state_dict())
            replacement.train(child.training)
            setattr(module, name, replacement)
        else:
            _replace_batchnorms(child)


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


hgnetv2_stage3_mid_chs = 96
hgnetv2_stage3_out_chs = 384
hgnetv2_stage4_mid_chs = 96
hgnetv2_stage4_out_chs = 192

hgnetv2_b1_reduced = {
    "stem_type": "v2",
    "stem_chs": [24, 32],
    # in_chs, mid_chs, out_chs, blocks, downsample, light_block, kernel_size, layer_num
    "stage1": [32, 32, 64, 1, False, False, 3, 3],
    "stage2": [64, 48, 256, 1, True, False, 3, 3],
    "stage3": [256, hgnetv2_stage3_mid_chs, hgnetv2_stage3_out_chs, 2, True, True, 5, 3],
    "stage4": [hgnetv2_stage3_out_chs, hgnetv2_stage4_mid_chs, hgnetv2_stage4_out_chs, 1, True, True, 5, 3],
}


def _adapt_hgnetv2_b1_pretrained(state_dict):
    """
    Adapt the pretrained hgnetv2_b1 weights to reduced stage3/stage4 channels using
    L1-norm magnitude pruning.
    """
    print("Applying L1-norm pruning for hgnetv2_b1")

    # Original hgnetv2_b1 channels (timm)
    stage3_mid_orig = 96
    stage3_out_orig = 512
    stage4_mid_orig = 192
    stage4_out_orig = 1024

    # This adapter assumes all configured widths are <= original pretrained widths.
    stage3_agg0_new = hgnetv2_stage3_out_chs // 2
    stage4_agg0_new = hgnetv2_stage4_out_chs // 2

    def get_top_indices(weight, k):
        # Calculate L1 norm (magnitude) of each filter: sum(|w|) over (C_in, H, W)
        # weight shape is (C_out, C_in, H, W)
        if k >= weight.shape[0]:
            return torch.arange(weight.shape[0], device=weight.device)
        norms = weight.abs().sum(dim=(1, 2, 3))
        # Get indices of the top k filters
        _, indices = torch.topk(norms, k)
        # Sort indices to maintain filter ordering (optional, but cleaner)
        return torch.sort(indices)[0]

    def slice_bn(prefix, idx):
        for key in ["weight", "bias", "running_mean", "running_var"]:
            full_key = f"{prefix}.{key}"
            state_dict[full_key] = state_dict[full_key][idx]

    # --- Step 1: Prune Stage 3 (stages.2) ---

    # Keep consistent stage3 output channels across both blocks to preserve residual alignment.
    idx_s3_out = get_top_indices(state_dict["stages.2.blocks.1.aggregation.1.conv.weight"], hgnetv2_stage3_out_chs)

    # Stage3 block0: prune mid channels (96 -> stage3_mid_new)
    idx_s3_mid_b0 = get_top_indices(state_dict["stages.2.blocks.0.layers.0.conv1.conv.weight"], hgnetv2_stage3_mid_chs)

    # layers.0
    state_dict["stages.2.blocks.0.layers.0.conv1.conv.weight"] = state_dict["stages.2.blocks.0.layers.0.conv1.conv.weight"][
        idx_s3_mid_b0
    ]
    slice_bn("stages.2.blocks.0.layers.0.conv1.bn", idx_s3_mid_b0)

    state_dict["stages.2.blocks.0.layers.0.conv2.conv.weight"] = state_dict["stages.2.blocks.0.layers.0.conv2.conv.weight"][
        idx_s3_mid_b0
    ]
    slice_bn("stages.2.blocks.0.layers.0.conv2.bn", idx_s3_mid_b0)

    # layers.1/2
    for layer_i in [1, 2]:
        key_conv1 = f"stages.2.blocks.0.layers.{layer_i}.conv1.conv.weight"
        state_dict[key_conv1] = state_dict[key_conv1][idx_s3_mid_b0][:, idx_s3_mid_b0, :, :]
        slice_bn(f"stages.2.blocks.0.layers.{layer_i}.conv1.bn", idx_s3_mid_b0)

        key_conv2 = f"stages.2.blocks.0.layers.{layer_i}.conv2.conv.weight"
        state_dict[key_conv2] = state_dict[key_conv2][idx_s3_mid_b0]
        slice_bn(f"stages.2.blocks.0.layers.{layer_i}.conv2.bn", idx_s3_mid_b0)

    # Stage3 block0 aggregation.0 (out: 256 -> stage3_agg0_new)
    key_s3_b0_agg0_conv = "stages.2.blocks.0.aggregation.0.conv.weight"
    idx_s3_b0_agg0 = get_top_indices(state_dict[key_s3_b0_agg0_conv], stage3_agg0_new)
    # Input is [block input (256), 3 * mid]
    idx_s3_b0_agg0_in = torch.cat(
        [
            torch.arange(256, device=idx_s3_b0_agg0.device),
            256 + idx_s3_mid_b0,
            256 + stage3_mid_orig + idx_s3_mid_b0,
            256 + 2 * stage3_mid_orig + idx_s3_mid_b0,
        ]
    )
    state_dict[key_s3_b0_agg0_conv] = state_dict[key_s3_b0_agg0_conv][idx_s3_b0_agg0][:, idx_s3_b0_agg0_in, :, :]
    slice_bn("stages.2.blocks.0.aggregation.0.bn", idx_s3_b0_agg0)

    # Stage3 block0 aggregation.1 (out: 512 -> stage3_out_new)
    key_s3_b0_agg1_conv = "stages.2.blocks.0.aggregation.1.conv.weight"
    state_dict[key_s3_b0_agg1_conv] = state_dict[key_s3_b0_agg1_conv][idx_s3_out][:, idx_s3_b0_agg0, :, :]
    slice_bn("stages.2.blocks.0.aggregation.1.bn", idx_s3_out)

    # Stage3 block1: input channels come from stage3 output (idx_s3_out)
    idx_s3_mid_b1 = get_top_indices(state_dict["stages.2.blocks.1.layers.0.conv1.conv.weight"], hgnetv2_stage3_mid_chs)

    # layers.0
    key_s3_b1_l0_conv1 = "stages.2.blocks.1.layers.0.conv1.conv.weight"
    state_dict[key_s3_b1_l0_conv1] = state_dict[key_s3_b1_l0_conv1][idx_s3_mid_b1][:, idx_s3_out, :, :]
    slice_bn("stages.2.blocks.1.layers.0.conv1.bn", idx_s3_mid_b1)

    key_s3_b1_l0_conv2 = "stages.2.blocks.1.layers.0.conv2.conv.weight"
    state_dict[key_s3_b1_l0_conv2] = state_dict[key_s3_b1_l0_conv2][idx_s3_mid_b1]
    slice_bn("stages.2.blocks.1.layers.0.conv2.bn", idx_s3_mid_b1)

    # layers.1/2
    for layer_i in [1, 2]:
        key_conv1 = f"stages.2.blocks.1.layers.{layer_i}.conv1.conv.weight"
        state_dict[key_conv1] = state_dict[key_conv1][idx_s3_mid_b1][:, idx_s3_mid_b1, :, :]
        slice_bn(f"stages.2.blocks.1.layers.{layer_i}.conv1.bn", idx_s3_mid_b1)

        key_conv2 = f"stages.2.blocks.1.layers.{layer_i}.conv2.conv.weight"
        state_dict[key_conv2] = state_dict[key_conv2][idx_s3_mid_b1]
        slice_bn(f"stages.2.blocks.1.layers.{layer_i}.conv2.bn", idx_s3_mid_b1)

    # Stage3 block1 aggregation.0 (out: 256 -> stage3_agg0_new)
    key_s3_b1_agg0_conv = "stages.2.blocks.1.aggregation.0.conv.weight"
    idx_s3_b1_agg0 = get_top_indices(state_dict[key_s3_b1_agg0_conv], stage3_agg0_new)
    # Input is [block input (stage3_out_orig), 3 * mid]
    idx_s3_b1_agg0_in = torch.cat(
        [
            idx_s3_out,
            stage3_out_orig + idx_s3_mid_b1,
            stage3_out_orig + stage3_mid_orig + idx_s3_mid_b1,
            stage3_out_orig + 2 * stage3_mid_orig + idx_s3_mid_b1,
        ]
    )
    state_dict[key_s3_b1_agg0_conv] = state_dict[key_s3_b1_agg0_conv][idx_s3_b1_agg0][:, idx_s3_b1_agg0_in, :, :]
    slice_bn("stages.2.blocks.1.aggregation.0.bn", idx_s3_b1_agg0)

    # Stage3 block1 aggregation.1 (out: 512 -> stage3_out_new)
    key_s3_b1_agg1_conv = "stages.2.blocks.1.aggregation.1.conv.weight"
    state_dict[key_s3_b1_agg1_conv] = state_dict[key_s3_b1_agg1_conv][idx_s3_out][:, idx_s3_b1_agg0, :, :]
    slice_bn("stages.2.blocks.1.aggregation.1.bn", idx_s3_out)

    # --- Step 2: Prune Stage 4 (stages.3) ---

    # Stage4 downsample (input/output channels follow stage3 output)
    key_s4_down_conv = "stages.3.downsample.conv.weight"
    state_dict[key_s4_down_conv] = state_dict[key_s4_down_conv][idx_s3_out]
    slice_bn("stages.3.downsample.bn", idx_s3_out)

    # Stage4 block0 mid channels (192 -> stage4_mid_new)
    idx_s4_mid = get_top_indices(state_dict["stages.3.blocks.0.layers.0.conv1.conv.weight"], hgnetv2_stage4_mid_chs)

    # layers.0
    key_s4_l0_conv1 = "stages.3.blocks.0.layers.0.conv1.conv.weight"
    state_dict[key_s4_l0_conv1] = state_dict[key_s4_l0_conv1][idx_s4_mid][:, idx_s3_out, :, :]
    slice_bn("stages.3.blocks.0.layers.0.conv1.bn", idx_s4_mid)

    key_s4_l0_conv2 = "stages.3.blocks.0.layers.0.conv2.conv.weight"
    state_dict[key_s4_l0_conv2] = state_dict[key_s4_l0_conv2][idx_s4_mid]
    slice_bn("stages.3.blocks.0.layers.0.conv2.bn", idx_s4_mid)

    # layers.1/2
    for layer_i in [1, 2]:
        key_conv1 = f"stages.3.blocks.0.layers.{layer_i}.conv1.conv.weight"
        state_dict[key_conv1] = state_dict[key_conv1][idx_s4_mid][:, idx_s4_mid, :, :]
        slice_bn(f"stages.3.blocks.0.layers.{layer_i}.conv1.bn", idx_s4_mid)

        key_conv2 = f"stages.3.blocks.0.layers.{layer_i}.conv2.conv.weight"
        state_dict[key_conv2] = state_dict[key_conv2][idx_s4_mid]
        slice_bn(f"stages.3.blocks.0.layers.{layer_i}.conv2.bn", idx_s4_mid)

    # Stage4 block0 aggregation.0 (out: 512 -> stage4_agg0_new)
    key_s4_agg0_conv = "stages.3.blocks.0.aggregation.0.conv.weight"
    idx_s4_agg0 = get_top_indices(state_dict[key_s4_agg0_conv], stage4_agg0_new)
    # Input is [block input (stage3_out_orig), 3 * stage4_mid_orig]
    idx_s4_agg0_in = torch.cat(
        [
            idx_s3_out,
            stage3_out_orig + idx_s4_mid,
            stage3_out_orig + stage4_mid_orig + idx_s4_mid,
            stage3_out_orig + 2 * stage4_mid_orig + idx_s4_mid,
        ]
    )
    state_dict[key_s4_agg0_conv] = state_dict[key_s4_agg0_conv][idx_s4_agg0][:, idx_s4_agg0_in, :, :]
    slice_bn("stages.3.blocks.0.aggregation.0.bn", idx_s4_agg0)

    # Stage4 block0 aggregation.1 (out: 1024 -> stage4_out_new)
    key_s4_agg1_conv = "stages.3.blocks.0.aggregation.1.conv.weight"
    idx_s4_out = get_top_indices(state_dict[key_s4_agg1_conv], hgnetv2_stage4_out_chs)
    state_dict[key_s4_agg1_conv] = state_dict[key_s4_agg1_conv][idx_s4_out][:, idx_s4_agg0, :, :]
    slice_bn("stages.3.blocks.0.aggregation.1.bn", idx_s4_out)

    # Head input follows stage4 output channels.
    key_head_conv = "head.last_conv.0.weight"
    state_dict[key_head_conv] = state_dict[key_head_conv][:, idx_s4_out, :, :]

    return state_dict


class ChannelShuffle(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        g = self.groups
        x = x.view(b, g, c // g, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, c, h, w)
        return x


class FusionBlock(nn.Module):
    def __init__(self, channels: int, groups: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, groups=groups, bias=False)
        self.shuffle = ChannelShuffle(groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pointwise(x)
        x = self.shuffle(x)
        return x


class ImageFeatureExtractor(nn.Module):

    mean_tensor: torch.Tensor
    std_tensor: torch.Tensor

    def __init__(
        self,
        global_dim=256,
        d_model=256,
        n_heads: list[int] = [4, 6, 8],
        pretrained: bool | str = True,
    ):
        """
        Initializes the feature extractor.
        Args:
            pretrained: Whether to load pretrained weights. `True` loads pretrained
                weights from timm, `False` initializes randomly, and a string loads weights
                from the specified path.
            global_dim: Dimension of the global feature vector.
            d_model: Dimension to project feature maps to.
        """
        super().__init__()

        # Initialize backbone from timm, without the final classification layer
        self.out_indices = [-4, -3, -2, -1]

        self.backbone = timm.models.build_model_with_cfg(
            timm.models.hgnet.HighPerfGpuNet,
            "hgnetv2_b1.ssld_stage1_in22k_in1k",
            pretrained=pretrained if type(pretrained) == bool else False,
            use_lab=True,
            model_cfg=hgnetv2_b1_reduced,
            feature_cfg=dict(out_indices=self.out_indices, flatten_sequential=True),
            pretrained_filter_fn=_adapt_hgnetv2_b1_pretrained,
            features_only=True,
        )

        if not torch.compiler.is_exporting():
            _replace_batchnorms(self.backbone)

        # Normalization parameters for ImageNet (pretrained checkpoint is trained on ImageNet-1k)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.register_buffer("mean_tensor", torch.tensor(self.mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std_tensor", torch.tensor(self.std).view(1, 3, 1, 1), persistent=False)

        self.d_model = d_model
        self.feature_layers = [-4, -3, -2]
        self.num_feature_maps = len(self.feature_layers)

        self.feature_dims = [self.backbone.feature_info[i]["num_chs"] for i in self.out_indices]
        self.feature_map_reductions = [self.backbone.feature_info[i]["reduction"] for i in self.out_indices]

        self.projections = nn.ModuleList(
            [
                # Fast sparse 1x1 using a hardware-friendly group size
                (
                    nn.Sequential(
                        nn.Conv2d(fd, 128, kernel_size=1, bias=False, groups=4),
                        ChannelShuffle(groups=4),
                        nn.Conv2d(128, self.d_model, kernel_size=1, bias=False, groups=4),
                    )
                    if fd > self.d_model
                    else nn.Conv2d(fd, self.d_model, kernel_size=1, bias=False)
                )
                for fd, heads in zip(self.feature_dims[: len(self.feature_layers)], n_heads)
            ]
        )

        for proj in self.projections:
            if isinstance(proj, nn.Sequential):
                init(proj[0], nonlinearity="linear")
                init(proj[2], nonlinearity="linear")
            else:
                init(proj, nonlinearity="linear")

        # We have len(feature_layers) - 1 fusion operations.
        num_fusions = len(self.feature_layers) - 1
        assert len(n_heads) == num_fusions + 1, "Length of n_heads must match number of fusions + 1"

        self.fusion_conv = nn.ModuleList(FusionBlock(self.d_model, groups=4) for heads in n_heads[:-1])

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_proj = nn.Linear(self.feature_dims[-1], global_dim)
        init(self.global_proj, nonlinearity="linear")

        if type(pretrained) == str:
            self.load_state_dict(torch.load(pretrained))

    def freeze_backbone(self, freeze: bool = True):
        """
        Freeze or unfreeze the backbone parameters.
        Args:
            freeze: If True, freeze the backbone. If False, unfreeze it.
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze

    def forward(self, image: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        """
        Extract feature maps and global feature vector from the input image.

        Args:
            image: Input image tensor of shape (B, 3, H, W).
        Returns:
            A tuple containing
                - feature_maps: List of projected feature maps.
                - global_feat: Global feature vector of shape (B, global_dim).
        """
        image = image.clone().sub_(self.mean_tensor).div_(self.std_tensor)
        x = self.backbone(image)

        # Feature Pyramid Network, top-down fusion

        # Lateral projections (Change channels to d_model)
        projections = [proj(x[i]) for i, proj in enumerate(self.projections)]

        # Top-down fusion
        feature_maps = [projections[-1]]

        for i in reversed(range(len(projections) - 1)):
            high_res_map = projections[i]
            low_res_map = feature_maps[0]  # The map we just processed, shape (B, C, H_low, W_low)

            # Upsample low-res map to match high-res dimensions
            # Use bilinear interpolation for smooth gradients
            upsampled = F.interpolate(
                low_res_map, size=high_res_map.shape[-2:], mode="bilinear", align_corners=False
            )  # (B, C, H_high, W_high)

            fusion_conv = self.fusion_conv[i]
            m = fusion_conv(high_res_map + upsampled)
            feature_maps.insert(0, m)

        # Global Context
        global_map = x[-1]  # (B, C, H, W)
        global_feat = self.global_pool(global_map).flatten(1)  # (B, C)
        global_feat = self.global_proj(global_feat)  # (B, global_dim)

        return feature_maps, global_feat


class ImageFeatureCorrelator(nn.Module):
    def __init__(
        self,
        n_heads: list[int],
        n_feature_maps: int,
        resolutions: list[int],
        num_channels=256,
        query_dim=3,
    ):
        """
        Produce similarity maps between image feature maps and 3D query points.

        Args:
            d_model: Dimension of the image feature maps.
            n_heads: Number of correlation heads. This splits d_model into n_heads heads.
            n_feature_maps: Number of feature maps to process.
            resolutions: List of kernel resolutions for each feature map.
                Where >1 indicates convolution with that kernel size, 1 indicates a dot
                product, and 0 indicates dot product without projection first.
        """
        super().__init__()

        self.d_model = num_channels
        self.n_heads = n_heads
        self.n_feature_maps = n_feature_maps
        self.resolutions = [max(res, 1) for res in resolutions]
        self.query_dim = query_dim

        for n_head in n_heads:
            assert num_channels % n_head == 0, "d_model must be divisible by n_heads"
        self.C_heads = [num_channels // n_heads for n_heads in n_heads]

        assert (
            len(self.resolutions) == self.n_feature_maps
            and len(self.n_heads) == self.n_feature_maps
            and len(self.C_heads) == self.n_feature_maps
        ), "Length of resolutions and n_heads must match n_feature_maps"

        self.query_encoder = nn.Sequential(nn.Linear(query_dim, 128), nn.SiLU(), nn.Linear(128, 192), nn.SiLU())
        init(self.query_encoder[0], nonlinearity="relu")
        init(self.query_encoder[2], nonlinearity="relu")

        self.query_encoder_p = nn.ModuleList(
            [nn.Linear(192, n_head * C_head * (res**2)) for res, C_head, n_head in zip(resolutions, self.C_heads, n_heads)]
        )

        for layer in self.query_encoder_p:
            init(layer, nonlinearity="linear")

    def forward_fused_conv_sample(
        self,
        image_features: list[torch.Tensor],
        query_points: torch.Tensor,
        grid_size: list[tuple[int, int]],
        last_pred_coords: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Forward pass for the fused convolution + grid sampling kernel.

        Args:
            image_features: List of feature maps from the backbone, each of shape (B, C, H, W).
            query_points: 3D query points, shape (B, num_queries, 3).
            grid_size: List of (gH, gW) output grid sizes for each feature map.
            last_pred_coords: Last predicted 2D coordinates for each query point, shape (B, num_queries, 2).
            hidden_state: Global hidden state vector, shape (B, num_queries, 128).
            image_size: Original image size (height, width) passed into the backbone.
            feature_extractor: The ImageFeatureExtractor used to extract the features, passed to correct coordinate bias.
        Returns:
            Local features around each query point from all feature maps,
            list of shape (batch_size, num_queries, n_heads, gH, gW).
        """

        qe = self.query_encoder(query_points)  # (B, num_queries, 512)
        out_feat = []
        for i, (feat, gs) in enumerate(zip(image_features, grid_size)):
            res = self.resolutions[i]
            n_heads = self.n_heads[i]
            C_head = self.C_heads[i]

            qe_curr = self.query_encoder_p[i](qe)  # (B, num_queries, n_head * C_head * res^2)
            qe_curr = qe_curr.unflatten(-1, (n_heads, C_head, res, res))  # (B, num_queries, n_heads, C_head, res, res)

            # Apply coordinate shift correction if feature_extractor is provided
            pred_coords = last_pred_coords

            out = fused_conv_sample(
                feature_maps=feat,
                conv_kernels=qe_curr,
                query_points=pred_coords,
                grid_size=gs,
                sampling_mode="bilinear",
                padding_mode="border",
            )  # (B, num_queries, n_heads, gH, gW)
            out_feat.append(out)
        return out_feat

    def forward(
        self,
        image_features: list[torch.Tensor],
        query_points: torch.Tensor,
    ) -> list[torch.Tensor]:
        """
        Forward pass.
        Args:
            image_features: List of image features each with shape (batch_size, self.d_model, H, W) where H and W can vary.
            query_points: 3D query points, shape (batch_size, num_queries, 3).
            hidden_state: Global hidden state vector, shape (batch_size, num_queries, 128).
        Returns:
            List of similarity maps for each feature level, each with shape (batch_size,
            num_queries, n_heads, H, W) or if the kernel size is even (batch_size, num_queries, n_heads, H+1, W+1).
        """
        qe = self.query_encoder(query_points)  # (B, num_queries, 512)

        corr_maps = []
        for i, feat in enumerate(image_features):
            B, _, H, W = feat.shape

            res = self.resolutions[i]
            n_heads = self.n_heads[i]
            C_head = self.C_heads[i]
            # assert (
            #     C == self.d_model
            # ), f"Feature map channel dimension {C} does not match d_model {self.d_model}"

            qe_curr: torch.Tensor = self.query_encoder_p[i](qe)  # (B, num_queries, n_head * C_head * res^2)

            if res == 1:
                # Fast path for 1x1 kernels (dot product)
                # q: (B, N, h, C)
                # f: (B, h * C, H, W)
                if B == 1:
                    # ONNX export with fixed one batch size
                    # (1 * N, h, 1, C) @ (1, h, C, H * W) = (N, h, 1, H * W) -> (N, h, H, W)

                    qe_curr = qe_curr.view(-1, n_heads, 1, C_head)  # (B * num_queries, n_heads, 1, C_head)
                    feat = feat.view(1, n_heads, C_head, H * W)

                    out = qe_curr @ feat  # (N, h, 1, H * W)
                    out = out.view(1, -1, n_heads, H, W)  # (B, num_queries, n_heads, H, W)
                else:
                    qe_curr = qe_curr.view(B, -1, n_heads, C_head)  # (B, num_queries, n_heads, C_head)
                    feat = feat.view(B, n_heads, C_head, H, W)  # (B, n_heads, C_head, H, W)
                    out = torch.einsum("bqnc,bnchw->bqnhw", qe_curr, feat)  # (B, num_queries, n_heads, H, W)
            else:
                # Formulate as Grouped Convolution

                if False and B == 1:  # TODO: probably not worth it.
                    # f: (1, h * C, H, W)
                    # w: (1, N, h * C * res * res)
                    qe_curr = qe_curr.view(-1, n_heads, C_head, res, res)  # (1 * num_queries, n_heads, C_head, res, res)

                    # Input: (1, C, H, W)
                    # Weights: (N, C, res, res)
                    out_heads = []
                    for h in range(n_heads):
                        feat_in = feat[:, h * C_head : (h + 1) * C_head, :, :]  # (1, C_head, H, W)
                        w = qe_curr[:, h, :, :, :]  # (1 * num_queries, C_head, res, res)
                        out_head = torch.conv2d(feat_in, w, padding=0)  # (1, num_queries, H_out, W_out)
                        _, _, H_out, W_out = out_head.shape
                        out_head = out_head.view(1, -1, 1, H_out, W_out)  # (1, num_queries, 1, H_out, W_out)
                        out_heads.append(out_head)
                    out = torch.cat(out_heads, dim=2)  # (1, num_queries, n_heads, H_out, W_out)

                else:
                    # Input: (1, B * n_heads * C_head, H, W)
                    # Weights: (B * n_heads * num_queries, C_head, res, res)
                    # Groups: B * n_heads

                    feat_in = feat.view(1, B * n_heads * C_head, H, W)
                    # Permute kernel to align with groups
                    w = (
                        qe_curr.unflatten(-1, (n_heads, C_head, res, res)).movedim(1, 2).flatten(0, 2)
                    )  # (B * n_heads * num_queries, C_head, res, res)

                    out = nn.functional.conv2d(
                        feat_in, w, padding=0, groups=B * n_heads
                    )  # (1, B * n_heads * num_queries, H_out, W_out)

                    _, _, H_out, W_out = out.shape

                    # Reshape back to (B, n_heads, Q, ...) then transpose
                    out = out.view(B, n_heads, -1, H_out, W_out)  # (B, n_heads, num_queries, H_out, W_out)
                    out = out.transpose(1, 2)  # (B, Q, n_heads, H_out, W_out)
                    # Make contiguous to save memory for later reshaping operations
                    out = out.contiguous()

            corr_maps.append(out)

        return corr_maps
