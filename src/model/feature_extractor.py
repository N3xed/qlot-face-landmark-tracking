import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from itertools import pairwise
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

    map_channels: list[int]  # Number of channels for each output feature map
    n_maps: int  # Number of output feature maps
    global_channels: int  # Number of channels for the global feature vector
    backbone_map_strides: list[int]  # Stride of each backbone feature map relative to the input image

    def __init__(self, map_channels: list[int] = [64, 128, 192], pretrained: bool | str = True, stream_safe: bool = False):
        """
        Initializes the feature extractor.
        Args:
            map_channels: List of output channels for each feature map.
            pretrained: Whether to load pretrained weights. `True` loads pretrained
                weights from timm, `False` initializes randomly, and a string loads weights
                from the specified path.
            stream_safe: If True, replaces batchnorm layers with stream-safe versions.
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

        if not torch.compiler.is_exporting() and stream_safe:
            _replace_batchnorms(self.backbone)
            batch_norm_cls = StreamSafeBatchNorm
        else:
            batch_norm_cls = nn.BatchNorm2d

        # Normalization parameters for ImageNet (pretrained checkpoint is trained on ImageNet-1k)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.register_buffer("mean_tensor", torch.tensor(self.mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std_tensor", torch.tensor(self.std).view(1, 3, 1, 1), persistent=False)

        backbone_map_layers = [-4, -3, -2]
        feature_dims = [self.backbone.feature_info[i]["num_chs"] for i in self.out_indices]
        self.backbone_map_strides = [self.backbone.feature_info[i]["reduction"] for i in self.out_indices]

        assert len(map_channels) == len(backbone_map_layers), "Length of map_channels must match number of feature layers"
        self.map_channels = map_channels
        self.global_channels = feature_dims[-1]
        self.n_maps = len(self.map_channels)

        self.proj = nn.ModuleList(
            [
                nn.Conv2d(fd_backbone, fd_down, kernel_size=1) if fd_backbone != fd_down else nn.Identity()
                for fd_down, fd_backbone in zip(map_channels, feature_dims[: self.n_maps])
            ]
        )
        self.down_proj = nn.ModuleList(
            [
                nn.Conv2d(fd_next_level, fd, kernel_size=1) if fd != fd_next_level else nn.Identity()
                for fd, fd_next_level in pairwise(self.map_channels)
            ]
        )

        for proj in self.proj:
            if isinstance(proj, nn.Identity):
                continue
            init(proj, nonlinearity="linear")
        for proj in self.down_proj:
            if isinstance(proj, nn.Identity):
                continue
            init(proj, nonlinearity="linear")

        self.fusion_conv = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(fd, fd, kernel_size=3, groups=fd, padding=1, bias=False),
                batch_norm_cls(fd),
                nn.SiLU(),
            )
            for fd in self.map_channels
        )
        for conv in self.fusion_conv:
            assert isinstance(conv, nn.Sequential)
            init(conv[0], nonlinearity="linear")

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

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
        projections = [proj(x[i]) for i, proj in enumerate(self.proj)]

        # Contains maps of shape (B, C_i, H_i, W_i) for each level i
        feature_maps = []

        for i in reversed(range(len(self.down_proj))):
            high_res_map = projections[i]

            if feature_maps:
                low_res_map = feature_maps[0]  # The map we just processed, shape (B, C, H_low, W_low)
            else:
                # Top-most level, no previous feature map to fuse with
                low_res_map = projections[i + 1]
                low_res_map = self.fusion_conv[i + 1](low_res_map)
                feature_maps.insert(0, low_res_map)
            low_res_map = self.down_proj[i](low_res_map)

            # Upsample low-res map to match high-res dimensions
            upsampled = F.interpolate(low_res_map, size=high_res_map.shape[-2:], mode="nearest")  # (B, C, H_high, W_high)

            fusion_conv = self.fusion_conv[i]
            m = fusion_conv(high_res_map + upsampled)
            feature_maps.insert(0, m)

        # Global Context
        global_map = x[-1]  # (B, C_3, H, W)
        global_feat = self.global_pool(global_map).flatten(1)  # (B, C)

        return feature_maps, global_feat
