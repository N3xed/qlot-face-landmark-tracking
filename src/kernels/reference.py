"""
Pure PyTorch reference implementation of fused conv + grid sample.

Mirrors the actual model pipeline: full per-query convolution (valid padding)
followed by grid_sample at unit-pixel-spaced positions around each query point.
"""

import torch
import torch.nn.functional as F


def fused_conv_sample_reference(
    feature_maps: torch.Tensor,
    conv_kernels: torch.Tensor,
    query_points: torch.Tensor,
    grid_size: tuple[int, int],
    sampling_mode: str = "bilinear",
    padding_mode: str = "border",
) -> torch.Tensor:
    """Reference: full conv (valid padding) then grid_sample.

    Args:
        feature_maps: (B, K*C, H, W) backbone feature maps.
        conv_kernels: (B, N, K, C, kH, kW) per-query dynamic convolution kernels.
        query_points: (B, N, 2) query point coordinates (x, y) normalised to [-1, 1]
                      relative to the full feature map sizes (H, W).
                      Uses align_corners=False convention.
        grid_size: (gH, gW) output grid resolution per query point.
        sampling_mode: 'bilinear' or 'nearest'.
        padding_mode: 'zeros' or 'border'.

    Returns:
        output: (B, N, K, gH, gW) sampled convolution results.
    """
    B, C_tot, H, W = feature_maps.shape
    _, N, K, C, kH, kW = conv_kernels.shape
    assert C_tot == K * C, f"Channel mismatch: feature_maps has {C_tot}, conv_kernels has {C} per K={K} kernels"
    gH, gW = grid_size

    # Valid padding conv map size
    H_out = H - kH + 1
    W_out = W - kW + 1

    # Full per-query convolution (grouped conv2d, valid padding)
    # See ImageFeatureCorrelator.forward()
    # Process each (batch, query) independently so groups=K is constant and
    # output_channels_per_group=1, ensuring cuDNN picks the same algorithm
    # regardless of B and N.
    conv_outs = []
    for b_idx in range(B):
        f_b = feature_maps[b_idx].unsqueeze(0)  # (1, K*C, H, W)
        batch_outs = []
        for n_idx in range(N):
            # Kernel for this query: (K, C, kH, kW) — one filter per group
            w_bn = conv_kernels[b_idx, n_idx]  # (K, C, kH, kW)
            out_bn = F.conv2d(f_b, w_bn, bias=None, padding=0, groups=K)  # (1, K, H_out, W_out)
            batch_outs.append(out_bn.squeeze(0))  # (K, H_out, W_out)
        conv_outs.append(torch.stack(batch_outs))  # (N, K, H_out, W_out)
    out = torch.stack(conv_outs).contiguous()  # (B, N, K, H_out, W_out)

    # Build sampling grid (unit-pixel spacing centred at query)
    # See get_cached_grid() in lmk_features.py
    yscale = (gH - 1) / H_out
    ty = torch.linspace(-yscale, yscale, gH, device=feature_maps.device)
    xscale = (gW - 1) / W_out
    tx = torch.linspace(-xscale, xscale, gW, device=feature_maps.device)
    ys, xs = torch.meshgrid(ty, tx, indexing="ij")  # (gH, gW)
    grid = torch.stack([xs, ys], dim=-1)  # (gH, gW, 2)

    # Translate grid to query points
    # See lookup_grid() in lmk_features.py

    query_points_scaled = torch.stack(
        [
            query_points[..., 0] * (W / W_out),  # x scale
            query_points[..., 1] * (H / H_out),  # y scale
        ],
        dim=-1,
    )  # (B, N, 2)
    grid = grid.unsqueeze(0).unsqueeze(0) + query_points_scaled.unsqueeze(2).unsqueeze(2)  # (B, N, gH, gW, 2)

    # Grid sample
    # See lookup_grid() in lmk_features.py
    out = F.grid_sample(
        out.view(B * N, K, H_out, W_out),  # (B*N, K, H_out, W_out)
        grid.view(B * N, gH, gW, 2),  # (B*N, gH, gW, 2)
        mode=sampling_mode,
        padding_mode=padding_mode,
        align_corners=False,
    )  # (B*N, K, gH, gW)

    return out.reshape(B, N, K, gH, gW)
