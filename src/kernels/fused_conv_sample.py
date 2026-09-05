"""
torch.autograd.Function wrapper for the fused conv + grid sample Triton kernel.

Provides ``fused_conv_sample`` — the public entry-point.
"""

from __future__ import annotations

import torch
from ._triton_kernels import (
    _fused_conv_sample_fwd_kernel,
    _fused_conv_sample_bwd_fused_kernel,
)

def _next_power_of_2(x: int) -> int:
    """Return smallest power-of-2 >= x (minimum 1)."""
    if x <= 0:
        return 1
    return 1 << (x - 1).bit_length()

_SAMPLING_MODE_MAP = {"bilinear": 0, "nearest": 1}
_PADDING_MODE_MAP = {"zeros": 0, "border": 1}

class FusedConvSample(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        feature_maps: torch.Tensor,
        conv_kernels: torch.Tensor,
        query_points: torch.Tensor,
        grid_size: tuple[int, int],
        sampling_mode: str = "bilinear",
        padding_mode: str = "border",
        native_acc: bool = False,
    ) -> torch.Tensor:
        assert feature_maps.ndim == 4
        assert conv_kernels.ndim == 6
        assert query_points.ndim == 3 and query_points.shape[-1] == 2
        
        B, C_tot, H, W = feature_maps.shape
        B_k, N, K, C, kH, kW = conv_kernels.shape
        gH, gW = grid_size
        
        sampling_int = _SAMPLING_MODE_MAP[sampling_mode]
        padding_int = _PADDING_MODE_MAP[padding_mode]

        # Use slightly larger grid for internal integer convolution in bilinear mode
        gH_p = gH + 1 if sampling_int == 0 else gH
        gW_p = gW + 1 if sampling_int == 0 else gW
        
        BLOCK_C = _next_power_of_2(C)
        BLOCK_GH = _next_power_of_2(gH_p)
        BLOCK_GW = _next_power_of_2(gW_p)
        
        output = torch.empty(B, N, K, gH, gW, device=feature_maps.device, dtype=feature_maps.dtype)

        # ACC_DTYPE_INT: 0=native (input dtype), 1=fp32, 2=fp64
        if feature_maps.dtype == torch.float64:
            acc_dtype_int = 2
            acc_dtype = torch.float64
        elif feature_maps.dtype == torch.float32 or native_acc:
            # fp32 input: native == fp32, no upcast needed
            # native_acc=True: use input dtype as-is (e.g. fp16)
            acc_dtype_int = 0
            acc_dtype = feature_maps.dtype
        else:
            # fp16/bf16: accumulate in fp32 by default
            acc_dtype_int = 1
            acc_dtype = torch.float32

        # Allocate conv_acc buffer to store intermediate integer convolution
        save_conv_acc = query_points.requires_grad
        if save_conv_acc or sampling_int == 0:
            # We ALWAYS need it for bilinear since Triton 3 doesn't support register shifting natively
            conv_acc = torch.empty(B, N, K, gH_p, gW_p, device=feature_maps.device, dtype=acc_dtype)
        else:
            # Dummy tensor to pass pointer check
            conv_acc = torch.empty(1, device=feature_maps.device, dtype=acc_dtype)

        grid = (B * N, K)
        _fused_conv_sample_fwd_kernel[grid](
            feature_maps, conv_kernels, query_points, output, conv_acc,
            B, N, H, W,
            feature_maps.stride(0), feature_maps.stride(1), feature_maps.stride(2), feature_maps.stride(3),
            conv_kernels.stride(0), conv_kernels.stride(1), conv_kernels.stride(2), conv_kernels.stride(3), conv_kernels.stride(4), conv_kernels.stride(5),
            query_points.stride(0), query_points.stride(1),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3), output.stride(4),
            conv_acc.stride(0) if conv_acc.ndim > 1 else 0, conv_acc.stride(1) if conv_acc.ndim > 1 else 0,
            conv_acc.stride(2) if conv_acc.ndim > 1 else 0, conv_acc.stride(3) if conv_acc.ndim > 1 else 0, conv_acc.stride(4) if conv_acc.ndim > 1 else 0,
            K=K, C=C, kH=kH, kW=kW, gH=gH, gW=gW,
            sampling_int=sampling_int, padding_int=padding_int,
            BLOCK_C=BLOCK_C, BLOCK_GH=BLOCK_GH, BLOCK_GW=BLOCK_GW,
            SAVE_CONV_ACC=True if conv_acc.ndim > 1 else False,
            ACC_DTYPE_INT=acc_dtype_int,
        )

        ctx.save_for_backward(feature_maps, conv_kernels, query_points, conv_acc)
        ctx.grid_size = grid_size
        ctx.sampling_int = sampling_int
        ctx.padding_int = padding_int
        ctx.acc_dtype_int = acc_dtype_int
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        feature_maps, conv_kernels, query_points, conv_acc = ctx.saved_tensors
        B, C_tot, H, W = feature_maps.shape
        _, N, K, C, kH, kW = conv_kernels.shape
        gH, gW = ctx.grid_size
        sampling_int = ctx.sampling_int
        padding_int = ctx.padding_int
        acc_dtype_int = ctx.acc_dtype_int

        gH_p = gH + 1 if sampling_int == 0 else gH
        gW_p = gW + 1 if sampling_int == 0 else gW
        BLOCK_C = _next_power_of_2(C)
        BLOCK_GH = _next_power_of_2(gH_p)
        BLOCK_GW = _next_power_of_2(gW_p)
        
        grad_output = grad_output.contiguous()
        need_fm, need_ck, need_qp = ctx.needs_input_grad[:3]

        grad_feature_maps = torch.zeros_like(feature_maps) if need_fm else torch.empty(1, device=feature_maps.device)
        grad_conv_kernels = torch.empty_like(conv_kernels) if need_ck else torch.empty(1, device=conv_kernels.device)
        grad_query_points = torch.zeros_like(query_points) if need_qp else torch.empty(1, device=query_points.device)

        grid = (B * N, K)
        _fused_conv_sample_bwd_fused_kernel[grid](
            feature_maps, conv_kernels, query_points, conv_acc,
            grad_output, grad_feature_maps, grad_conv_kernels, grad_query_points,
            B, N, H, W,
            feature_maps.stride(0), feature_maps.stride(1), feature_maps.stride(2), feature_maps.stride(3),
            conv_kernels.stride(0), conv_kernels.stride(1), conv_kernels.stride(2), conv_kernels.stride(3), conv_kernels.stride(4), conv_kernels.stride(5),
            query_points.stride(0), query_points.stride(1),
            grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3), grad_output.stride(4),
            grad_feature_maps.stride(0) if need_fm else 0, grad_feature_maps.stride(1) if need_fm else 0, grad_feature_maps.stride(2) if need_fm else 0, grad_feature_maps.stride(3) if need_fm else 0,
            grad_conv_kernels.stride(0) if need_ck else 0, grad_conv_kernels.stride(1) if need_ck else 0, grad_conv_kernels.stride(2) if need_ck else 0, grad_conv_kernels.stride(3) if need_ck else 0, grad_conv_kernels.stride(4) if need_ck else 0, grad_conv_kernels.stride(5) if need_ck else 0,
            grad_query_points.stride(0) if need_qp else 0, grad_query_points.stride(1) if need_qp else 0,
            conv_acc.stride(0) if conv_acc.ndim > 1 else 0, conv_acc.stride(1) if conv_acc.ndim > 1 else 0, conv_acc.stride(2) if conv_acc.ndim > 1 else 0, conv_acc.stride(3) if conv_acc.ndim > 1 else 0, conv_acc.stride(4) if conv_acc.ndim > 1 else 0,
            K=K, C=C, kH=kH, kW=kW, gH=gH, gW=gW,
            sampling_int=sampling_int, padding_int=padding_int,
            BLOCK_C=BLOCK_C, BLOCK_GH=BLOCK_GH, BLOCK_GW=BLOCK_GW,
            COMPUTE_DW=need_ck, COMPUTE_DQ=need_qp, COMPUTE_DF=need_fm,
            ACC_DTYPE_INT=acc_dtype_int,
        )

        return (
            grad_feature_maps if need_fm else None,
            grad_conv_kernels if need_ck else None,
            grad_query_points if need_qp else None,
            None, None, None, None
        )

def fused_conv_sample(
    feature_maps: torch.Tensor, conv_kernels: torch.Tensor, query_points: torch.Tensor,
    grid_size: tuple[int, int], sampling_mode: str = "bilinear", padding_mode: str = "border",
    native_acc: bool = False,
) -> torch.Tensor:
    return FusedConvSample.apply(feature_maps, conv_kernels, query_points, grid_size, sampling_mode, padding_mode, native_acc)
