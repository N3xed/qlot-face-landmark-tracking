"""
Fused convolution + grid sampling Triton kernels.

This package provides a fused kernel that combines per-query dynamic convolution
of backbone feature maps with grid sampling, avoiding the materialization of
full-resolution correlation maps.

Public API:
    fused_conv_sample: Autograd-compatible function for fused conv + grid sample.
    fused_conv_sample_reference: Pure PyTorch reference implementation for testing.
"""

from .fused_conv_sample import fused_conv_sample, FusedConvSample
from .reference import fused_conv_sample_reference

__all__ = [
    "fused_conv_sample",
    "fused_conv_sample_reference",
    "FusedConvSample",
]
