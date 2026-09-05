import triton
import triton.language as tl

_FWD_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=w, num_stages=s)
    for w in [2, 4, 8]
    for s in [1, 2, 3, 4]
]

_BWD_AUTOTUNE_CONFIGS = [
    triton.Config({}, num_warps=w, num_stages=s)
    for w in [2, 4, 8]
    for s in [1, 2, 3, 4]
]


@triton.autotune(configs=_FWD_AUTOTUNE_CONFIGS, key=['H', 'W'])
@triton.jit
def _fused_conv_sample_fwd_kernel(
    feature_maps_ptr, conv_kernels_ptr, query_points_ptr, output_ptr, conv_acc_ptr,
    B, N, H, W,
    stride_fm_b, stride_fm_c, stride_fm_h, stride_fm_w,
    stride_ck_b, stride_ck_n, stride_ck_k, stride_ck_c, stride_ck_kh, stride_ck_kw,
    stride_qp_b, stride_qp_n,
    stride_out_b, stride_out_n, stride_out_k, stride_out_h, stride_out_w,
    stride_ca_b, stride_ca_n, stride_ca_k, stride_ca_h, stride_ca_w,
    K: tl.constexpr, C: tl.constexpr, kH: tl.constexpr, kW: tl.constexpr,
    gH: tl.constexpr, gW: tl.constexpr,
    sampling_int: tl.constexpr, padding_int: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_GH: tl.constexpr, BLOCK_GW: tl.constexpr,
    SAVE_CONV_ACC: tl.constexpr, ACC_DTYPE_INT: tl.constexpr,
):
    pid_bn = tl.program_id(0)
    k = tl.program_id(1)
    b = pid_bn // N
    n = pid_bn % N

    # Spatial size of the feature map after theoretical full-image convolution
    H_out = H - kH + 1
    W_out = W - kW + 1

    # ---- 1. Map Queries to Grid ----
    store_dtype = feature_maps_ptr.dtype.element_ty
    # ACC_DTYPE_INT: 0=native (input dtype), 1=fp32, 2=fp64
    if ACC_DTYPE_INT == 2:
        acc_dtype = tl.float64
    elif ACC_DTYPE_INT == 1:
        acc_dtype = tl.float32
    else:
        acc_dtype = store_dtype
    # Coordinate math (floor, interpolation weights) needs at least fp32
    coord_dtype = tl.float64 if ACC_DTYPE_INT == 2 else tl.float32
    qp_base = query_points_ptr + b * stride_qp_b + n * stride_qp_n
    qx = tl.load(qp_base).to(coord_dtype)
    qy = tl.load(qp_base + 1).to(coord_dtype)

    # Convert query coordinates [-1, 1] into continuous spatial scales
    x_offset = (qx * W + W_out - gW) * 0.5
    y_offset = (qy * H + H_out - gH) * 0.5

    if sampling_int == 0:
        # Bilinear: Anchor at topleft integers (ix0, iy0) and store fractional remainder (dx, dy).
        # We fetch an integer block of size (gH+1, gW+1) so we can interpolate the +1 corners.
        ix0 = tl.math.floor(x_offset).to(tl.int32)
        iy0 = tl.math.floor(y_offset).to(tl.int32)
        dx = x_offset - ix0.to(coord_dtype)
        dy = y_offset - iy0.to(coord_dtype)
        gH_p, gW_p = gH + 1, gW + 1
    else:
        # Nearest: Simple rounding, compute exact (gH, gW) grid without fractions.
        ix0 = tl.math.floor(x_offset + 0.5).to(tl.int32)
        iy0 = tl.math.floor(y_offset + 0.5).to(tl.int32)
        dx, dy = 0.0, 0.0
        gH_p, gW_p = gH, gW

    # Generate offset arrays mapping purely to integer block patches
    offs_gy_p = tl.arange(0, BLOCK_GH)
    offs_gx_p = tl.arange(0, BLOCK_GW)
    mask_gp = (offs_gy_p[:, None] < gH_p) & (offs_gx_p[None, :] < gW_p)
    
    oy0_base = iy0 + offs_gy_p
    ox0_base = ix0 + offs_gx_p
    
    # Pre-compute boundary-safe clamping (padding='border' mode)
    oy0_safe = tl.minimum(tl.maximum(oy0_base, 0), H_out - 1)
    ox0_safe = tl.minimum(tl.maximum(ox0_base, 0), W_out - 1)
    
    fm_base = feature_maps_ptr + b * stride_fm_b
    ck_base = conv_kernels_ptr + b * stride_ck_b + n * stride_ck_n
    
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    
    fm_ptr_base = fm_base + (k * C + offs_c)[:, None, None] * stride_fm_c
    ck_k_base = ck_base + k * stride_ck_k
    
    row_h = oy0_safe * stride_fm_h
    col_w = ox0_safe * stride_fm_w

    if padding_int == 0:
        in_y0 = (oy0_base >= 0) & (oy0_base < H_out)
        in_x0 = (ox0_base >= 0) & (ox0_base < W_out)
        in_m = in_y0[:, None] & in_x0[None, :]

    # ---- 2. Inner Convolution Loop ----
    # Evaluate Bilinear(Conv(I, W), dx, dy) via the commutativity property:
    # First apply Conv(I, W) strictly along the exact pixel grids (0.0 offset).
    acc = tl.zeros((BLOCK_GH, BLOCK_GW), dtype=acc_dtype)

    for ky_i in tl.static_range(0, kH):
        for kx_i in tl.static_range(0, kW):
            ck_ptr = ck_k_base + offs_c * stride_ck_c + ky_i * stride_ck_kh + kx_i * stride_ck_kw
            ck_val = tl.load(ck_ptr, mask=mask_c, other=0.0).to(acc_dtype)

            r = row_h + ky_i * stride_fm_h
            c = col_w + kx_i * stride_fm_w
            
            # Read exact pixel indices, NO fractional lookups or overlapping loads.
            ptr = fm_ptr_base + r[None, :, None] + c[None, None, :]
            v = tl.load(ptr, mask=mask_c[:, None, None], other=0.0).to(acc_dtype)
            
            if padding_int == 0:
                v = tl.where(in_m[None, :, :], v, 0.0)
                
            acc += tl.sum(v * ck_val[:, None, None], axis=0)

    # ---- 3. Stash & Interpolate ----
    # Due to Triton's power-of-2 stationary register limitations, we cannot easily `[1:, 1:]` index 
    # the local accumulation grid `acc` directly to interpolate. 
    # Therefore, we stash it via L1 cache by bouncing it into the query `conv_acc_ptr` buffer.
    # This buffer is later used "for free" in the backwards pass to save re-computing the convolution entirely!
    ca_base = conv_acc_ptr + b * stride_ca_b + n * stride_ca_n + k * stride_ca_k
    if SAVE_CONV_ACC:
        # Save exact discrete chunk needed for backprop
        ca_ptr = ca_base + offs_gy_p[:, None] * stride_ca_h + offs_gx_p[None, :] * stride_ca_w
        tl.store(ca_ptr, acc, mask=mask_gp)
        # Ensure memory ops complete to read it back safely
        tl.debug_barrier()

    # Interpolate output
    offs_gy = tl.arange(0, BLOCK_GH)
    offs_gx = tl.arange(0, BLOCK_GW)
    mask_g = (offs_gy[:, None] < gH) & (offs_gx[None, :] < gW)
    
    if sampling_int == 0:
        # Load back 4 dynamically shifted views of the accumulated integer convolutions to do bilinear easily
        c_tl = tl.load(ca_base + offs_gy[:, None] * stride_ca_h + offs_gx[None, :] * stride_ca_w, mask=mask_g, other=0.0)
        c_tr = tl.load(ca_base + offs_gy[:, None] * stride_ca_h + (offs_gx[None, :] + 1) * stride_ca_w, mask=mask_g, other=0.0)
        c_bl = tl.load(ca_base + (offs_gy[:, None] + 1) * stride_ca_h + offs_gx[None, :] * stride_ca_w, mask=mask_g, other=0.0)
        c_br = tl.load(ca_base + (offs_gy[:, None] + 1) * stride_ca_h + (offs_gx[None, :] + 1) * stride_ca_w, mask=mask_g, other=0.0)
        out = c_tl * (1.0 - dy) * (1.0 - dx) + c_tr * (1.0 - dy) * dx + c_bl * dy * (1.0 - dx) + c_br * dy * dx
    else:
        out = acc

    out_base = output_ptr + b * stride_out_b + n * stride_out_n + k * stride_out_k
    out_ptr_k = out_base + offs_gy[:, None] * stride_out_h + offs_gx[None, :] * stride_out_w
    tl.store(out_ptr_k, out.to(store_dtype), mask=mask_g)


@triton.autotune(
    configs=_BWD_AUTOTUNE_CONFIGS, key=['H', 'W'],
    reset_to_zero=['grad_feature_maps_ptr', 'grad_query_points_ptr'],
)
@triton.jit
def _fused_conv_sample_bwd_fused_kernel(
    feature_maps_ptr, conv_kernels_ptr, query_points_ptr, conv_acc_ptr,
    grad_output_ptr, grad_feature_maps_ptr, grad_conv_kernels_ptr, grad_query_points_ptr,
    B, N, H, W,
    stride_fm_b, stride_fm_c, stride_fm_h, stride_fm_w,
    stride_ck_b, stride_ck_n, stride_ck_k, stride_ck_c, stride_ck_kh, stride_ck_kw,
    stride_qp_b, stride_qp_n,
    stride_go_b, stride_go_n, stride_go_k, stride_go_h, stride_go_w,
    stride_gfm_b, stride_gfm_c, stride_gfm_h, stride_gfm_w,
    stride_gck_b, stride_gck_n, stride_gck_k, stride_gck_c, stride_gck_kh, stride_gck_kw,
    stride_gqp_b, stride_gqp_n,
    stride_ca_b, stride_ca_n, stride_ca_k, stride_ca_h, stride_ca_w,
    K: tl.constexpr, C: tl.constexpr, kH: tl.constexpr, kW: tl.constexpr,
    gH: tl.constexpr, gW: tl.constexpr,
    sampling_int: tl.constexpr, padding_int: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_GH: tl.constexpr, BLOCK_GW: tl.constexpr,
    COMPUTE_DW: tl.constexpr, COMPUTE_DQ: tl.constexpr, COMPUTE_DF: tl.constexpr,
    ACC_DTYPE_INT: tl.constexpr,
):
    pid_bn = tl.program_id(0)
    k = tl.program_id(1)
    b = pid_bn // N
    n = pid_bn % N

    H_out = H - kH + 1
    W_out = W - kW + 1

    store_dtype = feature_maps_ptr.dtype.element_ty
    # ACC_DTYPE_INT: 0=native (input dtype), 1=fp32, 2=fp64
    if ACC_DTYPE_INT == 2:
        acc_dtype = tl.float64
    elif ACC_DTYPE_INT == 1:
        acc_dtype = tl.float32
    else:
        acc_dtype = store_dtype
    # Coordinate math (floor, interpolation weights) needs at least fp32
    coord_dtype = tl.float64 if ACC_DTYPE_INT == 2 else tl.float32
    qp_base = query_points_ptr + b * stride_qp_b + n * stride_qp_n
    qx = tl.load(qp_base).to(coord_dtype)
    qy = tl.load(qp_base + 1).to(coord_dtype)

    x_offset = (qx * W + W_out - gW) * 0.5
    y_offset = (qy * H + H_out - gH) * 0.5

    if sampling_int == 0:
        ix0 = tl.math.floor(x_offset).to(tl.int32)
        iy0 = tl.math.floor(y_offset).to(tl.int32)
        dx = x_offset - ix0.to(coord_dtype)
        dy = y_offset - iy0.to(coord_dtype)
        gH_p, gW_p = gH + 1, gW + 1
    else:
        ix0 = tl.math.floor(x_offset + 0.5).to(tl.int32)
        iy0 = tl.math.floor(y_offset + 0.5).to(tl.int32)
        dx, dy = 0.0, 0.0
        gH_p, gW_p = gH, gW

    offs_gy_p = tl.arange(0, BLOCK_GH)
    offs_gx_p = tl.arange(0, BLOCK_GW)

    # -------------------------------------------------------------------------
    # Backward Phase 1: Reverse Interpolation (d_out -> d_conv_acc)
    # The gradient splits proportionally onto local spatial integers similar to backwards bilinear.
    # -------------------------------------------------------------------------
    go_base = grad_output_ptr + b * stride_go_b + n * stride_go_n + k * stride_go_k
    
    if sampling_int == 0:
        # Load d_out shifted to 4 corners (essentially reading the 4 overlapping contributions
        # for a specific integer coordinate resulting from the forward pass spatial interpolation)
        mask_tl = (offs_gy_p[:, None] < gH) & (offs_gx_p[None, :] < gW)
        d_out_tl = tl.load(go_base + offs_gy_p[:, None]*stride_go_h + offs_gx_p[None, :]*stride_go_w, mask=mask_tl, other=0.0).to(acc_dtype)
        
        mask_tr = (offs_gy_p[:, None] < gH) & ((offs_gx_p[None, :] - 1) >= 0) & ((offs_gx_p[None, :] - 1) < gW)
        d_out_tr = tl.load(go_base + offs_gy_p[:, None]*stride_go_h + (offs_gx_p[None, :]-1)*stride_go_w, mask=mask_tr, other=0.0).to(acc_dtype)
        
        mask_bl = ((offs_gy_p[:, None] - 1) >= 0) & ((offs_gy_p[:, None] - 1) < gH) & (offs_gx_p[None, :] < gW)
        d_out_bl = tl.load(go_base + (offs_gy_p[:, None]-1)*stride_go_h + offs_gx_p[None, :]*stride_go_w, mask=mask_bl, other=0.0).to(acc_dtype)
        
        mask_br = ((offs_gy_p[:, None] - 1) >= 0) & ((offs_gy_p[:, None] - 1) < gH) & ((offs_gx_p[None, :] - 1) >= 0) & ((offs_gx_p[None, :] - 1) < gW)
        d_out_br = tl.load(go_base + (offs_gy_p[:, None]-1)*stride_go_h + (offs_gx_p[None, :]-1)*stride_go_w, mask=mask_br, other=0.0).to(acc_dtype)
        
        d_conv_acc = d_out_tl * (1-dx)*(1-dy) + d_out_tr * dx*(1-dy) + d_out_bl * (1-dx)*dy + d_out_br * dx*dy
    else:
        mask_nn = (offs_gy_p[:, None] < gH) & (offs_gx_p[None, :] < gW)
        d_conv_acc = tl.load(go_base + offs_gy_p[:, None]*stride_go_h + offs_gx_p[None, :]*stride_go_w, mask=mask_nn, other=0.0).to(acc_dtype)

    # -------------------------------------------------------------------------
    # Backward Phase 2: Query Gradients (d_Q)
    # Reusing the `conv_acc` (saved from Forward pass), we bypass needing to re-do 
    # the entire integer convolution loop just to find spatial differences.
    # -------------------------------------------------------------------------
    if COMPUTE_DQ and sampling_int == 0:
        ca_base = conv_acc_ptr + b * stride_ca_b + n * stride_ca_n + k * stride_ca_k
        mask_g = (offs_gy_p[:, None] < gH) & (offs_gx_p[None, :] < gW)
        
        # Load the 4 cached corners
        c_tl = tl.load(ca_base + offs_gy_p[:, None]*stride_ca_h + offs_gx_p[None, :]*stride_ca_w, mask=mask_g, other=0.0)
        c_tr = tl.load(ca_base + offs_gy_p[:, None]*stride_ca_h + (offs_gx_p[None, :]+1)*stride_ca_w, mask=mask_g, other=0.0)
        c_bl = tl.load(ca_base + (offs_gy_p[:, None]+1)*stride_ca_h + offs_gx_p[None, :]*stride_ca_w, mask=mask_g, other=0.0)
        c_br = tl.load(ca_base + (offs_gy_p[:, None]+1)*stride_ca_h + (offs_gx_p[None, :]+1)*stride_ca_w, mask=mask_g, other=0.0)
        
        # Taking finite derivatives in x and y using the stashed forward convolved tensor
        d_dx = (1.0 - dy)*(c_tr - c_tl) + dy*(c_br - c_bl)
        d_dy = (1.0 - dx)*(c_bl - c_tl) + dx*(c_br - c_tr)
        
        d_out_val = tl.load(go_base + offs_gy_p[:, None]*stride_go_h + offs_gx_p[None, :]*stride_go_w, mask=mask_g, other=0.0)
        
        # Sum pool over local grid
        grad_qx = tl.sum(d_out_val * d_dx)
        grad_qy = tl.sum(d_out_val * d_dy)
        
        gqp_base = grad_query_points_ptr + b * stride_gqp_b + n * stride_gqp_n
        tl.atomic_add(gqp_base, grad_qx * (W * 0.5))
        tl.atomic_add(gqp_base + 1, grad_qy * (H * 0.5))

    # -------------------------------------------------------------------------
    # Backward Phase 3: Commuted Correlation Loop (d_W / d_F)
    # Now we cross-correlate `d_conv_acc` against feature maps to get kernel grads (`d_W`), 
    # and convolve it against `conv_kernels` to scatter-add feature map grads (`d_F`).
    # -------------------------------------------------------------------------
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    
    oy0_base = iy0 + offs_gy_p
    ox0_base = ix0 + offs_gx_p
    
    oy0_safe = tl.minimum(tl.maximum(oy0_base, 0), H_out - 1)
    ox0_safe = tl.minimum(tl.maximum(ox0_base, 0), W_out - 1)
    
    fm_base = feature_maps_ptr + b * stride_fm_b
    fm_ptr_base = fm_base + (k * C + offs_c)[:, None, None] * stride_fm_c
    
    ck_base = conv_kernels_ptr + b * stride_ck_b + n * stride_ck_n
    ck_k_base = ck_base + k * stride_ck_k
    
    gck_base = grad_conv_kernels_ptr + b * stride_gck_b + n * stride_gck_n if COMPUTE_DW else None
    gck_k_base = gck_base + k * stride_gck_k if COMPUTE_DW else None
    
    gfm_base = grad_feature_maps_ptr + b * stride_gfm_b if COMPUTE_DF else None
    gfm_ptr_base = gfm_base + (k * C + offs_c)[:, None, None] * stride_gfm_c if COMPUTE_DF else None

    row_h = oy0_safe * stride_fm_h
    col_w = ox0_safe * stride_fm_w
    
    if COMPUTE_DF:
        gfm_row_h = oy0_safe * stride_gfm_h
        gfm_col_w = ox0_safe * stride_gfm_w
        
    in_m = None
    if padding_int == 0:
        in_y0 = (oy0_base >= 0) & (oy0_base < H_out)
        in_x0 = (ox0_base >= 0) & (ox0_base < W_out)
        in_m = in_y0[:, None] & in_x0[None, :]
        in_m = in_m[None, :, :]
        
    mask_gp = (offs_gy_p[:, None] < gH_p) & (offs_gx_p[None, :] < gW_p)
    d_conv_acc = tl.where(mask_gp, d_conv_acc, 0.0)

    for ky_i in tl.static_range(0, kH):
        for kx_i in tl.static_range(0, kW):
            ck_ptr = ck_k_base + offs_c * stride_ck_c + ky_i * stride_ck_kh + kx_i * stride_ck_kw
            ck_val = tl.load(ck_ptr, mask=mask_c, other=0.0).to(acc_dtype)
            
            if COMPUTE_DW:
                r = row_h + ky_i * stride_fm_h
                c = col_w + kx_i * stride_fm_w
                ptr = fm_ptr_base + r[None, :, None] + c[None, None, :]
                v = tl.load(ptr, mask=mask_c[:, None, None], other=0.0).to(acc_dtype)
                
                if padding_int == 0:
                    v = tl.where(in_m, v, 0.0)
                    
                d_w_val = tl.sum(tl.sum(v * d_conv_acc[None, :, :], axis=2), axis=1)
                gck_ptr = gck_k_base + offs_c * stride_gck_c + ky_i * stride_gck_kh + kx_i * stride_gck_kw
                tl.store(gck_ptr, d_w_val.to(store_dtype), mask=mask_c)

            if COMPUTE_DF:
                d_I_patch = d_conv_acc[None, :, :] * ck_val[:, None, None]
                if padding_int == 0:
                    d_I_patch = tl.where(in_m, d_I_patch, 0.0)
                    
                r_gfm = gfm_row_h + ky_i * stride_gfm_h
                c_gfm = gfm_col_w + kx_i * stride_gfm_w
                
                gfm_ptr = gfm_ptr_base + r_gfm[None, :, None] + c_gfm[None, None, :]
                
                mask_gfm = mask_c[:, None, None] & mask_gp[None, :, :]
                # Avoid out of bound writes by using in_m for atomics too!
                if padding_int == 0:
                    mask_gfm = mask_gfm & in_m
                    
                tl.atomic_add(gfm_ptr, d_I_patch.to(store_dtype), mask=mask_gfm)