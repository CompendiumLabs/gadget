from .libs import _libggml

from .constants import(
    GGMLQuantizationType
)

from .libs._libggml import (
    # initialization
    GGML_DEFAULT_GRAPH_SIZE,
    ggml_init_params,
    ggml_init,
    ggml_free,
    ggml_type_size,
    ggml_tensor_overhead,
    ggml_graph_overhead,
    # backend
    ggml_backend_cpu_init,
    ggml_backend_free,
    ggml_backend_alloc_ctx_tensors,
    ggml_backend_get_default_buffer_type,
    ggml_backend_cpu_set_n_threads,
    ggml_backend_graph_compute,
    # allocation
    ggml_gallocr_new,
    ggml_gallocr_reserve,
    ggml_gallocr_get_buffer_size,
    ggml_gallocr_alloc_graph,
    # graphs
    ggml_new_graph,
    ggml_build_forward_expand,
    ggml_graph_compute_with_ctx,
    # tensors
    ggml_new_tensor_1d,
    ggml_new_tensor_2d,
    ggml_new_tensor_3d,
    ggml_new_tensor_4d,
    ggml_set_name,
    # tensor ops
    ggml_dup,
    ggml_dup_inplace,
    ggml_add as _ggml_add,
    ggml_add_inplace,
    ggml_add_cast,
    ggml_add1,
    ggml_add1_inplace,
    ggml_acc,
    ggml_acc_inplace,
    ggml_sub,
    ggml_sub_inplace,
    ggml_mul,
    ggml_mul_inplace,
    ggml_div,
    ggml_div_inplace,
    ggml_sqr,
    ggml_sqr_inplace,
    ggml_sqrt,
    ggml_sqrt_inplace,
    ggml_log,
    ggml_log_inplace,
    ggml_sum,
    ggml_sum_rows,
    ggml_mean,
    ggml_argmax,
    ggml_repeat,
    ggml_repeat_back,
    ggml_concat,
    ggml_abs,
    ggml_abs_inplace,
    ggml_sgn,
    ggml_sgn_inplace,
    ggml_neg,
    ggml_neg_inplace,
    ggml_step,
    ggml_step_inplace,
    ggml_tanh,
    ggml_tanh_inplace,
    ggml_elu,
    ggml_elu_inplace,
    ggml_relu,
    ggml_leaky_relu,
    ggml_relu_inplace,
    ggml_sigmoid,
    ggml_sigmoid_inplace,
    ggml_gelu,
    ggml_gelu_inplace,
    ggml_gelu_quick,
    ggml_gelu_quick_inplace,
    ggml_silu,
    ggml_silu_inplace,
    ggml_silu_back,
    ggml_hardswish,
    ggml_hardsigmoid,
    ggml_norm,
    ggml_norm_inplace,
    ggml_rms_norm,
    ggml_rms_norm_inplace,
    ggml_group_norm,
    ggml_group_norm_inplace,
    ggml_rms_norm_back,
    ggml_mul_mat as _ggml_mul_mat,
    ggml_mul_mat_set_prec,
    ggml_mul_mat_id,
    ggml_out_prod,
    ggml_scale,
    ggml_set,
    ggml_set_inplace,
    ggml_scale_inplace,
    ggml_set_1d,
    ggml_set_1d_inplace,
    ggml_set_2d,
    ggml_set_2d_inplace,
    ggml_cpy,
    ggml_cast,
    ggml_cont,
    ggml_cont_1d,
    ggml_cont_2d,
    ggml_cont_3d,
    ggml_cont_4d,
    ggml_reshape,
    ggml_reshape_1d,
    ggml_reshape_2d,
    ggml_reshape_3d,
    ggml_reshape_4d,
    ggml_view_1d,
    ggml_view_2d,
    ggml_view_3d,
    ggml_view_4d,
    ggml_permute,
    ggml_transpose,
    ggml_get_rows as _ggml_get_rows,
    ggml_get_rows_back,
    ggml_diag,
    ggml_diag_mask_inf,
    ggml_diag_mask_inf_inplace,
    ggml_diag_mask_zero,
    ggml_diag_mask_zero_inplace,
    ggml_soft_max,
    ggml_soft_max_inplace,
    ggml_soft_max_ext,
    ggml_soft_max_back,
    ggml_soft_max_back_inplace,
    ggml_rope,
    ggml_rope_inplace,
    ggml_rope_ext,
    ggml_rope_ext_inplace,
    ggml_rope_yarn_corr_dims,
    ggml_rope_back,
    ggml_clamp,
    ggml_im2col,
    ggml_conv_depthwise_2d,
    ggml_conv_1d,
    ggml_conv_1d_ph,
    ggml_conv_transpose_1d,
    ggml_conv_2d,
    ggml_conv_2d_sk_p0,
    ggml_conv_2d_s1_ph,
    ggml_conv_transpose_2d_p0,
    ggml_pool_1d,
    ggml_pool_2d,
    ggml_upscale,
    ggml_upscale_ext,
    ggml_pad,
    ggml_timestep_embedding,
    ggml_argsort,
    ggml_arange,
    ggml_top_k,
    ggml_flash_attn_ext,
    ggml_flash_attn_ext_set_prec,
    ggml_flash_attn_back,
    ggml_ssm_conv,
    ggml_ssm_scan,
    ggml_win_part,
    ggml_win_unpart,
    ggml_unary,
    ggml_unary_inplace,
    ggml_get_rel_pos,
    ggml_add_rel_pos,
    ggml_add_rel_pos_inplace,
)

# pre-emptive type checking and naming

def trim_shape(shape):
    dims = 1 + max([
        i for i, d in enumerate(shape) if d > 1
    ], default=0)
    return shape[:dims]

def get_tensor_shape(tensor, raw=False):
    value = tensor.contents
    shape = tuple(value.ne[:4])
    if not raw:
        shape = trim_shape(shape)[::-1]
    return shape

def get_tensor_type(tensor):
    value = tensor.contents
    return value.type

# this is inlined
def ggml_can_mul_mat(t0, t1):
    shape0 = get_tensor_shape(t0, raw=True)
    shape1 = get_tensor_shape(t1, raw=True)
    return (
        (shape0[0] == shape1[0]     ) and
        (shape0[2]  % shape1[2] == 0) and
        (shape0[3]  % shape1[3] == 0)
    )

# this is inlined
def ggml_can_repeat(t0, t1):
    shape0 = get_tensor_shape(t0, raw=True)
    shape1 = get_tensor_shape(t1, raw=True)
    return (
        (shape1[0] % shape0[0] == 0) and
        (shape1[1] % shape0[1] == 0) and
        (shape1[2] % shape0[2] == 0) and
        (shape1[3] % shape0[3] == 0)
    )

def ggml_can_get_rows(t0, t1):
    shape0 = get_tensor_shape(t0, raw=True)
    shape1 = get_tensor_shape(t1, raw=True)
    ttype1 = get_tensor_type(t1)
    ttype_i32 = GGMLQuantizationType.I32.value
    return (
        (shape0[2] == shape1[1]) and
        (shape1[3] == 1        ) and
        (ttype1    == ttype_i32)
    )

def ggml_mul_mat(ctx, a, b, name=None):
    if not ggml_can_mul_mat(a, b):
        shape_a, shape_b = get_tensor_shape(a), get_tensor_shape(b)
        raise ValueError(f'ggml_mul_mat: bad tensor dimensions {shape_a} and {shape_b}')
    c = _ggml_mul_mat(ctx, a, b)
    if name is not None:
        ggml_set_name(c, name.encode('utf-8'))
    return c

def ggml_add(ctx, a, b, name=None):
    if not ggml_can_repeat(b, a):
        shape_a, shape_b = get_tensor_shape(a), get_tensor_shape(b)
        raise ValueError(f'ggml_add: bad tensor dimensions {shape_a} and {shape_b}')
    c = _ggml_add(ctx, a, b)
    if name is not None:
        ggml_set_name(c, name.encode('utf-8'))
    return c

def ggml_get_rows(ctx, a, b, name=None):
    if not ggml_can_get_rows(a, b):
        shape_a, shape_b = get_tensor_shape(a), get_tensor_shape(b)
        raise ValueError(f'ggml_get_rows: bad tensor dimensions {shape_a} and {shape_b}')
    c = _ggml_get_rows(ctx, a, b)
    if name is not None:
        ggml_set_name(c, name.encode('utf-8'))
    return c
