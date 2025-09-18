# common layers for reuse

from math import sqrt

from ..ggml import (
    LlamaPoolingType,
    ggml_dup,
    ggml_norm,
    ggml_rms_norm,
    ggml_mul,
    ggml_add,
    ggml_gelu,
    ggml_silu,
    ggml_rope_ext,
    ggml_mul_mat,
    ggml_permute,
    ggml_transpose,
    ggml_cont,
    ggml_cont_2d,
    ggml_reshape_2d,
    ggml_reshape_3d,
    ggml_soft_max_ext,
    ggml_set_output,
)
from ..tensor import get_tensor_shape

def linear_layer(ctx, x, weight, bias=None, name=None):
    x = ggml_mul_mat(ctx, weight, x)
    if bias is not None:
        x = ggml_add(ctx, x, bias)
    return ggml_dup(ctx, x, name=name)

def norm_layer(ctx, x, weight, bias=None, eps=0.0, rms=False, name=None):
    norm_func = ggml_rms_norm if rms else ggml_norm
    x = norm_func(ctx, x, eps)
    x = ggml_mul(ctx, x, weight)
    if bias is not None:
        x = ggml_add(ctx, x, bias)
    return ggml_dup(ctx, x, name=name)

def rope_extended(
    ctx, x, pos, n_dims, mode, freqs=None, n_ctx_orig=0, freq_base=10000.0, freq_scale=1.0,
    ext_factor=0.0, attn_factor=1.0, beta_fast=32.0, beta_slow=1.0, name=None
):
    return ggml_rope_ext(
        ctx, x, pos, freqs, n_dims, mode, n_ctx_orig, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow, name=name
    )

def attention_layer(
    ctx, x, n_heads, mask, wq, wk, wv, wo, bq=None, bk=None, bv=None, bo=None, wqn=None, wkn=None,
    n_heads_kv=None, rope_type=None, rope_freqs=None, freq_base=None, freq_scale=None, positions=None,
    alibi=0.0, layer_norm_eps=0.0, kv_cache=None, name=None
):
    # get n_heads_q and n_heads_kv
    n_heads_q = n_heads
    if n_heads_kv is None:
        n_heads_kv = n_heads

    # get dimensions
    embed_dim, batch_size = get_tensor_shape(x, trim=2)
    _, embed_dim_q = get_tensor_shape(wq, trim=2)
    _, embed_dim_k = get_tensor_shape(wk, trim=2)
    _, embed_dim_v = get_tensor_shape(wv, trim=2)

    # kv consistency
    if embed_dim_v != embed_dim_k:
        raise ValueError(f'embed_dim_v ({embed_dim_v}) must be equal to embed_dim_k ({embed_dim_k})')
    embed_dim_kv = embed_dim_k

    # check head divisibility
    if embed_dim_q % n_heads_q != 0:
        raise ValueError(f'embed_dim_q ({embed_dim_q}) must be divisble by n_heads_q ({n_heads_q})')
    if embed_dim_kv % n_heads_kv != 0:
        raise ValueError(f'embed_dim_kv ({embed_dim_kv}) must be divisble by n_heads_kv ({n_heads_kv})')

    # head dims match
    head_dim_q = embed_dim_q // n_heads_q
    head_dim_kv = embed_dim_kv // n_heads_kv
    if head_dim_q != head_dim_kv:
        raise ValueError(f'head_dim_q ({head_dim_q}) must be equal to head_dim_kv ({head_dim_kv})')
    head_dim = head_dim_q

    # compute query, key, value
    q = linear_layer(ctx, x, wq, bias=bq, name=f'{name}_q0')
    k = linear_layer(ctx, x, wk, bias=bk, name=f'{name}_k0')
    v = linear_layer(ctx, x, wv, bias=bv, name=f'{name}_v0')

    # reshape to head_dim
    q = ggml_reshape_3d(ctx, q, head_dim, n_heads_q, batch_size, name=f'{name}_q')
    k = ggml_reshape_3d(ctx, k, head_dim, n_heads_kv, batch_size, name=f'{name}_k')
    v = ggml_reshape_3d(ctx, v, head_dim, n_heads_kv, batch_size, name=f'{name}_v')

    # apply norm to q/k
    if wqn is not None:
        q = norm_layer(ctx, q, wqn, rms=True, eps=layer_norm_eps, name=f'{name}_q_norm')
    if wkn is not None:
        k = norm_layer(ctx, k, wkn, rms=True, eps=layer_norm_eps, name=f'{name}_k_norm')

    # apply rotary position embeddings
    if rope_type is not None:
        q = rope_extended(
            ctx, q, positions, head_dim, rope_type, freqs=rope_freqs, freq_base=freq_base,
            freq_scale=freq_scale, name=f'{name}_q_rope'
        )
        k = rope_extended(
            ctx, k, positions, head_dim, rope_type, freqs=rope_freqs, freq_base=freq_base,
            freq_scale=freq_scale, name=f'{name}_k_rope'
        )

    # apply kv cache
    if kv_cache is not None:
        k, v = kv_cache.update(k, v)

    # permute dimensions
    q = ggml_permute(ctx, q, 0, 2, 1, 3)
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3))

    # compute interactions
    head_wgt = 1.0/sqrt(head_dim)
    kq = ggml_mul_mat(ctx, k, q, name=f'{name}_pre_scores')
    kq = ggml_soft_max_ext(ctx, kq, mask, head_wgt, alibi, name=f'{name}_scores')

    # pull in values
    v = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3))
    kqv = ggml_mul_mat(ctx, v, kq)

    # merge dimensions
    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3)
    kqv = ggml_cont_2d(ctx, kqv, embed_dim_q, batch_size)

    # apply output layer
    out = linear_layer(ctx, kqv, wo, bias=bo, name=f'{name}_out')

    # return output
    return out

activations = {
    'gelu': ggml_gelu,
    'silu': ggml_silu,
}

# without gate layer, just a plain feed forward network
# with gate layer, it's a gated feed forward network
def feed_forward_layer(ctx, x, wu, wg, wd, bu=None, bg=None, bd=None, act='gelu', name=None):
    y = linear_layer(ctx, x, wg, bias=bg, name=f'{name}_gate')
    y = activations[act](ctx, y)
    g = linear_layer(ctx, x, wu, bias=bu, name=f'{name}_up')
    y = ggml_mul(ctx, y, g)
    y = linear_layer(ctx, y, wd, bias=bd, name=f'{name}_down')
    return y
