# llama implementation

import numpy as np

from .ggml import (
    ggml_element_size,
    ggml_add_inplace,
    ggml_get_rows,
    ggml_transpose,
    ggml_view_1d,
    ggml_view_2d,
    ggml_cont,
)
from .tensor import get_tensor_info
from .cache import KVCache
from .layers import (
    linear_layer,
    norm_layer,
    attention_layer,
    feed_forward_layer,
)
from .model import GgmlModel, Parameter, State, Tensor

##
## llama model
##

def get_head_dim_kv(gguf):
    n_head_kv = gguf.get_field('llama.attention.head_count_kv')
    embed_size_kv = gguf.get_tensor_shape('blk.0.attn_k.weight')[1]
    assert embed_size_kv % n_head_kv == 0
    return embed_size_kv // n_head_kv

class LlamaModel(GgmlModel):
    batch_size    : Parameter('llama.context_length')
    context_length: Parameter('llama.context_length')
    head_dim_kv   : Parameter(get_head_dim_kv)

    n_tokens: State(None)
    n_past  : State(0)

    tokens   : Tensor('I32', ('batch_size',))
    positions: Tensor('I32', ('batch_size',))
    mask     : Tensor('F32', ('context_length', 'context_length'))
    kcache   : Tensor('F32', ('head_dim_kv', 'llama.attention.head_count_kv', 'context_length', 'llama.block_count'))
    vcache   : Tensor('F32', ('head_dim_kv', 'llama.attention.head_count_kv', 'context_length', 'llama.block_count'))

    # perform param validation here
    def __init__(self, params, tensors, **kwargs):
        # validate batch_size and context_length
        if (bs := params['batch_size']) > (cl := params['context_length']):
            raise ValueError('batch_size ({bs}) > context_length ({cl})')
        if (cl := params['context_length']) > (cl0 := params['llama.context_length']):
            raise ValueError('context_length ({cl}) > maximum context_length ({cl0})')

        # pass to model constructor
        super().__init__(params, tensors, **kwargs)

        # make kv cache
        self.kv_cache = KVCache(self.tensors['kcache'], self.tensors['vcache'])

    def __call__(self, tokens, positions, n_tokens):
        self.state['n_tokens'] = n_tokens
        self.state['n_past'] = self.kv_cache.n_past
        output = super().__call__(tokens=tokens, positions=positions)
        self.kv_cache.increment_past(n_tokens)
        return output

    def set_mask(self, mask):
        self.set_input('mask', mask)

    # llama model function
    def forward(self):
        ctx = self.ctx_graph

        # get runtime state
        n_tokens = self.state['n_tokens']

        # get params
        n_layers, n_heads_q, n_heads_kv, rope_base, layer_norm_rms_eps = self.params[
            'llama.block_count'            , 'llama.attention.head_count',
            'llama.attention.head_count_kv', 'llama.rope.freq_base'      ,
            'llama.attention.layer_norm_rms_epsilon',
        ]

        # get embed tensors
        etok, rope_freqs = self.tensors['token_embd.weight', 'rope_freqs.weight']

        # get input tensors
        tokens, positions, mask = self.tensors['tokens', 'positions', 'mask']

        # select used input tokens
        tokens = ggml_view_1d(ctx, tokens, n_tokens, 0, name='tokens_batch')
        positions = ggml_view_1d(ctx, positions, n_tokens, 0, name='positions_batch')

        # select used mask (contiguous for ggml_soft_max_ext)
        context_length = self.params['context_length']
        mask_stride = ggml_element_size(mask) * context_length
        mask = ggml_view_2d(ctx, mask, n_tokens, n_tokens, mask_stride, 0, name='mask_batch')
        mask = ggml_cont(ctx, mask)

        # get token embeddings
        cur = ggml_get_rows(ctx, etok, tokens, name='embed=tok')

        # loop over layers
        for i in range(n_layers):
            last = cur

            # get layer tensors
            wq, wk, wv, wo, wan, wu, wd, wg, wn, = self.tensors[
                f'blk.{i}.attn_q.weight'     , f'blk.{i}.attn_k.weight'   , f'blk.{i}.attn_v.weight'  ,
                f'blk.{i}.attn_output.weight', f'blk.{i}.attn_norm.weight', f'blk.{i}.ffn_up.weight'  ,
                f'blk.{i}.ffn_down.weight'   , f'blk.{i}.ffn_gate.weight' , f'blk.{i}.ffn_norm.weight',
            ]

            # get attention interactions
            cache = self.kv_cache.layer_view(ctx, self.graph, i)
            att = norm_layer(ctx, cur, wan, rms=True, eps=layer_norm_rms_eps, name=f'attn{i}_norm')
            att = attention_layer(
                ctx, att, n_heads_q, mask, wq, wk, wv, wo, positions=positions, n_heads_kv=n_heads_kv,
                rope_freqs=rope_freqs, rope_base=rope_base, eps=layer_norm_rms_eps, kv_cache=cache,
                name=f'attn{i}'
            )

            # add layer input to attention
            att = ggml_add_inplace(ctx, att, last)

            # feed forward network on current
            cur = norm_layer(ctx, att, wn, rms=True, eps=layer_norm_rms_eps, name=f'ffn{i}_norm')
            cur = feed_forward_layer(ctx, cur, wg, wd, wg=wu, act='silu', name=f'ffn{i}') # notice wg/wu flipped

            # add attention output to current tensor
            cur = ggml_add_inplace(ctx, cur, att)

        # get output tensors
        onw = self.tensors['output_norm.weight']
        ow = self.tensors.get('output.weight', etok)

        # generate output
        cur = norm_layer(ctx, cur, onw, rms=True, eps=layer_norm_rms_eps, name='output_norm')
        cur = linear_layer(ctx, cur, ow, name='output')

        # return logits
        return cur
