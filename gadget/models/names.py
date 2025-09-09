# model name mappings

# llama3.1 is the default
NAMES_LLAMA31 = {}

# qwen3 is similar
NAMES_QWEN3_EMBED = {
    'qwen3.context_length'                  : 'llama.context_length',
    'qwen3.block_count'                     : 'llama.block_count',
    'qwen3.attention.head_count'            : 'llama.attention.head_count',
    'qwen3.attention.head_count_kv'         : 'llama.attention.head_count_kv',
    'qwen3.rope.freq_base'                  : 'llama.rope.freq_base',
    'qwen3.attention.layer_norm_rms_epsilon': 'llama.attention.layer_norm_rms_epsilon',
}

# final name map
NAMES = {
    'LlamaForCausalLM': NAMES_LLAMA31,
    'Qwen3ForCausalLM': NAMES_QWEN3_EMBED,
}
