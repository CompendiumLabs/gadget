# global constants and enums

from enum import IntEnum

# file format constants
GGUF_MAGIC             = 0x46554747  # "GGUF"
GGUF_VERSION           = 3
GGUF_DEFAULT_ALIGNMENT = 32
GGML_MAX_DIMS          = 4

# field data types
class GGUFValueType(IntEnum):
    UINT8   = 0
    INT8    = 1
    UINT16  = 2
    INT16   = 3
    UINT32  = 4
    INT32   = 5
    FLOAT32 = 6
    BOOL    = 7
    STRING  = 8
    ARRAY   = 9
    UINT64  = 10
    INT64   = 11
    FLOAT64 = 12

# tensor data types
class GGMLQuantizationType(IntEnum):
    F32     = 0
    F16     = 1
    Q4_0    = 2
    Q4_1    = 3
    Q5_0    = 6
    Q5_1    = 7
    Q8_0    = 8
    Q8_1    = 9
    Q2_K    = 10
    Q3_K    = 11
    Q4_K    = 12
    Q5_K    = 13
    Q6_K    = 14
    Q8_K    = 15
    IQ2_XXS = 16
    IQ2_XS  = 17
    IQ3_XXS = 18
    IQ1_S   = 19
    IQ4_NL  = 20
    IQ3_S   = 21
    IQ2_S   = 22
    IQ4_XS  = 23
    I8      = 24
    I16     = 25
    I32     = 26
    I64     = 27
    F64     = 28
    IQ1_M   = 29
    BF16    = 30
    TQ1_0   = 34
    TQ2_0   = 35
    MXFP4   = 39
    COUNT   = 40

# embed pooling types
class LlamaPoolingType(IntEnum):
    UNSPECIFIED = -1,
    NONE = 0,
    MEAN = 1,
    CLS  = 2,
    LAST = 3,

# rope types
class LlamaRopeType(IntEnum):
    NONE   = -1,
    NORM   = 0,
    NEOX   = 2,
    MROPE  = 8,
    VISION = 24,

# arch -> rope type mapping
ARCH_ROPE_TYPE = {
    # these models do not use RoPE
    'gpt2': LlamaRopeType.NONE,
    'gptj': LlamaRopeType.NONE,
    'mpt': LlamaRopeType.NONE,
    'refact': LlamaRopeType.NONE,
    'bloom': LlamaRopeType.NONE,
    'mamba': LlamaRopeType.NONE,
    'mamba2': LlamaRopeType.NONE,
    'jamba': LlamaRopeType.NONE,
    'jina_bert_v2': LlamaRopeType.NONE,
    't5': LlamaRopeType.NONE,
    't5encoder': LlamaRopeType.NONE,
    'jais': LlamaRopeType.NONE,
    'rwkv6': LlamaRopeType.NONE,
    'rwkv6qwen2': LlamaRopeType.NONE,
    'rwkv7': LlamaRopeType.NONE,
    'arwkv7': LlamaRopeType.NONE,
    'wavtokenizer_dec': LlamaRopeType.NONE,
    'nemotron_h': LlamaRopeType.NONE,

    # use what we call a normal RoPE, operating on pairs of consecutive head values
    'llama': LlamaRopeType.NORM,
    'llada': LlamaRopeType.NORM,
    'llama4': LlamaRopeType.NORM,
    'deci': LlamaRopeType.NORM,
    'baichuan': LlamaRopeType.NORM,
    'starcoder': LlamaRopeType.NORM,
    'internlm2': LlamaRopeType.NORM,
    'minicpm': LlamaRopeType.NORM,
    'xverse': LlamaRopeType.NORM,
    'command_r': LlamaRopeType.NORM,
    'cohere2': LlamaRopeType.NORM,
    'olmo': LlamaRopeType.NORM,
    'arctic': LlamaRopeType.NORM,
    'deepseek': LlamaRopeType.NORM,
    'deepseek2': LlamaRopeType.NORM,
    'plm': LlamaRopeType.NORM,
    'chatglm': LlamaRopeType.NORM,
    'glm4': LlamaRopeType.NORM,
    'granite': LlamaRopeType.NORM,
    'granite_moe': LlamaRopeType.NORM,
    'granite_hybrid': LlamaRopeType.NORM,
    'chameleon': LlamaRopeType.NORM,
    'bailingmoe': LlamaRopeType.NORM,
    'neo_bert': LlamaRopeType.NORM,
    'smallm3': LlamaRopeType.NORM,
    'arcee': LlamaRopeType.NORM,
    'ernie4_5': LlamaRopeType.NORM,
    'ernie4_5_moe': LlamaRopeType.NORM,

    # the pairs of head values are offset by n_rot/2
    'falcon': LlamaRopeType.NEOX,
    'falcon_h1': LlamaRopeType.NEOX,
    'grok': LlamaRopeType.NEOX,
    'dbrx': LlamaRopeType.NEOX,
    'bert': LlamaRopeType.NEOX,
    'jina_bert_v3': LlamaRopeType.NEOX,
    'nomic_bert': LlamaRopeType.NEOX,
    'nomic_bert_moe': LlamaRopeType.NEOX,
    'stablelm': LlamaRopeType.NEOX,
    'bitnet': LlamaRopeType.NEOX,
    'qwen': LlamaRopeType.NEOX,
    'qwen2': LlamaRopeType.NEOX,
    'dream': LlamaRopeType.NEOX,
    'qwen2moe': LlamaRopeType.NEOX,
    'qwen3': LlamaRopeType.NEOX,
    'qwen3moe': LlamaRopeType.NEOX,
    'olmo2': LlamaRopeType.NEOX,
    'olmoe': LlamaRopeType.NEOX,
    'phi2': LlamaRopeType.NEOX,
    'phi3': LlamaRopeType.NEOX,
    'phimoe': LlamaRopeType.NEOX,
    'plamo2': LlamaRopeType.NEOX,
    'gemma': LlamaRopeType.NEOX,
    'gemma2': LlamaRopeType.NEOX,
    'gemma3': LlamaRopeType.NEOX,
    'gemma3n': LlamaRopeType.NEOX,
    'gemma_embedding': LlamaRopeType.NEOX,
    'starcoder2': LlamaRopeType.NEOX,
    'openelm': LlamaRopeType.NEOX,
    'gptneox': LlamaRopeType.NEOX,
    'codeshell': LlamaRopeType.NEOX,
    'orion': LlamaRopeType.NEOX,
    'nemotron': LlamaRopeType.NEOX,
    'exaone': LlamaRopeType.NEOX,
    'exaone4': LlamaRopeType.NEOX,
    'minicpm3': LlamaRopeType.NEOX,
    'dots1': LlamaRopeType.NEOX,
    'hunyuan_moe': LlamaRopeType.NEOX,
    'openai_moe': LlamaRopeType.NEOX,
    'hunyuan_dense': LlamaRopeType.NEOX,
    'lfm2': LlamaRopeType.NEOX,
    'smallthinker': LlamaRopeType.NEOX,
    'glm4_moe': LlamaRopeType.NEOX,
    'seed_oss': LlamaRopeType.NEOX,

    # use M-RoPE, operating on groups of 4 consecutive head values
    'qwen2vl': LlamaRopeType.MROPE,

    # all model arches should be listed explicitly here
    'unknown': LlamaRopeType.NONE,
}
