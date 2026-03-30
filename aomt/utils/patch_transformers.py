import transformers.modeling_utils
import transformers.models.llama.modeling_llama

if not hasattr(transformers.modeling_utils, 'ALL_ATTENTION_FUNCTIONS'):
    # Mock ALL_ATTENTION_FUNCTIONS for compatibility with older VeOmni code
    # This is a map of attention implementation names to their forward functions
    transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS = {
        "eager": transformers.models.llama.modeling_llama.LlamaAttention,
        "flash_attention_2": transformers.models.llama.modeling_llama.LlamaFlashAttention2,
        "sdpa": transformers.models.llama.modeling_llama.LlamaSdpaAttention,
    }
    print("Patched transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS")
