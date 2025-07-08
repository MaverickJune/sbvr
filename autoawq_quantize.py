from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# Replace with the Hugging Face ID for the full-precision
# model_path = "meta-llama/Llama-3.2-1B"
model_path = "meta-llama/Llama-3.1-8B"  # Example for Llama-3.1-8B


# Where to save your AWQ-quantized 
# quant_path = "Llama-3.2-1B-AWQ-gemv"
quant_path = "Llama-3.1-8B-AWQ-gemv"  # Example for Llama-3.1-8B

# AWQ quantization settings
quant_config = {
    "zero_point": True,      # enable zero-point quantization
    "q_group_size": 128,     # group size for per-group quantization
    "w_bit": 4,              # 4-bit weights
    "version": "GEMV"        # use the GEMM-fused kernels
}

# 1) Load the full-precision Llama-3.2-1B model & tokenizer
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    trust_remote_code=True
)

# 2) Perform AWQ quantization
model.quantize(tokenizer, quant_config=quant_config)

# 3) Save the quantized model & tokenizer
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
