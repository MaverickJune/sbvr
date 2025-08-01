#! /bin/bash

##### model names #####
# meta-llama/Llama-3.2-1B
# Qwen/Qwen3-0.6B

# meta-llama/Llama-3.2-1B-Instruct
# ./quantized_model/meta-llama_Llama-3.2-1B-Instruct_4_16_16_w_rotate


export PATH="$CONDA_PREFIX/bin:$PATH"

python -m sbvr_e2e \
    --root_sbvr_path "./quantized_model/meta-llama_Llama-3.2-1B-Instruct_4_16_16_w_rotate" \
    --input_model "meta-llama/Llama-3.2-1B-Instruct" \
    --weight_bvr_len 128 \
    --weight_num_sums 4 \
    --rtn_group_size 128 \
    --rtn_bits 7 \
    --test_cudagraph
