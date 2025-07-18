#! /bin/bash

##### model names #####
# meta-llama/Llama-3.2-1B
# meta-llama/Llama-3.2-3B
# meta-llama/Meta-Llama-3-8B
# meta-llama/Meta-Llama-3-70B

##### sbvr_root_paths #####
# /home/nxclab/wonjun/sbvr/quantized_model/meta-llama_Llama-3.2-1B_4_16_16

##### sbvrizer_paths #####
# /home/nxclab/wonjun/sbvr/input_profile/meta-llama_Llama-3.2-1B_4_16_16/per_state_encoding

python -m sbvr_e2e \
    --root_sbvr_path "/home/nxclab/wonjun/sbvr/quantized_model/Qwen_Qwen3-0.6B_4_16_16_w_rotate" \
    --input_model "Qwen/Qwen3-0.6B" \
    --weight_bvr_len 128 \
    --weight_num_sums 4 \
    --rtn_group_size 128 \
    --rtn_bits 7 \
    --measure_commonqa \
    # --measure_ppl \

    # --measure_latency \
    
