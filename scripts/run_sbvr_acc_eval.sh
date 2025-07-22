#! /bin/bash

export PATH="$CONDA_PREFIX/bin:$PATH"

python -m sbvr_acc_eval \
    --root_sbvr_path "./quantized_model/Qwen_Qwen3-0.6B_4_16_16_w_rotate" \
    --input_model "Qwen/Qwen3-0.6B" \
    --weight_bvr_len 128 \
    --weight_num_sums 4 \
    --rtn_group_size 128 \
    --rtn_bits 7 \
    --measure_ppl \
    --measure_lm_eval \
    
