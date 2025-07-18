#! /bin/bash

# set variables
export PATH="$CONDA_PREFIX/bin:$PATH"
name=$(echo $1 | sed 's|/|_|g')
path="quantized_model/${name}_$2_$3_$4"
mkdir -p $path

# meta-llama/Llama-3.2-1B
# meta-llama/Llama-3.2-3B
# meta-llama/Llama-3.1-8B
# meta-llama/Llama-3.1-70B
# Qwen/Qwen3-0.6B-Base

# launch the python script
python -m ptq_eff_multi_gpu \
--input_model $1 \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 1 \
--model_max_length 2048 \
--fp16 False \
--bf16 True \
--save_safetensors False \
--w_bits $2 \
--a_bits $3 \
--k_bits $4 \
--v_bits $4 \
--w_clip \
--a_asym \
--k_asym \
--v_asym \
--k_groupsize 128 \
--v_groupsize 128 \
--w_sbvr \
--bvr_groupsize 128 \
--gptq_blockwise \
--rotate \
--rotate_mode "hadamard" \
--save_qmodel_path "$path"