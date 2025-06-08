# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# nnodes determines the number of GPU nodes to utilize (usually 1 for an 8 GPU node)
# nproc_per_node indicates the number of GPUs per node to employ.

# meta-llama/Llama-3.2-1B
# meta-llama/Llama-3.2-3B
# meta-llama/Meta-Llama-3-8B
# meta-llama/Meta-Llama-3-70B

name=$(echo $1 | sed 's|/|_|g')
path="quantized_model/${name}_$2_$3_$4"
input_profile_path="input_profile/${name}_$2_$3_$4"

mkdir -p $path
mkdir -p $input_profile_path
torchrun --nnodes=1 --nproc_per_node=$5 ptq.py \
--input_model $1 \
--do_train False \
--do_eval True \
--per_device_eval_batch_size 4 \
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
--load_qmodel_path "$path" \
--sbvrize_input

