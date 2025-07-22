# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import torch
import transformers
import sys

from eval_utils import gptq_utils, rotation_utils
from utils import data_utils, fuse_norm_utils, hadamard_utils, quant_utils, utils, profile_utils
from utils.convert_to_executorch import (
    sanitize_checkpoint_from_spinquant,
    write_model_llama,
)
from utils.quant_utils import ActQuantWrapper

def ptq_model(args, model, model_args=None):
    transformers.set_seed(args.seed)
    model.eval()
    print(utils.b_str("Running PTQ training..."))
    # Rotate the weights
    if args.rotate:
        fuse_norm_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbose=True)

        quant_utils.add_actquant(model)  # Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if "down_proj" in name:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(
            model
        )  # Add Activation Wrapper to the model as the rest of the code assumes it is present

    if args.w_bits < 16:
        print(utils.b_str("Running PTQ weight quantization..."))
        save_dict = {}
        if args.load_qmodel_path:  # Load Quantized Rotated Model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert (
                not args.save_qmodel_path
            ), "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path + "/quantized_model.pt")
            model.load_state_dict(save_dict["model"])
        elif not args.w_rtn:  # GPTQ Weight Quantization
            trainloader = data_utils.get_wikitext2(
                nsamples=args.nsamples,
                seed=args.seed,
                model=model_args.input_model,
                seqlen=2048,
                eval_mode=False,
            )
            if args.export_to_et:
                # quantize lm_head and embedding with 8bit per-channel 
                # quantization with rtn for executorch
                quantizers = gptq_utils.rtn_fwrd(
                    model,
                    "cuda",
                    args,
                    custom_layers=[model.model.embed_tokens, model.lm_head],
                )
            # quantize other layers with gptq
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, "cuda", args)
            save_dict["w_quantizers"] = quantizers
        else:  # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, "cuda", args)
            save_dict["w_quantizers"] = quantizers

        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            if args.export_to_et:
                save_dict = write_model_llama(
                    model.state_dict(), model.config, num_shards=1
                )[0]  # Export num_shards == 1 for executorch
                save_dict = sanitize_checkpoint_from_spinquant(
                    save_dict, group_size=args.w_groupsize
                )
            local_rank = utils.get_local_rank()
            if local_rank == 0:
                print("saving quantized model to {}".format(args.save_qmodel_path + "/quantized_model.pt"))
                torch.save(save_dict, args.save_qmodel_path + "/quantized_model.pt")

    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        print(utils.b_str("Adding Input Quantization..."))
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0:
            down_proj_groupsize = utils.llama_down_proj_groupsize(
                model, args.a_groupsize
            )

        for name in qlayers:
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not (args.a_asym)
            layer_a_clip = args.a_clip_ratio

            num_heads = model.config.num_attention_heads
            model_dim = model.config.hidden_size
            head_dim = model_dim // num_heads

            if "v_proj" in name and args.v_bits < 16:  # Set the v_proj precision
                v_groupsize = head_dim
                qlayers[name].out_quantizer.configure(
                    bits=args.v_bits,
                    groupsize=v_groupsize,
                    sym=not (args.v_asym),
                    clip_ratio=args.v_clip_ratio,
                )

            if "o_proj" in name:
                layer_groupsize = head_dim

            if "lm_head" in name:  # Skip lm_head quantization
                layer_input_bits = 16

            if "down_proj" in name:  # Set the down_proj precision
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

            qlayers[name].quantizer.configure(
                bits=layer_input_bits,
                groupsize=layer_groupsize,
                sym=layer_a_sym,
                clip_ratio=layer_a_clip,
            )

    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = "apply_rotary_pos_emb"
            layers = model.model.layers
            k_quant_config = {
                "k_bits": args.k_bits,
                "k_groupsize": args.k_groupsize,
                "k_sym": not (args.k_asym),
                "k_clip_ratio": args.k_clip_ratio,
            }
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                    layer.self_attn,
                    rope_function_name,
                    config=model.config,
                    **k_quant_config,
                )
    
    # # debugging section
    # local_rank = utils.get_local_rank()
    # if local_rank == 0:
    #     print(model)
    #     for i,layer in enumerate(model.model.layers):
    #         print(f"layer {i}")
    #         print(layer.mlp.down_proj.online_full_had)
    #         print(layer.mlp.down_proj.online_partial_had)
    # sys.exit(0)

    return model

@ torch.inference_mode()
def sbvrize_model(model, ptq_args=None, model_args=None, forward_mode='naive'):
    '''
    WARNING: this function should be called only after the ptq_model() is called
    '''
    w_bits = ptq_args.w_bits
    a_bits = ptq_args.a_bits
    kv_bits = ptq_args.k_bits
    model_name = model_args.input_model
    
    target_attn_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    target_mlp_modules = ["gate_proj", "up_proj","down_proj"]
    name_convert_table = {
        "q_proj": "qkv_proj",
        "k_proj": "qkv_proj",
        "v_proj": "qkv_proj",
        "o_proj": "o_proj",
        "gate_proj": "mlp_upgate",
        "up_proj": "mlp_upgate",
        "down_proj": "mlp_down"
    }
    
    for idx, layer in enumerate(model.model.layers):
        # set sbvr_input_wrapper for each module
        device = layer.self_attn.q_proj.weight.device
        layer.self_attn.sbvr_input_wrapper = profile_utils.sbvr_input_wrapper(idx, model_name, w_bits, a_bits, kv_bits, device)
        layer.mlp.sbvr_input_wrapper = profile_utils.sbvr_input_wrapper(idx, model_name, w_bits, a_bits, kv_bits, device)
        
        # set variables in ActQuantWrapper for sbvr wrapping
        for target_module in target_attn_modules:
            target = getattr(layer.self_attn, target_module)
            if not isinstance(target, ActQuantWrapper):
                raise ValueError(f"Target module {idx}_{target_module} is not an ActQuantWrapper")
            target.e2e_sbvr_applied = True
            target.sbvr_forward_mode = forward_mode
            target.input_quantizer = layer.self_attn.sbvr_input_wrapper.quantizer_dict[name_convert_table[target_module]]
            
        for target_module in target_mlp_modules:
            target = getattr(layer.mlp, target_module)
            if not isinstance(target, ActQuantWrapper):
                raise ValueError(f"Target module {idx}_{target_module} is not an ActQuantWrapper")
            target.e2e_sbvr_applied = True
            target.sbvr_forward_mode = forward_mode
            target.input_quantizer = layer.mlp.sbvr_input_wrapper.quantizer_dict[name_convert_table[target_module]]
            if target_module == "down_proj":
                target.sbvrize_input_on_forward = True
    
    return model