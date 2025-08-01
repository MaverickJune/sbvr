import torch
import transformers
import sys
import os

from eval_utils import gptq_utils_4_44_2, rotation_utils
# from eval_utils import gptq_utils_4_53_2, rotation_utils
from utils import data_utils, fuse_norm_utils, hadamard_utils, quant_utils, utils, profile_utils
from utils.convert_to_executorch import (
    sanitize_checkpoint_from_spinquant,
    write_model_llama,
)
from utils.quant_utils import ActQuantWrapper

@torch.inference_mode()
def sbvrize_model(args, model, model_args=None):
    transformers.set_seed(args.seed)
    print(utils.b_str("Preprocessing the model for SBVR..."))
    
    # 1. first, rotate the model weights
    if args.rotate:
        fuse_norm_utils.fuse_layer_norms(model, eff_multi_gpu=True)
        rotation_utils.rotate_model(model, args, eff_multi_gpu=True)
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
    
    # 2. sbvrize the model
    if args.w_bits < 16:
        print(utils.b_str("Running SBVR PTQ weight quantization..."))
        if args.w_rtn:
            raise ValueError("RTN is not supported for SBVR PTQ")
        else:
            # get the dataset for blockwise-GPTQ
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            trainloader = data_utils.get_wikitext2(
                nsamples=args.nsamples,
                seed=args.seed,
                model=model_args.input_model,
                seqlen=2048,
                eval_mode=False,
            )
            if args.export_to_et:
                raise ValueError("Export to executorch is not supported for SBVR PTQ")
            
            # call the sbvr_ptq function
            gptq_utils_4_44_2.sbvrize_fwrd(model, dataloader=trainloader, args=args)
            
    if args.a_bits < 16 or args.v_bits < 16:
        raise ValueError("Activation and Value quantization is not supported for SBVR PTQ")
        
    if args.k_bits < 16:
        raise ValueError("K quantization is not supported for SBVR PTQ")