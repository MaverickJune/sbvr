import datetime
from logging import Logger

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import LlamaTokenizerFast
import transformers
from eval_utils.modeling_llama import LlamaForCausalLM
from eval_utils.modeling_qwen3 import Qwen3ForCausalLM
from utils import data_utils, eval_utils, utils
from utils.process_args import process_args_ptq
from utils.model_utils import capture_layer_io, get_layer_io_save_path
from paper_eval_package.ppl_benchmark import evaluate_ppl
from eval_utils.multi_gpu_sbvr_main import sbvrize_model

import sys
from sbvr_e2e_utils.eval_ppl import eval_e2e_sbvr_ppl

log: Logger = utils.get_logger("spinquant")

def sbvr_spinquant_proess() -> None:
    '''
    First, load the whole model onto the CPU.
    Then, use torch multiprocessing to perform layer-wise quantization on each GPU.
    '''
    model_args, training_args, ptq_args = process_args_ptq()
    config = transformers.AutoConfig.from_pretrained(model_args.input_model)
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    
        
    config._attn_implementation = "sdpa"
    
    # load the model onto CPU
    model = Qwen3ForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_args.input_model,
        config=config,
        torch_dtype=dtype,
        device_map="cpu",
        low_cpu_mem_usage=True
    ).eval()
    
    print(f"model parameters are now loaded onto the shared memory of CPU")
    
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    sbvrize_model(ptq_args, model, model_args)
        

if __name__ == '__main__':
    sbvr_spinquant_proess()