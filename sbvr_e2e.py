import torch
import argparse
import sys
import random
import os

DEFAULT_SEED = 42
def set_deterministic_seed(seed: int = DEFAULT_SEED):
    """Fix all random sources so that every run produces identical values."""
    random.seed(seed)
    torch.manual_seed(seed)          # covers both CPU and CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set CUBLAS workspace config for deterministic cuBLAS calls
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(f"[seed] All random sources fixed with seed={seed}")
    
set_deterministic_seed()

try:
    from eval_utils.modeling_llama_sbvr_4_44_2 import LlamaForSbvrLM
    _LLAMA_IMPORT_ERROR = None
except Exception as exc:
    LlamaForSbvrLM = None
    _LLAMA_IMPORT_ERROR = exc
try:
    from eval_utils.modeling_qwen3_sbvr_4_53_2 import Qwen3ForSbvrLM
    _QWEN_IMPORT_ERROR = None
except Exception as exc:  # Keep Llama path working even if optional Qwen deps mismatch.
    Qwen3ForSbvrLM = None
    _QWEN_IMPORT_ERROR = exc
try:
    from eval_utils.modeling_qwen2_sbvr_4_53_2 import Qwen2ForSbvrLM
    _QWEN2_IMPORT_ERROR = None
except Exception as exc:  # Keep Llama path working even if optional Qwen deps mismatch.
    Qwen2ForSbvrLM = None
    _QWEN2_IMPORT_ERROR = exc

# from eval_utils.modeling_llama_sbvr_4_44_2 import LlamaForSbvrLM
# from eval_utils.modeling_qwen3_sbvr_4_53_2 import Qwen3ForSbvrLM
# from eval_utils.modeling_qwen2_sbvr_4_53_2 import Qwen2ForSbvrLM
from transformers import AutoTokenizer, AutoConfig, LlamaTokenizerFast
from sbvr_e2e_utils.eval_ppl import r_str, g_str, y_str, b_str
from paper_eval_package.cudagraph_utils import attach_cudagraph_generate
from paper_eval_package.ppl_benchmark import evaluate_ppl
from paper_eval_package.latency_benchmark import evaluate_latency
from paper_eval_package.commonqa_benchmark import evaluate_commonqa, get_dataset_configs
from paper_eval_package.lm_eval_benchmark import evaluate_lm_eval_benchmark
from sbvr_e2e_utils.utils import get_partial_state

# for CUDA graph
from transformers import GenerationConfig
import contextlib
from transformers.cache_utils import StaticCache
# from cudagraph_utils import attach_cudagraph_generate

# (wjbang, 2026.03.06)
# This function is almost identical with main(), but only returns the sbvrized model without running any evaluation.
def load_sbvr_qwen2_model(
    root_sbvr_path: str,
    input_model: str,
    weight_bvr_len: int = 128,
    weight_num_sums: int = 4,
    rtn_group_size: int = 128,
    rtn_bits: int = 7
):
    # Check the transformer version (only allows transformers v4.53.2 for now)
    import transformers
    assert transformers.__version__ == "4.53.2", (
        f"This function requires transformers v4.53.2, but found v{transformers.__version__}"
    )
    
    args = argparse.Namespace(
        root_sbvr_path=root_sbvr_path,
        input_model=input_model,
        weight_bvr_len=weight_bvr_len,
        weight_num_sums=weight_num_sums,
        rtn_group_size=rtn_group_size,
        rtn_bits=rtn_bits,
        flash_attn=False
    )
    
    filtered_state = get_partial_state(args)
    sbvr_state_dict = {
        "weight_bvr_len": args.weight_bvr_len,
        "weight_num_sums": args.weight_num_sums,
        "rtn_group_size": args.rtn_group_size,
        "rtn_bits": args.rtn_bits,
    }
    config = AutoConfig.from_pretrained(args.input_model)
    
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, 
    # clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True

    model = Qwen2ForSbvrLM(config=config, sbvr_state_dict=sbvr_state_dict)
    model.load_sbvr_weights(args.root_sbvr_path)
    # fill partial state
    missing, unexpected = model.load_state_dict(
        filtered_state,
        strict=False
    )
    if len(unexpected) > 0:
        raise ValueError(f"Unexpected keys: {unexpected}")
    
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    # convert the model to float16
    model.convert_model_dtype(dtype=torch.float16)
    model = model.to("cuda:0")
    model.eval()
    model.config._attn_implementation = "flash_attention" if args.flash_attn else "sdpa"
    
    return model

# (wjbang, 2026.03.05)
# This function is almost identical with main(), but only returns the sbvrized model without running any evaluation.
def load_sbvr_llama_model(
    root_sbvr_path: str,
    input_model: str,
    weight_bvr_len: int = 128,
    weight_num_sums: int = 4,
    rtn_group_size: int = 128,
    rtn_bits: int = 7
):
    # Check the transformer version (only allows transformers v4.44.2 for now)
    import transformers
    assert transformers.__version__ == "4.44.2", (
        f"This function requires transformers v4.44.2, but found v{transformers.__version__}"
    )
    
    args = argparse.Namespace(
        root_sbvr_path=root_sbvr_path,
        input_model=input_model,
        weight_bvr_len=weight_bvr_len,
        weight_num_sums=weight_num_sums,
        rtn_group_size=rtn_group_size,
        rtn_bits=rtn_bits,
        flash_attn=False
    )
    
    filtered_state = get_partial_state(args)
    sbvr_state_dict = {
        "weight_bvr_len": args.weight_bvr_len,
        "weight_num_sums": args.weight_num_sums,
        "rtn_group_size": args.rtn_group_size,
        "rtn_bits": args.rtn_bits,
    }
    config = AutoConfig.from_pretrained(args.input_model)
    
    # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, 
    # clone lm_head from embed_tokens
    process_word_embeddings = False
    if config.tie_word_embeddings:
        config.tie_word_embeddings = False
        process_word_embeddings = True
    
    model = LlamaForSbvrLM(config=config, sbvr_state_dict=sbvr_state_dict)
    # fill partial state
    missing, unexpected = model.load_state_dict(
        filtered_state,
        strict=False
    )
    if len(unexpected) > 0:
        raise ValueError(f"Unexpected keys: {unexpected}")
    
    if process_word_embeddings:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model.load_sbvr_weights(args.root_sbvr_path)
    # convert the model to float16
    model.convert_model_dtype(dtype=torch.float16)
    model.preprocess_model()
    model = model.to("cuda:0")
    model.eval()
    model.config._attn_implementation = "flash_attention" if args.flash_attn else "sdpa"
    
    return model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_sbvr_path", type=str, default=None,
                        help="The root path of the sbvr weights")
    parser.add_argument("--input_model", type=str, default=None,
                        help="the name of the original model")
    parser.add_argument("--load_qmodel_path", type=str, default=None,
                        help="load the sbvr model from the given path")
    parser.add_argument("--save_qmodel_path", type=str, default=None,
                        help="save the quantized model to the specified path")
    parser.add_argument("--weight_bvr_len", type=int, default=128,
                        help="the bvr length of the sbvr weight")
    parser.add_argument("--weight_num_sums", type=int, default=4,
                        help="the number of sums of the sbvr weight")
    parser.add_argument("--rtn_group_size", type=int, default=128,
                        help="rtn group size of the sbvr input")
    parser.add_argument("--rtn_bits", type=int, default=7,
                        help="the number of bits of the sbvr input")
    parser.add_argument("--flash_attn", type=bool, default=False,
                        help="whether to use flash attention")
    parser.add_argument("--measure_ppl", action="store_true",
                        help="whether to measure the ppl of the model")
    parser.add_argument("--measure_lm_eval", action="store_true",
                        help="whether to measure the lm eval of the model")
    parser.add_argument("--test_cudagraph", action="store_true",
                        help="whether to test cudagraph")
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    # sanity check
    if args.root_sbvr_path is None and args.load_qmodel_path is None:
        raise ValueError("Either root_sbvr_path or load_qmodel_path must be provided")
    
    if args.load_qmodel_path is None:
        # config setups
        filtered_state = get_partial_state(args)
        
        sbvr_state_dict = {
            "weight_bvr_len": args.weight_bvr_len,
            "weight_num_sums": args.weight_num_sums,
            "rtn_group_size": args.rtn_group_size,
            "rtn_bits": args.rtn_bits,
        }
        config = AutoConfig.from_pretrained(args.input_model)
        
        # Llama v3.2 specific: Spinquant is not compatiable with tie_word_embeddings, 
        # clone lm_head from embed_tokens
        process_word_embeddings = False
        if config.tie_word_embeddings:
            config.tie_word_embeddings = False
            process_word_embeddings = True
    
    # prepare the model
    if args.load_qmodel_path is not None:
        if "Llama" in args.input_model:
            model = LlamaForSbvrLM.from_pretrained(
                args.load_qmodel_path,
                torch_dtype="auto",
                low_cpu_mem_usage=False
            )
        elif "Qwen3" in args.input_model:
            model = Qwen3ForSbvrLM.from_pretrained(
                args.load_qmodel_path,
                torch_dtype="auto",
                low_cpu_mem_usage=False
            )
        elif "Qwen2" in args.input_model:
            model = Qwen2ForSbvrLM.from_pretrained(
                args.load_qmodel_path,
                torch_dtype="auto",
                low_cpu_mem_usage=False
            )
    else:
        if "Llama" in args.input_model:
            model = LlamaForSbvrLM(config=config, sbvr_state_dict=sbvr_state_dict)
        elif "Qwen3" in args.input_model:
            model = Qwen3ForSbvrLM(config=config, sbvr_state_dict=sbvr_state_dict)
        elif "Qwen2" in args.input_model:
            model = Qwen2ForSbvrLM(config=config, sbvr_state_dict=sbvr_state_dict)
        else:
            print(args.input_model)
            raise ValueError(f"Unsupported model type: {args.input_model}")
        
        # fill partial state
        missing, unexpected = model.load_state_dict(
            filtered_state,
            strict=False
        )
        if len(unexpected) > 0:
            raise ValueError(f"Unexpected keys: {unexpected}")
        
        if process_word_embeddings:
            model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
        model.load_sbvr_weights(args.root_sbvr_path)
        
        # convert the model to float16
        model.convert_model_dtype(dtype=torch.float16)
        model.preprocess_model()
        model = model.to("cuda:0")
        model.eval()
    print(b_str("Model loaded"))
    model.config._attn_implementation = "flash_attention" if args.flash_attn else "sdpa"
    print(b_str(f"attn_implementation: {model.config._attn_implementation}"))
    # sys.exit(0)
        
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.input_model,
        padding_side="right",
        use_fast=True,
    )

    if args.save_qmodel_path:
        print(b_str("Saving the quantized model..."))
        model.save_pretrained(
            args.save_qmodel_path,
            safe_serialization=False,
            max_shard_size="4GB",
        )
        tokenizer.save_pretrained(args.save_qmodel_path)
        
    if args.measure_ppl:
        ppl = evaluate_ppl(
            model=model,
            tokenizer=tokenizer
        )
    
    if args.measure_lm_eval:
            results = evaluate_lm_eval_benchmark(
                model=model,
                tokenizer=tokenizer,
                model_path=args.input_model,
            )
            print(g_str("CommonQA evaluation completed"))
    
    if args.test_cudagraph:
        attach_cudagraph_generate(model, tokenizer,device="cuda:0", dtype=torch.float16)
        latency_result = evaluate_latency(
            model=model,
            tokenizer=tokenizer
        )
        
if __name__ == "__main__":
    main()
        
    
    
    
    
