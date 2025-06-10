import torch
import argparse
from eval_utils.modelling_llama_sbvr import LlamaForSbvrLM
from transformers import AutoTokenizer, AutoConfig, LlamaTokenizerFast
from sbvr_e2e_utils.eval_ppl import eval_e2e_sbvr_ppl, r_str, g_str, y_str, b_str
from sbvr_e2e_utils.utils import get_partial_state

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_sbvr_path", type=str, default=None,
                        help="The root path of the sbvr weights")
    parser.add_argument("--sbvrizer_path", type=str, default=None,
                        help="The path of the sbvrizer weights")
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
    parser.add_argument("--input_bvr_len", type=int, default=128,
                        help="the bvr length of the sbvr input")
    parser.add_argument("--input_num_sums", type=int, default=8,
                        help="the number of sums of the sbvr input")
    parser.add_argument("--input_set_size", type=int, default=4,
                        help="the set size of the sbvr input")
    parser.add_argument("--flash_attn", type=bool, default=False,
                        help="whether to use flash attention")
    parser.add_argument("--measure_ppl", action="store_true",
                        help="whether to measure the ppl of the model")
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    # sanity check
    if args.root_sbvr_path is None and args.load_qmodel_path is None:
        raise ValueError("Either root_sbvr_path or load_qmodel_path must be provided")
    if args.root_sbvr_path is not None and args.sbvrizer_path is None:
        raise ValueError("sbvrizer_path must be provided if root_sbvr_path is provided")
    if args.sbvrizer_path is not None and args.load_qmodel_path is not None:
        raise ValueError("sbvrizer_path must be provided if load_qmodel_path is provided")
    
    filtered_state = get_partial_state(args)
    
    sbvr_state_dict = {
        "weight_bvr_len": args.weight_bvr_len,
        "weight_num_sums": args.weight_num_sums,
        "input_bvr_len": args.input_bvr_len,
        "input_num_sums": args.input_num_sums,
        "input_set_size": args.input_set_size,
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
        raise NotImplementedError("Loading quantized model is not supported yet")
    else:
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
        model = model.to(torch.bfloat16)
        model.load_sbvr_weights(args.root_sbvr_path, args.sbvrizer_path)
        model.preprocess_model()
        model = model.to("cuda:0")
        model.eval()
    print(b_str("Model loaded"))
        
    tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=args.input_model,
        cache_dir=None,
        model_max_length=2048,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )
        
    if args.measure_ppl:
        ppl = eval_e2e_sbvr_ppl(model, tokenizer, device="cuda:0",
                                decode_only=False, prefill_mode=0)
        
if __name__ == "__main__":
    main()
        
    
    
    
    
