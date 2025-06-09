import torch
import argparse
from eval_utils.modelling_llama_sbvr import LlamaForSbvrLM
from transformers import AutoTokenizer, AutoConfig

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
    parser.add_argument("--measure_ppl", type=bool, default=False,
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
    
    sbvr_state_dict = {
        "weight_bvr_len": args.weight_bvr_len,
        "weight_num_sums": args.weight_num_sums,
        "input_bvr_len": args.input_bvr_len,
        "input_num_sums": args.input_num_sums,
        "input_set_size": args.input_set_size,
    }
    config = AutoConfig.from_pretrained(args.input_model)
    
    # prepare the model
    if args.load_qmodel_path is not None:
        raise NotImplementedError("Loading quantized model is not supported yet")
    else:
        model = LlamaForSbvrLM(config=config, sbvr_state_dict=sbvr_state_dict)
        model.load_sbvr_weights(args.root_sbvr_path, args.sbvrizer_path)
        
    print(model)
    
if __name__ == "__main__":
    main()
        
    
    
    
    
