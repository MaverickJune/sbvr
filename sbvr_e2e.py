import torch
import argparse
from eval_utils.modelling_llama_sbvr import LlamaForSbvrLM
from transformers import AutoTokenizer, AutoConfig, LlamaTokenizerFast
from sbvr_e2e_utils.eval_ppl import r_str, g_str, y_str, b_str
from paper_eval_package.ppl_benchmark import evaluate_ppl
from paper_eval_package.latency_benchmark import evaluate_latency
from sbvr_e2e_utils.utils import get_partial_state

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
    parser.add_argument("--measure_latency", action="store_true",
                        help="whether to measure the latency of the model")
    args = parser.parse_args()
    
    return args

def main():
    args = parse_args()
    
    # sanity check
    if args.root_sbvr_path is None and args.load_qmodel_path is None:
        raise ValueError("Either root_sbvr_path or load_qmodel_path must be provided")
    
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
        model.load_sbvr_weights(args.root_sbvr_path)
        
        # convert the model to float16
        model.convert_model_dtype(dtype=torch.float16)
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

    layer_idx = 4  # 원하는 레이어 인덱스
    target_layer = model.model.layers[layer_idx]

    # 예시: Attention의 q_proj
    mlp_module = target_layer.mlp
    sbvr_module = target_layer.mlp.down_proj
    hidden_dim = mlp_module.hidden_size

    # seq_len=1, batch_size=1
    dummy_x = torch.randn(1, 1, hidden_dim, device="cuda:0", dtype=torch.float16)
    dummy_x_flat = dummy_x.view(1, -1)  # (seq_len, hidden_dim) = (1, hidden_dim)

    # d_forward는 내부에서 _sbvr_input_transfrom을 사용하므로, 입력 shape는 (1, hidden_dim) 권장
    with torch.no_grad():
        out = sbvr_module.d_forward(dummy_x_flat)
    print(f"output shape: {out.shape}")
        
    if args.measure_ppl:
        ppl = evaluate_ppl(
            model=model,
            tokenizer=tokenizer
        )

    if args.measure_latency:
        latency_result = evaluate_latency(
            model=model,
            tokenizer=tokenizer
        )
        
if __name__ == "__main__":
    main()
        
    
    
    
    
