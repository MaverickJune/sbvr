#!/usr/bin/env python3
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from paper_eval_package.ppl_benchmark import evaluate_ppl
from paper_eval_package.commonqa_benchmark import evaluate_commonqa, get_dataset_configs

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Evaluate PPL and CommonQA benchmarks on Qwen3 models"
    )
    parser.add_argument(
        '-s', '--size',
        choices=['0.6B', '1.7B', '8B'],
        default='0.6B',
        help="Specify the Qwen3 model size (e.g., 0.6B, 1.7B, 8B)"
    )
    args = parser.parse_args()

    # Construct model path based on size
    model_path = f"Qwen/Qwen3-{args.size}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    print(f"Loading model: {model_path}\n")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=True
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # Evaluate perplexity
    ppl = evaluate_ppl(
        model=model,
        tokenizer=tokenizer,
        device=device
    )
    print(f"Perplexity: {ppl}\n")

    # Evaluate CommonQA
    dataset_configs = get_dataset_configs()
    evaluate_commonqa(
        model=model,
        tokenizer=tokenizer,
        dataset_configs=dataset_configs
    )

if __name__ == "__main__":
    main()
