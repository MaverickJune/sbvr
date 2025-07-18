import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from paper_eval_package.ppl_benchmark import evaluate_ppl
from paper_eval_package.commonqa_benchmark import evaluate_commonqa, get_dataset_configs
import os

MODEL_PATH = "Qwen/Qwen3-0.6B"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {device}\n")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    trust_remote_code=True
).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

ppl = evaluate_ppl(
    model=model,
    tokenizer=tokenizer,
    device=device
)

NUM_SAMPLES_PER_DATASET = 100
dataset_configs = get_dataset_configs()

evaluate_commonqa(
    model=model,
    tokenizer=tokenizer,
    dataset_configs=dataset_configs
)