import torch
from transformers import AutoTokenizer, AutoConfig, LlamaTokenizerFast
from transformers import LlamaForCausalLM
from paper_eval_package.ppl_benchmark import evaluate_ppl


MODEL_NAME = "meta-llama/Llama-3.2-1b"
config = config = AutoConfig.from_pretrained(MODEL_NAME)
model = LlamaForCausalLM.from_pretrained(
    MODEL_NAME,
    config=config,
    torch_dtype=torch.float16,
    device_map="cuda:0",
    trust_remote_code=True
)

tokenizer = LlamaTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,
        cache_dir=None,
        model_max_length=2048,
        padding_side="right",
        use_fast=True,
        add_eos_token=False,
        add_bos_token=False,
    )

ppl = evaluate_ppl(
    model=model,
    tokenizer=tokenizer
)
