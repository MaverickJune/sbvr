import logging
import os
import sys

import torch
from tqdm import tqdm

from utils import data_utils
from transformers.cache_utils import DynamicCache

def r_str(s):
    return "\033[91m" + str(s) + "\033[0m"
def g_str(s):
    return "\033[92m" + str(s) + "\033[0m"
def y_str(s):
    return "\033[93m" + str(s) + "\033[0m"
def b_str(s):
    return "\033[94m" + str(s) + "\033[0m"

@torch.inference_mode()
def get_e2e_sbvr_ppl(model, testenc, device="cuda:0",
            decode_only=False, decode_len=-1,
            prefill_mode=0, decode_mode=0):
    model.eval()
    model = model.to(device)
    model.seqlen = 2048
    use_cache = model.config.use_cache
    
    if decode_only:
        print("Measuring the ppl only at the decoding stage")
        if decode_len == -1:
            raise ValueError("decode_len must be provided")
        model.config.use_cache = True
        
        # prepare the dataset
        input_ids = testenc.input_ids
        nsamples = input_ids.numel() // model.seqlen
        input_ids = input_ids[:, :nsamples * model.seqlen].view(1, -1).to(device)
        prefill_len = model.seqlen - decode_len
        nlls = []
        for i in tqdm(range(nsamples), desc="evaluating decode only ppl"):
            batch = input_ids[:, (i * model.seqlen) : ((i + 1) * model.seqlen)]
            kv_cache = DynamicCache()
            
            # perform prefill
            with torch.no_grad():
                # print(f"shape: {batch[:, :prefill_len].shape}")
                # sys.exit(0)
                if prefill_mode == -1:
                    output = model(batch[:, :prefill_len], past_key_values=kv_cache)
                else:
                    output = model(batch[:, :prefill_len], past_key_values=kv_cache, mode=prefill_mode)
                lm_logits = output.logits
                kv_cache = output.past_key_values
            
                sample_nlls = []
                for i in range(decode_len - 1):
                    next_token = batch[:, prefill_len + i].unsqueeze(0)
                    label = batch[:, prefill_len + i + 1].unsqueeze(0)
                    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                    
                    if decode_mode == -1:
                        output = model(next_token, past_key_values=kv_cache)
                    else:
                        output = model(next_token, past_key_values=kv_cache, mode=decode_mode)
                    token_logits = output.logits
                    kv_cache = output.past_key_values
                    loss = loss_fct(
                        token_logits.view(-1, token_logits.size(-1)),
                        label.view(-1)
                    )
                    neg_log_likelihood = loss.float()
                    
                    # print(f"ppl: {neg_log_likelihood.item()}")
                    # sys.exit(0)
                    
                    sample_nlls.append(neg_log_likelihood)
                nlls.append(torch.cat(sample_nlls))
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * (decode_len - 1)))
    else:
        model.config.use_cache = False
        
        # prepare the dataset
        input_ids = testenc.input_ids
        nsamples = input_ids.numel() // model.seqlen
        input_ids = input_ids[:, :nsamples * model.seqlen].view(1, -1).to(device)
        nlls = []
        
        for i in tqdm(range(nsamples), desc="evaluating ppl"):
            batch = input_ids[:, (i * model.seqlen) : ((i + 1) * model.seqlen)]
            with torch.no_grad():
                if prefill_mode == -1:
                    lm_logits = model(batch).logits
                else:
                    lm_logits = model(batch, mode=prefill_mode).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = batch[:, 1:]
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float()
            nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    
    model.config.use_cache = use_cache
    print(f"\n WikiText2 PPL: {ppl.item():.3f}")
    return ppl.item()

def eval_e2e_sbvr_ppl(model, tokenizer, device="cuda:0",
                      decode_only=True, decode_len=1024, prefill_mode=0, decode_mode=0):
    testloader = data_utils.get_wikitext2(
        seed=0,
        seqlen=2048,
        tokenizer=tokenizer,
        eval_mode=True,
    )
    
    dataset_ppl = get_e2e_sbvr_ppl(model, testloader, device=device,
        decode_only=decode_only, decode_len=decode_len, prefill_mode=prefill_mode, decode_mode=decode_mode)
    
    return dataset_ppl