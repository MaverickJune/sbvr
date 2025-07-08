from datasets import load_dataset, load_from_disk, Dataset
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

def compute_perplexity(model, tokenizer, dataset_name, dataset_config,
                       split="validation", max_length=4096, stride=512,
                       num_examples=10000):
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computing perplexity on {dataset_name} (config: {dataset_config}), "
          f"split: {split}, max_length={max_length}, stride={stride}...")

    # Ensure the base directory for datasets exists
    base_datasets_dir = "datasets" # Or any preferred base path
    os.makedirs(base_datasets_dir, exist_ok=True)
    
    # Construct a more specific path if dataset_config is used
    config_str = f"_{dataset_config}" if dataset_config else ""
    split_str = f"_{split}" if split else ""
    custom_dataset_path = os.path.join(
        base_datasets_dir, f"{dataset_name}{config_str}{split_str}"
    )

    texts = []
    text_key = None

    if os.path.exists(custom_dataset_path):
        print(f"Loading pre-processed dataset from {custom_dataset_path}...")
        # This is a Dataset object, not IterableDataset
        processed_dataset = load_from_disk(custom_dataset_path)
        print (f"Loaded {len(processed_dataset)} examples from disk.")
        
        if not processed_dataset:
            raise ValueError(
                f"Loaded dataset from {custom_dataset_path} is empty."
            )
        
        # Determine text_key from the loaded disk dataset's first example
        # Assuming all examples in the saved dataset share this structure
        try:
            first_example_disk = processed_dataset[0]
            text_key = next(
                (k for k in first_example_disk
                 if isinstance(first_example_disk[k], str)), None
            )
            if text_key is None:
                raise ValueError(
                    "No text field found in the dataset loaded from disk."
                )
        except IndexError:
             raise ValueError(
                f"Dataset loaded from {custom_dataset_path} appears empty."
            )

        for i, example in enumerate(processed_dataset):
            texts.append(example[text_key])
            # Allow taking fewer examples than what's saved if needed
            if num_examples is not None and (i + 1) >= num_examples:
                break
    else:
        print(f"Streaming and saving dataset: {dataset_name} "
              f"(config: {dataset_config}, split: {split})...")
        iterable_dataset = load_dataset(
            dataset_name,
            name=dataset_config, # Use 'name' for dataset_config
            split=split,
            streaming=True,
            trust_remote_code=True
        )

        collected_examples = []
        
        for i, example in enumerate(iterable_dataset):
            if i == 0: # Determine text_key from the first streamed example
                text_key = next(
                    (k for k in example if isinstance(example[k], str)),
                    None
                )
                if text_key is None:
                    raise ValueError(
                        "No text field found in first example of stream."
                    )
            
            # Store the whole example to preserve its structure for saving
            collected_examples.append(example)
            
            # Stop after collecting the desired number of examples
            if num_examples is not None and (i + 1) >= num_examples:
                break
        
        if not collected_examples:
            # This case might occur if num_examples is 0 or stream is empty
            print("No examples collected from the stream.")
            # Depending on desired behavior, either raise error or proceed
            # with empty texts
            # For now, let's ensure texts is empty and proceed.
        else:
            # Convert the list of collected examples to a Dataset object
            dataset_to_save = Dataset.from_list(collected_examples)

            # Save the new Dataset object (not the IterableDataset)
            # The path custom_dataset_path is a directory for save_to_disk
            dataset_to_save.save_to_disk(custom_dataset_path)
            print(f"Saved {len(dataset_to_save)} examples to "
                  f"{custom_dataset_path}.")

        # Populate texts from the collected examples
        # (text_key was determined from the first example)
        for example in collected_examples:
            texts.append(example[text_key])

    if not texts and num_examples > 0 : # num_examples could be None
        # If num_examples was 0, texts would be empty, which is fine.
        # If num_examples > 0 but texts is empty, it's an issue.
        print(f"Warning: No texts loaded or collected for PPL calculation, "
              f"though {num_examples} examples were requested.")
    elif texts:
        print(f"Using {len(texts)} examples for PPL calculation.")
    else: # texts is empty, num_examples was 0 or None and stream was empty
        print("No texts available for PPL calculation.")
        return float('nan') # Or handle as appropriate

    # Proceed with PPL calculation using the 'texts' list
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt",
                          add_special_tokens=False)
    input_ids = encodings.input_ids[0] # Take the first (and likely only) sequence

    if input_ids.numel() == 0:
        print("Warning: No tokens to calculate perplexity on.")
        return float('nan') # Or other appropriate error/default value

    nlls = []
    actual_eval_count = 0
    for i in tqdm(range(0, input_ids.size(0) - max_length + 1, stride), 
                  desc="Perplexity Chunks"):
        input_chunk = input_ids[i: i + max_length]
        # target_chunk = input_chunk.clone() # For Causal LM, labels are shifted
        
        # Labels are typically input_ids for Causal LM.
        # PyTorch CrossEntropyLoss handles shifting internally if logits match input_ids length.
        # Or, ensure target_chunk is appropriately formed if model doesn't shift.
        # For models like GPT-2, input_ids are used as labels directly.
        target_chunk_labels = input_chunk.clone()


        input_chunk_batch = input_chunk.unsqueeze(0).to(model.device)
        target_chunk_labels_batch = target_chunk_labels.unsqueeze(0).to(model.device)

        vocab_size = model.get_input_embeddings().num_embeddings
        if input_chunk_batch.max() >= vocab_size:
            invalid_ids = input_chunk_batch[input_chunk_batch >= vocab_size]
            raise ValueError(
                f"Invalid token IDs >= vocab size ({vocab_size}). "
                f"Found: {invalid_ids.unique().tolist()}"
            )

        with torch.no_grad():
            outputs = model(input_chunk_batch, labels=target_chunk_labels_batch)
            # The loss is usually the average NLL per token.
            # Multiply by the number of tokens in the label to get sum of NLLs for the chunk.
            # The number of tokens contributing to the loss is often seq_len - 1 for causal LMs
            # or seq_len depending on model and loss calculation.
            # If outputs.loss is mean NLL, then NLL_sum = outputs.loss * number_of_tokens_in_loss_calc
            # Let's assume labels are effectively of length max_length for loss calculation.
            neg_log_likelihood = outputs.loss * max_length # As in original code
        nlls.append(neg_log_likelihood)
        actual_eval_count += 1
    
    if not nlls:
        print(f"Warning: No NLLs computed. This might happen if total "
              f"tokens < max_length ({input_ids.numel()} < {max_length}).")
        return float('nan')

    # Total NLL sum / total number of tokens used in NLL calculation
    # Each NLL in nlls is for 'max_length' tokens.
    # So total tokens = len(nlls) * max_length
    ppl = torch.exp(torch.stack(nlls).sum() / (actual_eval_count * max_length))
    return ppl.item()


def evaluate_ppl(model, tokenizer):
    datasets = {
        "wikitext": ("wikitext", "wikitext-2-raw-v1", "validation"),
        # "ptb": ("ptb_text_only", "penn_treebank", "test"),
        # "c4": ("allenai/c4", "en", "validation")
    }

    scores = {}
    for name, (dataset, config, split) in datasets.items():
        print(f"Evaluating perplexity on {name}...")
        ppl = compute_perplexity(
            model=model,
            tokenizer=tokenizer,
            dataset_name=dataset,
            dataset_config=config,
            split=split,
            max_length=min(2048, model.config.max_position_embeddings),
            stride=512, # 512 was the original value
            num_examples=3600 if name == "c4" else None  # subset only for c4
        )
        print(f"{name}: {ppl:.2f}")
        scores[name] = ppl
    return scores