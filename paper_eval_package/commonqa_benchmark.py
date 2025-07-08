import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
from tqdm import tqdm
from dataclasses import dataclass
from typing import List, Callable, Optional, Any, Dict, Union

# --- Dataset Configuration (remains the same) ---
@dataclass
class DatasetTaskConfig:
    name: str
    hf_id: str
    get_correct_answer_index: Callable[[Dict[str, Any], List[str]], int]
    hf_config: Optional[str] = None
    split: str = 'validation'
    question_field: Union[str, Callable[[Dict], str]] = "question"
    choices_field: Union[str, Callable[[Dict], List[str]]] = "choices"
    prompt_choice_labels: Optional[List[str]] = None
    max_input_length: int = 1024
    max_new_tokens: int = 10

DEFAULT_CHOICE_LABELS = [chr(ord('A') + i) for i in range(26)]
N_DEBUG_SAMPLES = 1

# --- Predefined Dataset Configurations (remains the same) ---
def get_dataset_configs() -> List[DatasetTaskConfig]:
    configs = [
        DatasetTaskConfig(
            name="CommonsenseQA",
            hf_id="tau/commonsense_qa",
            question_field="question",
            choices_field=lambda item: item['choices']['text'],
            get_correct_answer_index=lambda item, c_texts:
                item['choices']['label'].index(str(item['answerKey']))
        ),
        DatasetTaskConfig(
            name="ARC-Challenge",
            hf_id="ai2_arc",
            hf_config="ARC-Challenge",
            question_field="question",
            choices_field=lambda item: item['choices']['text'],
            get_correct_answer_index=lambda item, c_texts:
                item['choices']['label'].index(str(item['answerKey']))
        ),
        DatasetTaskConfig(
            name="ARC-Easy",
            hf_id="ai2_arc",
            hf_config="ARC-Easy",
            question_field="question",
            choices_field=lambda item: item['choices']['text'],
            get_correct_answer_index=lambda item, c_texts:
                item['choices']['label'].index(str(item['answerKey']))
        ),
        DatasetTaskConfig(
            name="HellaSwag",
            hf_id="hellaswag",
            question_field=lambda item: (
                f"Context: {item['ctx']}\n"
                f"Which of the following is the most plausible continuation?"
            ),
            choices_field="endings",
            prompt_choice_labels=['A', 'B', 'C', 'D'],
            get_correct_answer_index=lambda item, c_texts: int(item['label'])
        ),
        DatasetTaskConfig(
            name="PIQA",
            hf_id="piqa",
            split="validation",
            question_field="goal",
            choices_field=lambda item: [item['sol1'], item['sol2']],
            prompt_choice_labels=['A', 'B'],
            get_correct_answer_index=lambda item, c_texts: int(item['label'])
        ),
        DatasetTaskConfig(
            name="Winogrande (M)",
            hf_id="winogrande",
            hf_config="winogrande_m",
            split="validation",
            question_field=lambda item: (
                f"Sentence: {item['sentence'].replace('_', '______')}\n"
                f"Which option correctly fills the blank?"
            ),
            choices_field=lambda item: [item['option1'], item['option2']],
            prompt_choice_labels=['A', 'B'],
            get_correct_answer_index=lambda item, c_texts: int(item['answer']) - 1
        ),
    ]
    return configs

def generate_with_forward(model, tokenizer, prompt, max_new_tokens=20, 
                          device='cuda'):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    generated = input_ids

    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated)
        logits = outputs.logits  # shape: (batch_size, seq_len, vocab_size)
        next_token_logits = logits[:, -1, :]  # last token logits

        # Greedy decoding: pick token with highest logit
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        # Append predicted token to sequence
        generated = torch.cat([generated, next_token], dim=-1)

        # Stop if EOS token generated
        if next_token.item() == tokenizer.eos_token_id:
            break

    return generated

# --- Helper Functions ---
def format_prompt_for_model( # Renamed for clarity, adjusted for base models
    question_text: str,
    choices_texts: List[str],
    choice_labels: List[str]
) -> str:
    formatted_choices = []
    for label, text in zip(choice_labels, choices_texts):
        formatted_choices.append(f"{label}. {text}")
    choices_str = "\n".join(formatted_choices)
    
    # Simplified prompt for base models, ending with a clear cue for completion
    prompt = (
        f"Question: {question_text}\n\n"
        f"Choices:\n{choices_str}\n\n"
        f"Answer:" # Encourage the model to complete with the letter
    )
    return prompt

def get_predicted_choice_label(
    generated_text: str,
    valid_labels: List[str]
) -> Optional[str]:
    if not generated_text:
        return None
    stripped_text = generated_text.strip().upper()
    if not stripped_text:
        return None
    # Check if the stripped_text starts with any of the valid labels followed by a non-alpha char or EOL
    for label in valid_labels:
        if stripped_text.startswith(label):
            if len(stripped_text) == len(label) or \
               not stripped_text[len(label)].isalpha(): # e.g. "A." or "A "
                return label
    # Fallback: if the very first char is a label (e.g. model just outputs "A")
    if stripped_text and stripped_text[0] in valid_labels:
         return stripped_text[0]
    return None
# --- End Helper Functions ---

def evaluate_single_dataset(
    model,
    tokenizer,
    config: DatasetTaskConfig,
    device: torch.device,
    num_samples: Optional[int] = None
) -> Dict[str, Any]:
    print(f"\n--- Evaluating on: {config.name} ---")
    print(f"  Using plain prompt format (assuming base model).")
    # ... (initial prints for dataset ID, model/tokenizer EOS/PAD IDs)

    try:
        dataset = load_dataset(
            config.hf_id,
            name=config.hf_config,
            split=config.split,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"  Error loading dataset {config.name}: {e}")
        return {"error": str(e), "accuracy": 0.0}

    # ... (dataset subsetting logic remains same) ...
    if num_samples is not None and num_samples > 0 \
       and num_samples < len(dataset):
        dataset = dataset.select(range(num_samples))
        print(f"  Evaluating on a subset of {num_samples} samples.")
    else:
        num_samples = len(dataset)
        print(f"  Evaluating on {num_samples} samples.")

    predictions_indices = []
    references_indices = []
    invalid_predictions_count = 0
    debug_samples_output = []

    for i, item in enumerate(tqdm(dataset, desc=f"Processing {config.name}")):
        if callable(config.question_field):
            question_text = config.question_field(item)
        else:
            question_text = item[config.question_field]

        if callable(config.choices_field):
            choice_texts = config.choices_field(item)
        else:
            choice_texts = item[config.choices_field]

        current_prompt_labels = config.prompt_choice_labels or \
                                DEFAULT_CHOICE_LABELS[:len(choice_texts)]

        # Use the direct prompt format suitable for base models
        direct_prompt = format_prompt_for_model( # Using the renamed function
            question_text,
            choice_texts,
            current_prompt_labels
        )
        
        inputs = tokenizer(
            direct_prompt, # Pass the direct prompt string
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=config.max_input_length 
        ).to(device)

        with torch.no_grad():
            outputs = generate_with_forward(
                model,
                tokenizer,
                prompt=direct_prompt,
                device=device,
                max_new_tokens=config.max_new_tokens,
            )
        
        input_len = inputs["input_ids"].shape[1]
        generated_ids_for_decode = outputs[0, input_len:]
        
        if i < N_DEBUG_SAMPLES:
            print(f"\n    --- Token Debug for Sample {i+1} ({config.name}) ---")
            print(f"    Formatted Prompt (Direct for Base Model):\n\"\"\"\n{direct_prompt}\n\"\"\"")
            # ... (rest of token debug prints: Input IDs shape, Generated token IDs, Decoded with special) ...
            print(f"    Input IDs shape: {inputs['input_ids'].shape}")
            print(f"    Generated token IDs (raw slice): {generated_ids_for_decode.tolist()}")
            decoded_with_special = tokenizer.decode(generated_ids_for_decode, skip_special_tokens=False)
            print(f"    Decoded (with special tokens): \"{decoded_with_special}\"")


        generated_text = tokenizer.decode(
            generated_ids_for_decode,
            skip_special_tokens=True
        )
            
        # ... (rest of prediction and reference processing, and debug_samples_output append) ...
        predicted_label_str = get_predicted_choice_label(
            generated_text,
            current_prompt_labels
        )
        predicted_idx = -1
        if predicted_label_str is not None:
            try:
                predicted_idx = current_prompt_labels.index(predicted_label_str)
            except ValueError:
                invalid_predictions_count += 1
        else:
            invalid_predictions_count += 1
        predictions_indices.append(predicted_idx)

        reference_idx = -2 # Default for error in reference processing
        correct_answer_text_for_debug = "N/A (error)"
        try:
            # <<< ADD DEBUG PRINT FOR WINOGRANDE HERE >>>
            if "winogrande" in config.hf_id.lower(): # Check if it's a Winogrande dataset
                if i < N_DEBUG_SAMPLES: # Only print for the first few samples
                    print(f"    DEBUG Winogrande (Sample {i+1}): item['answer'] = '{item.get('answer', 'FIELD_MISSING')}', type: {type(item.get('answer'))}")
            # <<< END DEBUG PRINT >>>

            reference_idx = config.get_correct_answer_index(item, choice_texts)
            if 0 <= reference_idx < len(choice_texts):
                correct_answer_text_for_debug = choice_texts[reference_idx]
            else:
                if i < N_DEBUG_SAMPLES: # Print warning only for debug samples to reduce noise
                    print(f"  Warning (Sample {i+1}): Reference index {reference_idx} out of bounds "
                          f"for item with {len(choice_texts)} choices. Item ID: {item.get('id', 'N/A')}")
                reference_idx = -2 # Mark as problematic reference
        except Exception as e:
            if i < N_DEBUG_SAMPLES: # Print error only for debug samples
                print(f"  Error (Sample {i+1}) getting reference index for item "
                      f"{item.get('id', 'N/A')}: {e}")
            reference_idx = -2
        references_indices.append(reference_idx)

        if i < N_DEBUG_SAMPLES:
            debug_samples_output.append({
                "id": item.get('id', f"sample_{i}"),
                "choices_texts": choice_texts,
                "raw_model_output_text": generated_text,
                "parsed_prediction_str": predicted_label_str,
                "predicted_idx": predicted_idx,
                "reference_idx": reference_idx,
                "correct_answer_label": current_prompt_labels[reference_idx] if 0 <= reference_idx < len(current_prompt_labels) else "N/A",
                "correct_answer_text": correct_answer_text_for_debug,
            })
            
    # ... (rest of the function: warnings, debug summary, accuracy calculation) ...
    if invalid_predictions_count > 0:
        print(f"  Warning: {invalid_predictions_count}/{len(dataset)} samples had "
              f"unparseable predictions for {config.name}.")
    # ... (problematic references warning) ...

    if debug_samples_output:
        print(f"\n  --- Debug Summary for First {len(debug_samples_output)} Samples for {config.name} ---")
        for k, debug_info in enumerate(debug_samples_output):
            print(f"  Sample {k+1} (ID: {debug_info['id']}):")
            # ... (debug summary prints)
            print(f"    Choices: {debug_info['choices_texts']}")
            print(f"    Correct Answer: [{debug_info['correct_answer_label']}] \"{debug_info['correct_answer_text']}\" (Ref Idx: {debug_info['reference_idx']})")
            print(f"    Model Raw Output (after skip_special): \"{debug_info['raw_model_output_text']}\"")
            print(f"    Parsed Prediction: Str='{debug_info['parsed_prediction_str']}', Idx={debug_info['predicted_idx']}")
            print("-" * 30)

    accuracy_metric = evaluate.load("accuracy")
    # ... (accuracy calculation) ...
    try:
        results = accuracy_metric.compute(
            predictions=predictions_indices,
            references=references_indices
        )
        accuracy = results.get('accuracy', 0.0)
    except Exception as e:
        print(f"  Error computing accuracy for {config.name}: {e}")
        accuracy = 0.0
        
    print(f"  Accuracy for {config.name}: {accuracy:.4f}")
    return {"accuracy": accuracy, "name": config.name,
            "samples": len(references_indices)}

dataset_configs = get_dataset_configs()

# --- evaluate_commonqa function (remains mostly the same) ---
def evaluate_commonqa(
    model,
    tokenizer,
    dataset_configs: List[DatasetTaskConfig] = dataset_configs,
    num_samples_per_dataset: Optional[int] = 200
):
    # ... (device setup, model.eval(), tokenizer/model config for pad/eos) ...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        print(f"Warning: tokenizer.pad_token_id is None. Setting to eos_token_id ({tokenizer.eos_token_id}).")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token 

    if model.config.pad_token_id != tokenizer.pad_token_id:
        print(f"Aligning model.config.pad_token_id ({model.config.pad_token_id}) "
              f"with tokenizer.pad_token_id ({tokenizer.pad_token_id}).")
        model.config.pad_token_id = tokenizer.pad_token_id
    
    if model.config.eos_token_id is None:
        model.config.eos_token_id = tokenizer.eos_token_id
    elif isinstance(tokenizer.eos_token_id, list): 
        if model.config.eos_token_id not in tokenizer.eos_token_id:
             model.config.eos_token_id = tokenizer.eos_token_id[0]
    elif model.config.eos_token_id != tokenizer.eos_token_id :
        model.config.eos_token_id = tokenizer.eos_token_id
    
    print(f"Final Model Config after setup: EOS ID: {model.config.eos_token_id}, PAD ID: {model.config.pad_token_id}")

    all_results = []
    for config_obj in dataset_configs:
        result = evaluate_single_dataset(
            model,
            tokenizer,
            config_obj,
            device,
            num_samples_per_dataset
        )
        if "error" not in result:
            all_results.append(result)
        else:
            print(f"Skipping {config_obj.name} due to error: {result['error']}")
    # ... (overall results printing) ...
    print("\n\n--- Overall Benchmark Results ---")
    total_accuracy = 0
    num_valid_datasets = 0
    if not all_results:
        print("No datasets were successfully evaluated.")
        return

    for res in all_results:
        print(f"  {res['name']}: {res['accuracy']:.4f}")
        total_accuracy += res['accuracy']
        num_valid_datasets += 1
    
    if num_valid_datasets > 0:
        average_accuracy = total_accuracy / num_valid_datasets
        print(f"\nAverage Zeroshot: {average_accuracy:.4f}")
    else:
        print("No datasets were successfully evaluated to calculate an average.")


# --- __main__ block ---
if __name__ == "__main__":
    MODEL_NAME = "meta-llama/Llama-3.2-3B" # Assuming this is a BASE model
    TRUST_REMOTE_CODE = True
    # TORCH_DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else None
    TORCH_DTYPE = torch.float16
    NUM_SAMPLES_PER_DATASET = 100

    print("Starting benchmark evaluation script...")
    # ... (pip install message)

    print(f"\nLoading model and tokenizer: {MODEL_NAME}...")
    try:
        tokenizer_args = {"trust_remote_code": TRUST_REMOTE_CODE}
        model_args = {"trust_remote_code": TRUST_REMOTE_CODE}
        if TORCH_DTYPE:
            model_args["torch_dtype"] = TORCH_DTYPE
        
        loaded_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, **tokenizer_args
        )
        loaded_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, **model_args
        )
        print("Model and tokenizer loaded successfully.")

        # No manual chat template setting needed/desired for a base model
        if loaded_tokenizer.chat_template is not None:
            print("Note: Loaded tokenizer HAS a chat_template. If this is a base model, "
                  "this template will NOT be used by this script's current logic.")
        else:
            print("Note: Loaded tokenizer does not have a chat_template, which is expected for a base model.")

    except Exception as e:
        print(f"Fatal Error loading model or tokenizer: {e}")
        exit()

    # dataset_eval_configs = get_dataset_configs()
    dataset_eval_configs = [
        DatasetTaskConfig(
            name="CommonsenseQA",
            hf_id="tau/commonsense_qa",
            question_field="question",
            choices_field=lambda item: item['choices']['text'],
            get_correct_answer_index=lambda item, c_texts:
                item['choices']['label'].index(str(item['answerKey']))
        )
    ]
    evaluate_commonqa(
        model=loaded_model,
        tokenizer=loaded_tokenizer,
        dataset_configs=dataset_eval_configs,
        num_samples_per_dataset=NUM_SAMPLES_PER_DATASET
    )
    print("\nBenchmark evaluation finished.")