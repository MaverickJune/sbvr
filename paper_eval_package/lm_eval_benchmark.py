
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import lm_eval

current_dir = os.path.dirname(os.path.realpath(__file__))

@torch.no_grad()
def run_lm_eval(tokenizer, model, tasks, verbose=True):

    model.eval()

    results = {}
    model_lm = lm_eval.models.huggingface.HFLM(pretrained=model, tokenizer=tokenizer)
    eval_results = lm_eval.simple_evaluate(model=model_lm, tasks=tasks)

    for task in tasks:
        results[f"{task}"] = eval_results['results'][task]

    return results

def save_results(results_dict, output_file):
    def recursive_sort_dict(d):
        if isinstance(d, dict):
            return {k: recursive_sort_dict(v) for k, v in sorted(d.items())}
        return d

    sorted_results = recursive_sort_dict(results_dict)

    with open(output_file, 'w') as f:
        json.dump(sorted_results, f, indent=2)



def evaluate_lm_eval_benchmark(model, tokenizer, model_path, tasks=None, verbose=True):
    """
    Evaluate the model using lm_eval benchmarks.
    
    Args:
        model: The model to evaluate.
        tokenizer: The tokenizer for the model.
        model_path: Path to the model.
        tasks: List of tasks to evaluate. If None, defaults to a predefined set.
        verbose: If True, prints detailed output.
    Returns:
        results: Dictionary containing evaluation results for each task.
    """

    model_name = os.path.basename(model_path)
    output_file = f"{model_name}_benchmark_results.json"

    if tasks is None:
        tasks = ['commonsense_qa', 'arc_challenge', 'arc_easy', 'hellaswag', 'piqa', 'winogrande']

    if verbose:
        print(f"Evaluating model on tasks: {', '.join(tasks)}")

    results = run_lm_eval(tokenizer, model, tasks, verbose)

    if verbose:
        print("Evaluation results:")
        for task, result in results.items():
            print(f"{task}: {result}")


    save_results(results, output_file)
    if verbose:
        print(f"Results saved to {output_file}")
    
    return results

def evaluate_lm_eval_benchmark_awq(
    model,
    tokenizer,
    model_path,
    tasks=None,
    batch_size: int = 1,
    device: str = "cuda:0",
    output_dir: str = None,
    verbose: bool = True,
):
    """
    Evaluate a quantized AWQ model using lm-eval-harness multiple-choice benchmarks.

    Args:
        model: Hugging Face `PreTrainedModel` instance (quantized AWQ).
        tokenizer: Corresponding HF tokenizer.
        model_path: Path or repo ID of the model (used to name output file).
        tasks: List of task names (e.g., ['commonsense_qa', ...]).
        batch_size: Evaluation batch size.
        device: Device string (e.g., 'cuda:0').
        output_dir: Directory to save results JSON. Defaults to cwd.
        verbose: Print progress and results if True.
    Returns:
        results: Dict of evaluation metrics per task.
    """
    # Determine default tasks
    if tasks is None:
        tasks = [
            'commonsense_qa',
            'arc_challenge',
            'arc_easy',
            'hellaswag',
            'piqa',
            'winogrande',
        ]

    model_name = os.path.basename(model_path.rstrip("/"))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"{model_name}_benchmark_results.json")
    else:
        output_file = f"{model_name}_benchmark_results.json"

    if verbose:
        print(f"Evaluating '{model_name}' on tasks: {', '.join(tasks)}")

    # Wrap into HFLM to use simple_evaluate
    model_lm = lm_eval.models.huggingface.HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=batch_size,
    )

    # Run evaluation
    eval_results = lm_eval.simple_evaluate(
        model=model_lm,
        tasks=tasks,
    )

    results = {}

    for task in tasks:
        results[f"{task}"] = eval_results['results'][task]


    save_results(results, output_file)
    if verbose:
        print(f"Results saved to {output_file}")

    return results