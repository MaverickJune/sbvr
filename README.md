# SBVR : Summation of Bit-Vector Representation

**Tested on CUDA 12.4, Python 3.10**

## Installation

```bash
conda create -n sbvr-supplementary python=3.10 -y
conda activate  sbvr-supplementary
pip install torch==2.6.0
pip install third_party/fast-hadamard-transform --no-build-isolation
pip install -e . —no-build-isolation
```

## Quantization

```bash
./scripts/run_sbvr_quantize.sh {hf_model_id}  # ex Qwen/Qwen3-0.6B
```

## Accuracy Evaluation (ppl, lm\_eval benchmark)

```bash
./scripts/run_sbvr_acc_eval.sh
```

## Latency Evaluation

```bash
./scripts/run_sbvr_latency_eval.sh
```

## Note on Transformers Version
For convenience, we unified the transformers version to 4.51.3 to support evaluation across models like LLaMA3 and Qwen3.
However, to reproduce the exact results reported in the paper, please use the following versions per model:

- LLaMA3: transformers==4.44.2

- Qwen3: transformers==4.53.2

- Deepseek-R1-Distilled-Qwen2: transformers==4.51.3

Correspondingly, you must also adjust the gptq_utils implementation to match each transformers version, as compatibility issues may arise otherwise.