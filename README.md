# SBVR : Summation of Bit-Vector Representation

**Tested on CUDA 12.4, Python 3.10**

## Installation

```bash
git clone https://github.com/cakeng/sbvr.git
git submodule update --init --recursive
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

To be updatated ..
