# SBVR : Summation of Bit-Vector Representation

**Tested on CUDA 12.4, Python 3.10**

## Installation

```bash
git submodule update --init --recursive
conda create -n sbvr-supplementary python=3.10 -y
conda activate  sbvr-supplementary
pip install torch==2.6.0
pip install third_party/fast-hadamard-transform --no-build-isolation
pip install -e . --no-build-isolation
```

## Quantization

```bash
./scripts/run_sbvr_quantize.sh {hf_model_id}  # ex) meta-llama/Llama-3.2-1B-Instruct
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

* For Llama3 models, testing was performed with `transformers==4.44.2`.
* For Qwen3 models, testing was performed with `transformers==4.53.2`.
* The default installation uses `transformers==4.44.2`.

## Updating for Qwen3 Accuracy Evaluation

To evaluate accuracy on Qwen3 models, update the following files:

1. **`sbvr_e2e_utils/utils.py`** (line 2)

   ```diff
   - from transformers import LlamaForCausalLM
   + from transformers import Qwen3ForCausalLM
   ```

2. **`sbvr_e2e.py`** (line 3)

   ```diff
   - from eval_utils.modeling_llama_sbvr_4_44_2 import LlamaForSbvrLM
   + from eval_utils.modeling_qwen3_sbvr_4_51_3 import Qwen3ForSbvrLM
   ```

3. **`eval_utils/multi_gpu_sbvr_main.py`** (line 6)

   ```diff
   - from eval_utils import gptq_utils_4_44_2, rotation_utils
   + from eval_utils import gptq_utils_4_53_2, rotation_utils
   ```

4. **'requirements.txt'** (line 2)

    ```diff
    - transformers==4.44.2
    + transformers==4.53.2
    ```

```bash
pip install -e . --no-build-isolation
```
