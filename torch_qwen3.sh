#!/bin/bash

# 출력 파일
OUTFILE="torch_qwen3_bf16_ppl_all_models.txt"

# 기존 결과 삭제
rm -f "$OUTFILE"

# 모델별 실행
for model in "0.6B" "1.7B" "8B"; do
    echo "========== Running model: $model ==========" | tee -a "$OUTFILE"
    python torch_ppl_benchmark.py -s $model 2>&1 | tee -a "$OUTFILE"
    echo "" | tee -a "$OUTFILE"
done
