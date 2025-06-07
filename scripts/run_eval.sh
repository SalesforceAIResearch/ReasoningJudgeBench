#!/bin/bash

# !! This script assumes you have served a model on port=8001 (feel free to change)
# Before running, serve the model with, e.g., serve_model.sh!

cd /export/xgen-finance/austinxu/ReasoningJudgeBench

export HF_TOKEN=<YOUR_HF_TOKEN_HERE>
# judge_model_name=<FULL_JUDGE_PATH_OR_HF_NAME>
judge_model_short=R-I-S-E/RISE-Judge-Qwen2.5-7B # for naming filepath only
port=8001

python main.py \
    --output_dir /export/xgen-finance/austinxu/ReasoningJudgeBench/results/ \
    --judge_model ${judge_model_short} \
    --evaluation_protocol pairwise \
    --dataset_name reasoningjudgebench \
    --base_url http://localhost:${port}/v1 \
    --num_proc=64 \
    --force_rerun

python aggregate.py --eval_dir /export/xgen-finance/austinxu/ReasoningJudgeBench/results/${judge_model_short}/standard/vanilla/reasoningjudgebench
