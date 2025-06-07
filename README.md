# J4R: Learning to Judge with Equivalent Initial State Group Relative Policy Optimization 

This is the codebase for J4R: Learning to Judge with Equivalent Initial State Group Relative Policy Optimization
- 📚 Paper: [arXiv](https://arxiv.org/abs/2505.13346)
- 🔬 ReasoningJudgeBench: [Huggingface Dataset](https://huggingface.co/datasets/Salesforce/ReasoningJudgeBench)

ReasoningJudgeBench is released as CC-BY-NC-4.0.

## 🔍 About J4R
To keep pace with the increasing pace of large language models (LLM) development, model output evaluation has transitioned away from time-consuming human evaluation to automatic evaluation, where LLMs themselves are tasked with assessing and critiquing other model outputs. LLM-as-judge models are a class of generative evaluators that excel in evaluating relatively simple domains, like chat quality, but struggle in reasoning intensive domains where model responses contain more substantive and challenging content. To remedy existing judge shortcomings, we explore training judges with reinforcement learning (RL). We make three key contributions: (1) We propose the Equivalent Initial State Group Relative Policy Optimization (EIS-GRPO) algorithm, which allows us to train our judge to be robust to positional biases that arise in more complex evaluation settings. (2) We introduce ReasoningJudgeBench, a benchmark that evaluates judges in diverse reasoning settings not covered by prior work. (3) We train Judge for Reasoning (J4R), a 7B judge trained with EIS-GRPO that outperforms GPT-4o and the next best small judge by 6.7% and 9%, matching or exceeding the performance of larger GRPO-trained judges on both JudgeBench and ReasoningJudgeBench.

## 🚀 Evaluating on ReasoningJudgeBench 
### Setup
This code was tested with Python 3.10.16 and PyTorch 2.5.1. 

```
conda create --name rjb python=3.10.16 -y
conda activate rjb
pip install uv
uv pip install torch==2.5.1
uv pip install -r requirements.txt
```

### Running evaluation
⚠️ Samples in the `aime_pairwise` split require 24K model context. All other splits were evaluated with 16K context in the paper.

ReasoningJudgeBench uses VLLM for inference, which is installed as a part of requirements.txt. First, serve your model with VLLM. An example script is provided in `scripts/serve_model.sh`
```
export HF_TOKEN=<YOUR_HF_TOKEN_HERE>

judge_model_name=<FULL_JUDGE_PATH_OR_HF_NAME>
CUDA_VISIBLE_DEVICES=0 vllm serve ${judge_model_name} \
    --tensor_parallel_size=1 \
    --max_model_len=16384 \
    --gpu_memory_utilization=0.8 \
    --port 8001 \
    --disable-log-stats \
    --disable-log-requests \
```

Next, run inference with `main.py`. An example script is provided in `scripts/run_eval.sh`

```
python main.py \
    --output_dir /path/to/output/dir \ # Defaults to ./results
    --judge_model <FULL_JUDGE_PATH_OR_HF_NAME> \ # For filepath naming purposes only
    --evaluation_protocol pairwise \
    --dataset_name reasoningjudgebench \
    --base_url http://localhost:${port}/v1 \ # Same port as model served above
    --num_proc=64 \ # Number of parallel requests
```

A few tips:
- After you have run evaluation, you can aggregate results using `aggregate.py`, which computes split-level and overall results. Example usage is in `scripts/run_eval.sh`. 
- You can also run inference with Together.ai by setting `--base_url https://api.together.xyz/v1/` and passing your Together AI api key via `--api_key` in `main.py`.
- You can modify sampling parameters (temperature, top_p, etc) by passing them to `main.py`. See args for examples.

## Evaluate your own judge
Adding your own judge takes 2 steps:
- Add a corresponding judge prompt in a separate `.py` file which contains the judge prompt, rendering functions, and judgment parsing code in `reasoning_judge_inf/prompts/`. See `reasoning_judge_inf/prompts/default_prompts.py` for examples. This step is only necessary if your judge has its own prompts / parsing code. Else, you can proceed to the next step.
- Add the (your judge name, prompt file name) pair to `model_to_file` in `reasoning_judge_inf/prompts/__init__.py`

Once done, you should be able to run your judge with the above inference code, changing judge model names appropriately.

## Ethical Considerations
This release is for research purposes only in support of an academic paper. Our datasets and code are not specifically designed or evaluated for all downstream purposes. We strongly recommend users evaluate and address potential concerns related to accuracy, safety, and fairness before model deployment. We encourage users to consider the common limitations of AI, comply with applicable laws, and leverage best practices when selecting use cases, particularly for high-risk scenarios where errors or misuse could significantly impact people’s lives, rights, or safety. For further guidance on use cases, refer to our [AUP](https://www.salesforce.com/content/dam/web/en_us/www/documents/legal/Agreements/policies/ExternalFacing_Services_Policy.pdf) and [AI AUP](https://www.salesforce.com/content/dam/web/en_us/www/documents/legal/Agreements/policies/ai-acceptable-use-policy.pdf). 

## Citation
If you find our project helpful, please consider citing our paper

```
@article{xu2025j4r,
  title={J4R: Learning to Judge with Equivalent Initial State Group Relative Policy Optimization},
  author={Xu, Austin and Zhou, Yilun and Nguyen, Xuan-Phi and Xiong, Caiming and Joty, Shafiq},
  journal={arXiv preprint arXiv:2505.13346},
  year={2025}
}
```



