import importlib

model_to_file = {
    'facebook/Self-taught-evaluator-llama3.1-70B': 'selftaught',
    'prometheus-eval/prometheus-7b-v2.0': 'prometheus',
    'prometheus-eval/prometheus-8x7b-v2.0': 'prometheus',
    'nuojohnchen/JudgeLRM-7B': 'judgelrm',
    'nuojohnchen/JudgeLRM-3B': 'judgelrm',
    'JudgeLRM-7B': 'judgelrm',
    'JudgeLRM-3B': 'judgelrm',
    'R-I-S-E/RISE-Judge-Qwen2.5-32B': 'risejudge',
    'R-I-S-E/RISE-Judge-Qwen2.5-7B': 'risejudge',
    'RISE-Judge-Qwen2.5-32B': 'risejudge',
    'RISE-Judge-Qwen2.5-7B': 'risejudge',
    'compassjudger_ft': 'compassjudger_ft',
    'gaotang/RM-R1-DeepSeek-Distilled-Qwen-7B': 'rmr1',
    'gaotang/RM-R1-DeepSeek-Distilled-Qwen-14B': 'rmr1',
    'gaotang/RM-R1-DeepSeek-Distilled-Qwen-32B': 'rmr1',
    'gaotang/RM-R1-DeepSeek-Distilled-Qwen-7B-rerun': 'rmr1',
    'gaotang/RM-R1-DeepSeek-Distilled-Qwen-14B-rerun': 'rmr1',
    'gaotang/RM-R1-DeepSeek-Distilled-Qwen-32B-rerun': 'rmr1',
    'deepseek-ai/DeepSeek-R1': 'deepseek', # deepseek-R1 requires <think> / <answer> prompt w/o system
    'deepseek-ai/DeepSeek-V3': 'default_prompts', # deepseek-v3 is a general chat model, so no special prompt
}

def load_prompter(model):
    print(f"Model: {model}")
    if 'opencompass/CompassJudger-1' in model:
        model = 'compassjudger_ft'
    
        
    model_short = model_to_file.get(model, 'default_prompts')

    print(f"Loading prompter from {model_short}.py")

    prompter = importlib.import_module('reasoning_judge_inf.prompts.{}'.format(model_short))
    return prompter


