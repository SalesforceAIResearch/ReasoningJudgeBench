import json,os,time,argparse,torch
import numpy as np

from reasoning_judge_inf.judge import VllmEndpointJudge, HfRewardModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tqdm import tqdm
from datasets import Dataset, DatasetDict, load_dataset

from vllm import LLM, SamplingParams
from model import VllmEndpoint
from utils.utils import compute_acc, read_jsonl, check_results_exist

HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)

def main(args):
    endpoint_config = {
        "base_url": args.base_url,
        "api_key": args.api_key,
        "model_name": args.judge_model,
    }
    judge = VllmEndpointJudge(endpoint_config)
    args.judge_model = judge.model_name
    print(f"Judge model name: {args.judge_model}")

    # Set output fpaths
    save_dir = os.path.join(args.output_dir, args.judge_model, args.prompt_strategy, args.dataset_name)
    if not os.path.isdir(save_dir):
        print(f"Creating directory: {args.output_dir}")
        os.makedirs(save_dir,exist_ok=True)

    # Inference!
    results_exist, output_paths = check_results_exist(args, save_dir)
    if not results_exist or args.force_rerun:
        ####################################
        # Run judge model locally with VLLM!
        ####################################

        # We set the following max tokens
        max_tokens=1024
        if 'grpo' in args.judge_model.lower():
            max_tokens = 1024
        elif 'gaotang' in args.judge_model.lower():
            max_tokens=4096 # RM-R1 has really long outputs
        elif 'deepseek' in args.judge_model.lower():
            max_tokens=8192
        
        from reasoning_judge_inf.scaling_method.standard import StandardJudging
        judging_task = StandardJudging(args.dataset_name, judge, max_tokens, debug=args.debug, num_proc=args.num_proc)

        #-----------------
        # Run task!
        updated_dataset = judging_task.run(args.evaluation_protocol, args.prompt_strategy)


    else:
        print(f"Path exists! Recomputing metrics")
        if len(output_paths) > 1:
            updated_dataset = {
                op.split('/')[-1].split('eval_detailed_')[-1].split('.jsonl')[0]: read_jsonl(op)
                for op in output_paths
            }
            updated_dataset =  DatasetDict(updated_dataset)
        else:
            updated_dataset = load_dataset("json", data_files=output_paths[0], split='train')
        

    if isinstance(updated_dataset, Dataset):
        updated_dataset = DatasetDict({args.dataset_name: updated_dataset})

    for split_name, updated_ds in updated_dataset.items():
        output_path = os.path.join(save_dir, f'eval_detailed_{split_name}.jsonl')
        output_result_path = os.path.join(save_dir, f'eval_result_{split_name}.json')
        if args.debug:
            output_path = output_path.replace('.jsonl', '.debug.jsonl')
            output_result_path = output_result_path.replace('.json', '.debug.json')

        judgements_1, judgements_2 = updated_ds['judgement_1'], updated_ds['judgement_2']
        outputs_1, outputs_2 = updated_ds['output_1'], updated_ds['output_2']
        
        with open(output_path, 'w', buffering=1) as fw:
            for i in range(len(updated_ds)):
                metadata = updated_ds[i].get('metadata', {})

                output = {
                    "votes": [judgements_1[i], judgements_2[i]],
                    "label": updated_ds[i]['label'],
                    "swap_inference1": outputs_1[i],
                    "swap_inference2": outputs_2[i],
                    "metadata": metadata,
                }
            
                fw.write(json.dumps(output)+"\n")

        metadata = {}
        if 'metadata' in updated_ds[0]:
            metadata = updated_ds['metadata']

        accuracy_consistent, consistency, accuracy_swap1, accuracy_swap2 = compute_acc(
            judgements_1, judgements_2, updated_ds['label'], metadata=metadata)
            
        output_evaluation = {
            'accuracy': float(np.average(accuracy_consistent)),
            'consistency': float(np.average(consistency)),
            'swap1_accuracy': float(np.average(accuracy_swap1)),
            'swap2_accuracy': float(np.average(accuracy_swap2)),
        }

        print(output_evaluation)

        with open(output_result_path, "w") as fw:
            json.dump(output_evaluation, fw, indent=4, sort_keys=True)

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Evaluation parameters
    parser.add_argument("--output_dir", type=str, default='results/', help="the directory for storing results")
    parser.add_argument("--dataset_name", type=str, default="reasoningjudgebench")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--force_rerun", action='store_true')
    parser.add_argument("--num_proc", type=int, default=10, help="number of threads to use for parallel processing of examples for openai api calls")

    # judge info
    parser.add_argument("--judge_model", type=str, help="model checkpoint or name")
    parser.add_argument("--prompt_strategy", type=str, default='vanilla', choices=['vanilla'], help='type of prompting for judge models. Used to enable prompt ablations, if needed')
    parser.add_argument("--evaluation_protocol", type=str, default="pairwise", choices=["pairwise", "single_rating"], help="How to conduct evaluation")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1", help="base url for served VLLM judge")
    parser.add_argument("--api_key", type=str, default="sample-api-key")

    # decoding strategy
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--num_sequences", default=1, type=int)
    parser.add_argument("--max_tokens", default=512, type=int, help="Only used for budget forcing task; otherwise set manually")
    parser.add_argument("--max_rounds", default=2, type=int, help="Only used for self reflection (number of reflection rounds)")

    args = parser.parse_args()

    main(args)