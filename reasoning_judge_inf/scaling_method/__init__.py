import numpy as np
from vllm import SamplingParams
from openai import OpenAI
from datasets import Dataset, DatasetDict, load_dataset
from reasoning_judge_inf.judge import Judge


dataset_to_path = {
    # pairwise datasets
    # Download benchmark data and store in /benchmark_data/ to run other benchmarks
    # Don't forget to change `input_key_maps` and `question_key_maps` if necessary! 
    'judgebench_gpt': './benchmark_data/judgebench_gpt.jsonl', 
    "ppe_gpqa": './benchmark_data/ppe_gpqa.jsonl',
    "ppe_mbpp": './benchmark_data/ppe_mbpp_plus.jsonl',
    "ppe_math": './benchmark_data/ppe_math.jsonl',
    # ReasoningJudgeBench
    'reasoningjudgebench': 'Salesforce/ReasoningJudgeBench',
}

input_key_maps = {
    # Other pairwise datasets
    'default': ['positive_response', 'negative_response'],
    'judgebench_gpt': ['response_A', 'response_B'],
    "ppe_gpqa": ['response_a', 'response_b'],
    "ppe_mbpp": ['response_a', 'response_b'],
    "ppe_math": ['response_a', 'response_b'],
    # ReasoningJudgeBench
    'reasoningjudgebench': ['positive_response', 'negative_response'],
}

question_key_maps = {
    # Other pairwise datasets
    'default': 'question',
    'judgebench_gpt': 'question',
    "ppe_gpqa": "question",
    "ppe_mbpp": "question",
    "ppe_math": "question",
    # ReasoningJudgeBench
    'reasoningjudgebench': 'instruction'
}

# If you want to implement custom eval criteria per benchmark, you can map benchmark -> criteria here
# Must update reasoning_judge_inf/prompts/[judge_model].py's render_pairwise_prompt function too
evaluation_criteria_map = {}


def get_dataset(dataset_name: str, num_proc: int):    
    data_path = dataset_to_path[dataset_name]
    if 'ReasoningJudgeBench/data' in dataset_to_path:
        eval_dataset = load_dataset("json", data_files=data_path, split='train')
    else:
        eval_dataset = load_dataset(data_path)
   
    standardize_label = None
    # Standardize labels
    if 'judgebench' in dataset_name:
        def standardize_label(example):
            if example['label'] == 'A>B':
                example['label'] = 1
            elif example['label'] == 'B>A':
                example['label'] = 2

            return example

    if standardize_label is not None:
        eval_dataset = eval_dataset.map(standardize_label, num_proc=num_proc)

    eval_criteria = evaluation_criteria_map.get(dataset_name, '')
    
    return eval_dataset, question_key_maps[dataset_name], input_key_maps[dataset_name], eval_criteria


class Task():
    def __init__(self, dataset_name: str, judge: Judge, debug: bool = False, num_proc: int = 10):
        self.judge = judge
        self.num_proc = num_proc
        self.dataset_name=dataset_name
        dataset, question_key, response_keys, eval_criteria = get_dataset(dataset_name, num_proc)

        if debug and isinstance(dataset, Dataset):
            dataset = dataset.select(range(10)) 
        # For debugging, we'll just grab the first split
        elif debug and isinstance(dataset, DatasetDict):
            dataset = dataset[list(dataset.keys())[0]]

        self.debug = debug
        self.question_key = question_key
        self.response_keys = response_keys
        self.dataset = dataset
        self.evaluation_criteria = eval_criteria        

    def get_pairwise_judgments(self, response_a: str, response_b: str, question: str, evaluation_criteria: str, prompt_strategy: str):
        raise NotImplementedError

    def get_single_rating_judgments(self, response: str, question: str, evaluation_criteria: str, prompt_strategy: str, instance_criteria: str = None):
        raise NotImplementedError

    def run_pairwise(self, prompt_strategy):
        def proc_example(example, question_key, response_keys, evaluation_criteria):
            response_a, response_b = example[response_keys[0]], example[response_keys[1]]
            question = example[question_key]
        
            outputs_1, outputs_2, judgments_1, judgments_2, metadata = self.get_pairwise_judgments(
                response_a, response_b, question, evaluation_criteria, prompt_strategy)

            output = {
                'judgement_1': judgments_1,
                'judgement_2': judgments_2,
                'output_1': outputs_1,
                'output_2': outputs_2,
                'metadata': metadata,
            }

            return output


        updated_dataset = self.dataset.map(
            proc_example, 
            num_proc = self.num_proc,
            fn_kwargs={
                "question_key": self.question_key, 
                "response_keys": self.response_keys, 
                "evaluation_criteria": self.evaluation_criteria}
                )

        return updated_dataset

    def run_single_rating(self, prompt_strategy):
        def proc_example(example, question_key, response_keys, evaluation_criteria):
            response = example[response_keys[0]]
            question = example[question_key]

            instance_criteria = None
            if prompt_strategy == 'instance_criteria':
                instance_criteria = example['instance_criteria']

            outputs, judgments = self.get_single_rating_judgments(response, question, prompt_strategy, instance_criteria=instance_criteria)

            output = {
                'judgement': judgments,
                'output': outputs
            }

            return output

        updated_dataset = self.dataset.map(
            proc_example, 
            num_proc = self.num_proc, 
            fn_kwargs={
                "question_key": self.question_key, 
                "response_keys": self.response_keys,
                "evaluation_criteria": self.evaluation_criteria})

        return updated_dataset

    def run(self, evaluation_protocol, prompt_strategy: str = 'vanilla'):
        if evaluation_protocol == "pairwise":
            return self.run_pairwise(prompt_strategy)
        elif evaluation_protocol == "single_rating":
            return self.run_single_rating(prompt_strategy)
        else:
            raise NotImplementedError(f"Protocol [{evaluation_protocol}] not implemented yet!")
    
