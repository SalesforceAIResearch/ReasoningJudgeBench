from vllm import SamplingParams
from openai import OpenAI
from datasets import Dataset
from reasoning_judge_inf.judge.vllm_utils import VllmEndpoint
from reasoning_judge_inf.prompts import load_prompter

class Judge():
    def __init__(self, prompter):
        self.prompter = prompter
        self.single_instance_rate_max_score = prompter.single_instance_rate_max_score

    def generate(self, messages: list):
        raise NotImplementedError

    def single_instance_rate(self, response: str, question: str, evaluation_criteria: str, 
        sampling_params: SamplingParams, prompt_strategy:str = 'vanilla', instance_criteria: str = None,
        return_comp_tokens=False, partial_judge_resp: str = None):
        messages = self.prompter.render_single_instance_rate_prompt(response, question, evaluation_criteria, prompt_strategy=prompt_strategy, instance_criteria=instance_criteria)
        if partial_judge_resp is not None:
            messages.append({'role': 'assistant', 'content': partial_judge_resp})

        response_texts, comp_tokens = self.generate(messages, sampling_params, return_comp_tokens=return_comp_tokens)
        return response_texts, comp_tokens

    def pairwise_compare(self, response_a: str, response_b: str, question: str, evaluation_criteria: str, sampling_params: SamplingParams, prompt_strategy:str = 'vanilla', 
        return_comp_tokens=False, partial_judge_resp: str = None, completion_gen: bool = False):
        messages = self.prompter.render_pairwise_prompt(response_a, response_b, question, evaluation_criteria, prompt_strategy=prompt_strategy)
        if partial_judge_resp is not None:
            messages.append({'role': 'assistant', 'content': partial_judge_resp})
        response_texts, comp_tokens = self.generate(messages, sampling_params, return_comp_tokens=return_comp_tokens, completion_gen=completion_gen)
        return response_texts, comp_tokens

    
class VllmEndpointJudge(Judge):
    def __init__(self, endpoint_config: dict, prompter=None):
        self.endpoint = VllmEndpoint(endpoint_config)
        self.model_name = self.endpoint.model_name

        if prompter is None:
            prompter = load_prompter(self.model_name)

        print(f"Pairwise Prompt: {prompter.PROMPT_PAIRWISE}")
        

        super().__init__(prompter)

    def generate(self, messages: list, sampling_params: SamplingParams, return_comp_tokens: bool = False, completion_gen: bool = False):
        return self.endpoint.generate(messages, sampling_params, return_comp_tokens = return_comp_tokens, completion_gen = completion_gen)



class HfRewardModel():
    def __init__(self, model, tokenizer, device, model_name, accelerator=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        inferencer = load_inferencer(model_name)
        self.inferencer = inferencer # anolgous to VLLM prompter
        self.accelerator = accelerator

    def single_instance_rate_batch(self, query_lst, response_lst):
        prompt_lst = [
            [{"role": "user", "content": q}, {"role": "assistant", "content": r}]
            for q, r in zip(query_lst, response_lst)
        ]
        scores_lst = self.generate_batch(prompt_lst)
        return scores_lst

    def generate_batch(self, conversations):
        return self.inferencer.generate_batch(conversations, self.tokenizer, self.model, self.device, accelerator=self.accelerator)