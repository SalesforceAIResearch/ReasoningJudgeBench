from vllm import SamplingParams
from openai import OpenAI
from reasoning_judge_inf.scaling_method import Task
from reasoning_judge_inf.judge import Judge


class StandardJudging(Task):
    def __init__(self, dataset_name: str, judge: Judge, max_tokens: int, debug: bool = False, num_proc: int = 10):

        self.sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
        )

        super().__init__(dataset_name, judge, debug=debug, num_proc=num_proc)


    #TODO
    def get_single_rating_judgments(self, response: str, question: str, evaluation_criteria: str, prompt_strategy: str, instance_criteria: str):
        outputs, _ = self.judge.single_instance_rate(response, question, self.sampling_params, prompt_strategy=prompt_strategy, instance_criteria=instance_criteria)
        judgments = [self.judge.prompter.parse_single_instance_rate(o) for o in outputs]

        return outputs, judgments

    def get_pairwise_judgments(self, response_a: str, response_b: str, question: str, evaluation_criteria: str, prompt_strategy: str):
        # Consistency run 1
        outputs_1, _ = self.judge.pairwise_compare(response_a, response_b, question, evaluation_criteria,
            self.sampling_params, prompt_strategy=prompt_strategy)
        judgments_1 = [self.judge.prompter.parse_pairwise_judgment(o) for o in outputs_1]

        # Consistency run 2
        outputs_2, _ = self.judge.pairwise_compare(response_b, response_a, question, evaluation_criteria,
            self.sampling_params, prompt_strategy=prompt_strategy)
        judgments_2 = [self.judge.prompter.parse_pairwise_judgment(o, flip=True) for o in outputs_2]

        return outputs_1, outputs_2, judgments_1, judgments_2, {}
