PROMPT_PAIRWISE="""
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user prompt displayed below. You will be given assistant A's answer and assistant B's answer. Your job is to evaluate which assistant's answer is better.

When evaluating the assistants' answers, identify any mistakes or inaccurate information. Focus on the content each response, and select the response that arrives at the correct response and with the most sound and logical reasoning. Avoid any biases, such as order of responses, length, or stylistic elements like formatting.

Before providing an your final verdict, think through the judging process and output your thoughts as an explanation

After providing your explanation, you must output only one of the following choices as your final verdict with a label:

1. Assistant A is better: [[A>B]]
2. Assistant B is better: [[B>A]]

Use the following template:
<think>Your detailed thought process as an explanation.</think>
<answer>[[A>B]] or [[B>A]]</answer>

<|User Prompt|>
{question}

<|The Start of Assistant A's Answer|>
{response_a}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{response_b}
<|The End of Assistant B's Answer|>
""".strip()

def render_pairwise_prompt(response_a, response_b, question, evaluation_criteria, prompt_strategy='vanilla'):
    prompt_template = PROMPT_PAIRWISE
    prompt_formatted = prompt_template.format(
        question=question,
        response_a=response_a,
        response_b=response_b,
    )
    return [{"role": "user", "content": prompt_formatted}]

def parse_pairwise_judgment(judge_output, flip=False):

    verdict = judge_output.split('<answer>')[-1].strip()
    if '[[A>B]]' in verdict and '[[B>A]]' not in verdict:
        judgement = 'A'
    elif '[[B>A]]' in verdict and '[[A>B]]' not in verdict:
        judgement = 'B'
    else:
        judgement = 'Error'
    
    if not flip:
        if judgement == 'A':
            return 1
        elif judgement == 'B':
            return 2
        else:
            return -1
    else:
        if judgement == 'A':
            return 2
        elif judgement == 'B':
            return 1
        else:
            return -1
    return -1

# Ignored; Pairwise judge only ===========================
single_instance_rate_max_score = 5 
