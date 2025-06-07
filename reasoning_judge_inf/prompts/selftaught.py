PROMPT_PAIRWISE_SYSTEM="""
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user\'s instructions and answers the user\'s question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \\"[[A]]\\" if assistant A is better, \\"[[B]]\\" if assistant B is better.
""".strip()


PROMPT_PAIRWISE="""
[User Question]
{question}

[The Start of Assistant A's Answer]
{response_a}
[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]
{response_b}
[The End of Assistant B's Answer]
""".strip()

def parse_pairwise_judgment(judge_output, flip=False):
    if not flip:
        if '[[A]]' in judge_output:
            return 1
        elif '[[B]]' in judge_output:
            return 2
        else:
            return -1
    else:
        if '[[A]]' in judge_output:
            return 2
        elif '[[B]]' in judge_output:
            return 1
        else:
            return -1

def render_pairwise_prompt(response_a, response_b, question, evaluation_criteria, prompt_strategy='vanilla'):
    prompt_template = PROMPT_PAIRWISE

    prompt_formatted = prompt_template.format(
        question=question,
        response_a=response_a,
        response_b=response_b,
    )
    return [{"role": "system", "content": PROMPT_PAIRWISE_SYSTEM}, {"role": "user", "content": prompt_formatted}]

# Ignored; Pairwise judge only ===========================
single_instance_rate_max_score = 5