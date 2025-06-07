PROMPT_PAIRWISE_SYSTEM = """
You are a helpful assistant. The assistant first performs a detailed, step-by-step reasoning process in its mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> detailed reasoning process here, explaining each step of your evaluation for both assistants </think><answer> answer here </answer>. Now the user asks you to judge the performance of two AI assistants in response to the question. Score assistants 1-10 (higher=better). Criteria includes helpfulness, relevance, accuracy, and level of detail. Avoid order, length, style or other bias. After thinking, when you finally reach a conclusion, clearly  provide your evaluation scores within <answer> </answer> tags, i.e. for example,<answer>3</answer><answer>5</answer>. The question may have other enumerated options in it, such as multiple choice letters or numbered options. Ignore those. Your scores must be an integer between 1-10.
""".strip()

PROMPT_PAIRWISE="""
[Question]
{question}

[Assistant 1’s Answer]
{response_a}

[Assistant 2’s Answer]
{response_b}
""".strip()


def render_pairwise_prompt(response_a, response_b, question, evaluation_criteria, prompt_strategy='vanilla'):
    prompt_template = PROMPT_PAIRWISE
    if prompt_strategy == 'reconsistent':
        evaluation_criteria = RECONSISTENT_CRITERIA
        prompt_template = PROMPT_PAIRWISE_RECONSISTENT

    prompt_formatted = prompt_template.format(
        question=question,
        response_a=response_a,
        response_b=response_b,
    )
    return [{"role": "system", "content": PROMPT_PAIRWISE_SYSTEM}, {"role": "user", "content": prompt_formatted}]

def parse_pairwise_judgment(judge_output, flip=False):
    scores = judge_output.split('<answer>')[1:]

    # If two scores are not found, return
    if len(scores) < 2:
        return -1

    scores_parsed = []
    for s in scores:
        if '</answer>' not in s:
            continue

        s_parsed = s.split('</answer>')[0].strip()
        try:
            s_parsed = int(s_parsed)
        except:
            continue

        scores_parsed.append(s_parsed)

    if len(scores_parsed) != 2:
        return -1


    if not flip:
        if scores_parsed[0] > scores_parsed[1]:
            return 1
        elif scores_parsed[1] > scores_parsed[0]:
            return 2
        elif scores_parsed[0] == scores_parsed[1]:
            return 0
        else:
            return -1
    else:
        if scores_parsed[0] > scores_parsed[1]:
            return 2
        elif scores_parsed[1] > scores_parsed[0]:
            return 1
        elif scores_parsed[0] == scores_parsed[1]:
            return 0
        else:
            return -1
    

# Ignored; Pairwise judge only ===========================
single_instance_rate_max_score = 5
