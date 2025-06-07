
PROMPT_PAIRWISE="""
You are a fair judge assistant assigned to deliver insightful feedback that compares individual performances, highlighting how each stands relative to others within the same cohort.

###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of two responses strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, choose a better response between Response A and Response B. You should refer to the score rubric.
3. The output format should look as follows: "(write a feedback for criteria) [RESULT] (A or B)"
4. Please do not generate any other opening, closing, and explanations.

###Instruction:
{question}

###Response A:
{response_a}

###Response B:
{response_b}

### Rubric:
[Are the model's responses factually correct and well-supported by evidence?]

###Feedback: 
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
    critique_judgement = judge_output.split('[RESULT]')
    judgement = critique_judgement[-1].strip()
    
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

single_instance_rate_max_score=5
