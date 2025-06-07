PROMPT_PAIRWISE = """
Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the Client's question displayed below.

[Client Question]
{question}

[The Start of Chatbot A's Response]
{response_a}
[The End of Chatbot A's Response]

[The Start of Chatbot A's Response]
{response_b}
[The End of Chatbot A's Response]

Instructions
1. Begin your evaluation by generating the rubric criteria tailored to the Client's question and context.
Enclose the rubric in <rubric> . . . </rubric> tags.
2. Assign weights to each rubric item based on their relative importance.
3. Within <rubric>, include a <justify> . . . </justify> section explaining the rationale behind the chosen
criteria and weights.
4. Compare both Chatbot responses using the rubric.
5. Include your evaluation in <eval> . . . </eval> tags.
Support your analysis using:
- <quote_A> . . . </quote_A> for direct quotes from Chatbot A
- <summary_A> . . . </summary_A> for paraphrased summaries of Chatbot A
- <quote_B> . . . </quote_B> for direct quotes from Chatbot B
- <summary_B> . . . </summary_B> for paraphrased summaries of Chatbot B
6. Conclude with your final judgment using:
<answer>[[A]]</answer> or <answer>[[B]]</answer>
Important Notes:
- Be objective and base your evaluation strictly on the content of the responses.
- Do not let the response order, length, or Chatbot names bias your judgment.
- The question may have other enumerated options in it, such as multiple choice letters or numbered options. Ignore those. Your answer must be [[A]] or [[B]].
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
    verdict = judge_output.split('<answer>')[-1].split('</answer>')[0].strip()
    if '[[A]]' in verdict and '[[B]]' not in verdict:
        judgement = 'A'
    elif '[[B]]' in verdict and '[[A]]' not in verdict:
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
