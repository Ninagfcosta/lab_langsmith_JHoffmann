"""
Two evaluators for scoring the target function's answers:

- correctness_evaluator: openevals' pre-built LLM-as-judge prompt,
  compares the answer to the reference answer from the dataset.
- clarity_evaluator: a custom evaluator I wrote to check if the answer
  is actually easy to understand for someone new to AI, since that
  matters for a "tutor" use case and correctness alone doesn't cover it.
"""

from dotenv import load_dotenv
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

load_dotenv()

JUDGE_MODEL = "openai:gpt-4o-mini"

_correctness_judge = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    model=JUDGE_MODEL,
)


def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    return _correctness_judge(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs,
    )


CLARITY_PROMPT = """You are grading whether an AI tutor's answer is clear and
beginner-friendly for someone with no prior AI/ML knowledge.

<question>
{inputs}
</question>

<answer>
{outputs}
</answer>

Score from 1 to 5:
- 5: Very clear, simple words, no unexplained jargon.
- 3: Understandable but has at least one unexplained technical term, or is
     a bit too dense for a beginner.
- 1: Confusing or assumes knowledge a beginner would not have.

Give your reasoning, then end with a line formatted exactly as:
Score: <integer 1-5>
"""

_clarity_judge = create_llm_as_judge(
    prompt=CLARITY_PROMPT,
    feedback_key="clarity",
    model=JUDGE_MODEL,
)


def clarity_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    # reference_outputs isn't used here, clarity doesn't depend on the
    # "correct" answer, just accepted so the signature matches the other one
    return _clarity_judge(inputs=inputs, outputs=outputs)


if __name__ == "__main__":
    test_inputs = {"question": "What is a token?"}
    test_outputs = {
        "answer": "A token is a small chunk of text, like a word or part of "
        "a word, that a language model reads or writes one piece at a time."
    }
    test_reference = {"answer": "A token is a small unit of text used by LLMs."}

    print("Correctness:", correctness_evaluator(test_inputs, test_outputs, test_reference))
    print("Clarity:", clarity_evaluator(test_inputs, test_outputs, test_reference))
