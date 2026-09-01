"""
Extra comparison: runs the same evaluation with gpt-4o-mini and gpt-4o so
I can compare scores vs. cost between a cheaper and a more capable model.

Run run_evaluation.py first.
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from langsmith import Client, traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI

from create_dataset import DATASET_NAME
from evaluators import clarity_evaluator, correctness_evaluator
from target_function import SYSTEM_PROMPT

load_dotenv()

openai_client = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

# Rough public per-1M-token prices in USD (check platform.openai.com/docs/pricing
# for current numbers before quoting these in the report).
PRICING_PER_MILLION_TOKENS = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
}


def make_target(model_name):
    @traceable(name=f"ai_tutor_target_{model_name}")
    def target(inputs: dict) -> dict:
        question = inputs.get("question", "")
        try:
            response = openai_client.chat.completions.create(
                model=model_name,
                temperature=0,
                max_tokens=200,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
            )
            usage = response.usage
            return {
                "answer": response.choices[0].message.content.strip(),
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
            }
        except Exception as exc:
            return {"answer": "", "error": str(exc)}

    return target


def estimate_cost(model_name, prompt_tokens, completion_tokens):
    prices = PRICING_PER_MILLION_TOKENS[model_name]
    return (
        prompt_tokens * prices["input"] + completion_tokens * prices["output"]
    ) / 1_000_000


def run_for_model(client, model_name):
    today = datetime.now().strftime("%Y-%m-%d")
    prefix = f"ai-tutor-{model_name}-{today}"
    target = make_target(model_name)

    print(f"Running evaluation for {model_name}...")
    return client.evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[correctness_evaluator, clarity_evaluator],
        experiment_prefix=prefix,
        max_concurrency=3,
        metadata={"model": model_name, "temperature": 0},
    )


def main():
    client = Client()
    for model_name in ["gpt-4o-mini", "gpt-4o"]:
        run_for_model(client, model_name)

    print("Both experiments are done, compare them in the LangSmith UI.")


if __name__ == "__main__":
    main()
