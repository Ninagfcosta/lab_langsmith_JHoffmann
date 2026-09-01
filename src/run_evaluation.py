"""
Runs the full evaluation: pulls the dataset, calls the target function on
every example, scores the results with both evaluators, and logs it all
as one experiment in LangSmith.

Run create_dataset.py first.
"""

from datetime import datetime

from dotenv import load_dotenv
from langsmith import Client

from create_dataset import DATASET_NAME
from evaluators import clarity_evaluator, correctness_evaluator
from target_function import ai_tutor_target

load_dotenv()


def main():
    client = Client()

    today = datetime.now().strftime("%Y-%m-%d")
    experiment_prefix = f"ai-tutor-gpt4o-mini-{today}"

    print(f"Running evaluation on '{DATASET_NAME}'...")

    results = client.evaluate(
        ai_tutor_target,
        data=DATASET_NAME,
        evaluators=[correctness_evaluator, clarity_evaluator],
        experiment_prefix=experiment_prefix,
        max_concurrency=3,
        metadata={"model": "gpt-4o-mini", "temperature": 0},
    )

    print("Done. Check the experiment in the LangSmith UI for traces and scores.")
    print(results)


if __name__ == "__main__":
    main()
