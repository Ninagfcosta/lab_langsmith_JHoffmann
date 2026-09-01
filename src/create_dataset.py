"""
Creates the LangSmith dataset for this project and uploads my examples.

Run this once before the evaluation script.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

DATASET_NAME = "ai-concepts-tutor-qa-v1"
DATASET_DESCRIPTION = (
    "15 question/answer pairs about core Generative AI and LLM concepts "
    "(tokens, embeddings, RAG, prompting, evaluation, etc). Used to test "
    "an AI tutor that should explain each concept in a simple, "
    "beginner-friendly way."
)

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "qa_dataset.json"


def load_examples():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def get_or_create_dataset(client):
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Dataset '{DATASET_NAME}' already exists, reusing it.")
        return client.read_dataset(dataset_name=DATASET_NAME)

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=DATASET_DESCRIPTION,
    )
    print(f"Created dataset '{DATASET_NAME}' (id={dataset.id}).")
    return dataset


def upload_examples(client, dataset_id, examples):
    inputs = [{"question": ex["question"]} for ex in examples]
    outputs = [{"answer": ex["answer"]} for ex in examples]
    metadata = [
        {"category": ex["category"], "difficulty": ex["difficulty"]} for ex in examples
    ]

    client.create_examples(
        inputs=inputs,
        outputs=outputs,
        metadata=metadata,
        dataset_id=dataset_id,
    )
    print(f"Uploaded {len(examples)} examples.")


def main():
    assert os.getenv("LANGCHAIN_API_KEY"), "Missing LANGCHAIN_API_KEY in .env"

    client = Client()
    examples = load_examples()
    assert len(examples) >= 10

    dataset = get_or_create_dataset(client)
    upload_examples(client, dataset.id, examples)

    print(f"\nDone. Check dataset '{DATASET_NAME}' in the LangSmith UI.")


if __name__ == "__main__":
    main()
