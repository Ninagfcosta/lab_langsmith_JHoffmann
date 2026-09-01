"""
Target function: sends each dataset question to an LLM playing the role
of a beginner-friendly AI tutor and returns its answer.

Uses @traceable so LangSmith logs every call automatically.
"""

import os

from dotenv import load_dotenv
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from openai import OpenAI

load_dotenv()

openai_client = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

SYSTEM_PROMPT = (
    "You are a friendly AI/ML tutor who explains generative AI concepts to "
    "complete beginners. Answer in 2-4 short sentences. Be accurate and "
    "avoid jargon; if you must use a technical term, briefly explain it."
)

MODEL_NAME = "gpt-4o-mini"


@traceable(name="ai_tutor_target_function")
def ai_tutor_target(inputs: dict) -> dict:
    """inputs = {"question": "..."} -> returns {"answer": "..."}"""
    question = inputs.get("question", "")

    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            max_tokens=200,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        answer = response.choices[0].message.content.strip()
        return {"answer": answer}
    except Exception as exc:
        # Don't let one bad API call kill the whole evaluation run
        return {"answer": "", "error": str(exc)}


if __name__ == "__main__":
    # quick check that the function works before running the full eval
    sample_questions = [
        "What is a token in the context of large language models?",
        "What does RAG stand for and what problem does it solve?",
        "What is the temperature parameter used for?",
    ]
    for q in sample_questions:
        result = ai_tutor_target({"question": q})
        print(f"Q: {q}\nA: {result['answer']}\n")
