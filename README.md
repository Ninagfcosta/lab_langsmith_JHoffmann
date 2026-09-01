# LangSmith Lab - AI Concepts Tutor Evaluation

## Domain

For this lab I built an evaluation for an "AI tutor" that answers beginner
questions about Generative AI / LLM concepts (tokens, embeddings, RAG,
prompting, fine-tuning, temperature, hallucination, vector databases,
agents, LangSmith itself, etc). I picked this domain because it's the same
material we've been covering in class, so I could write accurate reference
answers myself and it's easy to judge if the model's answers are actually
correct.

Dataset: 15 question/answer pairs I wrote by hand, in `data/qa_dataset.json`.
Each item has a `question`, a reference `answer`, a `category` (topic) and a
`difficulty` (easy/medium/hard).

LangSmith dataset name: `ai-concepts-tutor-qa-v1`
Dataset link: *(add after creating it)*
Experiment link: *(add after running it)*

## What each file does

- `data/qa_dataset.json` - the 15 examples used to build the dataset
- `src/create_dataset.py` - creates the dataset in LangSmith and uploads the examples
- `src/target_function.py` - the target function, sends each question to gpt-4o-mini with a tutor system prompt
- `src/evaluators.py` - correctness evaluator (openevals built-in prompt) + a custom clarity evaluator I wrote to check if answers are actually beginner-friendly
- `src/run_evaluation.py` - runs the evaluation over the whole dataset
- `src/cost_performance.py` - optional part, compares gpt-4o-mini vs gpt-4o on score and cost
- `evaluation_summary.md` - short report on the results
- `optimization_summary.md` - short report on the cost/performance comparison

## How to run

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your OpenAI + LangSmith keys

python src/create_dataset.py
python src/run_evaluation.py
python src/cost_performance.py   # optional
```
