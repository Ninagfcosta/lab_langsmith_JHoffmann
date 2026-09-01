# Evaluation Summary

> Fill this in after you run `src/run_evaluation.py` and look at the results in the LangSmith UI. Everything in [brackets] needs a real number or observation from your run.

I evaluated an AI tutor target function (`gpt-4o-mini`, temperature 0) against a custom 15-example LangSmith dataset covering core Generative AI/LLM concepts (tokens, embeddings, RAG, prompting, evaluation, etc.), using two LLM-as-judge evaluators: a built-in **correctness** evaluator (openevals `CORRECTNESS_PROMPT`, comparing each answer to a reference answer) and a custom **clarity** evaluator scoring 1-5 how beginner-friendly each answer is. The average correctness score was **[X.XX]** and the average clarity score was **[X.XX]**; **[N/15]** examples scored below [threshold] on correctness, and the most common failure pattern was **[e.g. "answers on harder/'hard' difficulty examples like RAG and evaluation left out an important nuance" — replace with your real finding]**. The main limitations of this evaluation are the small dataset size (15 examples), the use of the same family of model (`gpt-4o-mini`) as both target and judge which can bias scoring, and reference answers written by one person rather than validated externally; a good next step would be **[e.g. "expand the dataset to 30+ examples per difficulty level and add a second judge model to cross-check scores"]**.

## Key metrics (fill in from LangSmith)

| Metric | Value |
|---|---|
| Dataset size | 15 |
| Model | gpt-4o-mini, temperature=0 |
| Avg. correctness score | [ ] |
| Avg. clarity score | [ ] |
| Lowest-scoring example(s) | [ ] |
| Highest-scoring example(s) | [ ] |

## Biggest failure pattern

[Describe the 1-2 most common ways answers scored low — e.g. missing nuance on "hard" examples, an unexplained technical term, too long/short.]

## Recommendation

[One sentence: what would you change next — bigger dataset, different model, refined system prompt, etc.]
