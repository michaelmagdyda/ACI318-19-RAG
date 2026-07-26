# 03 — Phase 3: Evaluation & Finalization

## Goal
Measure whether retrieval actually finds the right sources, then finalize the project
(documentation, cleanup, README). Evaluation turns "it looks like it works" into numbers you
can report and defend in a presentation.

## Files created
- `08_evaluation.py` — Precision@k, Recall@k, and MRR over a gold set.
- `eval/gold.json` — maps each question to the page numbers that truly contain the answer.

## Responsibilities
`08_evaluation.py` runs each gold question through the retriever, compares the retrieved page
numbers against the relevant ones, and averages the metrics across all questions.

## Workflow
1. Edit `eval/gold.json` with real question → relevant-page mappings for your document.
2. Run the evaluation; it retrieves for each question and computes metrics.
3. Read the averaged Precision@k, Recall@k, and MRR.

## Key algorithms / metrics
- **Precision@k** — of the top-k retrieved pages, the fraction that are relevant. "How clean
  are the results?"
- **Recall@k** — of all relevant pages, the fraction that appear in the top-k. "How complete
  are the results?"
- **MRR (Mean Reciprocal Rank)** — `1/rank` of the first relevant hit, averaged over
  questions. Rewards putting a correct source near the top.

## Interpreting results
- Low precision **and** recall → retrieval is off (check chunking, embeddings, or the store).
- Good recall, low precision → right pages are found but buried among noise → try reranking.
- Good precision, low recall → results are clean but incomplete → raise `k` or improve chunking.

## How to run
```bash
python 08_evaluation.py
```
Output example:
```
Evaluation over gold set (k=5):
  Precision@k : 0.720
  Recall@k    : 0.900
  MRR         : 0.850
```

## Honesty note
The shipped `gold.json` is a small starter set tied to the sample document. Metrics are only
meaningful once you edit it to match your real document and questions. A 5–10 question set is
fine for a diploma project but is too small to be statistically robust — treat the numbers as
indicative.

## Future improvements
- Add answer-quality metrics (token F1, semantic similarity, LLM-as-judge).
- Grow the gold set and split it into a held-out test set.
- Log per-question results to spot systematic failures.
