# 02 — Phase 2: Advanced Retrieval

## Goal
Improve retrieval quality beyond plain semantic search, since that is where most RAG quality
problems come from. Two upgrades: hybrid search and cross-encoder reranking. Both live inside
the existing `06_retrieve_context.py`, so the pipeline shape does not change.

## Files created / changed
- `06_retrieve_context.py` — adds `HybridRetriever` (dense + BM25 + RRF) and optional
  cross-encoder reranking.
- `07_prompting.py` — modular prompting (separate functions for system prompt, context
  formatting, and assembly) so the prompt is easy to read and adjust.

## Responsibilities
`HybridRetriever` owns all retrieval logic: it queries Chroma for dense results, scores the
same chunks with BM25, fuses the two rankings, and (optionally) reranks the top candidates.

## Workflow
```
query
  ├─ dense ranking  (Chroma vector search)      ─┐
  ├─ sparse ranking (BM25 keyword scores)        ├─ Reciprocal Rank Fusion → top candidates
  └───────────────────────────────────────────── ┘
                                                     ↓ (optional)
                                          cross-encoder reranking → top-k
```

## Key algorithms
- **Dense retrieval** — semantic similarity; strong on paraphrase, weak on exact tokens.
- **BM25 sparse retrieval** — exact term matching; strong on clause numbers and symbols
  (e.g. `f'c`), weak on paraphrase.
- **Reciprocal Rank Fusion (RRF)** — `score(chunk) = Σ 1 / (k + rank)` across both lists.
  Rank-based, so it needs no score normalization between the two very different scales.
  `k` (default 60) controls how quickly rank influence flattens.
- **Cross-encoder reranking** — a model reads `(query, chunk)` together and scores relevance
  directly. More accurate than embedding similarity, but slower, so it runs only on the small
  fused candidate pool, not the whole corpus.

## Why hybrid matters here
Engineers often search with exact identifiers (a clause number, a bar size). Pure embeddings
can miss those; BM25 catches them. Fusing both improves recall without hurting semantic hits.

## How to run
```bash
python 06_retrieve_context.py     # prints top hits for a sample query
```
In the Streamlit sidebar, toggle **"Use cross-encoder reranking"** to compare results.

## Future improvements
- Tune the dense/sparse balance (currently equal via RRF).
- Try larger or domain-specific rerankers.
- Add fuzzy matching for mistyped clause numbers.
