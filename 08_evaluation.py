"""Stage 08 - Retrieval evaluation.

Measures how well the retriever finds the right sources using a small gold
set (``eval/gold.json``) that maps each question to the source pages that
actually contain the answer.

Metrics:
    * Precision@k - of the top-k retrieved, how many are relevant.
    * Recall@k    - of all relevant pages, how many appear in the top-k.
    * MRR         - reciprocal rank of the first relevant hit (averaged).

Run standalone (requires a built store)::

    python 08_evaluation.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set

from rag_utils import BASE_DIR, TOP_K, load_stage

retrieval_stage = load_stage("06_retrieve_context.py")

GOLD_PATH = BASE_DIR / "eval" / "gold.json"


def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Compute Precision@k for one query.

    Args:
        retrieved: Retrieved page numbers, in rank order.
        relevant: The set of relevant page numbers.
        k: Cut-off rank.

    Returns:
        Precision@k in the range ``[0, 1]``.
    """
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for p in top if p in relevant) / len(top)


def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Compute Recall@k for one query.

    Args:
        retrieved: Retrieved page numbers, in rank order.
        relevant: The set of relevant page numbers.
        k: Cut-off rank.

    Returns:
        Recall@k in the range ``[0, 1]``.
    """
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return sum(1 for p in relevant if p in top) / len(relevant)


def reciprocal_rank(retrieved: List[int], relevant: Set[int]) -> float:
    """Compute the reciprocal rank of the first relevant hit.

    Args:
        retrieved: Retrieved page numbers, in rank order.
        relevant: The set of relevant page numbers.

    Returns:
        ``1/rank`` of the first relevant page, or ``0.0`` if none found.
    """
    for rank, page in enumerate(retrieved, start=1):
        if page in relevant:
            return 1.0 / rank
    return 0.0


def load_gold(path: Path = GOLD_PATH) -> Dict[str, Set[int]]:
    """Load the gold question -> relevant-pages mapping.

    Args:
        path: Path to the gold JSON file.

    Returns:
        A dict mapping each question to a set of relevant page numbers.
    """
    raw = json.loads(Path(path).read_text())
    return {q: set(pages) for q, pages in raw.items() if not q.startswith("_")}


def evaluate(k: int = TOP_K, use_reranker: bool = False) -> Dict[str, float]:
    """Evaluate retrieval over the whole gold set.

    Args:
        k: Cut-off rank for the metrics.
        use_reranker: Whether to apply reranking during retrieval.

    Returns:
        A dict with mean ``precision``, ``recall``, and ``mrr``.
    """
    gold = load_gold()
    retriever = retrieval_stage.HybridRetriever()

    precisions, recalls, rrs = [], [], []
    for question, relevant in gold.items():
        chunks = retriever.retrieve(question, top_k=k, use_reranker=use_reranker)
        retrieved_pages = [c["page"] for c in chunks]
        precisions.append(precision_at_k(retrieved_pages, relevant, k))
        recalls.append(recall_at_k(retrieved_pages, relevant, k))
        rrs.append(reciprocal_rank(retrieved_pages, relevant))

    n = max(len(gold), 1)
    return {
        "precision@k": sum(precisions) / n,
        "recall@k": sum(recalls) / n,
        "mrr": sum(rrs) / n,
    }


def evaluate_detailed(k: int = TOP_K, use_reranker: bool = False) -> Dict[str, object]:
    """Evaluate retrieval and return per-question rows plus the summary.

    Unlike :func:`evaluate` (summary only) and :func:`report` (prints to the
    console), this returns structured data so a UI such as the Streamlit app
    can render the full breakdown.

    Args:
        k: Cut-off rank for the metrics.
        use_reranker: Whether to apply cross-encoder reranking.

    Returns:
        A dict with two keys:
            * ``rows``: a list of per-question dicts (question, retrieved
              pages, relevant pages, matched pages, precision, recall, rr).
            * ``summary``: mean ``precision@k``, ``recall@k``, and ``mrr``.
    """
    gold = load_gold()
    retriever = retrieval_stage.HybridRetriever()

    rows: List[dict] = []
    precisions, recalls, rrs = [], [], []
    for question, relevant in gold.items():
        chunks = retriever.retrieve(question, top_k=k, use_reranker=use_reranker)
        retrieved_pages = [c["page"] for c in chunks]
        p = precision_at_k(retrieved_pages, relevant, k)
        r = recall_at_k(retrieved_pages, relevant, k)
        rr = reciprocal_rank(retrieved_pages, relevant)
        precisions.append(p)
        recalls.append(r)
        rrs.append(rr)
        rows.append(
            {
                "question": question,
                "retrieved": retrieved_pages,
                "relevant": sorted(relevant),
                "matched": [pg for pg in retrieved_pages if pg in relevant],
                "precision": p,
                "recall": r,
                "rr": rr,
            }
        )

    n = max(len(gold), 1)
    summary = {
        "precision@k": sum(precisions) / n,
        "recall@k": sum(recalls) / n,
        "mrr": sum(rrs) / n,
    }
    return {"rows": rows, "summary": summary}


def report(k: int = TOP_K, use_reranker: bool = False) -> Dict[str, float]:
    """Run the evaluation and print a detailed, human-readable report.

    For each gold question it shows the pages the retriever returned, the
    pages the ground truth considers correct, which ones matched, and the
    per-question Precision@k / Recall@k / Reciprocal Rank. Ends with the
    averaged summary metrics.

    Args:
        k: Cut-off rank for the metrics.
        use_reranker: Whether to apply cross-encoder reranking.

    Returns:
        A dict with mean ``precision@k``, ``recall@k``, and ``mrr``.
    """
    gold = load_gold()
    retriever = retrieval_stage.HybridRetriever()

    print("=" * 70)
    print(f"RETRIEVAL EVALUATION REPORT  (k={k}, reranker={use_reranker})")
    print(f"Gold set: {GOLD_PATH}  |  {len(gold)} question(s)")
    print("=" * 70)

    precisions, recalls, rrs = [], [], []
    for idx, (question, relevant) in enumerate(gold.items(), start=1):
        chunks = retriever.retrieve(question, top_k=k, use_reranker=use_reranker)
        retrieved_pages = [c["page"] for c in chunks]
        p = precision_at_k(retrieved_pages, relevant, k)
        r = recall_at_k(retrieved_pages, relevant, k)
        rr = reciprocal_rank(retrieved_pages, relevant)
        precisions.append(p)
        recalls.append(r)
        rrs.append(rr)

        matched = [pg for pg in retrieved_pages if pg in relevant]
        print(f"\n[{idx}] {question}")
        print(f"    retrieved pages : {retrieved_pages}")
        print(f"    relevant pages  : {sorted(relevant)}")
        print(f"    matched         : {matched if matched else 'none'}")
        print(f"    Precision@{k} = {p:.3f}   Recall@{k} = {r:.3f}   RR = {rr:.3f}")

    n = max(len(gold), 1)
    scores = {
        "precision@k": sum(precisions) / n,
        "recall@k": sum(recalls) / n,
        "mrr": sum(rrs) / n,
    }

    print("\n" + "=" * 70)
    print(f"SUMMARY (averaged over {n} question(s), k={k})")
    print(f"  Precision@k : {scores['precision@k']:.3f}")
    print(f"  Recall@k    : {scores['recall@k']:.3f}")
    print(f"  MRR         : {scores['mrr']:.3f}")
    print("=" * 70)
    return scores


if __name__ == "__main__":
    report(k=TOP_K, use_reranker=False)
