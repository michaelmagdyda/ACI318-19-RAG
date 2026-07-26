"""Stage 06 - Context retrieval (hybrid search + reranking).

Given a question, find the most relevant chunks. This stage combines three
ideas taught as AI-engineering improvements:

1. Dense retrieval - semantic search over the Chroma vector store. Good at
   meaning ("cover" ~ "protective layer").
2. BM25 sparse retrieval - classic keyword scoring. Good at exact terms like
   clause numbers or "f'c".
3. Reciprocal Rank Fusion (RRF) - merges the two ranked lists without needing
   to reconcile their different score scales.

Optionally, a cross-encoder reranks the fused candidates for higher precision.

Run standalone (requires a built store)::

    python 06_retrieve_context.py
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Dict, List

from rank_bm25 import BM25Okapi

from rag_utils import CANDIDATE_POOL, RERANKER_MODEL, TOP_K, load_stage

chroma_stage = load_stage("05_create_chroma_store.py")
vectors_stage = load_stage("04_vector_representation.py")


def _tokenize(text: str) -> List[str]:
    """Lowercase word/number tokenizer used by BM25."""
    return re.findall(r"[a-z0-9']+", text.lower())


def _load_all_chunks(collection) -> List[Dict]:
    """Read every chunk back out of Chroma to build the BM25 index.

    Args:
        collection: A Chroma collection.

    Returns:
        A list of chunk dicts with ``id``, ``text``, ``source``, ``page``.
    """
    data = collection.get(include=["documents", "metadatas"])
    chunks: List[Dict] = []
    for cid, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        chunks.append(
            {
                "id": cid,
                "text": text,
                "source": meta.get("source", "unknown"),
                "page": meta.get("page"),
            }
        )
    return chunks


@lru_cache(maxsize=2)
def get_reranker(model_name: str = RERANKER_MODEL):
    """Load (and cache) a cross-encoder reranking model.

    Args:
        model_name: The cross-encoder model identifier.

    Returns:
        A loaded CrossEncoder instance.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


class HybridRetriever:
    """Hybrid dense + BM25 retriever with optional cross-encoder reranking."""

    def __init__(self, collection_name: str | None = None, rrf_k: int = 60) -> None:
        """Initialize the retriever and build the in-memory BM25 index.

        Args:
            collection_name: Chroma collection to use (default from config).
            rrf_k: The RRF constant; larger values flatten rank influence.
        """
        self.collection = (
            chroma_stage.get_collection(collection_name)
            if collection_name
            else chroma_stage.get_collection()
        )
        self.chunks = _load_all_chunks(self.collection)
        self.rrf_k = rrf_k
        self._bm25 = BM25Okapi([_tokenize(c["text"]) for c in self.chunks])
        self._id_to_index = {c["id"]: i for i, c in enumerate(self.chunks)}

    def _dense_ranking(self, query: str, pool: int) -> List[str]:
        """Return chunk ids ranked by dense (vector) similarity."""
        query_vec = vectors_stage.embed_texts([query])[0]
        result = self.collection.query(
            query_embeddings=[query_vec.tolist()], n_results=pool
        )
        return list(result["ids"][0])

    def _sparse_ranking(self, query: str, pool: int) -> List[str]:
        """Return chunk ids ranked by BM25 keyword score."""
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i]["id"] for i in ranked[:pool] if scores[i] > 0]

    def _fuse_rrf(self, dense_ids: List[str], sparse_ids: List[str]) -> List[str]:
        """Merge two ranked id lists using Reciprocal Rank Fusion."""
        scores: Dict[str, float] = {}
        for rank, cid in enumerate(dense_ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, cid in enumerate(sparse_ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
        return sorted(scores, key=lambda cid: scores[cid], reverse=True)

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        pool: int = CANDIDATE_POOL,
        use_reranker: bool = False,
    ) -> List[Dict]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: The user question.
            top_k: How many chunks to return.
            pool: Candidate pool size per retriever before fusion.
            use_reranker: If True, apply cross-encoder reranking to the pool.

        Returns:
            A list of chunk dicts, best first, each including a ``score`` key.
        """
        dense_ids = self._dense_ranking(query, pool)
        sparse_ids = self._sparse_ranking(query, pool)
        fused_ids = self._fuse_rrf(dense_ids, sparse_ids)

        candidate_ids = fused_ids[: max(top_k, pool if use_reranker else top_k)]
        candidates = [self.chunks[self._id_to_index[cid]] for cid in candidate_ids]

        if use_reranker and candidates:
            reranker = get_reranker()
            pairs = [(query, c["text"]) for c in candidates]
            rerank_scores = reranker.predict(pairs)
            ranked = sorted(
                zip(candidates, rerank_scores), key=lambda x: x[1], reverse=True
            )
            return [{**c, "score": float(s)} for c, s in ranked[:top_k]]

        # Without reranking, score by fusion rank position (1.0 at the top).
        results = []
        for rank, cid in enumerate(candidate_ids[:top_k], start=1):
            chunk = self.chunks[self._id_to_index[cid]]
            results.append({**chunk, "score": 1.0 / rank})
        return results


if __name__ == "__main__":
    retriever = HybridRetriever()
    hits = retriever.retrieve("What is the minimum concrete cover?", use_reranker=False)
    for h in hits:
        print(f"[{h['score']:.3f}] {h['source']} p.{h['page']}: {h['text'][:90]}...")
