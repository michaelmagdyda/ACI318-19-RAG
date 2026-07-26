"""Stage 07 - Prompting and answer generation.

Builds a grounded prompt from the retrieved chunks and sends it to an LLM via
OpenRouter (an OpenAI-compatible API gateway). Prompting is modular: the
system instructions, the context formatting, and the final assembly are
separate functions so they are easy to read and adjust.

The model is told to answer only from the context and to cite sources, which
is what keeps a RAG answer trustworthy.

Run standalone (requires a built store and an API key)::

    python 07_prompting.py
"""
from __future__ import annotations

from typing import Dict, List

from openai import OpenAI

import rag_utils
from rag_utils import format_sources, load_stage

retrieval_stage = load_stage("06_retrieve_context.py")

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided context.\n"
    "Rules:\n"
    "1. Use only the retrieved context to answer.\n"
    "2. If the answer is not in the context, say the information is not available.\n"
    "3. Always cite the source and page number(s) you used."
)


def build_context_block(chunks: List[Dict]) -> str:
    """Format retrieved chunks into a numbered, citable context block.

    Args:
        chunks: Retrieved chunk dicts with ``source``, ``page``, ``text``.

    Returns:
        A single string with one labeled block per chunk.
    """
    blocks = []
    for i, c in enumerate(chunks, start=1):
        page = f" p.{c['page']}" if c.get("page") is not None else ""
        blocks.append(f"[{i}] Source: {c['source']}{page}\n{c['text']}")
    return "\n\n".join(blocks)


def build_prompt(query: str, chunks: List[Dict]) -> List[Dict]:
    """Assemble the chat messages for the LLM.

    Args:
        query: The user question.
        chunks: Retrieved chunks to ground the answer.

    Returns:
        A list of chat message dicts (system + user).
    """
    context = build_context_block(chunks)
    user_content = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer using only the context above and cite sources."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def get_openrouter_client() -> OpenAI:
    """Create an OpenAI-compatible client pointed at OpenRouter.

    Returns:
        A configured OpenAI client.

    Raises:
        ValueError: If no OpenRouter API key is configured.
    """
    if not rag_utils.OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. Add it to your environment, .env, "
            "or Streamlit secrets."
        )
    return OpenAI(
        base_url=rag_utils.OPENROUTER_BASE_URL,
        api_key=rag_utils.OPENROUTER_API_KEY,
    )


def generate_answer(query: str, chunks: List[Dict]) -> str:
    """Call the LLM to answer the query from the retrieved chunks.

    Args:
        query: The user question.
        chunks: Retrieved chunks used as grounding context.

    Returns:
        The model's answer text.
    """
    client = get_openrouter_client()
    messages = build_prompt(query, chunks)
    response = client.chat.completions.create(
        model=rag_utils.OPENROUTER_MODEL,
        messages=messages,
        temperature=0.0,
    )
    return response.choices[0].message.content


def answer_question(query: str, top_k: int = rag_utils.TOP_K, use_reranker: bool = True) -> Dict:
    """Full RAG answer: retrieve, prompt, generate, and attach citations.

    Args:
        query: The user question.
        top_k: Number of chunks to retrieve.
        use_reranker: Whether to apply cross-encoder reranking.

    Returns:
        A dict with keys ``question``, ``answer``, ``chunks``, ``sources``.
    """
    retriever = retrieval_stage.HybridRetriever()
    chunks = retriever.retrieve(query, top_k=top_k, use_reranker=use_reranker)
    try:
        answer = generate_answer(query, chunks)
    except Exception as exc:
        answer = f"[LLM error: {exc}]"
    return {
        "question": query,
        "answer": answer,
        "chunks": chunks,
        "sources": format_sources(chunks),
    }


if __name__ == "__main__":
    result = answer_question("What is the minimum concrete cover?", use_reranker=False)
    print("Q:", result["question"])
    print("A:", result["answer"])
    print("Sources:", result["sources"])
