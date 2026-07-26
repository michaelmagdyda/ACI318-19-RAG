"""Stage 03 - Chunking.

Splits cleaned documents into smaller overlapping chunks. Small chunks make
retrieval precise; the overlap keeps a requirement that sits on a boundary
intact in at least one chunk.

Each chunk carries metadata (source, page, chunk id) so answers can cite the
exact origin.

Run standalone::

    python 03_chunking.py
"""
from __future__ import annotations

from typing import Dict, List

from rag_utils import CHUNK_OVERLAP, CHUNK_SIZE, load_stage

preprocessing_stage = load_stage("02_preprocessing.py")


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split a single string into overlapping character windows.

    Args:
        text: The text to split.
        chunk_size: Maximum characters per chunk.
        overlap: Number of characters shared between consecutive chunks.

    Returns:
        A list of chunk strings.

    Raises:
        ValueError: If ``overlap`` is not smaller than ``chunk_size``.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    step = chunk_size - overlap
    chunks: List[str] = []
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


def chunk_documents(
    documents: List[Dict],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Dict]:
    """Chunk every document and attach metadata.

    Args:
        documents: Cleaned document dicts from stage 02.
        chunk_size: Maximum characters per chunk.
        overlap: Overlap in characters between consecutive chunks.

    Returns:
        A list of chunk dicts with keys ``id``, ``text``, ``source``, ``page``.
    """
    chunks: List[Dict] = []
    chunk_id = 0
    for doc in documents:
        for piece in split_into_chunks(doc["text"], chunk_size, overlap):
            chunks.append(
                {
                    "id": f"chunk_{chunk_id}",
                    "text": piece,
                    "source": doc["source"],
                    "page": doc["page"],
                }
            )
            chunk_id += 1
    return chunks


if __name__ == "__main__":
    raw = preprocessing_stage.documents_stage.load_documents()
    clean = preprocessing_stage.preprocess_documents(raw)
    chunks = chunk_documents(clean)
    print(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")
    if chunks:
        print("Example chunk:", chunks[0]["id"], "| source:", chunks[0]["source"])
        print(" ", chunks[0]["text"][:200], "...")
