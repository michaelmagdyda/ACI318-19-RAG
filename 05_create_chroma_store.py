"""Stage 05 - Create the Chroma vector store.

Builds a persistent ChromaDB collection from the chunks: embeds each chunk
(stage 04) and stores the vector, the original text, and the metadata so it
can be queried later.

Running this file rebuilds the store from scratch::

    python 05_create_chroma_store.py
"""
from __future__ import annotations

from typing import Dict, List

import chromadb

from rag_utils import CHROMA_COLLECTION, CHROMA_DIR, load_stage

chunking_stage = load_stage("03_chunking.py")
vectors_stage = load_stage("04_vector_representation.py")


def get_chroma_client() -> "chromadb.ClientAPI":
    """Create a persistent ChromaDB client rooted at ``CHROMA_DIR``.

    Returns:
        A ChromaDB persistent client.
    """
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def build_store(chunks: List[Dict], collection_name: str = CHROMA_COLLECTION):
    """Create (or replace) a Chroma collection and add all chunks.

    Args:
        chunks: Chunk dicts from stage 03.
        collection_name: Name of the Chroma collection to (re)create.

    Returns:
        The populated Chroma collection.

    Raises:
        ValueError: If ``chunks`` is empty.
    """
    if not chunks:
        raise ValueError("No chunks to index. Run stages 01-03 on real data first.")

    client = get_chroma_client()

    # Start clean so re-runs do not create duplicates.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass  # collection may not exist yet

    # Cosine space matches our normalized embeddings.
    collection = client.create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    texts = [c["text"] for c in chunks]
    embeddings = vectors_stage.embed_texts(texts)
    metadatas = [{"source": c["source"], "page": c["page"]} for c in chunks]
    ids = [c["id"] for c in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )
    return collection


def get_collection(collection_name: str = CHROMA_COLLECTION):
    """Open an existing Chroma collection for querying.

    Args:
        collection_name: Name of the collection to open.

    Returns:
        The Chroma collection.
    """
    client = get_chroma_client()
    return client.get_collection(collection_name)


def build_store_from_data() -> int:
    """Full offline build: load -> preprocess -> chunk -> embed -> store.

    Returns:
        The number of chunks indexed.
    """
    preprocessing_stage = chunking_stage.preprocessing_stage
    raw = preprocessing_stage.documents_stage.load_documents()
    clean = preprocessing_stage.preprocess_documents(raw)
    chunks = chunking_stage.chunk_documents(clean)
    build_store(chunks)
    return len(chunks)


if __name__ == "__main__":
    count = build_store_from_data()
    print(f"Chroma store built with {count} chunks at {CHROMA_DIR}")
