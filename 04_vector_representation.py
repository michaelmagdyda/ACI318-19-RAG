"""Stage 04 - Vector representation (embeddings).

Turns chunk text into dense vectors using a SentenceTransformer model. Similar
meanings produce nearby vectors, which is what makes semantic search possible.

The model is loaded once and cached at module level so repeated calls (e.g. in
the Streamlit app) do not reload it.

Run standalone::

    python 04_vector_representation.py
"""
from __future__ import annotations

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_utils import EMBEDDING_MODEL


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load (and cache) a SentenceTransformer embedding model.

    Args:
        model_name: The model identifier, e.g. ``"all-MiniLM-L6-v2"``.

    Returns:
        A loaded SentenceTransformer instance.
    """
    return SentenceTransformer(model_name)


def embed_texts(texts: List[str], model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    """Embed a list of strings into L2-normalized vectors.

    Normalizing means cosine similarity equals a simple dot product, which
    keeps the vector-store math clean.

    Args:
        texts: The strings to embed.
        model_name: Which embedding model to use.

    Returns:
        A float32 array of shape ``(len(texts), embedding_dim)``.
    """
    model = get_embedding_model(model_name)
    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype("float32")


if __name__ == "__main__":
    sample = ["minimum concrete cover", "compressive strength of concrete"]
    vecs = embed_texts(sample)
    print(f"Embedded {len(sample)} texts.")
    print("Vector shape:", vecs.shape)
    print("Cosine similarity between the two:", float(np.dot(vecs[0], vecs[1])))
