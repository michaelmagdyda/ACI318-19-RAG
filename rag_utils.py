"""Shared utilities for the RAG project.

This module holds configuration and small helper functions that several of the
numbered pipeline files need. Keeping them here avoids copy-pasting the same
code into ``01_documents.py``, ``06_retrieve_context.py``, and so on.

It also provides :func:`load_stage`, which lets us import the numbered files
(``01_documents.py`` ...) even though names starting with a digit cannot be
imported with a normal ``import`` statement.
"""
from __future__ import annotations

import importlib.util
import os
import types
from pathlib import Path
from typing import List

# Load variables from a local ``.env`` file (if present) into the process
# environment, so ``os.environ.get(...)`` below can see them during local
# development. Safe to call when python-dotenv or the file is absent.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_store"

# --------------------------------------------------------------------------- #
# Model / API configuration (overridable via environment variables)
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL: str = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
CHROMA_COLLECTION: str = os.environ.get("CHROMA_COLLECTION", "rag_documents")

# OpenRouter is an OpenAI-compatible gateway to many models.
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL: str = os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
OPENROUTER_BASE_URL: str = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Retrieval defaults
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", 500))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", 100))
TOP_K: int = int(os.environ.get("TOP_K", 5))
CANDIDATE_POOL: int = int(os.environ.get("CANDIDATE_POOL", 20))


def load_stage(filename: str) -> types.ModuleType:
    """Import a numbered stage file (e.g. ``01_documents.py``) as a module.

    Files whose names begin with a digit cannot be imported with a normal
    ``import`` statement, so we load them dynamically from their path.

    Args:
        filename: The stage file name, such as ``"03_chunking.py"``.

    Returns:
        The imported module object.

    Raises:
        FileNotFoundError: If the stage file does not exist.
    """
    path = BASE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Stage file not found: {path}")

    module_name = path.stem  # e.g. "03_chunking"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def format_sources(chunks: List[dict]) -> str:
    """Build a short, human-readable citation string from retrieved chunks.

    Args:
        chunks: Retrieved chunks, each a dict with ``source`` and ``page`` keys.

    Returns:
        A comma-separated citation string, e.g. ``"ACI_318.pdf p.5, p.9"``.
    """
    seen = []
    for c in chunks:
        source = c.get("source", "unknown")
        page = c.get("page")
        label = f"{source} p.{page}" if page is not None else source
        if label not in seen:
            seen.append(label)
    return ", ".join(seen)
