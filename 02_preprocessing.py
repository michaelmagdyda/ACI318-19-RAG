"""Stage 02 - Preprocessing.

Cleans the raw text produced by stage 01: normalizes whitespace, removes
control characters, and drops empty documents. Clean input makes better
chunks and better embeddings.

Run standalone::

    python 02_preprocessing.py
"""
from __future__ import annotations

import re
from typing import Dict, List

from rag_utils import load_stage

# Load stage 01 dynamically (its file name starts with a digit).
documents_stage = load_stage("01_documents.py")


def clean_text(text: str) -> str:
    """Normalize whitespace and strip control characters from a string.

    Args:
        text: Raw text to clean.

    Returns:
        The cleaned text with single spaces and no leading/trailing whitespace.
    """
    # Remove non-printable control characters (keep normal whitespace).
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    # Collapse any run of whitespace (spaces, tabs, newlines) into one space.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_documents(documents: List[Dict]) -> List[Dict]:
    """Clean every document and drop those that become empty.

    Args:
        documents: Document dicts from stage 01.

    Returns:
        A new list of cleaned document dicts (same keys, cleaned ``text``).
    """
    cleaned: List[Dict] = []
    for doc in documents:
        text = clean_text(doc["text"])
        if text:
            cleaned.append({**doc, "text": text})
    return cleaned


if __name__ == "__main__":
    raw = documents_stage.load_documents()
    clean = preprocess_documents(raw)
    print(f"Documents before: {len(raw)} | after cleaning: {len(clean)}")
    if clean:
        print("Example cleaned text:")
        print(" ", clean[0]["text"][:200], "...")
