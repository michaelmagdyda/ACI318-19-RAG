"""Stage 01 - Document loading.

Loads raw source documents from the ``data/`` folder. Supports PDF (via
PyMuPDF) and plain-text/Markdown files. Each document is returned with its
source file name and, for PDFs, its page number so later stages can cite it.

Run standalone to see what gets loaded::

    python 01_documents.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF

from rag_utils import DATA_DIR


def load_pdf(path: Path) -> List[Dict]:
    """Load one PDF into a list of page-level documents.

    Args:
        path: Path to the PDF file.

    Returns:
        A list of dicts with keys ``text``, ``source``, and ``page`` (1-based).
    """
    docs: List[Dict] = []
    with fitz.open(str(path)) as pdf:
        for i, page in enumerate(pdf):
            text = page.get_text("text")
            if text.strip():
                docs.append({"text": text, "source": path.name, "page": i + 1})
    return docs


def load_text(path: Path) -> List[Dict]:
    """Load one plain-text or Markdown file as a single document.

    Args:
        path: Path to the ``.txt`` or ``.md`` file.

    Returns:
        A single-item list with keys ``text``, ``source``, and ``page`` (None).
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    # Text files have no real pages; use page 1 so citations/evaluation still work.
    return [{"text": text, "source": path.name, "page": 1}] if text.strip() else []


def load_documents(data_dir: Path = DATA_DIR) -> List[Dict]:
    """Load every supported document in a directory.

    Args:
        data_dir: Folder to scan for ``.pdf``, ``.txt``, and ``.md`` files.

    Returns:
        A combined list of document dicts across all files.

    Raises:
        FileNotFoundError: If the data directory does not exist.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    documents: List[Dict] = []
    for path in sorted(data_dir.iterdir()):
        suffix = path.suffix.lower()
        try:
            if suffix == ".pdf":
                documents.extend(load_pdf(path))
            elif suffix in (".txt", ".md"):
                documents.extend(load_text(path))
        except Exception as exc:  # keep going if one file is broken
            print(f"[warn] could not load {path.name}: {exc}")

    return documents


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} document units from {DATA_DIR}")
    if docs:
        print("First unit preview:")
        print("  source:", docs[0]["source"], "| page:", docs[0]["page"])
        print("  text  :", docs[0]["text"][:200].replace("\n", " "), "...")
