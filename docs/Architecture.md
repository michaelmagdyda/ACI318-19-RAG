# Architecture

## Goal
Describe how the pieces fit together at runtime, so the design is easy to explain in a
presentation.

## High-level architecture
```
                        ┌──────────────────────────────────────────┐
                        │              Streamlit UI                 │
                        │           (streamlit_app.py)              │
                        └───────────────┬───────────────┬──────────┘
                            build store │               │ ask question
                                        ▼               ▼
        ┌───────────────── OFFLINE (indexing) ──────────────────────┐
        │  01 documents → 02 preprocess → 03 chunk → 04 embed → 05  │
        │                                              ChromaDB store │
        └───────────────────────────────────────────────────────────┘
                                        │
                                        ▼
        ┌───────────────── ONLINE (querying) ───────────────────────┐
        │  question → 06 hybrid retrieve (dense+BM25+RRF, rerank)    │
        │           → 07 build prompt → OpenRouter LLM → answer      │
        │           → citations from chunk metadata                 │
        └───────────────────────────────────────────────────────────┘

        08 evaluation reuses 06 to score retrieval offline.
        rag_utils.py provides shared config + the load_stage() importer.
```

## Components and responsibilities
| Component | Responsibility | Key libraries |
|-----------|----------------|---------------|
| `01_documents.py` | Read PDFs/text into `{text, source, page}` | PyMuPDF |
| `02_preprocessing.py` | Normalize and clean text | re (stdlib) |
| `03_chunking.py` | Overlapping chunks + metadata | — |
| `04_vector_representation.py` | Embed text into vectors | sentence-transformers |
| `05_create_chroma_store.py` | Persist vectors + metadata | chromadb |
| `06_retrieve_context.py` | Hybrid retrieval + reranking | chromadb, rank-bm25, sentence-transformers |
| `07_prompting.py` | Grounded prompt + LLM call | openai (via OpenRouter) |
| `08_evaluation.py` | Precision@k, Recall@k, MRR | — |
| `streamlit_app.py` | UI + orchestration | streamlit |
| `rag_utils.py` | Config, citations, dynamic stage import | importlib |

## Two phases at runtime
- **Offline / indexing** runs once (or when documents change): stages 01–05 build the
  ChromaDB store on disk. This is the expensive part (embedding the whole corpus).
- **Online / querying** runs per question: stages 06–07 retrieve and generate. This is fast
  because the vectors already exist.

Separating the two is why the app stays responsive: no re-embedding on every question.

## Data contract between stages
Every chunk is a dict with a stable shape:
```python
{"id": "chunk_0", "text": "...", "source": "file.pdf", "page": 5}
```
Retrieval adds a `"score"` field. Keeping this contract consistent is what lets stages be
developed and tested independently.

## Configuration and secrets
All tunables (model names, chunk size, top-k, API key) live in `rag_utils.py`, read from
environment variables / `.env` locally and from Streamlit secrets when deployed. No secret is
ever hard-coded.

## Key design decisions
- **Hybrid over dense-only**: exact identifiers (clause numbers) matter in technical docs.
- **RRF over weighted fusion**: no score-scale tuning needed; robust default.
- **Reranking optional**: accuracy/latency trade-off left to the user via a UI toggle.
- **`load_stage` importer**: satisfies the required numeric file names while keeping code DRY.

## Future improvements
- Structure-aware chunking; conversational memory; answer-quality metrics; caching of
  embeddings; a larger evaluation set.
