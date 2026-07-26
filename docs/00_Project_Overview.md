# 00 — Project Overview

## Goal
Build a simple, well-structured Retrieval-Augmented Generation (RAG) assistant that answers
questions using only the content of provided documents and cites its sources. The project
follows the lab sequence *documents → preprocessing → chunking → vector representation →
vector store → retrieval → prompting → Streamlit UI*, and adds a few practical AI-engineering
improvements (hybrid search, reranking, evaluation).

## What the project does
1. Loads documents (PDF or text) from `data/`.
2. Cleans and chunks the text.
3. Embeds chunks with a SentenceTransformer model.
4. Stores them in a persistent ChromaDB collection.
5. Retrieves relevant chunks for a question using **hybrid search** (dense + BM25) with
   optional **cross-encoder reranking**.
6. Builds a grounded prompt and calls an LLM through **OpenRouter**.
7. Serves everything through a **Streamlit** web app with source citations.
8. Evaluates retrieval quality with **Precision@k, Recall@k, and MRR**.

## Files created
The instructor-required pipeline files plus a few small supporting files:

| File | Role |
|------|------|
| `01_documents.py` | Load raw documents |
| `02_preprocessing.py` | Clean text |
| `03_chunking.py` | Split into overlapping chunks |
| `04_vector_representation.py` | Embed chunks |
| `05_create_chroma_store.py` | Build the ChromaDB store |
| `06_retrieve_context.py` | Hybrid retrieval + reranking |
| `07_prompting.py` | Prompt + OpenRouter LLM |
| `08_evaluation.py` | Precision@k, Recall@k, MRR |
| `streamlit_app.py` | Web UI |
| `rag_utils.py` | Shared config + helpers |

## Workflow
```
data/  →  01 → 02 → 03 → 04 → 05  (build the store, once)
                                   ↓
question →  06 (retrieve) → 07 (prompt + LLM) → answer + citations
                                   ↑
                          08 evaluation (offline quality check)
```

## How to run
```bash
pip install -r requirements.txt
cp .env.example .env            # add your OpenRouter key
python 05_create_chroma_store.py    # build the store
streamlit run streamlit_app.py      # launch the app
```

## Development phases
- **Phase 1 — Basic RAG:** stages 01–07 + Streamlit (documents to a working assistant).
- **Phase 2 — Advanced Retrieval:** hybrid search + cross-encoder reranking in stage 06,
  modular prompting in stage 07.
- **Phase 3 — Evaluation & Finalization:** stage 08 metrics, documentation, cleanup, README.

## Future improvements
- Structure-aware chunking (split on section/clause numbers).
- Conversational memory for follow-up questions.
- A larger, human-labeled evaluation set.
- Answer-quality metrics (F1, semantic similarity) in addition to retrieval metrics.
