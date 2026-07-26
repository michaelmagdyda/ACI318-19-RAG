# 01 — Phase 1: Basic RAG

## Goal
Get a working end-to-end RAG pipeline: from raw documents to an LLM answer served in
Streamlit, with source citations. This phase covers the core lab sequence.

## Files created
- `01_documents.py` — document loading
- `02_preprocessing.py` — text cleaning
- `03_chunking.py` — chunking with overlap and metadata
- `04_vector_representation.py` — SentenceTransformer embeddings
- `05_create_chroma_store.py` — ChromaDB persistent store
- `06_retrieve_context.py` — retrieval (dense part; hybrid added in Phase 2)
- `07_prompting.py` — prompt construction + OpenRouter LLM call
- `streamlit_app.py` — user interface
- `rag_utils.py` — shared configuration and helpers

## Responsibilities
Each numbered file is one pipeline stage with a small set of functions and a
`__main__` demo block so it can be run and explained on its own.

## Workflow
1. **Load** (`01`): read PDFs (PyMuPDF) and text files into dicts of
   `{text, source, page}`.
2. **Clean** (`02`): normalize whitespace, strip control characters, drop empties.
3. **Chunk** (`03`): split into ~500-character chunks with 100-character overlap;
   attach `id`, `source`, `page`.
4. **Embed** (`04`): encode chunk text into normalized vectors with
   `all-MiniLM-L6-v2`.
5. **Store** (`05`): create a cosine-space ChromaDB collection and add vectors,
   text, and metadata.
6. **Retrieve** (`06`): embed the query and fetch the nearest chunks.
7. **Prompt + generate** (`07`): format context, instruct the model to answer only
   from it and cite sources, call OpenRouter.
8. **Serve** (`streamlit_app.py`): input box, answer, and expandable source list.

## Key algorithms
- **Overlapping character chunking** — keeps boundary-spanning facts intact.
- **Cosine similarity search** — normalized embeddings + Chroma cosine space.
- **Grounded prompting** — system rules force context-only answers with citations.

## How to run
```bash
python 01_documents.py          # inspect what loads
python 05_create_chroma_store.py    # build the store
streamlit run streamlit_app.py      # ask questions
```

## Future improvements
- Add hybrid keyword search and reranking (see Phase 2).
- Cache embeddings to avoid recomputing on rebuilds.
