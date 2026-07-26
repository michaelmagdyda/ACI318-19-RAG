# 04 — Project Structure

## Directory tree
```
rag-project/
├── 01_documents.py             # Stage 1: load PDF/text documents
├── 02_preprocessing.py         # Stage 2: clean text
├── 03_chunking.py              # Stage 3: chunk with overlap + metadata
├── 04_vector_representation.py # Stage 4: SentenceTransformer embeddings
├── 05_create_chroma_store.py   # Stage 5: build ChromaDB store
├── 06_retrieve_context.py      # Stage 6: hybrid search + reranking
├── 07_prompting.py             # Stage 7: prompt + OpenRouter LLM
├── 08_evaluation.py            # Stage 8: Precision@k, Recall@k, MRR
├── streamlit_app.py            # Web UI (wires stages together)
├── rag_utils.py                # Shared config + helper functions
├── requirements.txt
├── .env.example                # template (real .env is gitignored)
├── .gitignore
├── README.md
├── .streamlit/
│   └── secrets.toml.example    # Streamlit secrets template (TOML)
├── data/
│   └── sample_aci318.txt       # placeholder doc; replace with your PDF
├── eval/
│   └── gold.json               # evaluation ground truth
└── docs/
    ├── 00_Project_Overview.md
    ├── 01_Phase_1_Basic_RAG.md
    ├── 02_Phase_2_Advanced_Retrieval.md
    ├── 03_Phase_3_Evaluation.md
    ├── 04_Project_Structure.md
    └── Architecture.md
```

## Why these files
The eight numbered files plus `streamlit_app.py` are the instructor-required structure; each
is one clear pipeline stage. Two small additions keep the code clean without adding
complexity:

- **`rag_utils.py`** — one place for configuration (models, keys, chunk sizes) and shared
  helpers (`format_sources`, and `load_stage`, which imports the numbered files). Without it,
  the same config and helper code would be copy-pasted across several files.
- **`08_evaluation.py`** — holds the requested evaluation metrics, separate from the core
  pipeline so it can be run independently.

## The `load_stage` helper
Python cannot `import 01_documents` because the name starts with a digit. `rag_utils.load_stage`
loads these files by path using `importlib`, so later stages and the Streamlit app can reuse
earlier stages' functions instead of duplicating them.

## Data flow between files
```
01 ──▶ 02 ──▶ 03 ──▶ 04 ──▶ 05 ──▶ (Chroma store on disk)
                                      │
                                      ▼
                               06 ──▶ 07 ──▶ answer
                                      ▲
                               08 (reads 06 for evaluation)
streamlit_app.py calls 05 (build) and 07 (answer).
```

## What is not committed
`.env`, `.streamlit/secrets.toml`, and the generated `chroma_store/` are gitignored. API keys
live only in local `.env` or Streamlit Cloud secrets — never in the code or the repo.
