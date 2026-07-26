# RAG Assistant (Diploma Project)

A simple, modular Retrieval-Augmented Generation (RAG) assistant. It answers questions using
**only** the content of your documents and **cites its sources**. Built from plain Python
files (no notebooks), with ChromaDB, SentenceTransformer embeddings, an OpenRouter LLM, and a
Streamlit UI — plus hybrid search, cross-encoder reranking, and retrieval evaluation.

## Pipeline
```
documents → preprocessing → chunking → vector representation → vector store
          → context retrieval → prompting → Streamlit UI
```

## Features
- **ChromaDB** persistent vector store
- **SentenceTransformer** embeddings (`all-MiniLM-L6-v2`)
- **Hybrid search**: dense (semantic) + BM25 (keyword), fused with Reciprocal Rank Fusion
- **Cross-encoder reranking** (optional, toggle in the UI)
- **OpenRouter** LLM (OpenAI-compatible), model configurable
- **Source citations** in every answer
- **Evaluation**: Precision@k, Recall@k, MRR
- **Streamlit** web interface

## Project structure
See [`docs/04_Project_Structure.md`](docs/04_Project_Structure.md). The eight instructor-
required files (`01_documents.py` … `08_evaluation.py` + `streamlit_app.py`) are the pipeline
stages; `rag_utils.py` holds shared config and helpers.

## Setup
```bash
git clone <your-repo-url>
cd rag-project
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then add your OpenRouter API key
```

Put your document(s) in `data/` (a placeholder `sample_aci318.txt` is included so the app runs
immediately — replace it with your real PDF).

## Run locally
```bash
python 05_create_chroma_store.py    # build the vector store from data/
streamlit run streamlit_app.py      # launch the UI
```
You can also run any stage on its own to see what it does, e.g. `python 03_chunking.py`.

## Evaluate
Edit `eval/gold.json` to map your questions to the pages that contain the answers, then:
```bash
python 08_evaluation.py
```

## API key rules (per assignment)
- **Never** put your real API key in a Python file.
- **Never** commit your real `.env` (it is gitignored).
- Use **Streamlit TOML secrets** for deployment.

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub (the `.gitignore` already excludes secrets, `.env`, and the
   generated `chroma_store/`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick your repo,
   branch, and `streamlit_app.py`.
3. Open **Manage app → Secrets** and paste (valid TOML):
   ```toml
   OPENROUTER_API_KEY = "your_openrouter_key_here"
   OPENROUTER_MODEL = "openai/gpt-4o-mini"
   ```
4. Deploy. On first load the app builds the vector store from `data/`, then you can ask
   questions. (Rebuild anytime with the sidebar button.)

The app reads the key from Streamlit secrets automatically:
```python
try:
    if not rag.OPENROUTER_API_KEY:
        rag.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    rag.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag.OPENROUTER_MODEL)
except Exception:
    pass
```

## Documentation
Full write-ups are in [`docs/`](docs/): project overview, the three development phases,
project structure, and architecture.

## Final submission checklist
- [x] All required Python files exist (`01`–`08`, `streamlit_app.py`).
- [x] `requirements.txt` exists.
- [x] Real API key is **not** in the ZIP or GitHub repo.
- [x] Streamlit secrets configured in valid TOML.
- [x] The Streamlit app runs successfully.
- [x] Answers use retrieved context.
- [x] Answers cite sources.

## Limitations
- The included document and `gold.json` are placeholders — replace with your real data.
- Retrieval evaluation only (no automated answer-quality scoring yet).
- No conversational memory; each question is independent.

## License / data note
Do not commit copyrighted source documents (e.g. the official ACI 318 PDF) to a public repo;
keep them local.
