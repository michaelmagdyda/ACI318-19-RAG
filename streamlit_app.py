"""Streamlit UI for the RAG assistant.

Ties the pipeline stages together behind a simple web interface: build the
Chroma store from documents, ask a question, and see the grounded answer with
its cited sources.

Run locally::

    streamlit run streamlit_app.py
"""
from __future__ import annotations

import streamlit as st

import rag_utils
from rag_utils import load_stage

# --------------------------------------------------------------------------- #
# Read API credentials from Streamlit secrets when deployed.
# (Locally, values come from the environment / .env via rag_utils.)
# --------------------------------------------------------------------------- #
try:
    if not rag_utils.OPENROUTER_API_KEY:
        rag_utils.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    rag_utils.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", rag_utils.OPENROUTER_MODEL)
except Exception:
    pass

# Load the stages we need in the UI.
chroma_stage = load_stage("05_create_chroma_store.py")
prompting_stage = load_stage("07_prompting.py")
evaluation_stage = load_stage("08_evaluation.py")

st.set_page_config(page_title="RAG Assistant", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner=False)
def ensure_store() -> int:
    """Build the Chroma store once per session if it is missing.

    Returns:
        The number of chunks indexed (0 if the store already existed).
    """
    try:
        chroma_stage.get_collection()
        return 0  # already built
    except Exception:
        return chroma_stage.build_store_from_data()


# --------------------------------- Sidebar --------------------------------- #
st.sidebar.title("📚 RAG Assistant")
st.sidebar.write("Ask questions and get answers grounded in your documents, with citations.")

top_k = st.sidebar.slider("Number of sources (top-k)", 1, 10, rag_utils.TOP_K)
use_reranker = st.sidebar.checkbox("Use cross-encoder reranking", value=False)

st.sidebar.divider()
if st.sidebar.button("Build / rebuild index"):
    with st.sidebar:
        with st.spinner("Building the vector store from documents in data/ ..."):
            count = chroma_stage.build_store_from_data()
        st.success(f"Indexed {count} chunks.")
    st.cache_resource.clear()

st.sidebar.divider()
if st.sidebar.button("Run retrieval evaluation"):
    with st.sidebar:
        with st.spinner("Evaluating retrieval against the gold set..."):
            st.session_state["eval_results"] = evaluation_stage.evaluate_detailed(
                k=top_k, use_reranker=use_reranker
            )

if not rag_utils.OPENROUTER_API_KEY:
    st.sidebar.warning("No OpenRouter API key found. Set it in Streamlit secrets or your .env.")

# --------------------------------- Main ------------------------------------ #
st.title("Retrieval-Augmented Generation Assistant")
st.caption("Answers come only from retrieved document context and cite their sources.")

# Make sure a store exists before answering.
try:
    built = ensure_store()
    if built:
        st.info(f"Built the vector store with {built} chunks on first run.")
except Exception as exc:
    st.error(f"Could not prepare the vector store: {exc}")
    st.stop()

# --------------------------- Evaluation results ---------------------------- #
if "eval_results" in st.session_state:
    results = st.session_state["eval_results"]
    summary = results["summary"]

    st.subheader("Retrieval evaluation")
    c1, c2, c3 = st.columns(3)
    c1.metric("Precision@k", f"{summary['precision@k']:.3f}")
    c2.metric("Recall@k", f"{summary['recall@k']:.3f}")
    c3.metric("MRR", f"{summary['mrr']:.3f}")

    st.dataframe(
        [
            {
                "Question": row["question"],
                "Retrieved pages": ", ".join(map(str, row["retrieved"])),
                "Relevant pages": ", ".join(map(str, row["relevant"])),
                "Matched": ", ".join(map(str, row["matched"])) or "none",
                "Precision": round(row["precision"], 3),
                "Recall": round(row["recall"], 3),
                "RR": round(row["rr"], 3),
            }
            for row in results["rows"]
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Ground truth comes from eval/gold.json. Metrics measure retrieval only "
        "(are the right source pages found), not answer wording."
    )
    st.divider()

query = st.text_input("Your question", placeholder="e.g. What is the minimum concrete cover?")

if st.button("Ask", type="primary") and query.strip():
    with st.spinner("Retrieving context and generating an answer..."):
        result = prompting_stage.answer_question(query, top_k=top_k, use_reranker=use_reranker)

    st.subheader("Answer")
    st.write(result["answer"])
    st.caption(f"Sources: {result['sources']}")

    st.subheader("Retrieved context")
    for i, chunk in enumerate(result["chunks"], start=1):
        page = f" · p.{chunk['page']}" if chunk.get("page") is not None else ""
        with st.expander(f"[{i}] {chunk['source']}{page} (score {chunk['score']:.3f})"):
            st.write(chunk["text"])
