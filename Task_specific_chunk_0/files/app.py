from __future__ import annotations

import json
import time

import requests
import streamlit as st

import config
from embedding_models import JinaSmallEmbedding, TfidfEmbedding
from ingest import SimpleVectorIndex, chunk_documents, load_pdfs

st.set_page_config(page_title="Medical RAG Assistant", page_icon="🩺", layout="wide")

SYSTEM_PROMPT_TEMPLATE = """You are a strict, precise, and professional medical AI assistant. Your ONLY job is to answer the user's question based strictly on the provided medical Context below.

RULES:
1. DO NOT use any external knowledge, personal knowledge, or information not explicitly stated in the Context.
2. If the Context does not contain the answer, you MUST reply with exactly: "I don't have enough information to answer this based on the provided documents."
3. Do not guess, infer, or make up any medical advice, diagnoses, or information.
4. If the question is conversational or unrelated to the provided medical documents, simply reply that you are a medical assistant restricted to answering questions about the provided documents.

Context:
{context_text}

Question: {question}

Answer:"""


@st.cache_resource(show_spinner="Building the vector index... this runs once.")
def get_index() -> SimpleVectorIndex:
    embedding_model = JinaSmallEmbedding(
        model_name=config.JINA_MODEL_NAME,
        fallback_model=TfidfEmbedding(max_features=config.TFIDF_MAX_FEATURES),
        api_key=config.JINA_API_KEY,
    )
    pages = load_pdfs(config.DATA_DIR)
    chunks = chunk_documents(pages)
    return SimpleVectorIndex(embedding_model, chunks)


def call_groq(prompt: str) -> str:
    if not config.GROQ_API_KEY:
        return "⚠️ GROQ_API_KEY is missing. Add it to your .env file."
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        return f"API Error: {response.status_code} - {response.text}"
    except Exception as exc:  # noqa: BLE001
        return f"Connection Error: {exc}"


def answer_question(index: SimpleVectorIndex, question: str) -> dict:
    results = index.hybrid_search_with_relevance_scores(
        question,
        k=config.TOP_K,
        semantic_weight=config.HYBRID_SEMANTIC_WEIGHT,
        keyword_weight=config.HYBRID_KEYWORD_WEIGHT,
    )

    context_text = ""
    chunks_used = []
    citations = []
    for rank, (doc, score) in enumerate(results, 1):
        citation = doc.metadata.get("citation", "Unknown")
        context_text += f"--- Document {rank} (Source: {citation}) ---\n{doc.page_content}\n\n"
        chunks_used.append({"citation": citation, "text": doc.page_content, "score": round(score, 3)})
        citations.append(citation)

    prompt = SYSTEM_PROMPT_TEMPLATE.format(context_text=context_text, question=question)
    answer = call_groq(prompt)

    return {
        "answer": answer,
        "citations": sorted(set(citations)),
        "chunks": chunks_used,
    }


def main() -> None:
    st.title("🩺 Medical RAG Assistant")
    st.caption("RAG over ADHD.pdf and parenting2015.pdf — answers are grounded strictly in the retrieved chunks.")

    with st.sidebar:
        st.header("Settings")
        st.write(f"**Chunk size:** {config.CHUNK_SIZE} tokens")
        st.write(f"**Chunk overlap:** {config.CHUNK_OVERLAP} tokens")
        st.write(f"**Top K:** {config.TOP_K}")
        st.write(f"**Hybrid weights:** semantic {config.HYBRID_SEMANTIC_WEIGHT} / keyword {config.HYBRID_KEYWORD_WEIGHT}")
        st.divider()
        jina_status = "✅ Jina API" if config.JINA_API_KEY else "⚠️ TF-IDF fallback (no JINA_API_KEY)"
        groq_status = "✅ configured" if config.GROQ_API_KEY else "⚠️ missing"
        st.write(f"**Embedding model:** {jina_status}")
        st.write(f"**Groq LLM key:** {groq_status}")
        if st.button("🔄 Rebuild index"):
            get_index.clear()
            st.rerun()

    try:
        index = get_index()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("citations"):
                with st.expander("📚 Citations & retrieved chunks"):
                    st.write(", ".join(message["citations"]))
                    for i, chunk in enumerate(message["chunks"], 1):
                        st.markdown(f"**[{i}] {chunk['citation']}** (score: {chunk['score']})")
                        st.text(chunk["text"][:500] + ("..." if len(chunk["text"]) > 500 else ""))

    question = st.chat_input("Ask a question about ADHD or parenting...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching & thinking..."):
                result = answer_question(index, question)
            st.markdown(result["answer"])
            is_declined = (
                "restricted to answering" in result["answer"]
                or "don't have enough information" in result["answer"]
            )
            if not is_declined and result["citations"]:
                with st.expander("📚 Citations & retrieved chunks"):
                    st.write(", ".join(result["citations"]))
                    for i, chunk in enumerate(result["chunks"], 1):
                        st.markdown(f"**[{i}] {chunk['citation']}** (score: {chunk['score']})")
                        st.text(chunk["text"][:500] + ("..." if len(chunk["text"]) > 500 else ""))

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "citations": result["citations"] if not is_declined else [],
                "chunks": result["chunks"] if not is_declined else [],
            }
        )


if __name__ == "__main__":
    main()
