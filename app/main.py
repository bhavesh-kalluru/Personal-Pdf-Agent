from __future__ import annotations

import os
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv

from pdf_ingest import pdf_to_page_texts
from perplexity_client import PerplexityClient
from rag import OpenAIRAG, retrieve_top_chunks
from utils import chunk_text_by_tokens, ensure_dir, load_env_path, stable_doc_id
from vectorstore import ChromaStore


def init_env() -> None:
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    load_dotenv(env_path)


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "ingested_doc_ids" not in st.session_state:
        st.session_state.ingested_doc_ids = set()
    if "store_ready" not in st.session_state:
        st.session_state.store_ready = False


def user_explicit_web_request(text: str) -> bool:
    t = (text or "").strip().lower()
    return t.startswith("web:") or t.startswith("latest:") or t.startswith("news:")


def ingest_pdfs_into_store(store: ChromaStore, rag: OpenAIRAG, files: List) -> Tuple[int, List[str]]:
    added_chunks = 0
    ingested_docs = []

    for f in files:
        pdf_bytes = f.getvalue()
        doc_name = f.name or "document.pdf"
        doc_id = stable_doc_id(doc_name, pdf_bytes)

        if doc_id in st.session_state.ingested_doc_ids:
            ingested_docs.append(f"{doc_name} (already ingested)")
            continue

        safe_name, pages = pdf_to_page_texts(doc_name, pdf_bytes)

        all_ids = []
        all_texts = []
        all_metas = []

        for p in pages:
            if not p.text.strip():
                continue

            chunks = chunk_text_by_tokens(
                p.text,
                chunk_tokens=1000,
                overlap_tokens=150,
                model_name=os.getenv("OPENAI_ANSWER_MODEL", "gpt-4o-mini"),
            )

            for idx, ch in enumerate(chunks):
                chunk_uid = f"{doc_id}:{p.page_num}:{idx}"
                all_ids.append(chunk_uid)
                all_texts.append(ch)
                all_metas.append(
                    {"doc_name": safe_name, "page": int(p.page_num), "chunk_index": int(idx), "doc_id": doc_id}
                )

        if not all_texts:
            ingested_docs.append(f"{doc_name} (no extractable text)")
            st.session_state.ingested_doc_ids.add(doc_id)
            continue

        embeddings = []
        batch_size = 64
        for i in range(0, len(all_texts), batch_size):
            embeddings.extend(rag.embed_texts(all_texts[i : i + batch_size]))

        store.add_chunks(ids=all_ids, texts=all_texts, metadatas=all_metas, embeddings=embeddings)
        added_chunks += len(all_texts)
        st.session_state.ingested_doc_ids.add(doc_id)
        ingested_docs.append(f"{doc_name} (ingested {len(all_texts)} chunks)")

    return added_chunks, ingested_docs


def main() -> None:
    st.set_page_config(page_title="Personal FAQ (PDF RAG)", page_icon="📄", layout="wide")

    init_env()
    init_state()

    st.title("📄 Personal FAQ — Chat with your PDFs (RAG)")

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is missing. Add it to app/.env and restart.")
        st.stop()

    store = ChromaStore(collection_name="personal_faq")
    rag = OpenAIRAG()
    pplx = PerplexityClient()

    with st.sidebar:
        st.header("Setup")

        st.write("**1) Upload PDFs (1–5)**")
        uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

        colA, colB = st.columns(2)
        with colA:
            if st.button("Ingest PDFs", type="primary", use_container_width=True):
                if not uploads:
                    st.warning("Upload at least 1 PDF.")
                elif len(uploads) > 5:
                    st.warning("Max 5 PDFs for MVP. Upload fewer.")
                else:
                    with st.spinner("Ingesting PDFs (extract → chunk → embed → store)..."):
                        added, notes = ingest_pdfs_into_store(store, rag, uploads)
                        st.session_state.store_ready = True
                    st.success(f"Done. Added {added} chunks.")
                    for n in notes:
                        st.caption(n)

        with colB:
            if st.button("Reset index", use_container_width=True):
                store.reset()
                st.session_state.messages = []
                st.session_state.ingested_doc_ids = set()
                st.session_state.store_ready = False
                st.success("Index cleared.")

        st.divider()
        st.subheader("Answer settings")
        top_k = st.slider("Top-K chunks to retrieve", min_value=6, max_value=12, value=10, step=1)
        show_retrieval = st.checkbox("Show retrieved chunks (debug)", value=False)

        st.divider()
        st.subheader("Optional web fallback (Perplexity)")
        allow_web = st.checkbox(
            "Enable web lookup when explicitly requested",
            value=False,
            help="Only used if you type web: ... / latest: ... / news: ...",
        )

        if allow_web and not pplx.is_configured():
            st.warning("PERPLEXITY_API_KEY is not set. Web lookup will fail.")

        st.caption(f"Indexed chunks: {store.count()}")

    if not st.session_state.store_ready and store.count() == 0:
        st.info("Upload 1–5 PDFs in the sidebar and click **Ingest PDFs** to start.")
        st.stop()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Ask a question about your uploaded PDFs… (use 'web:' for web lookup)")
    if not user_q:
        return

    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    explicit_web = user_explicit_web_request(user_q)
    use_web = bool(allow_web and explicit_web)

    with st.chat_message("assistant"):
        if use_web:
            q = user_q.split(":", 1)[1].strip() if ":" in user_q else user_q
            with st.spinner("Searching the web (Perplexity)…"):
                try:
                    web_ans = pplx.web_answer(q)
                    st.markdown(web_ans)
                    st.session_state.messages.append({"role": "assistant", "content": web_ans})
                except Exception as e:
                    err = f"Web lookup failed: {e}"
                    st.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})
            return

        with st.spinner("Searching your PDFs…"):
            retrieved = retrieve_top_chunks(store, rag, user_q, top_k=top_k)

        if show_retrieval:
            st.write("**Retrieved context (debug):**")
            for ch in retrieved:
                st.markdown(
                    f"- **{ch.doc_name} p.{ch.page} (chunk {ch.chunk_index})** score={ch.score:.4f}\n\n"
                    f"{ch.text[:600]}{'…' if len(ch.text) > 600 else ''}"
                )

        with st.spinner("Writing answer…"):
            answer = rag.answer(question=user_q, retrieved=retrieved, conversation_messages=st.session_state.messages)

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    ensure_dir(load_env_path())
    main()
