from __future__ import annotations

import os
from typing import Any, Dict, List

from openai import OpenAI

from prompts import SYSTEM_PROMPT, USER_TEMPLATE
from utils import RetrievedChunk, dedupe_chunks


class OpenAIRAG:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
        self.embed_model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        self.answer_model = os.getenv("OPENAI_ANSWER_MODEL", "gpt-4o-mini")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.embed_model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, query: str) -> List[float]:
        return self.embed_texts([query])[0]

    def build_context(self, retrieved: List[RetrievedChunk]) -> str:
        parts: List[str] = []
        for ch in retrieved:
            header = f"[doc_name={ch.doc_name} page={ch.page} chunk={ch.chunk_index}]"
            parts.append(f"{header}\n{ch.text}")
        return "\n\n---\n\n".join(parts).strip()

    def format_conversation(self, messages: List[Dict[str, str]], max_turns: int = 10) -> str:
        tail = messages[-max_turns:] if len(messages) > max_turns else messages
        lines = []
        for m in tail:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            if content:
                lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines).strip() or "(none)"

    def answer(self, question: str, retrieved: List[RetrievedChunk], conversation_messages: List[Dict[str, str]]) -> str:
        context = self.build_context(retrieved)
        conversation = self.format_conversation(conversation_messages)

        user_content = USER_TEMPLATE.format(context=context, conversation=conversation, question=question)

        resp = self.client.chat.completions.create(
            model=self.answer_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()


def metadoc_to_chunk(chunk_id: str, doc_text: str, meta: Dict[str, Any], score: float) -> RetrievedChunk:
    return RetrievedChunk(
        doc_name=str(meta.get("doc_name", "unknown")),
        page=int(meta.get("page", 0)),
        chunk_index=int(meta.get("chunk_index", 0)),
        text=doc_text,
        score=float(score),
        chunk_id=chunk_id,
    )


def retrieve_top_chunks(store, rag: OpenAIRAG, query: str, top_k: int = 10) -> List[RetrievedChunk]:
    q_emb = rag.embed_query(query)
    ids, metas, scores, docs = store.query(q_emb, top_k=top_k)

    chunks: List[RetrievedChunk] = []
    for cid, meta, score, doc in zip(ids, metas, scores, docs):
        if not doc or not meta:
            continue
        chunks.append(metadoc_to_chunk(cid, doc, meta, score))

    return dedupe_chunks(chunks, max_chunks=top_k)
