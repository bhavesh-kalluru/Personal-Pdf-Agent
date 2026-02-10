# Personal FAQ — PDF RAG Bot (Python + OpenAI + optional Perplexity)

A simple MVP web app where you upload 1–5 PDFs and chat with them.
Answers are *document-grounded* and include quotes + page-number citations.

## Features
- Upload 1–5 PDFs (resume/notes/manuals)
- Extract text with page boundaries (PyMuPDF)
- Chunking (token-based) with overlap, **kept within each page** for clean page citations
- Local persistent vector DB (Chroma)
- RAG answering via OpenAI
- Multi-turn chat history (Streamlit session)
- Optional Perplexity web fallback **only when explicitly requested**
  - user must type: `web: ...` or `latest: ...` or `news: ...`
  - and enable the toggle in the sidebar

---

## Quickstart

### 1) Create venv and install deps
```bash
cd personal-faq-rag
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows
# .venv\Scripts\activate

pip install -r requirements.txt
```

### 2) Add API keys
Edit `app/.env`:
- `OPENAI_API_KEY=...` (required)
- `PERPLEXITY_API_KEY=...` (optional)

### 3) Run
```bash
streamlit run app/main.py
```

Open the local URL Streamlit prints.

---

## How it works (high level)

1) **Ingestion**
- For each PDF:
  - Extract text per page
  - Remove repeated headers/footers (heuristic)
  - Chunk each page into ~1000 tokens with ~150 token overlap
  - Embed each chunk with OpenAI embeddings
  - Store in Chroma with metadata: `{doc_name, page, chunk_index}`

2) **Retrieval**
- Embed the user query
- Retrieve top-k chunks from Chroma
- Deduplicate similar chunks

3) **Answer generation**
- Provide retrieved chunks as `<context>`
- Force strict citations `[doc_name p.<page>]` and 1–3 short quotes

---

## Prompting

### System prompt (grounded)
The app uses a strict system prompt:
- Answer using ONLY `<context>`
- Cite every factual claim like `[doc_name p.<page>]`
- Include 1–3 short direct quotes with citations
- If not supported: “I don't know based on the provided PDFs.”

See `app/prompts.py`.

### User prompt template
The retrieved chunks are injected as:

```
<context>
[doc_name=... page=... chunk=...]
...text...
</context>

Conversation (for reference only; NOT a source of truth):
...

Question: ...
```

---

## API Keys

### Environment variables (recommended)
- `OPENAI_API_KEY` (required)
- `PERPLEXITY_API_KEY` (optional)
- `OPENAI_EMBED_MODEL` default: `text-embedding-3-small`
- `OPENAI_ANSWER_MODEL` default: `gpt-4o-mini`
- `PERPLEXITY_MODEL` default: `sonar-pro`

The app loads `app/.env` automatically.

---

## Expected behavior (examples)

### Example 1 (supported answer with citations)
**User:** "What is the candidate’s experience with Kubernetes?"

**Assistant (example style):**
Answer:
- The PDFs describe hands-on Kubernetes experience, including deploying services and managing clusters. [Resume.pdf p.2]

Supporting quotes:
- "Deployed microservices to Kubernetes clusters and managed Helm-based releases." [Resume.pdf p.2]

Sources:
- [Resume.pdf p.2]

### Example 2 (insufficient info)
**User:** "What is their current salary?"

Assistant:
Answer:
- I don't know based on the provided PDFs.
- If you want this answered, upload a document that contains compensation details or ask a different question about the existing documents.

Supporting quotes:
- (none)

Sources:
- (none)

### Example 3 (explicit web lookup)
**User:** "web: What’s the latest version of Python and when was it released?"

(Only if the sidebar toggle is enabled and PERPLEXITY_API_KEY is set.)

Assistant:
- Returns a web-based answer (Perplexity) and does NOT pretend it came from the PDFs.

---

## Notes / Tips
- Scanned-image PDFs may extract little/no text. You’d need OCR for that (not included in this MVP).
- Keeping chunks within a single page makes citations reliable.
- For better quality, you can add reranking later (e.g., a small cross-encoder).

---

## Troubleshooting
- If you see: `OPENAI_API_KEY is missing`, set it in `app/.env` and restart.
- If web lookup errors, verify `PERPLEXITY_API_KEY` and possibly adjust `PERPLEXITY_MODEL`.
