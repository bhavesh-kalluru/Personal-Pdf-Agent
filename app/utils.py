from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Iterable, List

import tiktoken


def load_env_path() -> str:
    """Returns a safe path for app data persistence."""
    return os.path.join(os.path.dirname(__file__), "app_data")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def normalize_whitespace(text: str) -> str:
    """Basic cleanup for messy PDF text: collapse whitespace, normalize newlines."""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_page_noise_lines(page_text: str, header_lines: Iterable[str], footer_lines: Iterable[str]) -> str:
    """Remove exact header/footer lines when found at top/bottom."""
    lines = [ln.strip() for ln in page_text.splitlines()]
    if not lines:
        return page_text

    header_set = {h.strip() for h in header_lines if h and h.strip()}
    footer_set = {f.strip() for f in footer_lines if f and f.strip()}

    new_lines = lines[:]
    for i in range(min(3, len(new_lines))):
        if new_lines[i] in header_set:
            new_lines[i] = ""

    for i in range(1, min(4, len(new_lines)) + 1):
        if new_lines[-i] in footer_set:
            new_lines[-i] = ""

    cleaned = "\n".join([ln for ln in new_lines if ln.strip()])
    return cleaned.strip()


def get_tokenizer(model_name: str = "gpt-4o-mini") -> tiktoken.Encoding:
    """Tokenizer for chunk sizing."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def chunk_text_by_tokens(
    text: str,
    chunk_tokens: int,
    overlap_tokens: int,
    model_name: str = "gpt-4o-mini",
) -> List[str]:
    """Sliding-window chunking by tokens."""
    enc = get_tokenizer(model_name)
    tokens = enc.encode(text)

    if not tokens:
        return []

    chunks: List[str] = []
    start = 0
    n = len(tokens)

    if overlap_tokens >= chunk_tokens:
        overlap_tokens = max(0, chunk_tokens // 4)

    while start < n:
        end = min(start + chunk_tokens, n)
        chunk = enc.decode(tokens[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap_tokens)

    return chunks


def stable_doc_id(doc_name: str, file_bytes: bytes) -> str:
    """Stable ID for a PDF to avoid re-ingestion duplicates."""
    h = hashlib.sha256()
    h.update(doc_name.encode("utf-8", errors="ignore"))
    h.update(file_bytes)
    return h.hexdigest()[:24]


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\-. ]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:120]


@dataclass(frozen=True)
class RetrievedChunk:
    doc_name: str
    page: int
    chunk_index: int
    text: str
    score: float
    chunk_id: str


def dedupe_chunks(chunks: List[RetrievedChunk], max_chunks: int = 12) -> List[RetrievedChunk]:
    """Deduplicate by normalized text hash, keep best score."""
    best = {}
    for ch in chunks:
        key = sha256_text(normalize_whitespace(ch.text).lower())
        if key not in best or ch.score > best[key].score:
            best[key] = ch
    out = sorted(best.values(), key=lambda x: x.score, reverse=True)
    return out[:max_chunks]
