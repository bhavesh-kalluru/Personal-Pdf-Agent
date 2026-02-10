from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

import fitz  # PyMuPDF

from utils import normalize_whitespace, safe_filename, strip_page_noise_lines


@dataclass
class PageText:
    page_num: int  # 1-indexed
    text: str


def _extract_pages(pdf_bytes: bytes) -> List[PageText]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: List[PageText] = []
    for i in range(len(doc)):
        page = doc[i]
        txt = page.get_text("text") or ""
        pages.append(PageText(page_num=i + 1, text=txt))
    doc.close()
    return pages


def _detect_common_headers_footers(pages: List[PageText]) -> Tuple[List[str], List[str]]:
    top_lines = []
    bottom_lines = []
    for p in pages:
        lines = [ln.strip() for ln in (p.text or "").splitlines() if ln.strip()]
        if not lines:
            continue
        top_lines.extend(lines[:2])
        bottom_lines.extend(lines[-2:])

    if not pages:
        return [], []

    top_counts = Counter(top_lines)
    bottom_counts = Counter(bottom_lines)

    threshold = max(2, int(0.5 * len(pages)))
    headers = [ln for ln, c in top_counts.items() if c >= threshold and len(ln) >= 4]
    footers = [ln for ln, c in bottom_counts.items() if c >= threshold and len(ln) >= 4]
    return headers[:6], footers[:6]


def extract_pdf_pages_clean(pdf_bytes: bytes) -> List[PageText]:
    pages = _extract_pages(pdf_bytes)
    headers, footers = _detect_common_headers_footers(pages)

    cleaned_pages: List[PageText] = []
    for p in pages:
        txt = p.text or ""
        txt = strip_page_noise_lines(txt, headers, footers)
        txt = normalize_whitespace(txt)
        cleaned_pages.append(PageText(page_num=p.page_num, text=txt))

    return cleaned_pages


def pdf_to_page_texts(doc_name: str, pdf_bytes: bytes) -> Tuple[str, List[PageText]]:
    safe_name = safe_filename(doc_name) or "document.pdf"
    pages = extract_pdf_pages_clean(pdf_bytes)
    return safe_name, pages
