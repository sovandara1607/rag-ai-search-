"""
Ingestion: load raw documents from disk and split them into overlapping chunks.

Supports plain .txt files and .pdf files (text extraction via pypdf — scanned/
image-only PDFs won't yield usable text since there's no OCR step here).

Upgrade path (for your final project):
- Add HTML/Markdown loaders (e.g. BeautifulSoup) alongside .txt/.pdf
- Swap the naive word-count chunker below for a sentence- or token-aware chunker
- Store document metadata (source URL, author, date) alongside each chunk
"""

import os
from dataclasses import dataclass
from typing import List

from pypdf import PdfReader


@dataclass
class Chunk:
    chunk_id: str
    doc_title: str
    text: str


def derive_title(filename: str) -> str:
    """Turn a filename into a readable title without mangling existing acronyms/casing."""
    stem = os.path.splitext(filename)[0]
    if stem.lower().endswith(".pptx"):  # slides often get exported as "Name.pptx.pdf"
        stem = stem[: -len(".pptx")]
    words = stem.replace("_", " ").split()
    return " ".join(w.capitalize() if w.islower() else w for w in words)


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def _extract_pdf_text(source) -> str:
    """Extract text from a PDF given either a file path or a file-like object."""
    reader = PdfReader(source)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


def read_uploaded_file(uploaded) -> str:
    """Extract text from a Streamlit UploadedFile (.txt or .pdf)."""
    if uploaded.name.lower().endswith(".pdf"):
        return _extract_pdf_text(uploaded)
    return uploaded.getvalue().decode("utf-8", errors="ignore").strip()


def load_documents(folder: str) -> List[dict]:
    """Load every .txt/.pdf file in `folder` into {"title": ..., "text": ...} dicts."""
    docs = []
    for filename in sorted(os.listdir(folder)):
        path = os.path.join(folder, filename)
        lower = filename.lower()
        if lower.endswith(".txt"):
            text = _read_txt(path)
        elif lower.endswith(".pdf"):
            text = _extract_pdf_text(path)
        else:
            continue
        if not text:
            continue
        docs.append({"title": derive_title(filename), "text": text})
    return docs


def chunk_text(text: str, chunk_size: int = 80, overlap: int = 20) -> List[str]:
    """Split text into overlapping word-count chunks (simple, dependency-free)."""
    words = text.split()
    if not words:
        return []
    # overlap must be strictly less than chunk_size, or `start` never advances
    # and the loop below never terminates.
    overlap = max(0, min(overlap, chunk_size - 1))
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def build_chunk_records(docs: List[dict], chunk_size: int = 80, overlap: int = 20) -> List[Chunk]:
    """Turn loaded documents into a flat list of Chunk records ready for embedding."""
    records = []
    for doc in docs:
        pieces = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            records.append(Chunk(chunk_id=f"{doc['title']}::{i}", doc_title=doc["title"], text=piece))
    return records
