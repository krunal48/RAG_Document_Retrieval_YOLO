# app/ingest/chunker.py
from typing import List, Dict, Any

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """Character-based chunking with overlap."""
    text = (text or "").strip()
    if not text:
        return []
    chunks: List[str] = []
    i, n = 0, len(text)
    while i < n:
        end = min(i + chunk_size, n)
        chunks.append(text[i:end])
        if end == n:
            break
        i = max(0, end - overlap)
    return chunks

def build_records(
    doc_id: str,
    title: str,
    pages_text: List[str],
    chunk_size: int = 1200,
    overlap: int = 200
) -> List[Dict[str, Any]]:
    """Build per-page chunk records with metadata for vector upsert."""
    recs: List[Dict[str, Any]] = []
    for pno, page in enumerate(pages_text, start=1):
        for ci, chunk in enumerate(chunk_text(page, chunk_size, overlap), start=1):
            recs.append({
                "id": f"{doc_id}-p{pno}-c{ci}",
                "text": chunk,
                "metadata": {
                    "doc_id": doc_id,
                    "title": title,
                    "page": pno,
                    "chunk": ci,
                },
            })
    return recs
