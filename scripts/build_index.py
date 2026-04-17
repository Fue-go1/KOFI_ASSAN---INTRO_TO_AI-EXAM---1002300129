# Student: Kofi Assan | Index: 1002300129 | CS4241-Introduction to Artificial Intelligence
"""Build FAISS index from downloaded CSV + PDF."""
from __future__ import annotations

import sys
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.chunking import chunk_pdf_text, chunks_from_csv
from rag.embeddings import embed_texts
from rag.store import FaissStore

RAW = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "data" / "index"


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        parts.append(t)
    return "\n".join(parts)


def main() -> None:
    csv_path = RAW / "Ghana_Election_Result.csv"
    pdf_path = RAW / "2025-Budget-Statement-and-Economic-Policy_v4.pdf"
    if not csv_path.is_file() or not pdf_path.is_file():
        print("Run scripts/download_data.py first.", file=sys.stderr)
        sys.exit(1)

    chunks = []
    chunks.extend(chunks_from_csv(str(csv_path)))
    pdf_text = extract_pdf_text(pdf_path)
    chunks.extend(chunk_pdf_text(pdf_text, source_label="budget_2025"))

    texts = [c.text for c in chunks]
    print(f"Embedding {len(texts)} chunks…")
    vectors = embed_texts(texts)
    dim = vectors.shape[1]
    store = FaissStore(dim=dim)
    store.add(vectors, chunks)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save(INDEX_DIR)
    print(f"Saved index to {INDEX_DIR} ({len(chunks)} chunks, dim={dim}).")


if __name__ == "__main__":
    main()
