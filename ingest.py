import fitz  # PyMuPDF
from typing import List
import os
from backend.retriever import Retriever

def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    parts = []
    for page in doc:
        text = page.get_text("text", sort=True) # Added sort=True for better reading order
        if text:
            # Clean up excessive newlines and whitespace
            text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
            parts.append(text)
    return "\n\n".join(parts)

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    
    # --- ADDED THIS FIX ---
    # Remove duplicate chunks before returning
    unique_chunks = list(dict.fromkeys(chunks))
    return unique_chunks

def ingest_pdf_to_retriever(pdf_path: str, source_name: str = None, data_dir: str = "../data") -> list:
    if source_name is None:
        source_name = os.path.basename(pdf_path)
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)
    r = Retriever(data_dir)
    ids = r.add_chunks(source_name, chunks)
    return ids
