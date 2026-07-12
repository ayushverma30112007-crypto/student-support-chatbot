"""
Core RAG (Retrieval-Augmented Generation) utilities.

Pipeline:
  1. Load documents (.txt / .pdf)
  2. Split into overlapping chunks
  3. Embed chunks with a local sentence-transformer model (free, no API needed)
  4. Store embeddings in a FAISS vector index
  5. At query time: embed the question, retrieve top-k similar chunks,
     pass them as context to Groq's LLM to generate a grounded answer
"""

import os
import re
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, free, runs on CPU


def load_text_from_file(file_path: str) -> str:
    """Extract raw text from a .txt or .pdf file."""
    if file_path.lower().endswith(".pdf"):
        reader = PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    else:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150):
    """
    Split text into overlapping chunks (by characters).
    Overlap keeps context from being cut off between chunks.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


class VectorStore:
    """Wraps a FAISS index + the embedding model + the original chunk text."""

    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        self.index = None
        self.chunks = []
        self.sources = []

    def build(self, chunks: list[str], sources: list[str]):
        """Embed all chunks and build a fresh FAISS index."""
        self.chunks = chunks
        self.sources = sources
        embeddings = self.model.encode(chunks, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype("float32")
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product = cosine sim (since normalized)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 4):
        """Return the top_k most relevant chunks for a query."""
        if self.index is None or len(self.chunks) == 0:
            return []
        q_emb = self.model.encode([query], normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(q_emb, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append({
                "text": self.chunks[idx],
                "source": self.sources[idx],
                "score": float(score),
            })
        return results


def build_vector_store_from_files(file_paths: list[str]) -> VectorStore:
    """Convenience function: load + chunk + embed a list of files into a VectorStore."""
    all_chunks = []
    all_sources = []
    for path in file_paths:
        text = load_text_from_file(path)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        all_sources.extend([os.path.basename(path)] * len(chunks))

    store = VectorStore()
    store.build(all_chunks, all_sources)
    return store
