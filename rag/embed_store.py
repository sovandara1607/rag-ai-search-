"""
Vector store: turn chunks into vectors and support similarity search over them.

Backed by real sentence embeddings (`sentence-transformers`, model
`all-MiniLM-L6-v2`) and a FAISS index for similarity search, so it scales past
the in-memory TF-IDF + cosine-similarity approach this started from.

The `VectorStore` interface (`build`, `query`) is unchanged from the TF-IDF
version, so `app.py` doesn't need to know what's backing it.
"""

from typing import List, Tuple

import faiss
from sentence_transformers import SentenceTransformer

from .ingest import Chunk

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def _get_model() -> SentenceTransformer:
    """Lazily load the embedding model once and share it across VectorStore instances."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


class VectorStore:
    def __init__(self):
        self.index = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        """Embed all chunk text and index the vectors for cosine-similarity search."""
        self.chunks = chunks
        model = _get_model()
        texts = [c.text for c in chunks]
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product over normalized vectors = cosine similarity
        self.index.add(vectors)

    def query(self, query_text: str, top_k: int = 3) -> List[Tuple[Chunk, float]]:
        """Return the top_k (chunk, similarity_score) pairs for a query string."""
        if self.index is None:
            raise RuntimeError("VectorStore.build() must be called before query().")
        model = _get_model()
        query_vec = model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)
        top_k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_vec, top_k)
        return [(self.chunks[i], float(score)) for i, score in zip(indices[0], scores[0]) if i != -1]
