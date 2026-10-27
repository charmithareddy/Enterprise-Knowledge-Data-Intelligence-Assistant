"""
Vector Retrieval & Semantic Search Engine using FastEmbed and Cosine Similarity.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from fastembed import TextEmbedding

from src.config.settings import (
    EMBEDDING_MODEL_NAME,
    SIMILARITY_THRESHOLD,
    SCORE_MARGIN,
    TOP_K_CHUNKS,
)

class EnterpriseVectorStore:
    """Manages document embeddings and semantic vector retrieval with confidence thresholds."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._embedding_model = None
        self.chunks: List[Dict[str, Any]] = []
        self.vectors: Optional[np.ndarray] = None

    @property
    def embedding_model(self) -> TextEmbedding:
        """Lazy load embedding model."""
        if self._embedding_model is None:
            self._embedding_model = TextEmbedding(model_name=self.model_name)
        return self._embedding_model

    def build_index(self, chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Embed all document chunks and store normalized vectors for fast dot-product similarity.
        """
        if not chunks:
            self.chunks = []
            self.vectors = np.empty((0, 384), dtype=np.float32)
            return self.chunks, self.vectors

        self.chunks = chunks
        texts = [chunk["content"] for chunk in chunks]

        # Generate vectors
        raw_vectors = np.asarray(list(self.embedding_model.embed(texts)), dtype=np.float32)

        # L2 Normalize for cosine similarity via dot product
        norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
        self.vectors = raw_vectors / np.maximum(norms, 1e-12)

        return self.chunks, self.vectors

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_CHUNKS,
        threshold: float = SIMILARITY_THRESHOLD,
        margin: float = SCORE_MARGIN
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve the top most relevant chunks for a user query.
        Returns:
            context_string: Formatted context for LLM prompt.
            selected_chunks: List of chunk metadata dicts with 'score'.
        """
        if self.vectors is None or len(self.chunks) == 0:
            return "", []

        # Embed user query
        query_vec = np.asarray(list(self.embedding_model.embed([query]))[0], dtype=np.float32)
        query_norm = float(np.linalg.norm(query_vec))
        query_vec = query_vec / max(query_norm, 1e-12)

        # Cosine similarity scores
        scores = np.dot(self.vectors, query_vec)
        ranked_indices = np.argsort(scores)[::-1]

        best_score = float(scores[ranked_indices[0]])
        effective_threshold = max(threshold, best_score - margin)

        selected = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score < effective_threshold:
                continue

            item = dict(self.chunks[idx])
            item["score"] = score
            selected.append(item)

            if len(selected) >= top_k:
                break

        # Keyword boost fallback if specific codes are present (e.g. Sev-1, Q4, ARR, MFA)
        if not selected:
            query_words = set(query.lower().split())
            for idx, chunk in enumerate(self.chunks):
                content_lower = chunk["content"].lower()
                matches = sum(1 for w in query_words if len(w) > 3 and w in content_lower)
                if matches >= 2:
                    item = dict(chunk)
                    item["score"] = 0.45
                    selected.append(item)
                    if len(selected) >= top_k:
                        break

        if not selected:
            return "", []

        # Build clean formatted context block
        context_parts = []
        for item in selected:
            context_parts.append(
                f"DOCUMENT: {item['source']}\n"
                f"SECTION: {item['section']}\n"
                f"RELEVANCE SCORE: {item['score']:.4f}\n"
                f"CONTENT:\n{item['content']}"
            )

        context_string = "\n\n---\n\n".join(context_parts)
        return context_string, selected

# Singleton instance
vector_store = EnterpriseVectorStore()
