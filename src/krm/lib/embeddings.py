"""Embedding utilities for KRM pipeline.

Wrapper around sentence-transformers with on-disk caching to avoid
re-encoding the same texts across pipeline runs.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np


class EmbeddingCache:
    """On-disk cache for text embeddings keyed by text hash."""

    def __init__(self, cache_dir: Path | str = "data/embeddings_cache") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get(self, text: str) -> np.ndarray | None:
        path = self.cache_dir / f"{self._key(text)}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)  # noqa: S301 — trusted local cache
        return None

    def set(self, text: str, embedding: np.ndarray) -> None:
        path = self.cache_dir / f"{self._key(text)}.pkl"
        with open(path, "wb") as f:
            pickle.dump(embedding, f)

    def batch_get(self, texts: list[str]) -> tuple[list[str], list[np.ndarray], list[int]]:
        """Return (uncached_texts, cached_embeddings, cached_indices)."""
        uncached: list[str] = []
        cached_embs: list[np.ndarray] = []
        cached_idx: list[int] = []
        for i, text in enumerate(texts):
            emb = self.get(text)
            if emb is not None:
                cached_embs.append(emb)
                cached_idx.append(i)
            else:
                uncached.append(text)
        return uncached, cached_embs, cached_idx

    def batch_set(self, texts: list[str], embeddings: np.ndarray) -> None:
        for text, emb in zip(texts, embeddings, strict=True):
            self.set(text, emb)


class Embedder:
    """Lazily-loaded sentence-transformer with caching."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large-instruct",
        cache_dir: Path | str = "data/embeddings_cache",
    ) -> None:
        self.model_name = model_name
        self.cache = EmbeddingCache(cache_dir)
        self._model: object | None = None

    @property
    def model(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
        return self._model

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts with caching. E5 models need 'query: ' prefix."""
        prefixed = [f"query: {t}" for t in texts]

        uncached, cached_embs, cached_idx = self.cache.batch_get(prefixed)

        if not uncached:
            # All cached — reassemble
            result = np.zeros((len(texts), self.dim), dtype=np.float32)
            for idx, emb in zip(cached_idx, cached_embs, strict=True):
                result[idx] = emb
            return result

        new_embs: np.ndarray = self.model.encode(
            uncached,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        self.cache.batch_set(uncached, new_embs)

        # Merge cached + new
        result = np.zeros((len(texts), self.dim), dtype=np.float32)
        for idx, emb in zip(cached_idx, cached_embs, strict=True):
            result[idx] = emb
        new_idx = 0
        for i in range(len(texts)):
            if i not in cached_idx:
                result[i] = new_embs[new_idx]
                new_idx += 1

        return result

    @property
    def dim(self) -> int:
        return self.model.get_sentence_embedding_dimension() or 1024
