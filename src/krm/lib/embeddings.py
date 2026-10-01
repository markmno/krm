"""Embedding utilities for KRM pipeline.

Wrapper around sentence-transformers with on-disk caching to avoid
re-encoding the same texts across pipeline runs.

Supports text-enrichment grounding: characteristic labels can be prepended
to job title text before E5 encoding via :func:`build_title_characteristics`
and :meth:`Embedder.enrich_titles`, so the model integrates attribute signal
into token-level embeddings.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


def build_title_characteristics(
    dimensions_df: pd.DataFrame, top_n: int = 2
) -> dict[str, list[str]]:
    """Build a mapping from job title to its top characteristic labels.

    Groups a DataFrame of (vacancy_id, title, characteristic_label, confidence)
    by title, ranking characteristics by their mean confidence.

    Args:
        dimensions_df: DataFrame with columns ``title``, ``characteristic_label``,
            and ``confidence``.
        top_n: Number of top characteristic labels to select per title.

    Returns:
        ``dict[title, [label1, label2, ...]]`` — titles with no characteristics
        are omitted from the result.
    """
    required_cols = {"title", "characteristic_label", "confidence"}
    missing = required_cols - set(dimensions_df.columns)
    if missing:
        msg = f"DataFrame missing required columns: {missing}"
        raise KeyError(msg)

    grouped = (
        dimensions_df.groupby(["title", "characteristic_label"])["confidence"]
        .mean()
        .reset_index()
    )
    grouped = grouped.sort_values(
        ["title", "confidence"], ascending=[True, False]
    )

    result: dict[str, list[str]] = {}
    for title, grp in grouped.groupby("title"):
        labels = grp["characteristic_label"].head(top_n).tolist()
        result[str(title)] = labels
    return result


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
    """Lazily-loaded sentence-transformer wrapper with a disk cache.

    Supports text-enrichment grounding: :meth:`enrich_titles` prepends
    characteristic labels to title text before E5 encoding, so the model
    integrates attribute signal into token-level embeddings.
    """

    def __init__(
        self,
        model_name: str = "deepvk/USER-bge-m3",
        cache_dir: Path | str = "data/embeddings_cache",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache = EmbeddingCache(cache_dir)
        self._device = device
        self._model: SentenceTransformer | None = None

    def _resolve_device(self) -> str:
        if self._device is not None:
            return self._device
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self._resolve_device(),
                trust_remote_code=True,
            )
        return self._model

    @staticmethod
    def enrich_titles(
        titles: list[str],
        title_characteristics: dict[str, list[str]],
    ) -> list[str]:
        """Prepend characteristic labels to title strings for text-enrichment grounding.

        For each title, looks up its characteristic labels in
        ``title_characteristics`` and produces an enriched string of the form
        ``"characteristics: {label1}, {label2}; passage: {title}"``.

        Titles with no characteristics entry are left as ``"passage: {title}"``
        without enrichment.

        Args:
            titles: Raw job title strings.
            title_characteristics: Mapping from title → list of characteristic
                label strings, e.g.
                ``{"Химик-аналитик": ["Экспериментальный опыт", "Анализ данных"]}``.

        Returns:
            Enriched text strings, one per input title, ready for E5 encoding.
        """
        enriched: list[str] = []
        for title in titles:
            chars = title_characteristics.get(title)
            if chars:
                char_str = ", ".join(chars)
                enriched.append(f"characteristics: {char_str}; {title}")
            else:
                enriched.append(title)
        return enriched

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        *,
        enriched_titles: list[str] | None = None,
    ) -> np.ndarray:
        """Encode texts with caching and optional text-enrichment grounding.

        Args:
            texts: Raw texts to encode.
            batch_size: Number of texts per model forward pass.
            enriched_titles: Pre-enriched title strings produced by
                :meth:`enrich_titles`.  When provided, these are encoded
                and cached instead of the ``"passage: {t}"`` prefixes
                (so different enrichment of the same base title produces
                independent cache entries).

        Returns:
            Normalised embedding matrix of shape ``(len(texts), dim)``.
        """
        prefixed = enriched_titles if enriched_titles is not None else list(texts)

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
        model = self.model
        if hasattr(model, "get_embedding_dimension"):
            return model.get_embedding_dimension()
        return model.get_sentence_embedding_dimension()
