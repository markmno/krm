"""Russian word lemmatization using pymorphy3."""

from __future__ import annotations

from functools import lru_cache

from pymorphy3 import MorphAnalyzer


class Lemmatizer:
    """Lazily-initialized pymorphy3 lemmatizer with caching.

    pymorphy3's MorphAnalyzer is expensive to create (~500ms dictionary load),
    so we use a class that initializes once and caches frequent lookups.
    """

    def __init__(self) -> None:
        self._morph = MorphAnalyzer()

    def lemmatize(self, word: str) -> str:
        """Return normal form (lemma) of a Russian word.

        Example: 'физиками' → 'физик', 'инженерами' → 'инженер'
        """
        return self.lemmatize_with_pos(word)[0]

    # Known safe: Lemmatizer is used as a module-level singleton via NLPPipeline,
    # so lru_cache on this method does not cause memory leaks in practice.
    @lru_cache(maxsize=4096)  # noqa: B019
    def lemmatize_with_pos(self, word: str) -> tuple[str, str]:
        """Return (lemma, POS) tuple with LRU caching.

        POS values from pymorphy3: NOUN, ADJF, ADJS, COMP, VERB, INFN, PRTF,
        PRTS, GRND, NUMR, ADVB, NPRO, PRED, PREP, CONJ, PRCL, INTJ, LATN.
        """
        parsed = self._morph.parse(word)[0]
        pos = parsed.tag.POS
        return parsed.normal_form, (pos if pos else "UNKN")

    def get_pos(self, word: str) -> str:
        """Return POS tag for a word."""
        return self.lemmatize_with_pos(word)[1]
