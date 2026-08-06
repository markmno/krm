"""NLP pipeline for vacancy text: tokenization → lemmatization → collocations → filtering."""

from __future__ import annotations

from collections import Counter

from razdel import tokenize

from hh_competency.nlp.lemmatizer import Lemmatizer
from hh_competency.nlp.stopwords import StopwordFilter

# POS tags to keep for skill extraction (nouns, proper nouns, adjectives, latin words)
_POS_TO_KEEP = frozenset({"NOUN", "PROPN", "LATN"})
# POS tags that can form the HEAD of a collocation (noun-like)
_COLLOCATION_HEAD_POS = frozenset({"NOUN", "PROPN", "LATN"})
# POS tags that can PRECEDE a head in a collocation (adjectives, nouns)
_COLLOCATION_PRE_POS = frozenset({"ADJF", "NOUN", "PROPN", "LATN"})
# Minimum word length for collocation component
_MIN_COLLOC_WORD_LEN = 3

# Known Russian STEM collocations that should always be treated as compounds
_KNOWN_COLLOCATIONS: frozenset[str] = frozenset({
    # IT/Data (existing)
    "машинное обучение",
    "анализ данных",
    "численные методы",
    "искусственный интеллект",
    "большие данные",
    "обработка данных",
    "временные ряды",
    "нейронные сети",
    "компьютерное зрение",
    "базы данных",
    "глубокое обучение",
    "прикладная математика",
    # Physics
    "лабораторное оборудование",
    "оптические системы",
    "вакуумные системы",
    "лазерные установки",
    "математическое моделирование",
    "чистые помещения",
    "рентгеновская дифракция",
    "электронная микроскопия",
    "ядерный магнитный резонанс",
    "квантовая механика",
    "квантовая химия",
    "молекулярная динамика",
    "конденсированное состояние",
    "вычислительная физика",
    "физическая химия",
    "аттестация оборудования",
    "метрологическое обеспечение",
    "дифференциальные уравнения",
    "теоретическая физика",
    "экспериментальная физика",
    "физика плазмы",
    "ускорительная техника",
    "радиационная безопасность",
    # Biology/Chemistry
    "молекулярная биология",
    "клинические исследования",
    "полевые исследования",
    "генная инженерия",
    "полимеразная цепная",
    "клеточная биология",
    "органический синтез",
    "аналитическая химия",
    "масс-спектрометрия",
    "хроматографический анализ",
    "спектральный анализ",
    "высокоэффективная жидкостная",
    # Engineering
    "проектная документация",
    "техническая документация",
    "охрана труда",
    "техника безопасности",
    "техническое обслуживание",
    "конструкторская документация",
    "инженерные изыскания",
    "пусконаладочные работы",
    "технологический процесс",
    # Scientific work
    "научные исследования",
    "естественные науки",
    "научная статья",
    "научная деятельность",
    "экспериментальные данные",
    "научная литература",
    "анализ результатов",
    "обзор литературы",
    # Management/Communication
    "управление проектами",
    "управление командой",
    "ведение переговоров",
    "деловая переписка",
    "публичное выступление",
    "международное сотрудничество",
    "научное руководство",
    "грантовая деятельность",
})


# Pre-compute lemma forms of known collocations for matching inflected text.
# Russian adjectives lemmatize to masculine nominative (e.g. машинный),
# but known collocations use the form that matches the noun gender.
# We generate both so either form matches during detection.
def _build_lemma_collocations() -> frozenset[str]:
    from pymorphy3 import MorphAnalyzer

    morph = MorphAnalyzer()
    lemma_set: set[str] = set()
    for colloc in _KNOWN_COLLOCATIONS:
        parts = colloc.split()
        try:
            l0 = morph.parse(parts[0])[0].normal_form
            l1 = morph.parse(parts[1])[0].normal_form
            lemma_set.add(f"{l0} {l1}")
        except (IndexError, Exception):
            lemma_set.add(colloc)
    return frozenset(lemma_set)


_KNOWN_COLLOCATIONS_LEMMA: frozenset[str] = _build_lemma_collocations()



class NLPPipeline:
    """End-to-end Russian NLP pipeline for vacancy text analysis.

    Pipeline: razdel tokenize → collocation merge → pymorphy3 lemmatize
              → stopword filter → POS filter.
    """

    def __init__(self) -> None:
        self._lemmatizer = Lemmatizer()
        self._stopwords = StopwordFilter()

    def extract_keywords(self, text: str) -> list[tuple[str, str]]:
        """Extract meaningful (lemma, POS) tuples including named collocations.

        Filters out:
        - Words shorter than 3 characters
        - General and domain stopwords
        - Non-noun POS tags (keeps NOUN, PROPN, LATN)

        Also detects multi-word collocations like 'машинное обучение' and
        lemmatizes them as compounds (e.g., 'машинный обучение').

        Args:
            text: Raw vacancy description text in Russian.

        Returns:
            List of (lemma, POS) tuples for meaningful content words.
        """
        if not text or not text.strip():
            return []

        # First pass: collect all tokens with their POS tags
        tokens: list[str] = [t.text for t in tokenize(text)]
        token_pos: list[tuple[str, str]] = [
            self._lemmatizer.lemmatize_with_pos(t)
            for t in tokens
        ]

        # Merge detected collocations into compound tokens
        used: set[int] = set()
        lemmas: list[tuple[str, str]] = []

        i = 0
        while i < len(tokens):
            if i in used:
                i += 1
                continue

            # Try to match bigram collocation: PRE_POS + HEAD_POS
            colloc = self._try_collocation(tokens, token_pos, i, used)
            if colloc is not None:
                lemmas.append(colloc)
                i += 2
                continue

            # Try trigram: PRE_POS + PRE_POS + HEAD
            colloc = self._try_trigram_collocation(tokens, token_pos, i, used)
            if colloc is not None:
                lemmas.append(colloc)
                i += 3
                continue

            # Skip single words — keep only collocations
            i += 1

        return lemmas

    def _try_collocation(
        self,
        tokens: list[str],
        pos_list: list[tuple[str, str]],
        idx: int,
        used: set[int],
    ) -> tuple[str, str] | None:
        """Try to form a 2-word collocation at position idx.

        Pattern: ADJF/NOUN/PROPN + NOUN/PROPN (e.g., 'машинное обучение')
        Both words must be ≥ min length and not stopwords.
        Also checks against known collocation patterns.
        """
        if idx + 1 >= len(tokens):
            return None

        w0, w1 = tokens[idx], tokens[idx + 1]
        p0 = pos_list[idx]
        p1 = pos_list[idx + 1]

        # Both must be long enough
        if len(w0) < _MIN_COLLOC_WORD_LEN or len(w1) < _MIN_COLLOC_WORD_LEN:
            return None

        # Pattern: preceding POS + head POS
        if p0[1] not in _COLLOCATION_PRE_POS or p1[1] not in _COLLOCATION_HEAD_POS:
            return None

        # Lemmatize: adjective agrees with noun in case, use normalized forms
        lemma0, lemma1 = p0[0], p1[0]
        compound = f"{lemma0} {lemma1}"

        # Check if this is a known collocation FIRST (before stopword filter).
        # Known compounds like "машинное обучение" contain words that are
        # individually stopwords (e.g., "обучение").
        orig_bigram = f"{w0.lower()} {w1.lower()}"
        lemma_bigram = f"{lemma0.lower()} {lemma1.lower()}"
        is_known = (
            orig_bigram in _KNOWN_COLLOCATIONS
            or lemma_bigram in _KNOWN_COLLOCATIONS
            or lemma_bigram in _KNOWN_COLLOCATIONS_LEMMA
        )

        # For known collocations, skip stopword check entirely
        if is_known:
            used.add(idx)
            used.add(idx + 1)
            return (compound, "COLLOCATION")

        # Neither should be a stopword (only for unknown/non-known bigrams)
        if self._stopwords.is_stopword(p0[0]) or self._stopwords.is_stopword(p1[0]):
            return None

        # For non-known bigrams, only accept strong patterns:
        # ADJF+NOUN is most reliable for Russian
        if p0[1] == "ADJF" and p1[1] in ("NOUN", "PROPN"):
            used.add(idx)
            used.add(idx + 1)
            return (compound, "COLLOCATION")

        return None

    def _try_trigram_collocation(
        self,
        tokens: list[str],
        pos_list: list[tuple[str, str]],
        idx: int,
        used: set[int],
    ) -> tuple[str, str] | None:
        """Try to form a 3-word collocation (e.g., 'ядерный магнитный резонанс')."""
        if idx + 2 >= len(tokens):
            return None

        # Only check known 3-gram collocations for reliability
        trigram = f"{tokens[idx].lower()} {tokens[idx + 1].lower()} {tokens[idx + 2].lower()}"
        if trigram not in _KNOWN_COLLOCATIONS:
            return None

        w0, w1, w2 = tokens[idx], tokens[idx + 1], tokens[idx + 2]
        if (
            len(w0) < _MIN_COLLOC_WORD_LEN
            or len(w1) < _MIN_COLLOC_WORD_LEN
            or len(w2) < _MIN_COLLOC_WORD_LEN
        ):
            return None

        p0 = pos_list[idx]
        p1 = pos_list[idx + 1]
        if self._stopwords.is_stopword(p0[0]) or self._stopwords.is_stopword(p1[0]):
            return None

        compound = f"{p0[0]} {p1[0]} {tokens[idx + 2].lower()}"
        used.add(idx)
        used.add(idx + 1)
        used.add(idx + 2)
        return (compound, "COLLOCATION")

    def get_skill_frequencies(
        self,
        texts: list[str],
        nlp_keywords: list[str] | None = None,
    ) -> Counter:
        """Compute lemma frequencies across multiple vacancy descriptions.

        Args:
            texts: List of vacancy description strings.
            nlp_keywords: Optional list of keywords to filter for specific roles.
                          If None, all extracted keywords are counted.

        Returns:
            Counter mapping lemmas to their frequency across all texts.
        """
        all_lemmas: Counter = Counter()
        for text in texts:
            keywords = self.extract_keywords(text)
            lemma_set = {lemma for lemma, _pos in keywords}
            filtered = (
                [lemma for lemma in lemma_set if lemma in nlp_keywords]
                if nlp_keywords
                else list(lemma_set)
            )
            all_lemmas.update(filtered)

        return all_lemmas

    def extract_collocations(self, text: str) -> list[str]:
        """Extract only collocations (multi-word terms) from text.

        Useful for discovering domain-specific compound terms in vacancies.

        Args:
            text: Raw vacancy description text.

        Returns:
            List of collocation compound strings found in text.
        """
        if not text or not text.strip():
            return []

        keywords = self.extract_keywords(text)
        return [lemma for lemma, pos in keywords if str(pos) == "COLLOCATION"]

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text with razdel (useful for debugging)."""
        return [t.text for t in tokenize(text)]
