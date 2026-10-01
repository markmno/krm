"""Tests for phase_2b_extract_skills.py — Russian skill phrase extraction.

Tests cover noun-phrase extraction from a spaCy dependency tree, phrase
filtering, deduplication clustering, and the end-to-end pipeline (happy path,
empty input, short text skip).

No torch, sentence-transformers, spaCy, or DuckDB required for pure unit
tests: noun-phrase extraction is tested against a fake token tree, and the
end-to-end tests mock the DB connection.  spaCy only loads lazily when a real
description needs parsing, which these tests avoid.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _reset_nlp_caches() -> None:
    """Clear module-level NLP singletons between tests."""
    from krm.phase_2b_extract_skills import _reset_nlp_caches

    _reset_nlp_caches()


# ---------------------------------------------------------------------------
# Fake spaCy Doc/token helpers (no model download required)
# ---------------------------------------------------------------------------


class _FakeToken:
    def __init__(self, i: int, text: str, lemma: str, pos: str, dep: str) -> None:
        self.i = i
        self.text = text
        self.lemma_ = lemma
        self.pos_ = pos
        self.dep_ = dep
        self.children: list[_FakeToken] = []


def _make_doc(
    specs: list[tuple[str, str, str, str, int | None]],
) -> list[_FakeToken]:
    """Build a fake spaCy Doc (list of tokens) from dependency specs.

    Each spec is ``(text, lemma, pos, dep, head_index)``; ``head_index`` is
    the index of the parent token in ``specs`` (``None`` = root).
    """
    tokens = [
        _FakeToken(i, text, lemma, pos, dep)
        for i, (text, lemma, pos, dep, _) in enumerate(specs)
    ]
    for tok, (_, _, _, _, head) in zip(tokens, specs):
        if head is not None:
            tokens[head].children.append(tok)
    return tokens


# ---------------------------------------------------------------------------
# Test: _extract_noun_phrases
# ---------------------------------------------------------------------------


class TestExtractNounPhrases:
    """Tests for dependency-tree noun-phrase extraction."""

    def test_simple_noun_with_adjective(self) -> None:
        from krm.phase_2b_extract_skills import _extract_noun_phrases

        doc = _make_doc([
            ("лабораторных", "лабораторный", "ADJ", "amod", 1),
            ("исследований", "исследование", "NOUN", "root", None),
        ])
        phrases = _extract_noun_phrases(doc)
        assert phrases == [("лабораторных исследований", "лабораторный исследование")]

    def test_noun_with_participle_modifier(self) -> None:
        """A participle (VERB with amod dep) is included in the phrase."""
        from krm.phase_2b_extract_skills import _extract_noun_phrases

        doc = _make_doc([
            ("полученных", "получить", "VERB", "amod", 1),
            ("данных", "данные", "NOUN", "root", None),
        ])
        phrases = _extract_noun_phrases(doc)
        assert phrases == [("полученных данных", "получить данные")]

    def test_conjunction_filters_out(self) -> None:
        """Conjunct nouns stay separate; the conjunction is not merged."""
        from krm.phase_2b_extract_skills import _extract_noun_phrases

        # "анализ и синтез": синтез is a conj child of анализ.
        doc = _make_doc([
            ("анализ", "анализ", "NOUN", "root", None),
            ("и", "и", "CCONJ", "cc", 2),
            ("синтез", "синтез", "NOUN", "conj", 0),
        ])
        phrases = _extract_noun_phrases(doc)
        phrase_texts = {p[0] for p in phrases}
        assert "анализ" in phrase_texts
        assert "синтез" in phrase_texts

    def test_nested_nmod_forms_phrase(self) -> None:
        from krm.phase_2b_extract_skills import _extract_noun_phrases

        doc = _make_doc([
            ("анализ", "анализ", "NOUN", "root", None),
            ("данных", "данные", "NOUN", "nmod", 0),
        ])
        phrases = _extract_noun_phrases(doc)
        assert ("анализ данных", "анализ данные") in phrases

    def test_no_nouns_returns_empty(self) -> None:
        from krm.phase_2b_extract_skills import _extract_noun_phrases

        doc = _make_doc([
            ("и", "и", "CCONJ", "cc", None),
            ("в", "в", "ADP", "case", None),
        ])
        assert _extract_noun_phrases(doc) == []

    def test_empty_doc_returns_empty(self) -> None:
        from krm.phase_2b_extract_skills import _extract_noun_phrases

        assert _extract_noun_phrases([]) == []

    def test_realistic_dependency_tree(self) -> None:
        from krm.phase_2b_extract_skills import _extract_noun_phrases

        # "Проведение лабораторных исследований и анализ полученных данных"
        doc = _make_doc([
            ("Проведение", "проведение", "NOUN", "root", None),
            ("лабораторных", "лабораторный", "ADJ", "amod", 2),
            ("исследований", "исследование", "NOUN", "nmod", 0),
            ("и", "и", "CCONJ", "cc", 4),
            ("анализ", "анализ", "NOUN", "conj", 0),
            ("полученных", "получить", "VERB", "amod", 6),
            ("данных", "данные", "NOUN", "nmod", 4),
        ])
        phrases = _extract_noun_phrases(doc)
        phrase_texts = {p[0] for p in phrases}
        expected = {
            "проведение лабораторных исследований",
            "лабораторных исследований",
            "анализ полученных данных",
            "полученных данных",
        }
        assert phrase_texts == expected, f"Got: {phrase_texts}"

    def test_cyclic_dependency_tree_terminates(self) -> None:
        """Cyclic dependency tree must terminate instead of infinite-looping."""
        from krm.phase_2b_extract_skills import _extract_noun_phrases

        doc = _make_doc([
            ("анализ", "анализ", "NOUN", "nmod", 1),
            ("циклический", "циклический", "ADJ", "amod", 0),
        ])
        phrases = _extract_noun_phrases(doc)

        assert len(phrases) == 1
        text, lemma = phrases[0]
        assert set(text.split()) == {"анализ", "циклический"}
        assert set(lemma.split()) == {"анализ", "циклический"}

    def test_boilerplate_filtered(self) -> None:
        from krm.phase_2b_extract_skills import _extract_noun_phrases

        doc = _make_doc([
            ("опыт", "опыт", "NOUN", "root", None),
            ("работы", "работа", "NOUN", "nmod", 0),
        ])
        phrases = _extract_noun_phrases(doc)
        assert ("опыт работы", "опыт работа") not in phrases


# ---------------------------------------------------------------------------
# Test: _filter_phrases
# ---------------------------------------------------------------------------


class TestFilterPhrases:
    """Tests for word-count filtering."""

    def test_keeps_phrases_in_range(self) -> None:
        from krm.phase_2b_extract_skills import _filter_phrases

        phrases = [
            ("a", "a"),  # 1 word → out if min=2
            ("a b", "a b"),  # 2 words → in
            ("a b c d", "a b c d"),  # 4 words → in
            ("a b c d e", "a b c d e"),  # 5 words → out if max=4
        ]
        result = _filter_phrases(phrases, min_words=2, max_words=4)
        assert result == [("a b", "a b"), ("a b c d", "a b c d")]

    def test_min_words_1_allows_single(self) -> None:
        from krm.phase_2b_extract_skills import _filter_phrases

        phrases = [("спектроскопия", "спектроскопия")]
        result = _filter_phrases(phrases, min_words=1, max_words=4)
        assert len(result) == 1

    def test_empty_input(self) -> None:
        from krm.phase_2b_extract_skills import _filter_phrases

        assert _filter_phrases([], min_words=1, max_words=4) == []


# ---------------------------------------------------------------------------
# Test: _deduplicate_phrases
# ---------------------------------------------------------------------------


class TestDeduplicatePhrases:
    """Tests for embedding-based deduplication."""

    def _mock_embedder(self, lemmas: list[str], sim_matrix: np.ndarray) -> MagicMock:
        """Build a mock Embedder that returns the given similarity matrix."""
        emb = MagicMock()
        # We mock encode to return vectors with the desired pairwise dot products.
        eigvals, eigvecs = np.linalg.eigh(sim_matrix)
        eigvals = np.maximum(eigvals, 0)  # make PSD
        vectors = eigvecs @ np.diag(np.sqrt(eigvals))
        emb.encode = MagicMock(return_value=vectors.astype(np.float32))
        return emb

    def test_single_phrase_passthrough(self) -> None:
        """A single phrase returns as-is."""
        from krm.phase_2b_extract_skills import _deduplicate_phrases

        counter: Counter[tuple[str, str]] = Counter()
        counter[("хроматография", "хроматография")] = 5
        emb = self._mock_embedder(["хроматография"], np.array([[1.0]]))
        result = _deduplicate_phrases(counter, emb, threshold=0.85)
        assert result == [("хроматография", "хроматография", 5)]

    def test_similar_phrases_merged(self) -> None:
        """Two highly similar phrases merge into the more frequent one."""
        from krm.phase_2b_extract_skills import _deduplicate_phrases

        counter: Counter[tuple[str, str]] = Counter()
        counter[("хроматография", "хроматография")] = 3
        counter[("хроматографический анализ", "хроматографический анализ")] = 7

        sim = np.array([[1.0, 0.9], [0.9, 1.0]])
        emb = self._mock_embedder(
            ["хроматография", "хроматографический анализ"], sim
        )
        result = _deduplicate_phrases(counter, emb, threshold=0.85)

        assert len(result) == 1
        assert result[0][0] == "хроматографический анализ"
        assert result[0][2] == 10  # merged freq

    def test_dissimilar_phrases_kept_separate(self) -> None:
        """Two dissimilar phrases are NOT merged."""
        from krm.phase_2b_extract_skills import _deduplicate_phrases

        counter: Counter[tuple[str, str]] = Counter()
        counter[("python", "python")] = 5
        counter[("биология", "биология")] = 3

        sim = np.array([[1.0, 0.3], [0.3, 1.0]])
        emb = self._mock_embedder(["python", "биология"], sim)
        result = _deduplicate_phrases(counter, emb, threshold=0.85)

        assert len(result) == 2
        texts = {r[0] for r in result}
        assert texts == {"python", "биология"}

    def test_empty_input(self) -> None:
        from krm.phase_2b_extract_skills import _deduplicate_phrases

        emb = MagicMock()
        result = _deduplicate_phrases(Counter(), emb, threshold=0.85)
        assert result == []

    def test_at_threshold_boundary(self) -> None:
        """Phrases at exactly threshold=0.85 are merged."""
        from krm.phase_2b_extract_skills import _deduplicate_phrases

        counter: Counter[tuple[str, str]] = Counter()
        counter[("a", "a")] = 2
        counter[("b", "b")] = 3

        sim = np.array([[1.0, 0.85], [0.85, 1.0]])
        emb = self._mock_embedder(["a", "b"], sim)
        result = _deduplicate_phrases(counter, emb, threshold=0.85)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Test: _compute_tfidf
# ---------------------------------------------------------------------------


class TestComputeTfidf:
    def test_basic_tfidf(self) -> None:
        from krm.phase_2b_extract_skills import _compute_tfidf

        vac_map = {
            "v1": ["анализ", "хроматография"],
            "v2": ["анализ", "спектроскопия"],
            "v3": ["анализ"],
        }
        lemma_map = {"анализ": "анализ", "хроматография": "хроматография",
                      "спектроскопия": "спектроскопия"}

        weights = _compute_tfidf(vac_map, lemma_map)

        assert ("v1", "анализ") in weights
        assert ("v2", "анализ") in weights
        assert ("v3", "анализ") in weights

        w_common = weights[("v2", "анализ")]
        w_rare = weights[("v2", "спектроскопия")]
        assert w_rare > w_common, f"Expected rare phrase weight ({w_rare}) > common ({w_common})"

    def test_single_vacancy(self) -> None:
        from krm.phase_2b_extract_skills import _compute_tfidf

        vac_map = {"v1": ["анализ", "данные"]}
        lemma_map = {"анализ": "анализ", "данные": "данные"}

        weights = _compute_tfidf(vac_map, lemma_map)
        assert len(weights) == 2
        assert weights[("v1", "анализ")] == weights[("v1", "данные")]


# ---------------------------------------------------------------------------
# Test: extract_skills (end-to-end, mocked)
# ---------------------------------------------------------------------------


class TestExtractSkillsPipeline:
    """Happy path: extract_skills with mocked dependencies."""

    def _config(self) -> MagicMock:
        config = MagicMock()
        type(config).output_dir = PropertyMock(return_value=Path("/tmp/test_skills"))
        type(config).skill_extraction_min_phrase_length = PropertyMock(return_value=1)
        type(config).skill_extraction_max_phrase_length = PropertyMock(return_value=4)
        type(config).skill_extraction_similarity_threshold = PropertyMock(return_value=0.85)
        type(config).skill_extraction_min_doc_frequency = PropertyMock(return_value=1)
        type(config).skill_extraction_max_skills_per_vacancy = PropertyMock(return_value=30)
        type(config).skill_extraction_embedding_model = PropertyMock(
            return_value="deepvk/USER-bge-m3"
        )
        type(config).llm_base_url = PropertyMock(return_value="http://localhost:8080/v1")
        type(config).llm_model = PropertyMock(return_value="Qwen/Qwen3-8B-Instruct")
        type(config).llm_concurrency = PropertyMock(return_value=16)
        type(config).llm_temperature = PropertyMock(return_value=0.0)
        type(config).llm_max_tokens = PropertyMock(return_value=256)
        return config

    def _db(self) -> MagicMock:
        import duckdb

        db = duckdb.connect(":memory:")
        db.execute("""
            CREATE TABLE raw_vacancies (
                id TEXT, run_id TEXT, data JSON,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        return db

    def test_empty_db_returns_empty_output(self) -> None:
        """When DuckDB has no vacancies, return empty DataFrames."""
        from krm.phase_2b_extract_skills import extract_skills

        config = self._config()
        with patch("krm.phase_2b_extract_skills.get_connection") as mock_conn:
            mock_conn.return_value = self._db()
            skills_df, vocab_df = extract_skills(config)

        assert len(skills_df) == 0
        assert len(vocab_df) == 0
        assert list(skills_df.columns) == [
            "vacancy_id", "skill_phrase", "frequency", "tfidf_weight"
        ]
        assert list(vocab_df.columns) == ["skill_phrase", "doc_count"]

    def test_vacancy_with_short_description_skipped(self) -> None:
        """Descriptions shorter than MIN_DESC_CHARS are skipped."""
        from krm.phase_2b_extract_skills import extract_skills

        config = self._config()
        with patch("krm.phase_2b_extract_skills.get_connection") as mock_conn:
            db = self._db()
            db.execute(
                "INSERT INTO raw_vacancies (id, run_id, data) VALUES (?, ?, ?)",
                ["v001", "test", json.dumps({"description": "Коротко", "name": "тест"})],
            )
            mock_conn.return_value = db

            skills_df, vocab_df = extract_skills(config)

        # "Коротко" is only 7 chars < 20 → skipped.
        assert len(skills_df) == 0
        assert len(vocab_df) == 0
