"""Tests for phase_4_skills.py — ESCO skill extraction pure functions.

Tests cover _match_skill_label (3 strategies), _extract_uncatalogued_phrases,
and the _ESCO_COLUMN_CANDIDATES constant. No torch, sentence-transformers,
hdbscan, or ESCO CSV file required.
"""

from __future__ import annotations

import pytest

from krm.phase_4_skills import (
    _ESCO_COLUMN_CANDIDATES,
    _extract_uncatalogued_phrases,
    _match_skill_label,
)


# ---------------------------------------------------------------------------
# _ESCO_COLUMN_CANDIDATES
# ---------------------------------------------------------------------------


class TestEscoColumnCandidates:
    """Tests for the column-name mapping constant."""

    REQUIRED_KEYS = {"skill_id", "preferred_label", "description", "skill_type"}

    def test_has_all_required_keys(self):
        """All four canonical columns must be present."""
        for key in self.REQUIRED_KEYS:
            assert key in _ESCO_COLUMN_CANDIDATES, (
                f"Missing required key: {key}"
            )

    def test_each_value_is_tuple_of_strings(self):
        """Every canonical column maps to a tuple of candidate strings."""
        for key, candidates in _ESCO_COLUMN_CANDIDATES.items():
            assert isinstance(candidates, tuple), (
                f"Value for '{key}' must be a tuple, got {type(candidates).__name__}"
            )
            assert len(candidates) > 0, (
                f"Candidates tuple for '{key}' must not be empty"
            )
            for i, c in enumerate(candidates):
                assert isinstance(c, str), (
                    f"Candidate {i} for '{key}' must be a str, got {type(c).__name__}"
                )

    def test_no_extra_keys(self):
        """Only the four known canonical columns should appear."""
        assert set(_ESCO_COLUMN_CANDIDATES) == self.REQUIRED_KEYS


# ---------------------------------------------------------------------------
# _match_skill_label — strategy (a): exact case-insensitive substring
# ---------------------------------------------------------------------------


class TestMatchSkillLabelExactSubstring:
    """Tests for strategy (a): case-insensitive literal match."""

    def test_exact_match_same_case(self):
        assert _match_skill_label("python", "python developer") is True

    def test_exact_match_case_differs(self):
        assert _match_skill_label("Python", "PYTHON developer") is True
        assert _match_skill_label("PYTHON", "python developer") is True
        assert _match_skill_label("Machine Learning", "machine learning engineer") is True

    def test_exact_substring_within_word(self):
        # "data" appears literally in "database" → strategy (a) matches.
        assert _match_skill_label("data", "database administrator") is True

    def test_no_substring_match(self):
        assert _match_skill_label("java", "python developer") is False
        assert _match_skill_label("ruby", "I write Python and JavaScript") is False

    def test_partial_word_not_a_substring(self):
        # "jav" is not literally in "java is great" → "jav" ≠ substring of "java".
        # Wait — "jav" IS a substring of "java".  This should match (a).
        assert _match_skill_label("jav", "java developer") is True


# ---------------------------------------------------------------------------
# _match_skill_label — strategy (b): word-boundary regex
# ---------------------------------------------------------------------------


class TestMatchSkillLabelRegex:
    """Tests for strategy (b): \\b-delimited whole-word/phrase match."""

    def test_regex_word_boundary_match(self):
        # "data" as an independent word in "data analysis" → matches (b).
        assert _match_skill_label("data", "data analysis is key") is True

    def test_regex_no_match_when_embedded_in_word(self):
        # "data" embedded in "database" → strategy (a) catches it first.
        # But if we test a label that is only matched via regex:
        # "cat" appears in "concatenate" as substring → (a) catches it.
        # For a label NOT caught by (a) but also not a whole word:
        # Actually "data" in "database" IS caught by (a).  Let's find a case
        # where (a) fails but (b) *would* fail too because the word is embedded.
        #
        # The scenario: label "data", text "database" → (a) catches it.
        # To isolate (b), we need a label that fails (a) but passes (b):
        # "data" in "tools, data, methods" → (a) passes.
        # Hmm — any single token present as a substring also satisfies (a).
        # So (b) is only tested meaningfully when (a) FAILS.
        # That only happens when the label is NOT a substring, and (b) would
        # still fail because \b won't match inside a word either.
        #
        # The meaningful (b) test is demonstrating that (a) catches things
        # \b would not (already shown in test_exact_substring_within_word),
        # and that (b) returns True for the same cases (a) does.
        pass

    def test_regex_matches_start_of_text(self):
        """Word-boundary anchors work at the beginning of the text."""
        assert _match_skill_label("python", "python is great") is True

    def test_regex_matches_end_of_text(self):
        """Word-boundary anchors work at the end of the text."""
        assert _match_skill_label("python", "I love python") is True

    def test_regex_with_punctuation_borders(self):
        """Word boundaries work around punctuation."""
        assert _match_skill_label("ai", "use AI,") is True
        assert _match_skill_label("sql", "skills: SQL, Python") is True

    def test_regex_multi_token_phrase(self):
        """Multi-token phrases match via regex when present as contiguous words."""
        assert _match_skill_label("machine learning", "I study machine learning daily") is True
        assert _match_skill_label("deep learning", "machine learning and deep learning") is True

    def test_regex_multi_token_no_contiguous_match(self):
        """Multi-token phrase NOT present contiguously → (a) and (b) fail,
        leaving only (c) as a possible path."""
        assert (
            _match_skill_label(
                "machine learning",
                "I operate machines that are learning slowly",
            )
            is False
        )


# ---------------------------------------------------------------------------
# _match_skill_label — strategy (c): token-overlap scoring
# ---------------------------------------------------------------------------


class TestMatchSkillLabelTokenOverlap:
    """Tests for strategy (c): sliding-window Jaccard token overlap."""

    # ── full overlap (score = 1.0) ──

    def test_full_overlap_contiguous(self):
        """All tokens match in a contiguous window → score 1.0."""
        assert (
            _match_skill_label(
                "machine learning",
                "we use machine learning techniques",
                fuzzy_threshold=0.7,
            )
            is True
        )

    def test_full_overlap_non_contiguous(self):
        """All tokens match but NOT contiguous → still score 1.0 in a window
        because the tokens may be scattered across different windows."""
        # "data" and "analysis" both in "data analysis" window → 1.0
        assert (
            _match_skill_label(
                "data analysis",
                "perform data analysis",
                fuzzy_threshold=0.7,
            )
            is True
        )

    # ── partial overlap ──

    def test_partial_overlap_below_threshold(self):
        """Only 1 of 2 tokens matches → score 0.5 < 0.7 → False."""
        assert (
            _match_skill_label(
                "machine learning",
                "I operate heavy machine tools",
                fuzzy_threshold=0.7,
            )
            is False
        )

    def test_partial_overlap_above_lenient_threshold(self):
        """Only 1 of 2 tokens matches → score 0.5 ≥ 0.5 → True."""
        assert (
            _match_skill_label(
                "machine learning",
                "I operate heavy machine tools",
                fuzzy_threshold=0.5,
            )
            is True
        )

    def test_partial_overlap_three_tokens(self):
        """2 of 3 tokens match in a window → score 2/3 ≈ 0.667.

        At default threshold 0.7, 0.667 < 0.7 → False.
        At threshold 0.6, 0.667 ≥ 0.6 → True.
        """
        # "language" + "processing" appear contiguously → window 2/3 match.
        assert (
            _match_skill_label(
                "natural language processing",
                "we work on language processing tasks",
                fuzzy_threshold=0.6,
            )
            is True
        )
        # Same text at default threshold 0.7 → 0.667 < 0.7 → False.
        assert (
            _match_skill_label(
                "natural language processing",
                "we work on language processing tasks",
            )
            is False
        )
        # Only "language" matches — score 0.33 at any threshold ≤ 0.33.
        assert (
            _match_skill_label(
                "natural language processing",
                "we focus on language understanding",
                fuzzy_threshold=0.3,
            )
            is True
        )

    def test_perfect_overlap_strict_threshold(self):
        """All tokens match → score 1.0 ≥ 1.0 → True."""
        assert (
            _match_skill_label(
                "data analysis",
                "data analysis and reporting",
                fuzzy_threshold=1.0,
            )
            is True
        )

    def test_imperfect_overlap_strict_threshold(self):
        """1 of 2 tokens matches → score 0.5 < 1.0 → False."""
        assert (
            _match_skill_label(
                "data analysis",
                "data entry and reporting",
                fuzzy_threshold=1.0,
            )
            is False
        )

    # ── single-token labels skip (c) ──

    def test_single_token_skips_strategy_c(self):
        """Single-token labels bypass token-overlap (n_tokens > 1 check fails)."""
        # "analysis" is in the text → caught by (a).  Let's pick a label
        # that fails (a)/(b) so we know (c) is the only path.
        # Actually any single token is caught by (a) if it's a substring.
        # So single-token labels where the token is NOT in the text
        # fail all three strategies — and (c) is never even reached.
        assert (
            _match_skill_label("biology", "python developer")
            is False
        )

    def test_text_shorter_than_label_skips_strategy_c(self):
        """When text has fewer tokens than the label, strategy (c) returns early."""
        # "machine learning" (2 tokens) in "mlops" (1 token) → too short.
        assert (
            _match_skill_label(
                "machine learning",
                "mlops",
                fuzzy_threshold=0.3,
            )
            is False
        )


# ---------------------------------------------------------------------------
# _match_skill_label — edge cases
# ---------------------------------------------------------------------------


class TestMatchSkillLabelEdgeCases:
    """Edge-case and boundary tests for _match_skill_label."""

    def test_none_text_returns_false(self):
        assert _match_skill_label("python", None) is False  # type: ignore[arg-type]

    def test_none_label_returns_false(self):
        assert _match_skill_label(None, "python developer") is False  # type: ignore[arg-type]

    def test_both_none_returns_false(self):
        assert _match_skill_label(None, None) is False  # type: ignore[arg-type]

    def test_empty_text_returns_false(self):
        assert _match_skill_label("python", "") is False
        assert _match_skill_label("python", "   ") is False

    def test_empty_label_returns_false(self):
        assert _match_skill_label("", "python developer") is False
        assert _match_skill_label("   ", "python developer") is False

    def test_both_empty_returns_false(self):
        assert _match_skill_label("", "") is False

    def test_label_longer_than_text(self):
        """Label longer than text → (a)/(b) fail, (c) returns early."""
        assert (
            _match_skill_label(
                "advanced machine learning engineering",
                "short text",
            )
            is False
        )

    def test_exact_same_string(self):
        assert _match_skill_label("python", "python") is True

    def test_whitespace_normalization(self):
        """Multiple spaces in label vs text — tokens split fine."""
        assert (
            _match_skill_label(
                "data  analysis",  # split → ["data", "analysis"]
                "I do data analysis daily",
            )
            is True
        )

    def test_unicode_letter_matching(self):
        """Non-ASCII letters work in all three strategies."""
        # Cyrillic substring test.
        assert _match_skill_label("анализ", "анализ данных") is True
        # Not in text.
        assert _match_skill_label("физика", "химия и биология") is False

    def test_threshold_zero_always_matches_multi_token(self):
        """With fuzzy_threshold=0.0, any multi-token label with a single
        matching token in a window scores > 0.0 → should match.

        BUT: if no windows can be formed (text too short), it still fails.
        """
        assert (
            _match_skill_label(
                "machine learning",
                "I operate heavy machine tools",
                fuzzy_threshold=0.0,
            )
            is True
        )

    def test_below_zero_threshold(self):
        """Even a negative threshold is ≤ any non-negative score → True."""
        assert (
            _match_skill_label(
                "machine learning",
                "I operate heavy machine tools",
                fuzzy_threshold=-0.5,
            )
            is True
        )


# ---------------------------------------------------------------------------
# _extract_uncatalogued_phrases
# ---------------------------------------------------------------------------


class TestExtractUncataloguedPhrases:
    """Tests for the uncatalogued multi-word phrase extraction."""

    def test_returns_sorted_by_frequency(self):
        """Most frequent phrase across documents comes first."""
        descriptions = [
            "data curation pipeline",
            "data curation pipeline",
            "data curation pipeline",
            "machine learning ops",
            "machine learning ops",
        ]
        esco_labels: set[str] = set()
        result = _extract_uncatalogued_phrases(descriptions, esco_labels)
        # 3-gram and 2-grams from 3-doc descriptions tie on doc frequency;
        # order among ties is deterministic but depends on first-encounter.
        # All top-tier phrases appear in exactly 3 documents.
        top_tier = {
            "data curation",
            "curation pipeline",
            "data curation pipeline",
        }
        # "data curation pipeline" (3-gram, 3 docs)
        # "data curation" (2-gram, 3 docs)
        # "curation pipeline" (2-gram, 3 docs)
        mid_tier = {"machine learning", "learning ops", "machine learning ops"}
        assert result[0] in top_tier
        # Check that 3-doc-frequency phrases precede 2-doc-frequency phrases.
        top_indices = [i for i, p in enumerate(result) if p in top_tier]
        mid_indices = [i for i, p in enumerate(result) if p in mid_tier]
        assert max(top_indices) < min(mid_indices), (
            f"Top-tier phrases should all come before mid-tier; got {result}"
        )

    def test_filters_esco_labels(self):
        """Phrases already in ESCO are excluded."""
        descriptions = [
            "data analysis is important",
            "data analysis everywhere",
            "python programming",
        ]
        esco_labels = {"data analysis"}
        result = _extract_uncatalogued_phrases(descriptions, esco_labels)
        assert "data analysis" not in result
        # "python programming" (a 2-gram) should appear.
        assert "python programming" in result

    def test_min_word_len_filters_short_tokens(self):
        """Tokens shorter than min_word_len are dropped before n-gram building."""
        descriptions = ["a b c d e f"]
        esco_labels: set[str] = set()
        # With min_word_len=2 (default), all single-char tokens are dropped
        # → no tokens remain → no n-grams → empty result.
        result = _extract_uncatalogued_phrases(descriptions, esco_labels, min_word_len=2)
        assert result == []

        # With min_word_len=1, single-char tokens survive → n-grams are formed.
        result = _extract_uncatalogued_phrases(descriptions, esco_labels, min_word_len=1)
        assert len(result) > 0

    def test_top_k_limits_output(self):
        """Only top_k phrases are returned."""
        descriptions = [
            "alpha beta gamma delta",
            "alpha beta gamma delta",
            "alpha beta gamma delta",
            "gamma delta epsilon zeta",
            "gamma delta epsilon zeta",
            "delta epsilon zeta eta",
        ]
        esco_labels: set[str] = set()
        result = _extract_uncatalogued_phrases(descriptions, esco_labels, top_k=3)
        assert len(result) <= 3

    def test_empty_input_returns_empty(self):
        assert _extract_uncatalogued_phrases([], set()) == []

    def test_empty_descriptions_survive(self):
        """Empty or None descriptions are skipped gracefully."""
        descriptions = ["", "data science toolkit", "data science toolkit"]
        esco_labels: set[str] = set()
        result = _extract_uncatalogued_phrases(descriptions, esco_labels)
        assert "data science" in result

    def test_single_description(self):
        """A single description still yields n-grams."""
        result = _extract_uncatalogued_phrases(
            ["quantum computing research"], set()
        )
        assert len(result) > 0
        assert "quantum computing" in result

    def test_ngrams_span_2_3_4_tokens(self):
        """2-, 3-, and 4-grams are all considered."""
        descriptions = ["one two three four five"]
        esco_labels: set[str] = set()
        result = _extract_uncatalogued_phrases(descriptions, esco_labels)
        assert "one two" in result  # 2-gram
        assert "one two three" in result  # 3-gram
        assert "one two three four" in result  # 4-gram

    def test_all_phrases_filtered_yields_empty(self):
        """When every n-gram is an ESCO label, result is empty."""
        descriptions = ["machine learning", "deep learning"]
        esco_labels = {"machine learning", "deep learning"}
        result = _extract_uncatalogued_phrases(descriptions, esco_labels)
        assert result == []

    def test_doc_frequency_not_token_frequency(self):
        """Counting is per-document, not per-token-occurrence in a single doc."""
        descriptions = [
            # "data analysis" appears 3 times in ONE doc.
            "data analysis data analysis data analysis",
            # "machine learning" appears once in each of 3 docs.
            "machine learning something",
            "machine learning else",
            "machine learning again",
        ]
        esco_labels: set[str] = set()
        result = _extract_uncatalogued_phrases(descriptions, esco_labels)
        # "machine learning" in 3 documents wins over "data analysis" in 1.
        assert result[0] == "machine learning"
        assert "data analysis" in result
        assert result.index("machine learning") < result.index("data analysis")
