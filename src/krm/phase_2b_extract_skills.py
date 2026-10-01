"""Phase 2b: Russian Skill Phrase Extraction from Vacancy Descriptions.

Extracts skill phrases from Russian vacancy descriptions using either the
shared LLM backend (``krm.lib.llm.LLMClient``) or a deterministic spaCy
noun-phrase fallback, and deduplicates via embedding-based clustering.

Pipeline:
    1. Reads raw vacancies from ``data/krm.duckdb`` (``raw_vacancies`` table).
    2. Filters each description to skill-bearing sections (requirements +
       responsibilities) via ``_filter_to_skill_sections``.
    3. Extracts skills per vacancy via the LLM backend, merging explicit
       ``key_skills``.
    4. Deduplicates globally using ``deepvk/USER-bge-m3`` embeddings +
       cosine-similarity greedy clustering.
    5. Computes TF-IDF weights across the full corpus.
    6. Writes ``extracted_skills.parquet`` and ``skill_vocabulary.parquet``.

Usage:
    from krm.config import Config
    from krm.phase_2b_extract_skills import extract_skills

    config = Config()
    extracted_df, vocab_df = extract_skills(config)
"""

from __future__ import annotations

import asyncio
import json
import math
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, field_validator

from krm.config import Config
from krm.lib.embeddings import Embedder
from krm.lib.io import get_connection
from krm.lib.llm import LLMPrompt, build_llm

# ---------------------------------------------------------------------------
# LLM skill-extraction constants (relocated from ``krm.lib.llm_skills``)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "Ты — эксперт по анализу вакансий. Извлеки из описания вакансии все "
    "профессиональные навыки и компетенции (hard skills), которые требуются от "
    "кандидата. Не включай условия труда, зарплату, бенефиты, график работы, "
    "информацию о компании, требования к образованию или стажу. Каждый навык — "
    "короткая фраза из 1–4 слов на русском. Верни строго JSON без пояснений: "
    '{"skills": ["навык1", "навык2", ...]}'
)

_MAX_DESC_CHARS = 3000
_SKILL_MIN_WORDS = 1
_SKILL_MAX_WORDS = 4


# ---------------------------------------------------------------------------
# Module-level NLP singleton (slow to initialise — cache forever)
# ---------------------------------------------------------------------------

_nlp: Any | None = None

# spaCy Universal Dependencies POS tags for content words in skill phrases.
_SKILL_POS_TAGS: frozenset[str] = frozenset({"NOUN", "PROPN", "ADJ"})

# High-frequency boilerplate phrases dropped at the source (extend iteratively).
_BOILERPLATE: frozenset[str] = frozenset({
    # employment terms & benefits
    "опыт работы",
    "обучение и развитие",
    "развитие сотрудников",
    "аренда квартиры",
    "аренда квартир",
    "добровольное медицинское страхование",
    "медицинское страхование",
    "раз в месяц",
    "раза в месяц",
    "год",
    "месяц",
    "официальное трудоустройство",
    "трудоустройство",
    "социальный пакет",
    "график работы",
    "заработная плата",
    "заработной платы",
    "зарплата",
    "рабочее место",
    "рабочем месте",
    "условия труда",
    "полный рабочий день",
    # qualifications & education
    "высшее образование",
    "среднее образование",
    "высшее профессиональное образование",
    "образование",
    "ученая степень",
    "ученой степени",
    "учёная степень",
    # generic filler
    "знание",
    "навык",
    "навыки",
    "умение",
    "умения",
    "компетенция",
    "компетенции",
    "требование",
    "требования",
    "стаж",
    "стаж работы",
    "опыт",
})


def _get_nlp() -> Any:
    """Return the module-level spaCy ``ru_core_news_lg`` pipeline singleton."""
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load("ru_core_news_lg")
    return _nlp


_section_clf: Any | None = None

_SKILL_SECTIONS: frozenset[str] = frozenset({"requirements", "responsibilities"})


def _get_section_clf() -> Any:
    """Return the Russian vacancy section classifier (rubert-tiny)."""
    global _section_clf
    if _section_clf is None:
        from transformers import pipeline

        _section_clf = pipeline(
            "text-classification",
            model="seninoseno/rubert-tiny-vacancy-information-extractor",
            device=-1,
        )
    return _section_clf


def _filter_to_skill_sections(desc: str) -> str:
    """Keep only requirements/responsibilities sentences, dropping terms/notes
    (benefits, schedule, company PR) which are the main boilerplate source."""
    doc = _get_nlp()(desc)
    clf = _get_section_clf()
    kept: list[str] = []
    for sent in doc.sents:
        text = sent.text.strip()
        if len(text) < 3:
            continue
        try:
            label = clf(text, truncation=True)[0]["label"].lower()
        except Exception:  # noqa: BLE001 — classification failure, keep sentence
            kept.append(text)
            continue
        if label in _SKILL_SECTIONS:
            kept.append(text)
    return " ".join(kept)


def _reset_nlp_caches() -> None:
    """Clear the module-level spaCy pipeline cache (for tests)."""
    global _nlp, _section_clf
    _nlp = None
    _section_clf = None


# ---------------------------------------------------------------------------
# Noun-phrase extraction from the spaCy dependency tree
# ---------------------------------------------------------------------------


def _extract_noun_phrases(doc: Any) -> list[tuple[str, str]]:
    """Extract noun phrases from a spaCy ``Doc`` via dependency-tree walking.

    For each NOUN/PROPN head (excluding function roles), collects its
    adjectival/nominal modifier children and returns the surface form and
    lemmatized form as ``(text, lemma)`` tuples.
    """
    phrases: list[tuple[str, str]] = []
    seen_spans: set[tuple[int, int]] = set()

    for token in doc:
        if token.pos_ not in ("NOUN", "PROPN"):
            continue

        # Walk the subtree rooted at this noun, collecting modifiers.
        subtree: list[Any] = [token]
        stack: list[Any] = [token]
        visited: set[int] = {token.i}
        while stack:
            head = stack.pop()
            for child in head.children:
                if child.dep_ == "conj":
                    continue
                if child.i in visited:
                    continue
                visited.add(child.i)
                is_content = child.pos_ in _SKILL_POS_TAGS or (
                    child.pos_ == "VERB" and child.dep_ == "amod"
                )
                if is_content:
                    subtree.append(child)
                if child.pos_ == "ADJ" or (child.pos_ == "VERB" and child.dep_ == "amod"):
                    stack.append(child)
                elif child.pos_ in ("NOUN", "PROPN") and child.dep_ == "nmod":
                    stack.append(child)

        subtree.sort(key=lambda t: t.i)
        span = (min(t.i for t in subtree), max(t.i for t in subtree))
        if span in seen_spans:
            continue
        seen_spans.add(span)

        text = " ".join(t.text for t in subtree).lower()
        lemma = " ".join(t.lemma_ for t in subtree).lower()
        if not text or text in _BOILERPLATE or lemma in _BOILERPLATE:
            continue
        phrases.append((text, lemma))

    return phrases


# ---------------------------------------------------------------------------
# Phrase filtering
# ---------------------------------------------------------------------------


def _filter_phrases(
    phrases: list[tuple[str, str]],
    min_words: int,
    max_words: int,
) -> list[tuple[str, str]]:
    """Keep only phrases whose word count is within the configured range.

    Args:
        phrases: ``(text, lemma)`` tuples.
        min_words: Minimum word count (inclusive).
        max_words: Maximum word count (inclusive).

    Returns:
        Filtered list.
    """
    return [
        (text, lemma)
        for text, lemma in phrases
        if min_words <= len(text.split()) <= max_words
    ]


# ---------------------------------------------------------------------------
# Skill extraction (LLM via shared client, or deterministic spaCy fallback)
# ---------------------------------------------------------------------------


class SkillsResponse(BaseModel):
    """Structured skill list returned by the LLM backend."""

    skills: list[str] = []

    @field_validator("skills", mode="before")
    @classmethod
    def _normalize_skills(cls, v: Any) -> list[str]:
        if not isinstance(v, list):
            return []
        result: list[str] = []
        seen: set[str] = set()
        for raw in v:
            if not isinstance(raw, str):
                continue
            s = raw.strip().lower()
            if not s or s in seen:
                continue
            if not (_SKILL_MIN_WORDS <= len(s.split()) <= _SKILL_MAX_WORDS):
                continue
            seen.add(s)
            result.append(s)
        return result


def _extract_skills_llm(descriptions: list[str], config: Config) -> list[list[str]]:
    """Extract skill lists per description via the shared :class:`LLMClient`.

    A ``None`` result (total failure for that item) maps to an empty list,
    matching the legacy ``_parse_skills`` behaviour.
    """
    client = build_llm(config)
    prompts = [
        LLMPrompt(system=_SYSTEM_PROMPT, user=desc[:_MAX_DESC_CHARS])
        for desc in descriptions
    ]
    results = asyncio.run(client.run_many(prompts, SkillsResponse))
    return [result.skills if result is not None else [] for result in results]


def _extract_skills_noun_phrases(
    descriptions: list[str],
) -> list[list[tuple[str, str]]]:
    """Deterministic fallback: spaCy noun phrases as ``(text, lemma)`` tuples."""
    nlp = _get_nlp()
    per_desc: list[list[tuple[str, str]]] = []
    for desc in descriptions:
        phrases = _extract_noun_phrases(nlp(desc))
        per_desc.append(_filter_phrases(phrases, _SKILL_MIN_WORDS, _SKILL_MAX_WORDS))
    return per_desc


def _extract_skills_per_desc(
    descriptions: list[str],
    config: Config,
) -> list[list[tuple[str, str]]]:
    """Dispatch LLM vs deterministic extraction into a uniform per-desc shape."""
    if config.use_llm_phase_2b:
        llm_skills = _extract_skills_llm(descriptions, config)
        return [[(s, s) for s in skills] for skills in llm_skills]
    return _extract_skills_noun_phrases(descriptions)


# ---------------------------------------------------------------------------
# Deduplication (embedding clustering)
# ---------------------------------------------------------------------------


def _deduplicate_phrases(
    phrase_counts: Counter[tuple[str, str]],
    embedder: Embedder,
    threshold: float,
) -> list[tuple[str, str, int]]:
    """Deduplicate similar phrases using cosine-similarity greedy clustering.

    Each unique phrase is embedded. Phrases with cosine similarity ≥
    ``threshold`` are merged into a cluster, and the most frequent
    surface form is kept as the canonical representation.

    Args:
        phrase_counts: ``Counter[(text, lemma)]`` mapping each unique
            phrase to its global frequency.
        embedder: Initialised :class:`~krm.lib.embeddings.Embedder`.
        threshold: Cosine-similarity threshold for merge (0.0–1.0).

    Returns:
        List of ``(canonical_text, canonical_lemma, merged_frequency)``.
    """
    unique = list(phrase_counts.keys())
    n = len(unique)

    if n == 0:
        return []
    if n == 1:
        t, lemma = unique[0]
        return [(t, lemma, phrase_counts[(t, lemma)])]

    # Embed texts (use lemma for semantic comparability).
    texts_to_embed = [lemma for _, lemma in unique]
    embeddings = embedder.encode(texts_to_embed)  # already L2-normalised

    assigned: list[bool] = [False] * n
    clusters: list[list[int]] = []

    for i in range(n):
        if assigned[i]:
            continue
        cluster = [i]
        assigned[i] = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim >= threshold:
                cluster.append(j)
                assigned[j] = True
        clusters.append(cluster)

    result: list[tuple[str, str, int]] = []
    for cluster in clusters:
        # Pick the most frequent surface form in this cluster.
        best_idx = max(cluster, key=lambda idx: phrase_counts[unique[idx]])
        best_text, best_lemma = unique[best_idx]
        total_freq = sum(phrase_counts[unique[idx]] for idx in cluster)
        result.append((best_text, best_lemma, total_freq))

    return result


# ---------------------------------------------------------------------------
# TF-IDF computation
# ---------------------------------------------------------------------------


def _compute_tfidf(
    vac_phrase_map: dict[str, list[str]],  # vacancy_id → list of lemmatized phrases
    canonical_lemma_map: dict[str, str],  # lemma → canonical text
) -> dict[tuple[str, str], float]:
    """Compute per-vacancy TF-IDF weights for each (vacancy_id, phrase) pair.

    TF = 1 / N (one occurrence per phrase per vacancy).
    IDF = log(1 + total_vacancies / doc_count).

    Args:
        vac_phrase_map: ``{vacancy_id: [lemma1, lemma2, ...]}``.
        canonical_lemma_map: Mapping from dedup lemma → canonical text.

    Returns:
        ``{(vacancy_id, canonical_text): tfidf_weight}``.
    """
    total_vacancies = len(vac_phrase_map)

    # Compute document frequency per phrase (lemma).
    doc_freq: Counter[str] = Counter()
    for phrases in vac_phrase_map.values():
        unique_lemmas = set(phrases)
        doc_freq.update(unique_lemmas)

    weights: dict[tuple[str, str], float] = {}
    for vac_id, phrases in vac_phrase_map.items():
        n_phrases = len(phrases)
        if n_phrases == 0:
            continue
        tf = 1.0 / n_phrases
        for lemma in phrases:
            canonical = canonical_lemma_map.get(lemma, lemma)
            idf = math.log(1 + total_vacancies / max(doc_freq[lemma], 1))
            weights[(vac_id, canonical)] = round(tf * idf, 6)

    return weights


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_skills(
    config: Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract Russian skill phrases from raw vacancy descriptions.

    Full pipeline:

    1. Reads ``raw_vacancies`` from ``data/krm.duckdb``.
    2. Filters each description to skill-bearing sections (requirements +
       responsibilities) via ``_filter_to_skill_sections``.
    3. Extracts skills per vacancy via the LLM backend or the deterministic
       spaCy fallback (dispatching on ``config.use_llm_phase_2b``), merging
       explicit ``key_skills``.
    4. Deduplicates globally via embedding clustering at
       ``similarity_threshold`` cosine.
    5. Filters low-frequency phrases (< ``min_doc_frequency`` vacancies).
    6. Computes TF-IDF weights.
    7. Writes ``extracted_skills.parquet`` and ``skill_vocabulary.parquet``
       to ``config.output_dir``.

    Args:
        config: Pipeline configuration providing ``output_dir`` and
            ``skill_extraction_*`` settings.

    Returns:
        ``(extracted_skills_df, skill_vocabulary_df)``.

        ``extracted_skills.parquet`` columns:
            ============== ===========================================
            ``vacancy_id`` str — source vacancy identifier.
            ``skill_phrase`` str — canonical skill phrase (Russian).
            ``frequency``    int — always 1 per vacancy.
            ``tfidf_weight`` float — TF-IDF weight for this phrase.
            ============== ===========================================

        ``skill_vocabulary.parquet`` columns:
            =============== =========================================
            ``skill_phrase`` str — canonical skill phrase.
            ``doc_count``    int — number of vacancies containing it.
            =============== =========================================
    """
    sim_threshold = config.skill_extraction_similarity_threshold
    min_doc_freq = config.skill_extraction_min_doc_frequency
    max_skills_per_vac = config.skill_extraction_max_skills_per_vacancy
    output_dir = config.output_dir

    # ------------------------------------------------------------------
    # 1. Load raw vacancies from DuckDB
    # ------------------------------------------------------------------
    conn = get_connection(config.db_path)
    try:
        rows = conn.execute(
            "SELECT id, data FROM raw_vacancies"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        logger.info("[Phase 2b] No vacancies in raw_vacancies — returning empty output")
        empty_skills, empty_vocab = _empty_output()
        _write_outputs(empty_skills, empty_vocab, output_dir)
        return empty_skills, empty_vocab

    logger.info(f"[Phase 2b] Loaded {len(rows)} raw vacancies from DuckDB")

    # ------------------------------------------------------------------
    # 2. Extract skills per vacancy (LLM client or deterministic spaCy)
    # ------------------------------------------------------------------
    # vac_phrases: {vacancy_id: [(skill, skill), ...]}
    vac_phrases: dict[str, list[tuple[str, str]]] = {}
    skipped_empty = 0
    skipped_short = 0

    MIN_DESC_CHARS = 20  # skip descriptions shorter than this

    # Gather usable descriptions (section-filtered) + explicit key_skills.
    descs: list[tuple[str, str, list[str]]] = []
    for vac_id, data_json in rows:
        try:
            data = json.loads(data_json)
        except (json.JSONDecodeError, TypeError):
            skipped_empty += 1
            continue

        desc = (data.get("description") or "").strip()
        if not desc or len(desc) < MIN_DESC_CHARS:
            skipped_short += 1
            continue

        # Keep only skill-bearing sections (requirements + responsibilities);
        # drops benefits/schedule/company-PR boilerplate.
        try:
            desc = _filter_to_skill_sections(desc)
        except Exception:  # noqa: BLE001 — keep full description on failure
            pass
        if not desc:
            skipped_empty += 1
            continue

        key_skills: list[str] = []
        for item in data.get("key_skills") or []:
            name = (item.get("name") or "").strip() if isinstance(item, dict) else str(item).strip()
            if name:
                key_skills.append(name.lower())

        descs.append((str(vac_id), desc, key_skills))

    # Batch skill extraction over the filtered descriptions (LLM or spaCy).
    per_desc_phrases: list[list[tuple[str, str]]] = []
    if descs:
        per_desc_phrases = _extract_skills_per_desc(
            [d for _, d, _ in descs], config
        )

    for (vac_id, _desc, key_skills), phrases in zip(descs, per_desc_phrases):
        merged: list[tuple[str, str]] = []
        seen: set[str] = set()
        for text, lemma in [*phrases, *((s, s) for s in key_skills)]:
            text = text.strip().lower()
            if not text or text in seen or len(text.split()) > 8:
                continue
            seen.add(text)
            merged.append((text, lemma))
        if merged:
            vac_phrases[vac_id] = merged

    n_with_phrases = len(vac_phrases)
    msg = (
        f"[Phase 2b] Extracted phrases from {n_with_phrases} vacancies"
    )
    if skipped_empty:
        msg += f" (skipped {skipped_empty} empty/invalid)"
    if skipped_short:
        msg += f" (skipped {skipped_short} too short < {MIN_DESC_CHARS} chars)"
    logger.info(msg)

    if not vac_phrases:
        empty_skills, empty_vocab = _empty_output()
        _write_outputs(empty_skills, empty_vocab, output_dir)
        return empty_skills, empty_vocab

    # ------------------------------------------------------------------
    # 3. Global deduplication via embedding clustering
    # ------------------------------------------------------------------
    # Collect all unique phrases across all vacancies.
    global_counter: Counter[tuple[str, str]] = Counter()
    for phrases in vac_phrases.values():
        # Count each unique phrase once per vacancy for doc frequency,
        # but for global dedup we count total occurrences.
        for p in set(phrases):
            global_counter[p] += 1

    n_unique_raw = len(global_counter)
    logger.info(f"[Phase 2b] {n_unique_raw} unique raw phrases before dedup")

    embedder = Embedder(model_name=config.skill_extraction_embedding_model)

    deduped = _deduplicate_phrases(global_counter, embedder, sim_threshold)
    logger.info(f"[Phase 2b] {len(deduped)} phrases after dedup (threshold={sim_threshold})")

    # Build lemma → canonical text mapping.
    lemma_to_canonical: dict[str, str] = {}
    for text, lemma, _ in deduped:
        if lemma not in lemma_to_canonical:
            lemma_to_canonical[lemma] = text

    # ------------------------------------------------------------------
    # 4. Map vacancies to canonical phrases
    # ------------------------------------------------------------------
    # Build a reverse mapping: raw (text, lemma) → canonical lemma.
    raw_to_canonical: dict[tuple[str, str], str] = {}

    # For each unique raw phrase, find its canonical lemma via nearest dedup cluster.
    # We need to embed all raw phrases and match them to canonical phrases.
    raw_unique = list({p for phrases in vac_phrases.values() for p in phrases})
    if len(raw_unique) > 0:
        raw_embeddings = embedder.encode([lemma for _, lemma in raw_unique])
        canonical_lemmas = [lemma for _, lemma, _ in deduped]
        canonical_embs = embedder.encode(canonical_lemmas)

        for i, (raw_text, raw_lemma) in enumerate(raw_unique):
            # Find nearest canonical phrase by cosine similarity.
            sims = np.dot(raw_embeddings[i], canonical_embs.T)
            best_j = int(np.argmax(sims))
            raw_to_canonical[(raw_text, raw_lemma)] = canonical_lemmas[best_j]

    # Replace raw phrases with canonical lemmas per vacancy.
    vac_canonical_lemmas: dict[str, list[str]] = {}
    for vac_id, phrases in vac_phrases.items():
        canonical_list: list[str] = []
        seen: set[str] = set()
        for p in phrases:
            canonical_lemma = raw_to_canonical.get(p, p[1])
            canonical_text = lemma_to_canonical.get(canonical_lemma, canonical_lemma)
            if canonical_text not in seen:
                canonical_list.append(canonical_lemma)
                seen.add(canonical_text)
            if len(canonical_list) >= max_skills_per_vac:
                break
        vac_canonical_lemmas[vac_id] = canonical_list

    # Filter by min_doc_frequency
    doc_phrase_counts: Counter[str] = Counter()
    for lemmas in vac_canonical_lemmas.values():
        doc_phrase_counts.update(set(lemmas))

    kept_lemmas: set[str] = {
        lemma
        for lemma, count in doc_phrase_counts.items()
        if count >= min_doc_freq
    }

    # Re-filter vacancies to only keep frequent phrases.
    vac_filtered: dict[str, list[str]] = {}
    for vac_id, lemmas in vac_canonical_lemmas.items():
        filtered_lemmas = [lemma for lemma in lemmas if lemma in kept_lemmas]
        if filtered_lemmas:
            vac_filtered[vac_id] = filtered_lemmas

    removed_count = sum(
        len(lemmas) for lemmas in vac_canonical_lemmas.values()
    ) - sum(len(lemmas) for lemmas in vac_filtered.values())
    logger.info(
        "[Phase 2b] Kept %d phrases appearing in >=%d vacancies (removed %d low-frequency)"
        % (len(kept_lemmas), min_doc_freq, removed_count)
    )

    # ------------------------------------------------------------------
    # 5. Compute TF-IDF weights
    # ------------------------------------------------------------------
    tfidf_weights = _compute_tfidf(vac_filtered, lemma_to_canonical)

    # ------------------------------------------------------------------
    # 6. Build output DataFrames
    # ------------------------------------------------------------------
    skill_rows: list[dict[str, Any]] = []
    for vac_id, lemmas in vac_filtered.items():
        for lemma in lemmas:
            canonical = lemma_to_canonical.get(lemma, lemma)
            weight = tfidf_weights.get((vac_id, canonical), 0.0)
            skill_rows.append(
                {
                    "vacancy_id": vac_id,
                    "skill_phrase": canonical,
                    "frequency": 1,
                    "tfidf_weight": weight,
                }
            )

    extracted_df = pd.DataFrame(
        skill_rows,
        columns=["vacancy_id", "skill_phrase", "frequency", "tfidf_weight"],
    )

    # Vocabulary: canonical phrase → doc_count
    vocab_doc_counts: dict[str, int] = {}
    for lemmas in vac_filtered.values():
        for lemma in set(lemmas):
            canonical = lemma_to_canonical.get(lemma, lemma)
            vocab_doc_counts[canonical] = vocab_doc_counts.get(canonical, 0) + 1

    vocab_rows = [
        {"skill_phrase": phrase, "doc_count": count}
        for phrase, count in sorted(vocab_doc_counts.items(), key=lambda x: -x[1])
    ]
    vocab_df = pd.DataFrame(vocab_rows, columns=["skill_phrase", "doc_count"])

    # ------------------------------------------------------------------
    # 7. Write output
    # ------------------------------------------------------------------
    _write_outputs(extracted_df, vocab_df, output_dir)

    logger.info(
        "[Phase 2b] Wrote %d skill-vacancy pairs (%d unique phrases) to %s"
        % (len(extracted_df), len(vocab_df), output_dir)
    )

    return extracted_df, vocab_df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _empty_output() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return empty DataFrames with correct schemas."""
    skills = pd.DataFrame(
        columns=["vacancy_id", "skill_phrase", "frequency", "tfidf_weight"]
    )
    vocab = pd.DataFrame(columns=["skill_phrase", "doc_count"])
    return skills, vocab


def _write_outputs(
    skills_df: pd.DataFrame,
    vocab_df: pd.DataFrame,
    output_dir: Any,
) -> None:
    """Write extracted_skills.parquet and skill_vocabulary.parquet."""
    from pathlib import Path

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    skills_df.to_parquet(out / "extracted_skills.parquet", index=False)
    vocab_df.to_parquet(out / "skill_vocabulary.parquet", index=False)
