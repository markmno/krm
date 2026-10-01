"""Phase 2.5: Broad Role Characteristics Extraction from Vacancy Descriptions.

Uses zero-shot natural language inference (NLI) to score each vacancy
against 7 unified characteristic hypotheses that describe broad role-level
requirements — experimental work, domain knowledge, management, scientific
communication, data analysis, and computational methods.

Years-of-experience is extracted separately via regex from the vacancy
description text.

**Position in the pipeline**: runs after classification (Phase 2) and
before role discovery (Phase 3). Results feed into temporal validation
(Phase 7).

**Input**: ``classified.parquet`` (from Phase 2).
**Output**: ``characteristics.parquet`` — one row per vacancy–characteristic
pair where the NLI confidence exceeds the configured threshold.

Usage::

    from krm.config import Config
    from krm.phase_2_5_characteristics import run_pipeline

    config = Config()
    df = run_pipeline(config)

Output schema
-------------

.. list-table::
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``vacancy_id``
     - str
     - HH.ru vacancy identifier.
   * - ``characteristic_id``
     - str
      - Hypothesis identifier from config (e.g. ``"experimental"``).
   * - ``characteristic_label``
     - str
     - Russian label (e.g. *Лабораторная работа*).
   * - ``confidence``
     - float
     - NLI entailment score in [0, 1].
   * - ``extracted_title``
     - str
     - Vacancy title from classified Parquet.
   * - ``experience_years``
     - float
     - Regex-extracted years of experience (NaN if not found).
"""

from __future__ import annotations

import asyncio
import re
from contextlib import suppress
from typing import Any

import pandas as pd
from pydantic import BaseModel, field_validator
from tqdm import tqdm

from krm.config import Config
from krm.lib.io import get_connection, read_parquet, write_parquet
from krm.lib.llm import LLMPrompt, build_llm

# ---------------------------------------------------------------------------
# Regex-based experience extraction
# ---------------------------------------------------------------------------


def extract_experience(description: str, patterns: list[str]) -> float | None:
    """Extract years of required experience from a vacancy description.

    Applies each regex pattern from ``config.experience_patterns`` to the
    description text. The first successful match is used. Both single
    numbers (``"3 года"``) and ranges (``"3–5 лет"``) are supported.
    Ranges are averaged.

    Edge cases handled::

        "опыт работы от 3 лет"
        "требуемый опыт 5–7 лет"
        "стаж работы 2 года"
        "опыт работы 3–5 лет"

    Args:
        description: Raw vacancy description text.
        patterns: List of regex strings from
            ``config.experience_patterns``.

    Returns:
        Years as a float, or ``None`` if no pattern matched.
    """
    if not description or not isinstance(description, str):
        return None

    description_lower = description.lower()

    for pat_str in patterns:
        m = re.search(pat_str, description_lower)
        if not m:
            continue

        groups = m.groups()
        if not groups:
            continue

        # First numeric group is always present.
        try:
            first_val = float(groups[0])
        except (ValueError, TypeError):
            continue

        # Check if there is a second numeric group (range).
        second_val: float | None = None
        if len(groups) >= 2 and groups[1] and groups[1].isdigit():
            with suppress(ValueError, TypeError):
                second_val = float(groups[1])

        if second_val is not None:
            return (first_val + second_val) / 2.0
        return first_val

    return None


# ---------------------------------------------------------------------------
# Zero-shot NLI characteristic scoring
# ---------------------------------------------------------------------------


def _build_characteristic_classifier(config: Config) -> Any:
    """Initialise the zero-shot classification pipeline.

    Lazy-imports ``transformers`` to avoid the dependency at module
    load time (consistent with Phase 2 and Phase 5 patterns).

    Args:
        config: Pipeline configuration providing
            ``characteristics_model`` and ``characteristics_device``.

    Returns:
        A HuggingFace ``zero-shot-classification`` pipeline.
    """
    from transformers import pipeline

    return pipeline(
        "zero-shot-classification",
        model=config.characteristics_model,
        device=config.characteristics_device,
    )


# ---------------------------------------------------------------------------
# LLM structured-output schema and prompt building
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = (
    "Ты — эксперт по анализу вакансий в научной и инженерной сфере. Оцени, "
    "насколько каждая из перечисленных ниже характеристик выражена в описании "
    "вакансии. Верни строго JSON со списком оценок: для каждой характеристики "
    "укажи её идентификатор (characteristic_id) и уверенность (confidence) в "
    "диапазоне от 0.0 до 1.0, где 0 означает полное отсутствие характеристики, "
    "а 1 — что она является явным центральным требованием. Используй только "
    "идентификаторы из списка."
)


class CharacteristicScore(BaseModel):
    """A single characteristic confidence score produced by the LLM."""

    characteristic_id: str
    confidence: float

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        """Clamp the self-assessed confidence into [0, 1]."""
        return min(max(value, 0.0), 1.0)


class CharacteristicScoresResponse(BaseModel):
    """LLM structured output: confidence per characteristic for one vacancy.

    Entries whose ``characteristic_id`` is not a configured hypothesis are
    ignored by the caller — the allow-list lives in config, not in the schema,
    so it is enforced at the dispatch boundary in
    :func:`_extract_characteristics_llm`.
    """

    characteristics: list[CharacteristicScore]


def _build_characteristic_prompt(
    hypotheses: dict[str, str],
    labels_ru: dict[str, str],
    description: str,
) -> LLMPrompt:
    """Build one LLM prompt listing all characteristic hypotheses as the schema.

    Args:
        hypotheses: Mapping ``characteristic_id → hypothesis`` from config.
        labels_ru: Mapping ``characteristic_id → Russian label`` from config.
        description: The (sanitised, non-empty) vacancy description.

    Returns:
        A single :class:`LLMPrompt` whose user payload lists the 7 hypotheses
        and asks for a confidence score per characteristic.
    """
    definitions = "\n".join(
        f"- {cid} ({labels_ru[cid]}): {text}" for cid, text in hypotheses.items()
    )
    user = (
        "Описание вакансии:\n"
        '"""\n'
        f"{description}\n"
        '"""\n\n'
        "Характеристики для оценки:\n"
        f"{definitions}\n\n"
        "Оцени каждую характеристику по шкале 0–1 и верни JSON со списком оценок."
    )
    return LLMPrompt(system=_LLM_SYSTEM_PROMPT, user=user)


def extract_characteristics(
    config: Config,
    descriptions: list[str],
    vacancy_ids: list[str],
) -> pd.DataFrame:
    """Score each vacancy against all characteristic hypotheses.

    Dispatches on ``config.use_llm_phase_2_5``:

    * ``False`` (default) — zero-shot NLI (bart-large-mnli): each vacancy is
      scored against every hypothesis in a single entailment pass.
    * ``True`` — one LLM call per vacancy over the shared :class:`LLMClient`,
      returning a confidence per characteristic. A vacancy whose LLM call
      totally fails (``None``) gracefully degrades to the NLI result for that
      vacancy only (per-item fallback, never a batch abort).

    Both paths emit one row per vacancy–characteristic pair whose confidence
    meets or exceeds ``config.characteristics_confidence_threshold``.

    Args:
        config: Pipeline configuration (model, hypotheses, thresholds).
        descriptions: List of vacancy description strings (same length
            as ``vacancy_ids``). ``None`` / empty values are treated as
            empty strings.
        vacancy_ids: List of vacancy identifiers corresponding to each
            description.

    Returns:
        DataFrame with columns: ``vacancy_id``, ``characteristic_id``,
        ``characteristic_label``, ``confidence``. May be empty if no
        confidence scores exceed the threshold.
    """
    hypotheses: dict[str, str] = config.characteristic_hypotheses
    labels_ru: dict[str, str] = config.characteristic_labels_ru

    # Sanitise descriptions: convert None/NaN to empty string.
    sanitised: list[str] = [
        d if isinstance(d, str) and d else "" for d in descriptions
    ]

    if config.use_llm_phase_2_5:
        return _extract_characteristics_llm(
            config, sanitised, vacancy_ids, hypotheses, labels_ru
        )

    return _extract_characteristics_nli(
        config, sanitised, vacancy_ids, hypotheses, labels_ru
    )


def _extract_characteristics_nli(
    config: Config,
    sanitised: list[str],
    vacancy_ids: list[str],
    hypotheses: dict[str, str],
    labels_ru: dict[str, str],
) -> pd.DataFrame:
    """Score vacancies against hypotheses via zero-shot NLI (deterministic)."""
    hypothesis_texts: list[str] = list(hypotheses.values())
    hypothesis_to_id: dict[str, str] = {text: cid for cid, text in hypotheses.items()}
    hypothesis_to_label: dict[str, str] = {
        text: labels_ru[cid] for cid, text in hypotheses.items()
    }

    threshold: float = config.characteristics_confidence_threshold
    batch_size: int = config.characteristics_batch_size

    print(
        f"[Phase 2.5] Loading NLI model: {config.characteristics_model} "
        f"(device={config.characteristics_device})"
    )
    classifier = _build_characteristic_classifier(config)
    print(f"[Phase 2.5] {len(hypothesis_texts)} characteristic hypotheses loaded")

    # ---- Batch processing ---------------------------------------------------
    all_rows: list[dict[str, Any]] = []

    for batch_start in tqdm(
        range(0, len(sanitised), batch_size),
        desc="[Phase 2.5] NLI characteristic scoring",
        unit="batch",
    ):
        batch_end = min(batch_start + batch_size, len(sanitised))
        batch_texts = sanitised[batch_start:batch_end]

        # Filter out empty texts; track their original indices for
        # correct vacancy_id mapping.
        non_empty_texts: list[str] = []
        non_empty_indices: list[int] = []
        for j, text in enumerate(batch_texts):
            if text:
                non_empty_texts.append(text)
                non_empty_indices.append(batch_start + j)

        if not non_empty_texts:
            continue

        results: list[dict[str, Any]] = classifier(
            non_empty_texts,
            hypothesis_texts,
        )

        for j, result in enumerate(results):
            global_idx = non_empty_indices[j]
            vacancy_id = vacancy_ids[global_idx]

            for label, score in zip(result["labels"], result["scores"], strict=False):
                if score >= threshold:
                    all_rows.append({
                        "vacancy_id": vacancy_id,
                        "characteristic_id": hypothesis_to_id.get(label, ""),
                        "characteristic_label": hypothesis_to_label.get(label, ""),
                        "confidence": float(score),
                    })

    # ---- Build output DataFrame ---------------------------------------------
    if not all_rows:
        return _empty_characteristics_df()

    return pd.DataFrame(all_rows)


def _extract_characteristics_llm(
    config: Config,
    sanitised: list[str],
    vacancy_ids: list[str],
    hypotheses: dict[str, str],
    labels_ru: dict[str, str],
) -> pd.DataFrame:
    """Score vacancies via one LLM call per vacancy, falling back to NLI per item."""
    threshold: float = config.characteristics_confidence_threshold

    # One prompt per non-empty vacancy. Empty descriptions produce no rows
    # (matching the NLI path) and are never sent to the model.
    prompts: list[LLMPrompt] = []
    prompt_indices: list[int] = []
    for idx, text in enumerate(sanitised):
        if not text:
            continue
        prompts.append(_build_characteristic_prompt(hypotheses, labels_ru, text))
        prompt_indices.append(idx)

    if not prompts:
        return _empty_characteristics_df()

    client = build_llm(config)
    print(
        f"[Phase 2.5] Scoring {len(prompts)} vacancies via LLM ({config.llm_model})"
    )
    results: list[CharacteristicScoresResponse | None] = asyncio.run(
        client.run_many(prompts, CharacteristicScoresResponse)
    )

    all_rows: list[dict[str, Any]] = []
    fallback_indices: list[int] = []

    for idx, result in zip(prompt_indices, results, strict=True):
        if result is None:
            # Total failure for this vacancy → per-item graceful degradation.
            fallback_indices.append(idx)
            continue

        for score in result.characteristics:
            cid = score.characteristic_id
            if cid not in hypotheses:
                continue  # unknown characteristic_id — ignore
            if score.confidence < threshold:
                continue
            all_rows.append({
                "vacancy_id": vacancy_ids[idx],
                "characteristic_id": cid,
                "characteristic_label": labels_ru[cid],
                "confidence": score.confidence,
            })

    if fallback_indices:
        fallback_descriptions = [sanitised[i] for i in fallback_indices]
        fallback_vacancy_ids = [vacancy_ids[i] for i in fallback_indices]
        fallback_df = _extract_characteristics_nli(
            config, fallback_descriptions, fallback_vacancy_ids, hypotheses, labels_ru
        )
        if not fallback_df.empty:
            all_rows.extend(fallback_df.to_dict("records"))

    if not all_rows:
        return _empty_characteristics_df()

    return pd.DataFrame(all_rows)


def _empty_characteristics_df() -> pd.DataFrame:
    """Return an empty DataFrame with the characteristic output schema."""
    return pd.DataFrame(
        columns=["vacancy_id", "characteristic_id", "characteristic_label", "confidence"]
    )


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


_MAX_TITLE_CHARS = 100


def _extract_title(
    title: str | None,
    description: str | None,
) -> str:
    """Return a usable title string.

    Uses the title column from classified Parquet when available.
    Falls back to the first 100 characters of the description,
    truncated at the last whitespace boundary.

    Args:
        title: Vacancy title from ``classified.parquet``.
        description: Vacancy description from ``classified.parquet``.

    Returns:
        A non-empty title string.
    """
    if isinstance(title, str) and title.strip():
        return title.strip()

    if isinstance(description, str) and description.strip():
        text = description.strip()
        if len(text) > _MAX_TITLE_CHARS:
            break_idx = text.rfind(" ", 0, _MAX_TITLE_CHARS)
            text = text[:break_idx] if break_idx > 0 else text[:_MAX_TITLE_CHARS]
        return text

    return ""


# ---------------------------------------------------------------------------
# Title-level characteristic aggregation
# ---------------------------------------------------------------------------


def _compute_title_characteristics(
    characteristics_df: pd.DataFrame,
) -> dict[str, list[str]]:
    """Aggregate characteristic scores to produce a per-title grounding dict.

    Groups the characteristics DataFrame by ``extracted_title`` and, for
    each unique title, selects the top‑2 characteristic labels by mean
    confidence score.  Titles with no characteristic rows above threshold
    are omitted.

    Args:
        characteristics_df: Output from :func:`extract_characteristics`
            with columns ``extracted_title``, ``characteristic_label``,
            ``confidence``.

    Returns:
        Mapping ``{title: [label_ru, label_ru]}`` where the two labels
        are ordered by descending mean confidence.  Titles that have
        fewer than 2 characteristic rows yield a shorter list.
    """
    if characteristics_df.empty or "extracted_title" not in characteristics_df.columns:
        return {}

    grouped = (
        characteristics_df.groupby(["extracted_title", "characteristic_label"])["confidence"]
        .mean()
        .reset_index()
    )

    title_characteristics: dict[str, list[str]] = {}
    for title, grp in grouped.groupby("extracted_title"):
        top = grp.nlargest(2, "confidence")["characteristic_label"].tolist()
        if top:
            title_characteristics[str(title)] = top

    return title_characteristics


def run_pipeline(config: Config) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Run Phase 2.5: extract characteristics and experience from vacancies.

    Pipeline steps:

    1. Reads ``classified.parquet`` from ``config.output_dir``.
    2. Runs zero-shot NLI to score each vacancy against 7 characteristic
       hypotheses, emitting rows where confidence ≥ threshold.
    3. Extracts years of experience from each vacancy description via regex.
    4. Merges characteristics, experience, and title into a single
       DataFrame.
    5. Computes per-title top-2 characteristic labels
       (``title_characteristics`` dict) for downstream grounding.
    6. Writes ``characteristics.parquet`` to ``config.output_dir``.

    Args:
        config: Pipeline configuration.

    Returns:
        Tuple of ``(characteristics_df, title_characteristics)``:

        * **characteristics_df** — DataFrame with columns: ``vacancy_id``,
          ``characteristic_id``, ``characteristic_label``, ``confidence``,
          ``extracted_title``, ``experience_years``, ``year``.
        * **title_characteristics** — ``dict[str, list[str]]`` mapping
          each ``extracted_title`` to its top‑2 Russian characteristic
          labels (ordered by mean confidence).

    Raises:
        FileNotFoundError: If ``classified.parquet`` does not exist.
    """
    # ------------------------------------------------------------------
    # 1. Load classified vacancies
    # ------------------------------------------------------------------
    classified_path = config.classified_path
    classified_df = read_parquet(classified_path)
    print(
        f"[Phase 2.5] Loaded {len(classified_df)} classified vacancies "
        f"from {classified_path}"
    )

    if classified_df.empty:
        result = pd.DataFrame(
            columns=[
                "vacancy_id",
                "characteristic_id",
                "characteristic_label",
                "confidence",
                "extracted_title",
                "experience_years",
            ]
        )
        write_parquet(result, config.characteristics_path)
        return result, {}

    # ------------------------------------------------------------------
    # 2. Extract characteristics via zero-shot NLI
    # ------------------------------------------------------------------
    vacancy_ids: list[str] = classified_df["vacancy_id"].astype(str).tolist()
    descriptions: list[str] = [
        d if isinstance(d, str) else "" for d in classified_df["description"].tolist()
    ]

    characteristics_df = extract_characteristics(config, descriptions, vacancy_ids)
    n_pairs = len(characteristics_df)
    print(f"[Phase 2.5] Extracted {n_pairs} characteristic assignments")

    # ------------------------------------------------------------------
    # 3. Extract years of experience per vacancy
    # ------------------------------------------------------------------
    patterns: list[str] = config.experience_patterns
    experience_map: dict[str, float] = {}

    for i in tqdm(
        range(len(vacancy_ids)),
        desc="[Phase 2.5] Experience extraction",
        unit="vacancy",
    ):
        years = extract_experience(descriptions[i], patterns)
        if years is not None:
            experience_map[vacancy_ids[i]] = years

    experience_series = pd.Series(experience_map, name="experience_years")
    print(
        f"[Phase 2.5] Experience extracted for "
        f"{len(experience_map)}/{len(vacancy_ids)} vacancies"
    )

    # ------------------------------------------------------------------
    # 4. Extract titles
    # ------------------------------------------------------------------
    titles_raw: list[str | None] = [
        classified_df["title"].iloc[i]
        if "title" in classified_df.columns
        else None
        for i in range(len(classified_df))
    ]
    descs_raw: list[str | None] = classified_df["description"].tolist()

    title_map: dict[str, str] = {}
    for i in range(len(vacancy_ids)):
        extracted = _extract_title(titles_raw[i], descs_raw[i])
        if extracted:
            title_map[vacancy_ids[i]] = extracted

    title_series = pd.Series(title_map, name="extracted_title")

    # ------------------------------------------------------------------
    # 4.5 Extract year per vacancy from DuckDB
    # ------------------------------------------------------------------
    year_map: dict[str, int] = {}
    try:
        conn = get_connection(config.db_path)
        year_rows = conn.execute(
            "SELECT id, EXTRACT(YEAR FROM fetched_at) AS year FROM raw_vacancies"
        ).fetchall()
        year_map = {
            str(row[0]): int(row[1])
            for row in year_rows
            if row[1] is not None
        }
    except Exception as exc:
        print(f"[Phase 2.5] WARNING: could not extract years from DuckDB: {exc}")

    # ------------------------------------------------------------------
    # 5. Merge characteristic rows with per-vacancy attributes
    # ------------------------------------------------------------------
    if characteristics_df.empty:
        # No characteristics above threshold — still output vacancy-level
        # experience and title data with placeholder characteristic columns.
        print(
            "[Phase 2.5] No characteristic assignments passed the "
            f"confidence threshold ({config.characteristics_confidence_threshold}). "
            "Writing parameter-only output."
        )
        final_df = pd.DataFrame({
            "vacancy_id": pd.Series(vacancy_ids, dtype=str),
        })
        final_df["characteristic_id"] = None
        final_df["characteristic_label"] = None
        final_df["confidence"] = pd.NA
        final_df["extracted_title"] = final_df["vacancy_id"].map(title_series)
        final_df["experience_years"] = final_df["vacancy_id"].map(experience_series)
        final_df["year"] = final_df["vacancy_id"].map(year_map)
        final_df = final_df.dropna(subset=["extracted_title"], how="all")
    else:
        final_df = characteristics_df.copy()
        final_df["extracted_title"] = final_df["vacancy_id"].map(title_series)
        final_df["experience_years"] = final_df["vacancy_id"].map(experience_series)
        final_df["year"] = final_df["vacancy_id"].map(year_map)

    # Enforce column ordering.
    final_df = final_df[[
        "vacancy_id",
        "characteristic_id",
        "characteristic_label",
        "confidence",
        "extracted_title",
        "experience_years",
        "year",
    ]]

    # ------------------------------------------------------------------
    # 6. Compute per-title characteristic grounding dict
    # ------------------------------------------------------------------
    title_characteristics = _compute_title_characteristics(final_df)
    n_titles = len(title_characteristics)
    print(
        f"[Phase 2.5] Computed title characteristics for "
        f"{n_titles} unique titles"
    )

    # ------------------------------------------------------------------
    # 7. Write output
    # ------------------------------------------------------------------
    write_parquet(final_df, config.characteristics_path)
    print(
        f"[Phase 2.5] Wrote {len(final_df)} rows "
        f"to {config.characteristics_path}"
    )

    # Per-vacancy experience for ALL classified vacancies. The characteristics
    # table above is a sparse long-format (vacancy × characteristic) view, so it
    # only carries experience on rows that also have an assignment above the
    # confidence threshold. Phase 6's Experience histogram needs one row per
    # vacancy, so persist the full vacancy→experience mapping separately.
    vacancy_experience = pd.DataFrame({"vacancy_id": vacancy_ids})
    vacancy_experience["experience_years"] = vacancy_experience["vacancy_id"].map(
        experience_series
    )
    write_parquet(vacancy_experience, config.vacancy_experience_path)

    return final_df, title_characteristics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = Config()
    df, title_chars = run_pipeline(config)
    print(df.head())
    print(f"\nTitle characteristics ({len(title_chars)} titles):")
    for title, labels in list(title_chars.items())[:5]:
        print(f"  {title}: {labels}")
