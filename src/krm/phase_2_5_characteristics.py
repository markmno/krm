"""Phase 2.5: Broad Role Characteristics Extraction from Vacancy Descriptions.

Uses zero-shot natural language inference (NLI) to score each vacancy
against 12 characteristic hypotheses that describe broad role-level
requirements — laboratory work, field work, team leadership, grant writing,
publication, teaching, project management, administrative duties, equipment
maintenance, international collaboration, industry partnership, and
regulatory compliance.

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
     - Hypothesis identifier from config (e.g. ``"laboratory_work"``).
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

import re
from typing import Any

import pandas as pd
from tqdm import tqdm

from krm.config import Config
from krm.lib.io import get_connection, read_parquet, write_parquet

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
            try:
                second_val = float(groups[1])
            except (ValueError, TypeError):
                pass

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


def extract_characteristics(
    config: Config,
    descriptions: list[str],
    vacancy_ids: list[str],
) -> pd.DataFrame:
    """Score each vacancy against all characteristic hypotheses using zero-shot NLI.

    Loads the NLI model (lazy import), processes descriptions in batches
    to avoid OOM, and emits one row per vacancy–characteristic pair
    where the entailment confidence meets or exceeds
    ``config.characteristics_confidence_threshold``.

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
    hypotheses_cfg: list[dict[str, str]] = config.characteristics_hypotheses
    hypothesis_texts: list[str] = [h["hypothesis"] for h in hypotheses_cfg]

    # Sanitise descriptions: convert None/NaN to empty string.
    sanitised: list[str] = [
        d if isinstance(d, str) and d else "" for d in descriptions
    ]

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

            for label, score in zip(result["labels"], result["scores"]):
                if score >= threshold:
                    # Find which hypothesis config entry matches this label.
                    # The label returned is the hypothesis text.
                    char_id: str = ""
                    char_label: str = ""
                    for h in hypotheses_cfg:
                        if h["hypothesis"] == label:
                            char_id = h["id"]
                            char_label = h["label_ru"]
                            break

                    all_rows.append({
                        "vacancy_id": vacancy_id,
                        "characteristic_id": char_id,
                        "characteristic_label": char_label,
                        "confidence": float(score),
                    })

    # ---- Build output DataFrame ---------------------------------------------
    if not all_rows:
        return pd.DataFrame(
            columns=["vacancy_id", "characteristic_id", "characteristic_label", "confidence"]
        )

    result_df = pd.DataFrame(all_rows)
    return result_df


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


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
        if len(text) > 100:
            break_idx = text.rfind(" ", 0, 100)
            if break_idx > 0:
                text = text[:break_idx]
            else:
                text = text[:100]
        return text

    return ""


def run_pipeline(config: Config) -> pd.DataFrame:
    """Run Phase 2.5: extract characteristics and experience from vacancies.

    Pipeline steps:

    1. Reads ``classified.parquet`` from ``config.output_dir``.
    2. Runs zero-shot NLI to score each vacancy against 12 characteristic
       hypotheses, emitting rows where confidence ≥ threshold.
    3. Extracts years of experience from each vacancy description via regex.
    4. Merges characteristics, experience, and title into a single
       DataFrame.
    5. Writes ``characteristics.parquet`` to ``config.output_dir``.

    Args:
        config: Pipeline configuration.

    Returns:
        DataFrame with columns: ``vacancy_id``, ``characteristic_id``,
        ``characteristic_label``, ``confidence``, ``extracted_title``,
        ``experience_years``.

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
        return result

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
        f"{experience_map.__len__()}/{len(vacancy_ids)} vacancies"
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
        conn = get_connection()
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
        # experience and title data.
        final_df = pd.DataFrame({
            "vacancy_id": pd.Series(vacancy_ids, dtype=str),
        })
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
    # 6. Write output
    # ------------------------------------------------------------------
    write_parquet(final_df, config.characteristics_path)
    print(
        f"[Phase 2.5] Wrote {len(final_df)} rows "
        f"to {config.characteristics_path}"
    )

    return final_df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = Config()
    df = run_pipeline(config)
    print(df.head())
