"""Phase 2: STEM/IT/Non-STEM classification for Russian vacancy text.

Four-tier classifier designed to distinguish genuine STEM research positions
from IT roles that appear alongside STEM keywords on the HH.ru platform.

**Classification tiers:**

1. **API professional_role filter** — vacancies whose HH.ru
   professional_role ID belongs to {96, 123, 124, 125, 126} are
   classified as PURE_IT immediately.

2. **Keyword triage** — vacancies whose title matches
   ``[Pp]rogramm``/``[Rr]азработчик`` and whose description contains
   *no* STEM keywords (лаборатор, эксперимент, исследовани, научн,
   анализ, синтез, моделирован) are classified as PURE_IT.

3. **Zero-shot NLI** — remaining vacancies are scored with
   ``facebook/bart-large-mnli`` against three hypotheses:
   *scientific research position requiring experimental work*,
   *software engineering or IT development role*, and
   *administrative or teaching position*.

4. **Interdisciplinary catch** — if both STEM and IT entailment scores
   exceed ``interdisciplinary_threshold``, the vacancy is classified
   as INTERDISCIPLINARY. Otherwise the highest-scoring hypothesis
   determines the category: STEM_RESEARCH, PURE_IT, or NON_STEM.

**Output schema** (columns in ``classified.parquet``):

.. list-table::
   :header-rows: 1

   * - Column
     - Type
     - Description
   * - ``vacancy_id``
     - str
     - HH.ru vacancy identifier.
   * - ``title``
     - str
     - Vacancy title from ``data.name``.
   * - ``description``
     - str (nullable)
     - Full vacancy description from ``data.description``.
   * - ``stem_category``
     - str
     - One of STEM_RESEARCH, PURE_IT, NON_STEM, INTERDISCIPLINARY.
   * - ``nli_scores_json``
     - str (nullable)
     - JSON-encoded entailment scores (null for Tier-1/2 decisions).

"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from krm.config import Config
from krm.lib.io import get_connection, write_parquet

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# HH.ru professional_role IDs that unequivocally denote IT work.
_PURE_IT_ROLE_IDS: frozenset[int] = frozenset({96, 123, 124, 125, 126})

# Tier 1 — categories assigned without running the NLI model.
STEM_RESEARCH: str = "STEM_RESEARCH"
PURE_IT: str = "PURE_IT"
NON_STEM: str = "NON_STEM"
INTERDISCIPLINARY: str = "INTERDISCIPLINARY"

# Tier 2 patterns.
_IT_TITLE_RE: re.Pattern[str] = re.compile(
    r"[Pp]rogramm|[Пп]рограмм|[RrРр]азработчик",
)
_STEM_KEYWORD_RE: re.Pattern[str] = re.compile(
    r"лаборатор|эксперимент|исследовани|научн|анализ|синтез|моделирован",
)

# Tier 3 hypothesis labels.
_NLI_HYPOTHESES: list[str] = [
    "scientific research position requiring experimental work",
    "software engineering or IT development role",
    "administrative or teaching position",
]

# Mapping from hypothesis index to category.
_NLI_CATEGORY_MAP: list[str] = [STEM_RESEARCH, PURE_IT, NON_STEM]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_vacancy_text(data: dict[str, Any]) -> tuple[str, str]:
    """Extract title and description from a raw vacancy data dict.

    Args:
        data: Raw HH.ru vacancy JSON dict as stored in DuckDB.

    Returns:
        ``(title, description)``. Missing values are normalised to ``""`` so
        downstream regex/NLI code never sees ``None``/``NaN``.
    """
    title = data.get("name") or ""
    description = data.get("description") or ""
    return title, description


def _parse_professional_roles(data: dict[str, Any]) -> list[int]:
    """Extract professional_role IDs from raw vacancy data.

    The HH.ru API embeds roles as ``[{"id": 96, "name": "..."}, ...]``.

    Args:
        data: Raw HH.ru vacancy JSON dict.

    Returns:
        List of integer role IDs. Empty list if the key is missing or
        the value is not a list.
    """
    roles: Any = data.get("professional_roles")
    if not isinstance(roles, list):
        return []
    result: list[int] = []
    for r in roles:
        if isinstance(r, dict):
            rid = r.get("id")
            if isinstance(rid, int):
                result.append(rid)
    return result


def _tier1_is_pure_it(role_ids: list[int]) -> bool:
    """Return True if any role ID is in the pure-IT set."""
    return bool(_PURE_IT_ROLE_IDS.intersection(role_ids))


def _tier2_is_pure_it(title: str | None, description: str | None) -> bool:
    """Return True if title looks IT and description has no STEM keywords.

    Args:
        title: Vacancy title (may be ``None``).
        description: Vacancy description (may be ``None``).

    Returns:
        ``True`` when the title matches an IT pattern and the
        description does *not* contain any STEM keyword.
    """
    if title is None or description is None:
        return False
    if not _IT_TITLE_RE.search(title):
        return False
    if _STEM_KEYWORD_RE.search(description):
        return False
    return True


def _build_nli_text(title: str | None, description: str | None) -> str:
    """Join title and description into a single NLI premise string.

    Args:
        title: Vacancy title.
        description: Vacancy description.

    Returns:
        A non-empty string suitable for zero-shot classification.
    """
    parts: list[str] = []
    if title:
        parts.append(title)
    if description:
        parts.append(description)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_vacancies(config: Config) -> pd.DataFrame:
    """Classify raw vacancies into STEM_RESEARCH / PURE_IT / NON_STEM / INTERDISCIPLINARY.

    Reads raw vacancies from DuckDB, applies a four-tier classification
    pipeline, and writes ``classified.parquet`` to
    ``config.output_dir``.

    *Tiers:*

    1. If **professional_role** ∈ {96, 123, 124, 125, 126} → PURE_IT.
    2. If **title** matches ``[Pp]rogramm`` / ``[Rr]азработчик`` and
       **description** has no STEM keywords → PURE_IT.
    3. Remaining vacancies are scored via zero-shot NLI
       (``facebook/bart-large-mnli``) against three hypotheses.
    4. If both STEM and IT entailment exceed
       ``config.interdisciplinary_threshold`` → INTERDISCIPLINARY;
       otherwise the highest-scoring hypothesis wins →
       STEM_RESEARCH | PURE_IT | NON_STEM.

    Thresholds and the model name are read from ``config`` — nothing
    is hardcoded.

    Args:
        config: Pipeline configuration (model, thresholds, paths).

    Returns:
        DataFrame with columns: ``vacancy_id``, ``title``,
        ``description``, ``stem_category``, ``nli_scores_json``.
    """
    # -- Stage 0: load raw vacancies ------------------------------------------
    conn = get_connection(config.db_path)
    try:
        df = conn.execute(
            "SELECT id AS vacancy_id, data FROM raw_vacancies"
        ).fetchdf()
    finally:
        conn.close()

    if df.empty:
        result = pd.DataFrame(
            columns=["vacancy_id", "title", "description", "stem_category", "nli_scores_json"]
        )
        write_parquet(result, config.output_dir / "classified.parquet")
        return result

    # Parse JSON blobs.
    parsed = df["data"].apply(json.loads)

    vacancy_ids: pd.Series = df["vacancy_id"]
    titles: pd.Series = parsed.apply(lambda d: _extract_vacancy_text(d)[0])
    descriptions: pd.Series = parsed.apply(lambda d: _extract_vacancy_text(d)[1])
    role_ids: pd.Series = parsed.apply(_parse_professional_roles)

    n = len(df)
    categories: list[str] = [""] * n
    nli_scores: list[str | None] = [None] * n

    # -- Tier 1: professional_role filter ------------------------------------
    tier2_mask = pd.Series([False] * n)
    for i in range(n):
        if _tier1_is_pure_it(role_ids.iloc[i]):
            categories[i] = PURE_IT
        else:
            tier2_mask.iloc[i] = True

    # -- Tier 2: keyword triage ----------------------------------------------
    tier3_indices: list[int] = []
    for i in range(n):
        if not tier2_mask.iloc[i]:
            continue
        if _tier2_is_pure_it(titles.iloc[i], descriptions.iloc[i]):
            categories[i] = PURE_IT
        else:
            tier3_indices.append(i)

    # -- Tier 3: zero-shot NLI -----------------------------------------------
    if tier3_indices:
        from transformers import pipeline

        classifier = pipeline(
            "zero-shot-classification",
            model=config.classification_model,
            device=config._data.get("classification", {}).get("device", "cpu"),
        )

        texts: list[str] = [
            _build_nli_text(titles.iloc[i], descriptions.iloc[i])
            for i in tier3_indices
        ]

        # Process in batches to avoid OOM on large datasets.
        batch_size: int = config._data.get("classification", {}).get("batch_size", 32)
        for batch_start in range(0, len(texts), batch_size):
            batch_end = min(batch_start + batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]

            results: list[dict[str, Any]] = classifier(
                batch_texts,
                _NLI_HYPOTHESES,
            )

            for j, r in enumerate(results):
                idx = tier3_indices[batch_start + j]
                scores_dict = dict(zip(r["labels"], r["scores"]))
                ordered_scores = [scores_dict.get(h, 0.0) for h in _NLI_HYPOTHESES]

                # Store full scores as JSON.
                nli_scores[idx] = json.dumps(
                    {_NLI_HYPOTHESES[k]: ordered_scores[k] for k in range(len(_NLI_HYPOTHESES))},
                    ensure_ascii=False,
                )

                stem_score: float = ordered_scores[0]
                it_score: float = ordered_scores[1]
                admin_score: float = ordered_scores[2]

                # Tier 4: interdisciplinary catch.
                if (
                    stem_score > config.interdisciplinary_threshold
                    and it_score > config.interdisciplinary_threshold
                ):
                    categories[idx] = INTERDISCIPLINARY
                else:
                    scores: list[float] = [stem_score, it_score, admin_score]
                    best: int = max(range(len(scores)), key=lambda k: scores[k])
                    categories[idx] = _NLI_CATEGORY_MAP[best]

    # -- Assemble output DataFrame -------------------------------------------
    result_df = pd.DataFrame({
        "vacancy_id": vacancy_ids,
        "title": titles,
        "description": descriptions,
        "stem_category": pd.Categorical(
            categories,
            categories=[STEM_RESEARCH, PURE_IT, NON_STEM, INTERDISCIPLINARY],
        ),
        "nli_scores_json": nli_scores,
    })

    # -- Persist --------------------------------------------------------------
    out_path: Path = config.output_dir / "classified.parquet"
    write_parquet(result_df, out_path)

    return result_df
