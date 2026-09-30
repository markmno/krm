"""Phase 4: Role-level skill aggregation (data-driven, no ESCO).

Aggregates the vacancy-level skills produced by Phase 2b
(``extracted_skills.parquet``, Russian NLP via pymorphy3 + natasha) up to the
role level, producing ``skills_per_role.parquet`` for Phase 5's
skill-to-characteristic mapping.

ESCO was deprecated — see :mod:`krm.phase_2b_extract_skills` for the underlying
data-driven extraction. This module only maps vacancies to roles and rolls the
per-vacancy skills up into per-role frequency + TF-IDF weights.

Usage:
    from krm.config import Config
    from krm.phase_4_skills import extract_skills

    config = Config()
    skills_df = extract_skills(config)
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from loguru import logger

from krm.config import Config
from krm.lib.embeddings import Embedder
from krm.lib.io import read_parquet, write_parquet

_OUTPUT_COLUMNS = ["role_id", "skill_canonical_name", "frequency", "tfidf_weight"]
_VACANCY_ROLES_COLUMNS = ["vacancy_id", "role_id"]


def _assign_vacancies_to_roles(
    classified_df: pd.DataFrame,
    roles_df: pd.DataFrame,
    embedder: Embedder,
) -> pd.Series:
    """Map each STEM_RESEARCH vacancy to its nearest role via centroid proximity.

    For each unique job title among STEM_RESEARCH vacancies, embeds the
    title and computes cosine similarity to every role centroid
    (deserialised from ``roles.parquet`` ``centroid_embedding``).  The
    title is assigned to the role with the highest similarity.

    Noise points (``role_id == -1``) are excluded from the candidate
    centroids and are assigned only when no other role matches.

    Args:
        classified_df: Classified vacancies (must already be filtered to
            ``STEM_RESEARCH``).
        roles_df: Role definitions from Phase 3.
        embedder: Initialised :class:`~src.krm.lib.embeddings.Embedder`.

    Returns:
        Series indexed by ``vacancy_id`` mapping to ``role_id`` (int).
    """
    # Deserialise centroids — skip noise row.
    centroids: list[tuple[int, np.ndarray]] = []
    for _, row in roles_df.iterrows():
        rid = int(row["role_id"])
        if rid == -1:
            continue
        emb = pickle.loads(row["centroid_embedding"])  # noqa: S301 — trusted
        if isinstance(emb, np.ndarray):
            centroids.append((rid, emb))

    if not centroids:
        # No non-noise roles — assign all vacancies to -1.
        return pd.Series(-1, index=classified_df["vacancy_id"])

    centroid_ids = np.array([c[0] for c in centroids], dtype=np.int64)
    centroid_matrix = np.stack([c[1] for c in centroids])  # (n_roles, dim)

    # Embed unique job titles.
    unique_titles = sorted(classified_df["title"].dropna().unique().tolist())
    title_embs = embedder.encode(unique_titles)  # L2-normalised

    # Cosine similarity → dot product (embeddings are already normalised).
    sims: np.ndarray = title_embs @ centroid_matrix.T  # (n_titles, n_roles)
    best_idx: np.ndarray = np.argmax(sims, axis=1)

    title_to_role: dict[str, int] = {}
    for i, title in enumerate(unique_titles):
        title_to_role[title] = int(centroid_ids[best_idx[i]])

    # Map back to vacancy_id — Series must be indexed by vacancy_id so that
    # _aggregate_skills can `.map()` extracted vacancy_ids onto role_ids.
    return (
        classified_df.set_index("vacancy_id")["title"]
        .map(title_to_role)
        .fillna(-1)
        .astype(int)
    )


def _aggregate_skills(
    extracted_df: pd.DataFrame,
    vacancy_role_map: pd.Series,
) -> pd.DataFrame:
    """Roll vacancy-level skills (Phase 2b) up to the role level.

    Args:
        extracted_df: Phase 2b output with columns ``vacancy_id``,
            ``skill_phrase``, ``frequency``, ``tfidf_weight``.
        vacancy_role_map: Series indexed by ``vacancy_id`` → ``role_id``.

    Returns:
        DataFrame with columns ``role_id``, ``skill_canonical_name``,
        ``frequency``, ``tfidf_weight`` — one row per (role, skill) pair.
    """
    extracted = extracted_df.copy()
    extracted["role_id"] = extracted["vacancy_id"].map(vacancy_role_map)
    # Drop vacancies that are noise (-1) or not in the STEM role map (NaN).
    extracted = extracted[extracted["role_id"].notna()]
    extracted = extracted[extracted["role_id"] != -1]
    extracted["role_id"] = extracted["role_id"].astype(int)

    agg = (
        extracted.groupby(["role_id", "skill_phrase"], as_index=False)
        .agg(frequency=("frequency", "sum"), tfidf_weight=("tfidf_weight", "sum"))
        .rename(columns={"skill_phrase": "skill_canonical_name"})
    )
    return agg[_OUTPUT_COLUMNS]


def _vacancy_roles_frame(vacancy_role_map: pd.Series) -> pd.DataFrame:
    """Build the two-column vacancy→role frame, dropping noise assignments.

    Args:
        vacancy_role_map: Series indexed by ``vacancy_id`` → ``role_id``.

    Returns:
        DataFrame with columns ``vacancy_id`` (str) and ``role_id`` (int),
        excluding every ``role_id == -1`` row.
    """
    frame = pd.DataFrame(
        {
            "vacancy_id": vacancy_role_map.index.astype(str),
            "role_id": vacancy_role_map.to_numpy().astype(int),
        }
    )
    return frame[frame["role_id"] != -1].reset_index(drop=True)


def _empty(config: Config) -> pd.DataFrame:
    result = pd.DataFrame(columns=_OUTPUT_COLUMNS)
    write_parquet(result, config.output_dir / "skills_per_role.parquet")
    write_parquet(pd.DataFrame(columns=_VACANCY_ROLES_COLUMNS), config.vacancy_roles_path)
    return result


def extract_skills(config: Config) -> pd.DataFrame:
    """Aggregate data-driven skills per role (no ESCO).

    Pipeline:
        1. Loads ``roles.parquet`` and ``classified.parquet``.
        2. Filters classified vacancies to ``STEM_RESEARCH``.
        3. Assigns each vacancy to its nearest role (embedding centroid).
        4. Ensures Phase 2b vacancy-level skills exist (runs Phase 2b if not).
        5. Aggregates per-vacancy skills into per-role frequency + TF-IDF.
        6. Writes ``skills_per_role.parquet``.

    Args:
        config: Pipeline configuration providing paths and the embedding model.

    Returns:
        DataFrame with columns ``role_id``, ``skill_canonical_name``,
        ``frequency``, ``tfidf_weight``.
    """
    roles_path = config.output_dir / "roles.parquet"
    classified_path = config.output_dir / "classified.parquet"

    if not roles_path.exists():
        raise FileNotFoundError(f"roles.parquet not found at {roles_path}")
    if not classified_path.exists():
        raise FileNotFoundError(f"classified.parquet not found at {classified_path}")

    roles_df = read_parquet(roles_path)
    classified_df = read_parquet(classified_path)
    logger.info(
        f"[Phase 4] Loaded {len(roles_df)} roles, {len(classified_df)} classified vacancies"
    )

    stem_df = classified_df[classified_df["stem_category"] == "STEM_RESEARCH"].copy()
    if stem_df.empty:
        logger.info("[Phase 4] No STEM_RESEARCH vacancies — returning empty skills table")
        return _empty(config)
    logger.info(f"[Phase 4] {len(stem_df)} STEM_RESEARCH vacancies for skill aggregation")

    # Assign vacancies to roles (embedding-based centroid similarity).
    embedder = Embedder(model_name=config.embedding_model)
    vacancy_role_map = _assign_vacancies_to_roles(stem_df, roles_df, embedder)
    n_assigned = int((vacancy_role_map != -1).sum())
    logger.info(f"[Phase 4] Assigned {n_assigned}/{len(stem_df)} vacancies to roles")

    # Persist the vacancy→role map (Phase 5b reads this for soft scoring).
    vacancy_roles = _vacancy_roles_frame(vacancy_role_map)
    write_parquet(vacancy_roles, config.vacancy_roles_path)
    logger.info(f"[Phase 4] Wrote {len(vacancy_roles)} vacancy→role mappings")

    # Ensure Phase 2b vacancy-level skills exist.
    extracted_path = config.output_dir / "extracted_skills.parquet"
    if not extracted_path.exists():
        from krm.phase_2b_extract_skills import extract_skills as extract_skills_2b

        extract_skills_2b(config)
    extracted_df = read_parquet(extracted_path)
    logger.info(f"[Phase 4] Loaded {len(extracted_df)} vacancy-level skill entries")

    skills_df = _aggregate_skills(extracted_df, vacancy_role_map)

    output_path = config.output_dir / "skills_per_role.parquet"
    write_parquet(skills_df, output_path)
    logger.info(
        f"[Phase 4] Wrote {len(skills_df)} role-skill entries across "
        f"{skills_df['role_id'].nunique() if not skills_df.empty else 0} roles"
    )
    return skills_df
