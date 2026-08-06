"""Phase 4: ESCO Skill Extraction via Dictionary-Based Fuzzy Matching.

Extracts skills per discovered role from aggregated vacancy descriptions
using the ESCO (European Skills, Competences, Qualifications and Occupations)
skill taxonomy as a reference dictionary.

**Pipeline:**

1. Loads ``roles.parquet`` and ``classified.parquet`` from
   ``config.output_dir``.
2. Maps each STEM_RESEARCH vacancy to its nearest role via
   centroid-embedding cosine similarity.
3. Loads the ESCO taxonomy CSV (columns: ``skill_id``,
   ``preferred_label``, ``description``, ``skill_type``).
4. For each role, aggregates all vacancy descriptions and matches
   ESCO skill labels using three strategies:
   (a) exact case-insensitive substring,
   (b) word-boundary regex match,
   (c) token-overlap scoring against ``config.fuzzy_threshold``.
5. Computes per-role frequencies (how many vacancies mention each skill)
   and TF-IDF weights via ``TfidfVectorizer`` on role-aggregated texts.
6. Identifies uncatalogued skills — frequent multi-word phrases not
   found in ESCO — and stores them as JSON arrays.
7. Writes ``skills_per_role.parquet`` to ``config.output_dir``.

Usage:
    from krm.config import Config
    from krm.phase_4_skills import extract_skills

    config = Config()
    skills_df = extract_skills(config)
"""

from __future__ import annotations

import json
import pickle
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from krm.config import Config
from krm.lib.embeddings import Embedder
from krm.lib.io import read_parquet, write_parquet

# ---------------------------------------------------------------------------
# ESCO loader
# ---------------------------------------------------------------------------

# Expected column names in the ESCO CSV — the file may use variations.
_ESCO_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "skill_id": ("skill_id", "conceptUri", "uri", "identifier"),
    "preferred_label": ("preferred_label", "preferredLabel", "label", "title"),
    "description": ("description", "description", "Description"),
    "skill_type": ("skill_type", "skillType", "type", "skillType"),
}


def _load_esco(path: str | Any) -> pd.DataFrame | None:
    """Load ESCO taxonomy CSV and normalise column names.

    Args:
        path: Filesystem path to the ESCO CSV file.

    Returns:
        DataFrame with columns ``[skill_id, preferred_label, description,
        skill_type]``, or ``None`` if the file does not exist.
    """
    import os

    p = str(path)
    if not os.path.isfile(p):
        print(f"[Phase 4] ESCO file not found: {p} — skill extraction skipped")
        return None

    df = pd.read_csv(p, dtype=str)

    # Map known column name variants to canonical names.
    rename: dict[str, str] = {}
    for canonical, candidates in _ESCO_COLUMN_CANDIDATES.items():
        for col in df.columns:
            if col in candidates:
                rename[col] = canonical
                break

    df = df.rename(columns=rename)

    # Ensure all required columns exist, filling with empty strings if missing.
    required = ["skill_id", "preferred_label", "description", "skill_type"]
    for col in required:
        if col not in df.columns:
            df[col] = ""

    # Keep only the canonical columns.
    df = df[required].copy()
    # Drop rows without a usable label.
    df = df.dropna(subset=["preferred_label"])
    df = df[df["preferred_label"].str.strip().str.len() > 0]

    print(f"[Phase 4] Loaded {len(df)} ESCO skill entries")
    return df


# ---------------------------------------------------------------------------
# Per-vacancy skill matching
# ---------------------------------------------------------------------------


def _match_skill_label(
    label: str,
    text: str,
    fuzzy_threshold: float = 0.7,
) -> bool:
    """Determine whether a skill label appears in a text description.

    Three matching strategies are applied in order — a match on any
    strategy returns ``True`` immediately:

    (a) **Exact substring** — case-insensitive literal presence.
    (b) **Word-boundary regex** — the label as a whole word/phrase
        delimited by ``\\b``.
    (c) **Token-overlap score** — sliding-window Jaccard overlap
        between the label's tokens and same-length windows in the
        text.  A score ≥ ``fuzzy_threshold`` counts as a match.

    Args:
        label: ESCO preferred label (skill name).
        text: Vacancy description text (may be ``None`` or empty).
        fuzzy_threshold: Minimum token-overlap score for strategy (c).

    Returns:
        ``True`` if the skill is detected in the text.
    """
    if not text or not label:
        return False

    text_lower = text.lower()
    label_lower = label.lower()
    label_tokens = label_lower.split()
    n_tokens = len(label_tokens)

    # (a) Exact case-insensitive substring.
    if label_lower in text_lower:
        return True

    # (b) Word-boundary regex.
    pattern = r"\b" + re.escape(label_lower) + r"\b"
    if re.search(pattern, text_lower):
        return True

    # (c) Token-overlap scoring.
    if n_tokens > 1:
        label_set = set(label_tokens)
        text_tokens = text_lower.split()
        if len(text_tokens) < n_tokens:
            return False

        for i in range(len(text_tokens) - n_tokens + 1):
            window = set(text_tokens[i : i + n_tokens])
            overlap = len(label_set & window) / len(label_set)
            if overlap >= fuzzy_threshold:
                return True

    return False


# ---------------------------------------------------------------------------
# Vacancy → Role assignment
# ---------------------------------------------------------------------------


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

    # Map back to vacancy_id.
    return classified_df["title"].map(title_to_role).fillna(-1).astype(int)


# ---------------------------------------------------------------------------
# Unatalogued skill extraction
# ---------------------------------------------------------------------------


def _extract_uncatalogued_phrases(
    descriptions: list[str],
    esco_labels: set[str],
    min_word_len: int = 2,
    top_k: int = 50,
) -> list[str]:
    """Find frequent multi-word phrases absent from ESCO.

    Extracts 2-, 3-, and 4-grams from the aggregated vacancy
    descriptions of a role, keeps those that appear in at least
    ``min_word_len`` vacancies, and returns the top-*k* phrases (by
    frequency) that are *not* present among ``esco_labels``.

    Args:
        descriptions: List of vacancy description strings.
        esco_labels: Lower-cased set of all ESCO preferred labels.
        min_word_len: Minimum phrase length in words.
        top_k: Maximum number of phrases to return.

    Returns:
        Sorted list of uncatalogued phrases (most frequent first).
    """
    # Tokenise each description into lower-cased words (keep only alpha).
    tokenised: list[list[str]] = []
    for desc in descriptions:
        if not desc:
            continue
        tokens = [
            t.lower()
            for t in re.findall(r"[a-zA-Zа-яА-ЯёЁ]+", desc)
            if len(t) >= min_word_len
        ]
        tokenised.append(tokens)

    # Count multi-word n-grams per document, then globally.
    doc_phrase_sets: list[set[str]] = []
    for tokens in tokenised:
        phrases: set[str] = set()
        for n in (2, 3, 4):
            if len(tokens) >= n:
                for i in range(len(tokens) - n + 1):
                    phrases.add(" ".join(tokens[i : i + n]))
        doc_phrase_sets.append(phrases)

    # Count how many documents each phrase appears in.
    global_counter: Counter[str] = Counter()
    for pset in doc_phrase_sets:
        global_counter.update(pset)

    # Filter: not in ESCO, keep top-k by document frequency.
    uncatalogued: list[tuple[str, int]] = []
    for phrase, count in global_counter.most_common():
        if phrase not in esco_labels:
            uncatalogued.append((phrase, count))
            if len(uncatalogued) >= top_k:
                break

    return [phrase for phrase, _ in uncatalogued]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_skills(config: Config) -> pd.DataFrame:
    """Extract ESCO skills per role from vacancy descriptions.

    Full pipeline:

    1. Loads ``roles.parquet`` and ``classified.parquet`` from
       ``config.output_dir``.
    2. Filters classified vacancies to ``STEM_RESEARCH``.
    3. Assigns each vacancy to its most similar role via centroid
       cosine similarity (embedding-based).
    4. Loads the ESCO taxonomy from ``config.esco_path``.
    5. For each role, aggregates all vacancy descriptions and scans
       for ESCO skill labels using three matching strategies.
    6. Computes per-role **frequency** (number of vacancies
       mentioning the skill).
    7. Fits a ``TfidfVectorizer`` on role-aggregated texts and
       computes per-skill TF-IDF weights.
    8. Extracts uncatalogued multi-word phrases (not in ESCO) per role.
    9. Writes ``skills_per_role.parquet`` to ``config.output_dir``.

    Args:
        config: Pipeline configuration providing ``esco_path``,
            ``fuzzy_threshold``, ``tfidf_max_features``, and paths.

    Returns:
        DataFrame with columns:

        ============================ ======================================
        column                       description
        ============================ ======================================
        ``role_id``                  int — HDBSCAN cluster label.
        ``skill_canonical_name``     str — ESCO ``preferred_label``.
        ``esco_id``                  str — ESCO ``skill_id``.
        ``frequency``                int — vacancies mentioning the skill.
        ``tfidf_weight``             float — TF-IDF weight for the skill
                                     in this role.
        ``uncatalogued_skills``      str — JSON array of top uncatalogued
                                     multi-word phrases for the role
                                     (identical across rows of the same
                                     role).
        ============================ ======================================

    Raises:
        FileNotFoundError: If ``roles.parquet`` or ``classified.parquet``
            does not exist in ``config.output_dir``.
    """
    # ------------------------------------------------------------------
    # 1. Load input data
    # ------------------------------------------------------------------
    roles_path = config.output_dir / "roles.parquet"
    classified_path = config.output_dir / "classified.parquet"

    if not roles_path.exists():
        msg = f"roles.parquet not found at {roles_path}"
        raise FileNotFoundError(msg)
    if not classified_path.exists():
        msg = f"classified.parquet not found at {classified_path}"
        raise FileNotFoundError(msg)

    roles_df = read_parquet(roles_path)
    classified_df = read_parquet(classified_path)

    print(f"[Phase 4] Loaded {len(roles_df)} roles, {len(classified_df)} classified vacancies")

    # ------------------------------------------------------------------
    # 2. Filter to STEM_RESEARCH
    # ------------------------------------------------------------------
    stem_df = classified_df[classified_df["stem_category"] == "STEM_RESEARCH"].copy()
    if stem_df.empty:
        print("[Phase 4] No STEM_RESEARCH vacancies — returning empty skills table")
        empty = pd.DataFrame(
            columns=[
                "role_id",
                "skill_canonical_name",
                "esco_id",
                "frequency",
                "tfidf_weight",
                "uncatalogued_skills",
            ]
        )
        write_parquet(empty, config.output_dir / "skills_per_role.parquet")
        return empty

    n_stem = len(stem_df)
    print(f"[Phase 4] {n_stem} STEM_RESEARCH vacancies for skill extraction")

    # ------------------------------------------------------------------
    # 3. Assign vacancies to roles
    # ------------------------------------------------------------------
    embedder = Embedder(model_name=config.embedding_model)
    vacancy_role_map = _assign_vacancies_to_roles(stem_df, roles_df, embedder)

    # Attach role_id to stem_df.
    stem_df["role_id"] = stem_df["vacancy_id"].map(vacancy_role_map)
    # Exclude noise assignments.
    stem_assigned = stem_df[stem_df["role_id"] != -1].copy()
    unique_roles = sorted(stem_assigned["role_id"].unique())
    print(
        f"[Phase 4] Assigned vacancies to {len(unique_roles)} roles "
        f"({len(stem_df) - len(stem_assigned)} unassigned/noise)"
    )

    if stem_assigned.empty:
        print("[Phase 4] No role-assigned vacancies — returning empty skills table")
        empty = pd.DataFrame(
            columns=[
                "role_id",
                "skill_canonical_name",
                "esco_id",
                "frequency",
                "tfidf_weight",
                "uncatalogued_skills",
            ]
        )
        write_parquet(empty, config.output_dir / "skills_per_role.parquet")
        return empty

    # ------------------------------------------------------------------
    # 4. Load ESCO taxonomy
    # ------------------------------------------------------------------
    esco_df = _load_esco(config.esco_path)
    if esco_df is None or esco_df.empty:
        # No ESCO — return empty with uncatalogued only.
        return _build_empty_with_uncatalogued(
            stem_assigned, set(), config
        )

    esco_labels_lower: set[str] = {
        lbl.lower().strip() for lbl in esco_df["preferred_label"] if lbl
    }
    # Build a fast lookup: lower-cased label → (name, skill_id).
    esco_lookup: dict[str, tuple[str, str]] = {}
    for _, row in esco_df.iterrows():
        key = str(row["preferred_label"]).lower().strip()
        if key:
            esco_lookup[key] = (str(row["preferred_label"]), str(row["skill_id"]))

    # ------------------------------------------------------------------
    # 5. Per-vacancy skill matching
    # ------------------------------------------------------------------
    n_vacancies_total = len(stem_assigned)
    label_list = list(esco_lookup)
    print(
        f"[Phase 4] Matching {len(label_list)} ESCO labels "
        f"against {n_vacancies_total} vacancy descriptions..."
    )

    # Pre-compile vacancy texts for fast access.
    vacancy_texts: dict[str, str] = dict(
        zip(stem_assigned["vacancy_id"], stem_assigned["description"].fillna(""), strict=False)
    )
    # For each skill label, record which vacancies matched.
    # Structure: {skill_label_lower: set(vacancy_ids)}
    skill_vacancy_hits: dict[str, set[str]] = {}
    for label_key, (canonical_name, _) in esco_lookup.items():
        matched_vacancies: set[str] = set()
        for vid, text in vacancy_texts.items():
            if _match_skill_label(canonical_name, text, config.fuzzy_threshold):
                matched_vacancies.add(vid)
        if matched_vacancies:
            skill_vacancy_hits[label_key] = matched_vacancies

    print(f"[Phase 4] Matched {len(skill_vacancy_hits)} ESCO skills across vacancies")

    # ------------------------------------------------------------------
    # 6. Aggregate to role level — frequency
    # ------------------------------------------------------------------
    role_descriptions: dict[int, list[str]] = {rid: [] for rid in unique_roles}
    for _, row in stem_assigned.iterrows():
        rid = row["role_id"]
        desc = str(row.get("description") or "")
        role_descriptions[rid].append(desc)

    # Build rows: one per (role, skill) pair.
    rows: list[dict[str, Any]] = []
    for role_id in unique_roles:
        role_vacancies = stem_assigned[stem_assigned["role_id"] == role_id]
        role_vacancy_set = set(role_vacancies["vacancy_id"])

        for label_key, (canonical_name, esco_id) in esco_lookup.items():
            hits = skill_vacancy_hits.get(label_key, set())
            role_hits = hits & role_vacancy_set
            if not role_hits:
                continue
            rows.append(
                {
                    "role_id": int(role_id),
                    "skill_canonical_name": canonical_name,
                    "esco_id": esco_id,
                    "frequency": len(role_hits),
                    "tfidf_weight": 0.0,  # filled in next step
                }
            )

    if not rows:
        return _build_empty_with_uncatalogued(stem_assigned, esco_labels_lower, config)

    skills_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 7. Compute TF-IDF weights
    # ------------------------------------------------------------------
    # Build one aggregated document per role.
    role_docs: dict[int, str] = {}
    for rid in unique_roles:
        role_docs[rid] = " ".join(role_descriptions[rid])

    doc_order: list[int] = sorted(role_docs)
    doc_texts: list[str] = [role_docs[rid] for rid in doc_order]

    vectorizer = TfidfVectorizer(
        max_features=config.tfidf_max_features,
        ngram_range=(1, 3),  # unigrams through trigrams
        lowercase=True,
        stop_words=None,  # keep all tokens for ESCO label matching
    )
    tfidf_matrix = vectorizer.fit_transform(doc_texts)  # (n_roles, n_features)
    vocab = vectorizer.vocabulary_  # term → index

    # For each skill in each role, look up the label in the TF-IDF vocabulary.
    # If the full label is not a single term in the vocabulary (e.g., multi-word
    # phrase not captured as an n-gram), average the TF-IDF of constituent tokens.
    role_index_map: dict[int, int] = {rid: i for i, rid in enumerate(doc_order)}

    for idx, row in skills_df.iterrows():
        role_id = int(row["role_id"])
        canonical_name = str(row["skill_canonical_name"])
        role_idx = role_index_map.get(role_id)
        if role_idx is None:
            continue

        label_lower = canonical_name.lower()
        # Try exact phrase match first.
        if label_lower in vocab:
            feature_idx = vocab[label_lower]
            weight = float(tfidf_matrix[role_idx, feature_idx])
        else:
            # Fall back to average of constituent token weights.
            tokens = label_lower.split()
            token_indices = [vocab[t] for t in tokens if t in vocab]
            if token_indices:
                weight = float(
                    np.mean([tfidf_matrix[role_idx, ti] for ti in token_indices])
                )
            else:
                weight = 0.0

        skills_df.at[idx, "tfidf_weight"] = weight

    # ------------------------------------------------------------------
    # 8. Extract uncatalogued skills per role
    # ------------------------------------------------------------------
    uncatalogued_per_role: dict[int, list[str]] = {}
    for rid in unique_roles:
        descs = role_descriptions[rid]
        uncat = _extract_uncatalogued_phrases(descs, esco_labels_lower)
        uncatalogued_per_role[rid] = uncat

    # Attach uncatalogued_skills as a JSON column (same per role).
    skills_df["uncatalogued_skills"] = skills_df["role_id"].apply(
        lambda rid: json.dumps(
            uncatalogued_per_role.get(int(rid), []), ensure_ascii=False
        )
    )

    # Ensure column order.
    skills_df = skills_df[
        [
            "role_id",
            "skill_canonical_name",
            "esco_id",
            "frequency",
            "tfidf_weight",
            "uncatalogued_skills",
        ]
    ]

    # ------------------------------------------------------------------
    # 9. Write output
    # ------------------------------------------------------------------
    output_path = config.output_dir / "skills_per_role.parquet"
    write_parquet(skills_df, output_path)

    n_skills = len(skills_df)
    n_roles = skills_df["role_id"].nunique()
    print(
        f"[Phase 4] Wrote {n_skills} skill entries across {n_roles} roles "
        f"to {output_path}"
    )

    return skills_df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_empty_with_uncatalogued(
    stem_df: pd.DataFrame,
    esco_labels: set[str],
    config: Config,
) -> pd.DataFrame:
    """Build empty skills DataFrame with only uncatalogued phrases.

    Used when ESCO taxonomy is unavailable but we still want to surface
    candidate skills from the text.
    """
    unique_roles = sorted(stem_df["role_id"].unique())
    rows: list[dict[str, Any]] = []

    for rid in unique_roles:
        role_descs = stem_df[stem_df["role_id"] == rid]["description"].fillna("").tolist()
        uncat = _extract_uncatalogued_phrases(role_descs, esco_labels)
        # Create one dummy row per role to carry uncatalogued_skills.
        rows.append(
            {
                "role_id": int(rid),
                "skill_canonical_name": "",
                "esco_id": "",
                "frequency": 0,
                "tfidf_weight": 0.0,
                "uncatalogued_skills": json.dumps(uncat, ensure_ascii=False),
            }
        )

    result = pd.DataFrame(
        rows,
        columns=[
            "role_id",
            "skill_canonical_name",
            "esco_id",
            "frequency",
            "tfidf_weight",
            "uncatalogued_skills",
        ],
    )

    output_path = config.output_dir / "skills_per_role.parquet"
    write_parquet(result, output_path)
    print(f"[Phase 4] Wrote {len(result)} empty-skill role rows to {output_path}")
    return result
