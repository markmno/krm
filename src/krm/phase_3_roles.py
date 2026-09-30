"""Phase 3: Role Discovery via HDBSCAN Clustering of STEM Job Titles.

Discovers natural groupings (roles) among STEM job titles using
embedding-based clustering with UMAP dimensionality reduction
and HDBSCAN clustering with soft membership support for
interdisciplinary role detection.

Usage:
    from krm.config import Config
    from krm.phase_3_roles import discover_roles

    config = Config()
    roles_df = discover_roles(config)
"""

from __future__ import annotations

import json
import pickle
from typing import Any

import hdbscan
import numpy as np
import pandas as pd
import umap

from krm.config import Config
from krm.lib.embeddings import Embedder, build_title_characteristics
from krm.lib.io import read_parquet, write_parquet
from krm.lib.metrics import compute_clustering_metrics


def _label_cluster(
    cluster_id: int,
    member_indices: np.ndarray,
    unique_titles: list[str],
    embeddings: np.ndarray,
    top_k: int = 3,
) -> tuple[str, list[str], np.ndarray]:
    """Name a cluster by its most central job titles.

    Computes the centroid of the cluster in the original embedding space
    and returns the *top_k* titles closest to that centroid.

    Args:
        cluster_id: Label assigned by HDBSCAN.
        member_indices: Indices into ``unique_titles`` belonging to this cluster.
        unique_titles: All unique job titles.
        embeddings: Original high-dimensional embeddings (n_titles, dim).
        top_k: Number of central titles to return.

    Returns:
        (role_label, top_titles, centroid) tuple where ``role_label`` is the
        single most central title, ``top_titles`` is the ordered list of the
        *top_k* most central titles, and ``centroid`` is the mean embedding.
    """
    cluster_embs = embeddings[member_indices]
    centroid = cluster_embs.mean(axis=0)

    # Embeddings are already L2-normalized by the embedder,
    # so cosine similarity is the dot product with the normalized centroid.
    centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
    similarities = cluster_embs @ centroid_norm
    sorted_order = np.argsort(-similarities)  # descending

    top_order = sorted_order[: min(top_k, len(sorted_order))]
    top_titles = [unique_titles[member_indices[i]] for i in top_order]

    role_label = top_titles[0] if top_titles else f"role_{cluster_id}"

    return role_label, top_titles, centroid


def _serialize_centroid(emb: np.ndarray) -> bytes:
    """Serialize a numpy embedding vector to bytes for Parquet storage."""
    return pickle.dumps(emb.astype(np.float32))


def discover_roles(config: Config) -> pd.DataFrame:
    """Discover roles by clustering STEM job titles with HDBSCAN.

    Full pipeline:

    1. Reads ``classified.parquet`` from ``config.output_dir`` and filters
       rows where ``stem_category == "STEM_RESEARCH"``.
    2. Extracts unique job titles and generates embeddings via
       :class:`~src.krm.lib.embeddings.Embedder`.
    3. Reduces dimensionality with UMAP (cosine metric, ``random_state=42``)
       using parameters from ``config``.
    4. Clusters the UMAP-reduced embeddings with HDBSCAN (Euclidean metric)
       using parameters from ``config``.
    5. Computes soft membership vectors via
       ``clusterer.all_points_membership_vectors_`` for interdisciplinary
       role analysis.
    6. Names each cluster by its most central job titles (top-3 closest
       to the centroid in original embedding space).
    7. Computes silhouette and Davies-Bouldin metrics via
       :func:`~src.krm.lib.metrics.compute_clustering_metrics`.
    8. Writes ``roles.parquet`` to ``config.output_dir``.

    Args:
        config: Pipeline configuration providing ``embedding_model``,
            ``umap_n_components``, ``hdbscan_min_cluster_size``, and
            ``hdbscan_cluster_selection_epsilon``.

    Returns:
        DataFrame with one row per discovered role (including a noise row):

        =========================== ============================================
        column                      description
        =========================== ============================================
        ``role_id``                 int — HDBSCAN cluster label (-1 for noise)
        ``role_label``              str — most central job title
        ``centroid_embedding``      bytes — pickled ``np.float32`` centroid array
        ``top_titles``              str — JSON array of top-3 central titles
        ``member_count``            int — number of unique titles in the role
        ``noise_flag``              bool — ``True`` only for the noise row
        ``characteristic_profile``  str — JSON dict ``{characteristic_id: mean_confidence}``
        =========================== ============================================

    Raises:
        ValueError: If no ``STEM_RESEARCH`` vacancies exist in the input.
    """
    # ------------------------------------------------------------------
    # 1. Load classified data and filter to STEM_RESEARCH
    # ------------------------------------------------------------------
    classified_path = config.output_dir / "classified.parquet"
    df = read_parquet(classified_path)

    df_stem = df[df["stem_category"] == "STEM_RESEARCH"].copy()
    if len(df_stem) == 0:
        msg = "No STEM_RESEARCH vacancies found in classified.parquet"
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # 2. Extract unique job titles
    # ------------------------------------------------------------------
    unique_titles = sorted(df_stem["title"].dropna().unique().tolist())
    n_titles = len(unique_titles)
    print(f"[Phase 3] Clustering {n_titles} unique STEM job titles")

    # ------------------------------------------------------------------
    # 2a. Text-enrichment grounding via characteristic labels
    # ------------------------------------------------------------------
    characteristics_path = config.characteristics_path
    characteristics_df: pd.DataFrame | None = None
    enriched_titles: list[str] | None = None

    if characteristics_path.exists():
        characteristics_df = read_parquet(characteristics_path)
        # build_title_characteristics expects 'title' column;
        # characteristics.parquet uses 'extracted_title'.
        dims_for_mapping = characteristics_df.rename(
            columns={"extracted_title": "title"}
        )
        title_map = build_title_characteristics(dims_for_mapping, top_n=2)
        enriched_titles = Embedder.enrich_titles(unique_titles, title_map)
        n_grounded = sum(
            1 for t in unique_titles
            if t in title_map
        )
        print(
            f"[Phase 3] Text-enrichment grounded: "
            f"{n_grounded}/{n_titles} titles with characteristic context"
        )
    else:
        print(
            "[Phase 3] Text-enrichment skipped: "
            f"{characteristics_path} not found (bare titles used)"
        )

    # ------------------------------------------------------------------
    # 3. Generate embeddings (with optional enrichment)
    # ------------------------------------------------------------------
    embedder = Embedder(model_name=config.embedding_model)
    embeddings = embedder.encode(unique_titles, enriched_titles=enriched_titles)
    print(f"[Phase 3] Generated embeddings: {embeddings.shape}")

    # ------------------------------------------------------------------
    # 4. UMAP dimensionality reduction
    # ------------------------------------------------------------------
    n_components = min(config.umap_n_components, n_titles - 1, embeddings.shape[1])
    if n_components < 2:
        n_components = 2

    reducer = umap.UMAP(
        n_components=n_components,
        metric="cosine",
        min_dist=0.0,
        random_state=42,
    )
    umap_embeddings = reducer.fit_transform(embeddings)
    print(f"[Phase 3] UMAP reduced to {n_components} dimensions")

    # ------------------------------------------------------------------
    # 5. HDBSCAN clustering
    # ------------------------------------------------------------------
    min_cluster_size = min(config.hdbscan_min_cluster_size, n_titles // 2)
    if min_cluster_size < 2:
        min_cluster_size = 2

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        cluster_selection_epsilon=config.hdbscan_cluster_selection_epsilon,
        gen_min_span_tree=True,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(umap_embeddings)

    # 5a. Soft membership for interdisciplinary support
    #     (attribute renamed in newer hdbscan versions — fall back gracefully)
    try:
        soft_membership: np.ndarray | None = clusterer.all_points_membership_vectors_
    except AttributeError:
        soft_membership = None

    unique_labels = set(labels)
    n_clusters_real = len(unique_labels - {-1})
    n_noise = int((labels == -1).sum())
    print(
        f"[Phase 3] Found {n_clusters_real} clusters, "
        f"{n_noise} noise points "
        f"(soft membership: {soft_membership.shape if soft_membership is not None else 'N/A'})"
    )

    # ------------------------------------------------------------------
    # 6. Compute clustering quality metrics
    # ------------------------------------------------------------------
    metrics = compute_clustering_metrics(umap_embeddings, labels)
    print(f"[Phase 3] Clustering metrics: {metrics}")

    # ------------------------------------------------------------------
    # 7. Build role records
    # ------------------------------------------------------------------
    roles: list[dict[str, Any]] = []

    for cluster_id in sorted(unique_labels):
        mask = labels == cluster_id
        member_indices = np.where(mask)[0]
        member_titles: list[str] = [unique_titles[i] for i in member_indices]

        role_label, top_titles, centroid = _label_cluster(
            cluster_id,
            member_indices,
            unique_titles,
            embeddings,
            top_k=3,
        )

        characteristic_profile: dict[str, float] = {}
        if characteristics_df is not None:
            cluster_chars = characteristics_df[
                characteristics_df["extracted_title"].isin(member_titles)
            ]
            if not cluster_chars.empty:
                characteristic_profile = (
                    cluster_chars.groupby("characteristic_id")["confidence"]
                    .mean()
                    .round(4)
                    .to_dict()
                )

        roles.append(
            {
                "role_id": int(cluster_id),
                "role_label": role_label,
                "centroid_embedding": _serialize_centroid(centroid),
                "top_titles": json.dumps(top_titles, ensure_ascii=False),
                "member_count": int(mask.sum()),
                "noise_flag": bool(cluster_id == -1),
                "characteristic_profile": json.dumps(
                    characteristic_profile, ensure_ascii=False
                ),
            }
        )

    roles_df = pd.DataFrame(roles)

    # ------------------------------------------------------------------
    # 8. Write output
    # ------------------------------------------------------------------
    output_path = config.output_dir / "roles.parquet"
    write_parquet(roles_df, output_path)
    print(f"[Phase 3] Wrote {len(roles_df)} roles to {output_path}")

    return roles_df
