"""Tests for phase_3_roles.py — HDBSCAN role clustering."""

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from krm import phase_3_roles
from krm.config import Config
from krm.lib.llm import LLMClient
from krm.lib.llm_role_names import RoleNameResponse, name_roles
from krm.phase_3_roles import (
    _label_cluster,
    _serialize_centroid,
)


class TestLabelCluster:
    def test_single_title(self):
        titles = ["физик-экспериментатор"]
        emb = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        label, top, centroid = _label_cluster(0, np.array([0]), titles, emb)
        assert "физик-экспериментатор" in label
        assert len(top) == 1

    def test_multiple_titles_returns_central(self):
        titles = ["физик-экспериментатор", "научный сотрудник", "химик"]
        emb = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        label, top, centroid = _label_cluster(0, np.array([0, 1, 2]), titles, emb)
        assert isinstance(label, str)
        assert len(label) > 0
        assert len(top) >= 1
        assert len(centroid) == 3


class TestSerializeCentroid:
    def test_returns_bytes(self):
        emb = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = _serialize_centroid(emb)
        assert isinstance(result, bytes)

    def test_roundtrip(self):
        import pickle

        emb = np.random.default_rng(42).random((16,)).astype(np.float32)
        serialized = _serialize_centroid(emb)
        restored = pickle.loads(serialized)
        assert np.allclose(emb, restored)


def _role_frame() -> pd.DataFrame:
    """Three roles: two real clusters plus the noise row (-1)."""
    return pd.DataFrame([
        {
            "role_id": 0,
            "role_label": "data scientist",
            "centroid_embedding": b"\x00\x01\x02",
            "top_titles": json.dumps(
                ["data scientist", "ml engineer", "research scientist"]
            ),
            "member_count": 3,
            "noise_flag": False,
            "characteristic_profile": json.dumps({"0": 0.9, "3": 0.6}),
        },
        {
            "role_id": 1,
            "role_label": "химик-аналитик",
            "centroid_embedding": b"\x00\x03\x04",
            "top_titles": json.dumps(["химик-аналитик"]),
            "member_count": 1,
            "noise_flag": False,
            "characteristic_profile": json.dumps({"0": 0.8}),
        },
        {
            "role_id": -1,
            "role_label": "role_-1",
            "centroid_embedding": b"\x00\x05",
            "top_titles": json.dumps([]),
            "member_count": 2,
            "noise_flag": True,
            "characteristic_profile": json.dumps({}),
        },
    ])


def _llm_config() -> MagicMock:
    """Config-like object with cache_dir=None so no on-disk cache is written."""
    config = MagicMock()
    config.llm_base_url = "http://localhost:8080/v1"
    config.llm_model = "test-model"
    config.llm_concurrency = 2
    config.llm_temperature = 0.0
    config.llm_max_tokens = 256
    config.llm_max_retries = 1
    config.llm_cache_dir = None
    return config


class TestLlmNameRoles:
    def test_llm_result_populates_label(self, monkeypatch):
        sent: list[str] = []

        async def fake_run_many(self, prompts, schema):
            assert schema is RoleNameResponse
            sent.extend(p.user for p in prompts)
            return [
                RoleNameResponse(label="наука о данных"),
                RoleNameResponse(label="аналитическая химия"),
            ]

        monkeypatch.setattr(LLMClient, "run_many", fake_run_many)
        labels = name_roles(_role_frame(), _llm_config())

        # (a) LLM label lands on the non-noise rows, in input order.
        assert labels.iloc[0] == "наука о данных"
        assert labels.iloc[1] == "аналитическая химия"
        # Noise row keeps its deterministic role_label and is never prompted.
        assert labels.iloc[2] == "role_-1"
        assert len(sent) == 2
        # Prompt carries the top-3 central titles and the characteristic profile.
        assert "data scientist" in sent[0]
        assert "Характеристический профиль" in sent[0]
        assert "0.9" in sent[0]

    def test_none_falls_back_to_role_label(self, monkeypatch):
        async def fake_run_many(self, prompts, schema):
            return [None, RoleNameResponse(label="")]

        monkeypatch.setattr(LLMClient, "run_many", fake_run_many)
        labels = name_roles(_role_frame(), _llm_config())

        assert labels.iloc[0] == "data scientist"
        assert labels.iloc[1] == "химик-аналитик"
        assert labels.iloc[2] == "role_-1"

    def test_does_not_mutate_input_columns(self, monkeypatch):
        async def fake_run_many(self, prompts, schema):
            return [RoleNameResponse(label="наука о данных") for _ in prompts]

        monkeypatch.setattr(LLMClient, "run_many", fake_run_many)
        roles = _role_frame()
        before = roles.copy(deep=True)
        name_roles(roles, _llm_config())
        pd.testing.assert_frame_equal(roles, before)


class _FakeEmbedder:
    def __init__(self, model_name=None):
        self.model_name = model_name

    def encode(self, titles, enriched_titles=None):
        n = len(titles)
        return np.arange(n * 4, dtype=np.float32).reshape(n, 4)


class _FakeUMAP:
    def __init__(self, **kwargs):
        pass

    def fit_transform(self, embeddings):
        n = embeddings.shape[0]
        return np.arange(n * 2, dtype=np.float64).reshape(n, 2)


class _FakeHDBSCAN:
    all_points_membership_vectors_ = None

    def __init__(self, **kwargs):
        pass

    def fit_predict(self, umap_embeddings):
        return np.array([0, 0, 1, 1, -1, -1])


def _classified_df() -> pd.DataFrame:
    return pd.DataFrame({
        "stem_category": ["STEM_RESEARCH"] * 6,
        "title": [
            "химик-аналитик", "биоинформатик", "физик-экспериментатор",
            "научный сотрудник", "инженер-исследователь", "лаборант",
        ],
    })


class TestDiscoverRolesLlm:
    def test_llm_column_added_without_touching_clustering(
        self, tmp_path, monkeypatch
    ):
        # Deterministic, stateless stand-ins for the clustering stack.
        monkeypatch.setattr(phase_3_roles, "read_parquet", lambda p: _classified_df())
        monkeypatch.setattr(phase_3_roles, "Embedder", _FakeEmbedder)
        monkeypatch.setattr(phase_3_roles.umap, "UMAP", _FakeUMAP)
        monkeypatch.setattr(phase_3_roles.hdbscan, "HDBSCAN", _FakeHDBSCAN)
        monkeypatch.setattr(
            phase_3_roles, "compute_clustering_metrics", lambda _e, _labels: {}
        )

        use_llm = {"enabled": False}
        monkeypatch.setattr(
            Config, "use_llm_phase_3", property(lambda self: use_llm["enabled"])
        )
        monkeypatch.setattr(Config, "output_dir", property(lambda self: tmp_path))
        monkeypatch.setattr(
            Config,
            "characteristics_path",
            property(lambda self: tmp_path / "nonexistent.parquet"),
        )
        monkeypatch.setattr(
            Config,
            "llm_cache_dir",
            property(lambda self: str(tmp_path / "llm_cache")),
        )

        config = Config()

        # Baseline: deterministic clustering, no LLM column.
        base = phase_3_roles.discover_roles(config)
        assert "llm_role_label" not in base.columns

        # Opt into LLM naming and stub the shared client's batch runner.
        async def fake_run_many(self, prompts, schema):
            assert schema is RoleNameResponse
            return [
                RoleNameResponse(label=f"llm-{i}") for i in range(len(prompts))
            ]

        use_llm["enabled"] = True
        monkeypatch.setattr(LLMClient, "run_many", fake_run_many)
        llm = phase_3_roles.discover_roles(config)

        # (a) + (b): llm_role_label populated, noise falls back to role_label.
        assert "llm_role_label" in llm.columns
        non_noise = llm[~llm["noise_flag"]]
        assert sorted(non_noise["llm_role_label"]) == ["llm-0", "llm-1"]
        noise = llm[llm["noise_flag"]]
        assert (noise["llm_role_label"] == noise["role_label"]).all()

        # (c) clustering/role_label/centroid_embedding/top_titles unchanged.
        pd.testing.assert_series_equal(base["role_label"], llm["role_label"])
        pd.testing.assert_series_equal(
            base["centroid_embedding"], llm["centroid_embedding"]
        )
        pd.testing.assert_series_equal(base["top_titles"], llm["top_titles"])
        pd.testing.assert_series_equal(
            base["member_count"], llm["member_count"]
        )
