"""Tests for phase_3_roles.py — HDBSCAN role clustering."""

import numpy as np
import pytest

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
