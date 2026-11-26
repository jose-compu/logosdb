import math

import numpy as np
import pytest

import logosdb


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "sp"
    d = logosdb.DB(str(path), dim=8, ef_search=32)
    vecs = [np.random.randn(8).astype(np.float32) for _ in range(6)]
    for v in vecs:
        v /= np.linalg.norm(v)
        d.put(v)
    return d, vecs


def test_search_uncapped_without_filters(db):
    d, vecs = db
    hits = d.search(vecs[0], top_k=5000)
    assert len(hits) >= 1


def test_deny_filter(db):
    d, vecs = db
    deny = np.array([0], dtype=np.uint64)
    hits = d.search(vecs[1], top_k=5, deny=deny)
    assert all(h.id != 0 for h in hits)


def test_batch_padding(db):
    d, vecs = db
    q = np.stack(vecs[:2], axis=0)
    ids, scores = d.search_batch(q, k=4, min_score=2.0)
    assert ids.shape == (2, 4)
    assert np.isnan(scores).any()


def test_get_many(db):
    d, vecs = db
    many = d.get_many(np.array([0, 1, 2], dtype=np.uint64))
    assert many.shape == (3, 8)
    one = d.get(1)
    assert np.allclose(many[1], one.embedding, atol=1e-4)