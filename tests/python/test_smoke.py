"""Smoke tests for the LogosDB Python bindings."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

import logosdb
from logosdb import DB, SearchHit


DIM = 64


def unit(seed: int, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "db"
    yield p
    shutil.rmtree(p, ignore_errors=True)


def test_version_exported():
    assert isinstance(logosdb.__version__, str)
    assert logosdb.__version__


def test_open_close(db_path: Path):
    db = DB(str(db_path), dim=DIM)
    assert db.dim == DIM
    assert db.count() == 0
    assert db.count_live() == 0
    assert len(db) == 0


def test_put_and_search(db_path: Path):
    db = DB(str(db_path), dim=DIM)

    v0 = unit(100)
    v1 = unit(200)
    v2 = unit(300)

    id0 = db.put(v0, text="fact zero", timestamp="2025-01-01T00:00:00Z")
    id1 = db.put(v1, text="fact one")
    id2 = db.put(v2)

    assert (id0, id1, id2) == (0, 1, 2)
    assert db.count() == 3
    assert db.count_live() == 3
    assert len(db) == 3

    hits = db.search(v0, top_k=3)
    assert len(hits) == 3
    assert isinstance(hits[0], SearchHit)
    assert hits[0].id == 0
    assert hits[0].text == "fact zero"
    assert hits[0].timestamp == "2025-01-01T00:00:00Z"
    assert hits[0].score > 0.9


def test_delete(db_path: Path):
    db = DB(str(db_path), dim=DIM)
    v0, v1, v2 = unit(1), unit(2), unit(3)
    db.put(v0, "a")
    db.put(v1, "b")
    db.put(v2, "c")

    db.delete(1)
    assert db.count() == 3
    assert db.count_live() == 2

    hits = db.search(v1, top_k=3)
    assert all(h.id != 1 for h in hits)


def test_delete_errors(db_path: Path):
    db = DB(str(db_path), dim=DIM)
    db.put(unit(10), "only")

    with pytest.raises(RuntimeError):
        db.delete(99)

    db.delete(0)
    with pytest.raises(RuntimeError):
        db.delete(0)


def test_update(db_path: Path):
    db = DB(str(db_path), dim=DIM)
    v0 = unit(1)
    v1 = unit(2)
    v_new = unit(99)
    db.put(v0, "old zero")
    db.put(v1, "keep one")

    new_id = db.update(0, v_new, text="new zero", timestamp="2025-07-01T00:00:00Z")
    assert new_id == 2
    assert db.count() == 3
    assert db.count_live() == 2

    hits = db.search(v_new, top_k=1)
    assert hits[0].id == new_id
    assert hits[0].text == "new zero"
    assert hits[0].timestamp == "2025-07-01T00:00:00Z"

    old_hits = db.search(v0, top_k=3)
    assert all(h.id != 0 for h in old_hits)


def test_dim_mismatch_raises(db_path: Path):
    db = DB(str(db_path), dim=DIM)
    wrong = np.zeros(DIM + 10, dtype=np.float32)
    wrong[0] = 1.0
    with pytest.raises(RuntimeError):
        db.put(wrong, "bad")


def test_non_1d_raises(db_path: Path):
    db = DB(str(db_path), dim=DIM)
    bad = np.zeros((2, DIM), dtype=np.float32)
    with pytest.raises(ValueError):
        db.put(bad, "bad")


def test_dtype_coercion(db_path: Path):
    db = DB(str(db_path), dim=DIM)
    v = unit(42).astype(np.float64)
    rid = db.put(v, "coerced")
    assert rid == 0
    hits = db.search(v.astype(np.float32), top_k=1)
    assert hits[0].id == 0


def test_raw_vectors_zero_copy(db_path: Path):
    db = DB(str(db_path), dim=DIM)
    v0 = unit(500)
    v1 = unit(600)
    db.put(v0, "a")
    db.put(v1, "b")

    raw = db.raw_vectors()
    assert raw.shape == (2, DIM)
    assert raw.dtype == np.float32
    assert not raw.flags.writeable
    assert np.allclose(raw[0], v0, atol=1e-6)
    assert np.allclose(raw[1], v1, atol=1e-6)


def test_persistence(tmp_path: Path):
    path = tmp_path / "persist"
    v0 = unit(1000)
    v1 = unit(2000)

    db1 = DB(str(path), dim=DIM)
    db1.put(v0, "A", "2025-04-01T00:00:00Z")
    db1.put(v1, "B", "2025-05-01T00:00:00Z")
    db1.delete(0)
    del db1

    db2 = DB(str(path), dim=DIM)
    assert db2.count() == 2
    assert db2.count_live() == 1

    hits = db2.search(v0, top_k=2)
    assert all(h.id != 0 for h in hits)

    hits_b = db2.search(v1, top_k=1)
    assert hits_b[0].text == "B"
    assert hits_b[0].timestamp == "2025-05-01T00:00:00Z"


def test_repr(db_path: Path):
    db = DB(str(db_path), dim=DIM)
    r = repr(db)
    assert "DB" in r and "dim=" in r and "count=" in r


def test_many_vectors(db_path: Path):
    db = DB(str(db_path), dim=32, max_elements=5000)
    n = 500
    vecs = [unit(i, dim=32) for i in range(n)]
    for i, v in enumerate(vecs):
        db.put(v, f"row_{i}")
    assert db.count() == n

    hit = db.search(vecs[123], top_k=1)[0]
    assert hit.id == 123
    assert hit.text == "row_123"
