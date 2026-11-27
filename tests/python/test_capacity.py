"""HNSW auto-grow and max_capacity (1.8.0)."""

from __future__ import annotations

import shutil
import tempfile
import threading

import numpy as np
import pytest

import logosdb


def _unit(dim: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-12
    return v


def test_grow_reopen_and_cap():
    dim = 8
    td = tempfile.mkdtemp(prefix="logosdb-cap-")
    try:
        db = logosdb.DB(td, dim=dim, max_elements=16, max_capacity=0)
        for i in range(20):
            db.put(_unit(dim, 100 + i))
        assert db.index_capacity() == 32
        db.sync()
        db2 = logosdb.DB(td, dim=dim, max_elements=16)
        assert db2.index_capacity() == 32
        for i in range(13):
            db2.put(_unit(dim, 200 + i))
        assert db2.index_capacity() == 64

        cap_path = tempfile.mkdtemp(prefix="logosdb-cap2-")
        try:
            dbc = logosdb.DB(cap_path, dim=dim, max_elements=16, max_capacity=20)
            for i in range(20):
                dbc.put(_unit(dim, 300 + i))
            with pytest.raises(RuntimeError, match="hnsw capacity cap exceeded"):
                dbc.put(_unit(dim, 999))
            assert dbc.count() == 20
        finally:
            shutil.rmtree(cap_path, ignore_errors=True)

        batch_path = tempfile.mkdtemp(prefix="logosdb-cap3-")
        try:
            dbb = logosdb.DB(batch_path, dim=dim, max_elements=16)
            flat = np.stack([_unit(dim, 400 + i) for i in range(40)], axis=0)
            ids = dbb.put_batch(flat)
            assert len(ids) == 40
            assert dbb.index_capacity() == 64
        finally:
            shutil.rmtree(batch_path, ignore_errors=True)
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_gil_released_during_search_batch():
    dim = 16
    td = tempfile.mkdtemp(prefix="logosdb-gil-")
    try:
        db = logosdb.DB(td, dim=dim, max_elements=4096, ef_search=64)
        for i in range(2000):
            db.put(_unit(dim, 500 + i))
        q = _unit(dim, 501)
        counter = {"n": 0}

        def tick():
            for _ in range(50_000):
                counter["n"] += 1

        th = threading.Thread(target=tick)
        th.start()
        batch_ids, batch_scores = db.search_batch(np.stack([q], axis=0), k=5)
        th.join()
        assert counter["n"] > 0
        one = db.search(q, 5)
        assert len(one) == len(batch_ids[0])
        for i, hit in enumerate(one):
            assert batch_ids[0, i] == hit.id
            assert abs(batch_scores[0, i] - hit.score) < 1e-5
    finally:
        shutil.rmtree(td, ignore_errors=True)