"""Tests for on-disk storage dtype (float16 / int8)."""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

import logosdb
from logosdb import DB, DTYPE_FLOAT16, DTYPE_FLOAT32, DTYPE_INT8


DIM = 32


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


def test_float16_create_stats_reopen(db_path: Path):
    v = unit(10)
    db = DB(str(db_path), dim=DIM, dtype=DTYPE_FLOAT16)
    db.put(v, text="a")
    hits = db.search(v, top_k=1)
    assert hits[0].id == 0
    assert db.stats()["storage_dtype"] == DTYPE_FLOAT16
    del db

    db2 = DB(str(db_path), dim=DIM, dtype=DTYPE_FLOAT16)
    assert db2.count_live() == 1
    hits2 = db2.search(v, top_k=1)
    assert hits2[0].id == 0


def test_reopen_wrong_dtype_raises(db_path: Path):
    v = unit(11)
    db = DB(str(db_path), dim=DIM, dtype=DTYPE_FLOAT16)
    db.put(v)
    del db

    with pytest.raises(RuntimeError, match="dtype mismatch"):
        DB(str(db_path), dim=DIM, dtype=DTYPE_FLOAT32)


def test_int8_create_and_search(db_path: Path):
    v = unit(22)
    db = DB(str(db_path), dim=DIM, dtype=DTYPE_INT8)
    db.put(v, text="int8")
    hits = db.search(v, top_k=1)
    assert len(hits) == 1
    assert db.stats()["storage_dtype"] == DTYPE_INT8


def test_dtype_constants_exported():
    assert logosdb.DTYPE_FLOAT32 == 0
    assert logosdb.DTYPE_FLOAT16 == 1
    assert logosdb.DTYPE_INT8 == 2