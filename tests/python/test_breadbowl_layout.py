"""Layout tests for BreadBowl-Embed routing rows in LogosDB (no checkpoint)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

import logosdb
from logosdb import DIST_IP

from examples.python.breadbowl_logosdb import (
    SLOTS,
    ROUTING_DIM,
    aggregate_slot_hits,
    parse_slot_text,
    put_routing,
    search_routing,
    slot_text,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "routing"
    yield p
    shutil.rmtree(p, ignore_errors=True)


def test_slot_text_round_trip():
    raw = slot_text("doc-42", 7)
    assert raw == '{"passage_id":"doc-42","slot":7}'
    assert parse_slot_text(raw) == ("doc-42", 7)


def test_two_passages_sixteen_rows_each(db_path: Path):
    db = logosdb.DB(
        str(db_path),
        dim=ROUTING_DIM,
        distance=DIST_IP,
        max_elements=2 * SLOTS,
        ef_search=64,
        M=16,
    )
    rng = np.random.default_rng(0)

    def unit(seed: int) -> np.ndarray:
        v = rng.standard_normal((SLOTS, ROUTING_DIM)).astype(np.float32)
        for s in range(SLOTS):
            n = np.linalg.norm(v[s])
            v[s] /= n if n > 0 else 1.0
        return v

    r0 = unit(1)
    r1 = unit(2)
    put_routing(db, "alpha", r0)
    put_routing(db, "beta", r1)
    assert db.count_live() == 2 * SLOTS

    q = r0[3]
    hits = db.search(q, top_k=5)
    texts = [h.text for h in hits]
    assert any(parse_slot_text(t)[0] == "alpha" for t in texts)


def test_aggregate_one_hit_rest_miss():
    one_slot = [("only", 0.9)]
    lists = [one_slot if i == 0 else [] for i in range(SLOTS)]
    ranked = aggregate_slot_hits(lists)
    assert ranked[0][0] == "only"
    expected = (0.9 + 15 * (-1.0)) / 16
    assert ranked[0][1] == pytest.approx(expected)


def test_search_routing_finds_passage(db_path: Path):
    db = logosdb.DB(
        str(db_path),
        dim=ROUTING_DIM,
        distance=DIST_IP,
        max_elements=SLOTS,
        ef_search=64,
    )
    rng = np.random.default_rng(1)
    routing = rng.standard_normal((SLOTS, ROUTING_DIM)).astype(np.float32)
    for s in range(SLOTS):
        n = np.linalg.norm(routing[s])
        routing[s] /= n if n > 0 else 1.0
    put_routing(db, "p1", routing)

    slot_lists = search_routing(db, routing, slot_hits=32)
    ranked = aggregate_slot_hits(slot_lists)
    assert ranked and ranked[0][0] == "p1"
    assert ranked[0][1] > 0.5