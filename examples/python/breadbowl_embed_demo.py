#!/usr/bin/env python3
"""BreadBowl-Embed v1.3 + LogosDB: encode, index routing slots, search, rerank.

Requires Python 3.12+ and breadbowl-embed from GitHub (not on PyPI):

    pip install "git+https://github.com/BreadBowlAI/breadbowl-embed.git"
    pip install .
    python3 examples/python/breadbowl_embed_demo.py
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

import logosdb
from logosdb import DIST_IP

from examples.python.breadbowl_logosdb import (
    ROUTING_DIM,
    SLOTS,
    aggregate_slot_hits,
    put_routing,
    rerank_passages,
    search_routing,
)

MODEL_ID = "dotproductx/breadbowl-embed-v1.3"

PASSAGES = {
    "octopus": (
        "The giant Pacific octopus has three hearts and blue blood. "
        "It can change color and texture to blend into rocky reefs."
    ),
    "ocean": (
        "The deep ocean holds most of Earth's water. "
        "Sunlight fades within the first few hundred meters."
    ),
    "reef": (
        "Coral reefs shelter thousands of species. "
        "Warm, clear water helps corals grow with their symbiotic algae."
    ),
}

QUESTION = "Which animal has blue blood and lives on reefs?"


def main() -> None:
    from breadbowl_embed import BreadBowl

    model = BreadBowl.from_pretrained(MODEL_ID)
    info = model.info()["architecture"]
    assert info["slots"] == SLOTS and info["d_k"] == ROUTING_DIM and info["d_v"] == ROUTING_DIM
    print(f"model {MODEL_ID}: slots={SLOTS} d_k=d_v={ROUTING_DIM}")

    texts = list(PASSAGES.values())
    ids = list(PASSAGES.keys())
    docs = model.encode_documents(texts, batch_size=1)
    q = model.encode_queries([QUESTION], batch_size=1)

    work = Path(tempfile.mkdtemp(prefix="logosdb-breadbowl-demo-"))
    slot_dir = work / "values"
    db_path = work / "routing"
    try:
        docs.without_tokens().save(slot_dir)
        stored = type(docs).load(slot_dir)

        n_rows = len(ids) * SLOTS
        db = logosdb.DB(
            str(db_path),
            dim=ROUTING_DIM,
            distance=DIST_IP,
            max_elements=n_rows,
            ef_search=512,
            M=16,
        )
        for i, pid in enumerate(ids):
            routing = np.asarray(docs.routing[i], dtype=np.float32)
            put_routing(db, pid, routing)
        print(f"indexed {len(ids)} passages ({db.count_live()} routing rows)")

        slot_lists = search_routing(db, np.asarray(q.routing[0], dtype=np.float32), slot_hits=64)
        ranked = aggregate_slot_hits(slot_lists)
        top_ids = [pid for pid, _ in ranked[: min(3, len(ranked))]]
        positions = [ids.index(pid) for pid in top_ids if pid in ids]
        scores = rerank_passages(model, q, stored, positions)
        order = np.argsort(-np.asarray(scores.combined))[0]
        best = ids[positions[int(order[0])]]
        print("routing top:", json.dumps(top_ids))
        print("rerank winner:", best)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()