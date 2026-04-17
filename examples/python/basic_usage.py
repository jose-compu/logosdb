"""Minimal LogosDB Python usage demo — no external embedding model required.

Run:
    pip install .
    python examples/python/basic_usage.py
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

import logosdb


def random_unit(rng: np.random.Generator, dim: int) -> np.ndarray:
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def main() -> None:
    dim = 128
    rng = np.random.default_rng(0)

    path = Path(tempfile.mkdtemp(prefix="logosdb-demo-"))
    try:
        db = logosdb.DB(str(path), dim=dim)
        print(f"opened {db}")

        # Insert a handful of labeled "memories".
        texts = [
            "the sun rises in the east",
            "the sun sets in the west",
            "a stitch in time saves nine",
            "rome was not built in a day",
            "to be or not to be",
        ]
        # put() returns a row id per vector (0, 1, 2, … for a fresh database).
        ids = [db.put(random_unit(rng, dim), text=t) for t in texts]
        print(f"inserted ids: {ids}")
        print(f"count={db.count()}  count_live={db.count_live()}")

        # Search: rebuild the same pseudo-random stream as above so "query"
        # matches the vector we stored as ids[1] (second text).
        rng2 = np.random.default_rng(0)
        _ = random_unit(rng2, dim)  # same draws as ids[0]
        query = random_unit(rng2, dim)  # same draw as ids[1]
        hits = db.search(query, top_k=3)
        print("top 3 hits:")
        for h in hits:
            print(f"  id={h.id} score={h.score:.4f} text={h.text!r}")

        # update(old_id, …) replaces one *live* row: it tombstones that id and
        # appends a new row; the return value is the NEW id (not the same as old_id).
        first_id = ids[0]  # same as 0 on a fresh DB — use the variable, not a magic 0
        new_id = db.update(first_id, random_unit(rng, dim), text="updated first memory")
        print(f"update({first_id}) -> new row id {new_id}")

        # delete(id) tombstones that row (it no longer appears in search).
        db.delete(ids[2])
        print(f"after update+delete: count={db.count()}  count_live={db.count_live()}")
        # count() is total rows ever stored (including tombstones); count_live() is active rows.

        # Zero-copy bulk view.
        vecs = db.raw_vectors()
        print(f"raw_vectors shape: {vecs.shape}  dtype: {vecs.dtype}  writeable: {vecs.flags.writeable}")

    finally:
        shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()
