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
        ids = [db.put(random_unit(rng, dim), text=t) for t in texts]
        print(f"inserted ids: {ids}")
        print(f"count={db.count()}  count_live={db.count_live()}")

        # Search near one of the inserted vectors (we recompute it for the demo).
        rng2 = np.random.default_rng(0)
        _ = random_unit(rng2, dim)  # id 0
        query = random_unit(rng2, dim)  # id 1
        hits = db.search(query, top_k=3)
        print("top 3 hits:")
        for h in hits:
            print(f"  id={h.id} score={h.score:.4f} text={h.text!r}")

        # Update and delete.
        new_id = db.update(0, random_unit(rng, dim), text="updated first memory")
        print(f"update(0) -> new id {new_id}")
        db.delete(2)
        print(f"after update+delete: count={db.count()}  count_live={db.count_live()}")

        # Zero-copy bulk view.
        vecs = db.raw_vectors()
        print(f"raw_vectors shape: {vecs.shape}  dtype: {vecs.dtype}  writeable: {vecs.flags.writeable}")

    finally:
        shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()
