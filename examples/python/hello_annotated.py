"""Smallest LogosDB Python path with teaching comments (open / put / search / close).

Run:
    pip install .
    python examples/python/hello_annotated.py

Comments explain public API usage only. Behavior is a minimal put+search demo.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

import logosdb


def unit_vector(dim: int, seed: int) -> np.ndarray:
    """Deterministic L2-normalized vector (demo stand-in for a real embedding)."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def main() -> None:
    dim = 64
    path = Path(tempfile.mkdtemp(prefix="logosdb-hello-"))
    try:
        # --- open DB ---
        # logosdb.DB(path, dim=...) creates or opens an on-disk index at `path`.
        # `dim` is the embedding width and must match every put/search vector.
        db = logosdb.DB(str(path), dim=dim)
        print(f"opened {db}")

        # --- put vector + text ---
        # put(embedding, text=...) appends one live row and returns its row id.
        # Text is optional metadata returned on search hits when present.
        emb = unit_vector(dim, seed=1)
        row_id = db.put(emb, text="hello from logosdb")
        print(f"put id={row_id} count_live={db.count_live()}")

        # --- search top_k ---
        # search(query, top_k=N) returns the N nearest *live* rows under the DB
        # distance metric. Each hit exposes id, score, and optional text.
        query = unit_vector(dim, seed=1)  # same vector => the row we just wrote ranks first
        hits = db.search(query, top_k=3)
        print("top_k hits:")
        for h in hits:
            print(f"  id={h.id} score={h.score:.4f} text={h.text!r}")

        # --- close / cleanup ---
        # Python releases the handle when `db` goes out of scope. Remove the temp
        # directory so repeated runs start from an empty database.
        del db
    finally:
        shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()
