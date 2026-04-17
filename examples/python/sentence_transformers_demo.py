"""LogosDB + sentence-transformers semantic-search demo.

Requires the optional `examples` extra:

    pip install ".[examples]"
    python examples/python/sentence_transformers_demo.py
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

import logosdb


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def main() -> None:
    from sentence_transformers import SentenceTransformer  # lazy import

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"loading model: {model_name}")
    model = SentenceTransformer(model_name)
    dim = model.get_sentence_embedding_dimension()

    corpus = [
        "Vector databases enable semantic search over text.",
        "HNSW is a graph-based approximate nearest neighbor algorithm.",
        "Memory-mapped files provide zero-copy access to disk data.",
        "Transformers revolutionized natural language processing in 2017.",
        "Inner product search on L2-normalized vectors equals cosine similarity.",
        "ChromaDB is a popular Python vector store built on SQLite.",
        "Apple Silicon uses an ARM64 architecture with unified memory.",
        "Rome was not built in a day.",
    ]

    path = Path(tempfile.mkdtemp(prefix="logosdb-st-"))
    try:
        db = logosdb.DB(str(path), dim=dim)
        print(f"opened {db}")

        embeddings = l2_normalize(np.asarray(model.encode(corpus)))
        for text, vec in zip(corpus, embeddings):
            db.put(vec, text=text)
        print(f"inserted {db.count_live()} rows")

        queries = [
            "What is HNSW?",
            "How do vector databases work?",
            "Tell me something about Italy.",
        ]
        query_embeddings = l2_normalize(np.asarray(model.encode(queries)))

        for query, q_vec in zip(queries, query_embeddings):
            print(f"\nquery: {query!r}")
            for h in db.search(q_vec, top_k=3):
                print(f"  score={h.score:.4f}  {h.text}")

    finally:
        shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    main()
