"""Integration adapter smoke tests for LogosDB wrappers."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from logosdb.camel import LogosDBVectorMemory
from logosdb.cognee import LogosDBVectorStore as CogneeVectorStore
from logosdb.crewai import LogosDBKnowledgeSource
from logosdb.superlinked import LogosDBVectorIndex


DIM = 32


def unit(seed: int, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def _cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def test_camel_vector_memory_add_search(tmp_path: Path) -> None:
    path = tmp_path / "camel_db"
    mem = LogosDBVectorMemory(uri=str(path), dim=DIM)
    v0 = unit(1)
    v1 = unit(2)
    mem.add(v0, "camel memory one")
    mem.add(v1, "camel memory two")
    hits = mem.search(v0, top_k=2)
    assert len(hits) == 2
    assert hits[0]["text"] == "camel memory one"
    assert mem.count() == 2
    _cleanup(path)


def test_crewai_knowledge_source_add_search(tmp_path: Path) -> None:
    path = tmp_path / "crewai_db"
    ks = LogosDBKnowledgeSource(uri=str(path), dim=DIM, collection="research_docs")
    v0 = unit(10)
    v1 = unit(20)
    ids = ks.add(
        texts=["crew doc one", "crew doc two"],
        embeddings=[v0, v1],
        metadatas=[{"timestamp": "2026-01-01"}, {"timestamp": "2026-01-02"}],
    )
    assert len(ids) == 2
    hits = ks.search(v0, top_k=1)
    assert hits[0]["text"] == "crew doc one"
    assert ks.count_live() == 2
    _cleanup(path)


def test_cognee_vector_store_upsert_search(tmp_path: Path) -> None:
    path = tmp_path / "cognee_db"
    store = CogneeVectorStore(uri=str(path), dim=DIM, collection="kg_nodes")
    v0 = unit(100)
    v1 = unit(200)
    ids = store.upsert(
        vectors=[v0, v1],
        texts=["node: authentication", "node: billing"],
    )
    assert len(ids) == 2
    hits = store.search(v0, top_k=2)
    assert hits[0]["text"] == "node: authentication"
    assert store.count() == 2
    _cleanup(path)


def test_superlinked_vector_index_upsert_query(tmp_path: Path) -> None:
    path = tmp_path / "superlinked_db"
    index = LogosDBVectorIndex(uri=str(path), dim=DIM, index_name="products")
    v0 = unit(1000)
    v1 = unit(2000)
    ids = index.upsert(
        vectors=[v0, v1],
        payloads=[
            {"text": "product: laptop", "timestamp": "2026-02-01"},
            {"text": "product: keyboard", "timestamp": "2026-02-02"},
        ],
    )
    assert len(ids) == 2
    hits = index.query(v0, top_k=1)
    assert hits[0]["text"] == "product: laptop"
    assert index.count() == 2
    _cleanup(path)

