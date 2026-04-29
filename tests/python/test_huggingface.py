"""Tests for Hugging Face integration."""

from __future__ import annotations

from pathlib import Path
from typing import List

from logosdb.huggingface import HuggingFaceVectorStore


def _fake_embedding(text: str, dim: int) -> List[float]:
    idx = abs(hash(("hf", text))) % dim
    vec = [0.0] * dim
    vec[idx] = 1.0
    return vec


class _DummyEmbedder:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        import numpy as np

        rows = [_fake_embedding(t, 64) for t in texts]
        return np.asarray(rows, dtype=np.float32)


def test_huggingface_store_and_search(tmp_path: Path, monkeypatch):
    # avoid importing sentence-transformers in tests
    monkeypatch.setattr(HuggingFaceVectorStore, "_build_embedder", staticmethod(lambda model, device: _DummyEmbedder()))

    store = HuggingFaceVectorStore(str(tmp_path / "hf_db"), dim=64)
    ids = store.add_texts(["red", "green", "blue"])
    assert ids == [0, 1, 2]
    assert store.count() == 3

    hits = store.search("green", top_k=3)
    assert len(hits) >= 1
    assert hits[0]["text"] == "green"
