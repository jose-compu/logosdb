"""Tests for Mistral integration."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from logosdb.mistral import MistralVectorStore


def _fake_embedding(text: str, dim: int) -> List[float]:
    # Deterministic one-hot-ish vector by text hash
    idx = abs(hash(text)) % dim
    vec = [0.0] * dim
    vec[idx] = 1.0
    return vec


def test_mistral_store_and_search(tmp_path: Path):
    store = MistralVectorStore(str(tmp_path / "mistral_db"), api_key="test-key", dim=64)

    # Mock API call to avoid network in tests
    store._request_embeddings = lambda texts: [_fake_embedding(t, 64) for t in texts]  # type: ignore[attr-defined]

    ids = store.add_texts(
        ["apple", "banana", "carrot"],
        metadatas=[{"timestamp": "2026-01-01T00:00:00Z"}, {}, {}],
    )
    assert ids == [0, 1, 2]
    assert store.count() == 3

    hits = store.search("banana", top_k=3)
    assert len(hits) >= 1
    assert hits[0]["text"] == "banana"


def test_mistral_missing_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Mistral API key required"):
        MistralVectorStore(str(tmp_path / "mistral_db"))
