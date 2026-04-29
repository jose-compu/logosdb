"""Tests for OpenAI integration."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from logosdb.openai import OpenAIVectorStore


def _fake_embedding(text: str, dim: int) -> List[float]:
    idx = abs(hash(("openai", text))) % dim
    vec = [0.0] * dim
    vec[idx] = 1.0
    return vec


def test_openai_store_and_search(tmp_path: Path):
    store = OpenAIVectorStore(str(tmp_path / "openai_db"), api_key="test-key", dim=64)
    store._request_embeddings = lambda texts: [_fake_embedding(t, 64) for t in texts]  # type: ignore[attr-defined]

    ids = store.add_texts(["alpha", "beta", "gamma"])
    assert ids == [0, 1, 2]
    assert store.count() == 3

    hits = store.search("beta", top_k=3)
    assert len(hits) >= 1
    assert hits[0]["text"] == "beta"


def test_openai_missing_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OpenAI API key required"):
        OpenAIVectorStore(str(tmp_path / "openai_db"))
