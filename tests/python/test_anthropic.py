"""Tests for Anthropic integration."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

from logosdb.anthropic import AnthropicVectorStore


def _fake_embedding(text: str, dim: int) -> List[float]:
    idx = abs(hash(("anthropic", text))) % dim
    vec = [0.0] * dim
    vec[idx] = 1.0
    return vec


def test_anthropic_store_and_search(tmp_path: Path):
    store = AnthropicVectorStore(str(tmp_path / "anthropic_db"), api_key="test-key", dim=64)
    store._request_embeddings = lambda texts: [_fake_embedding(t, 64) for t in texts]  # type: ignore[attr-defined]

    ids = store.add_texts(["lion", "tiger", "bear"])
    assert ids == [0, 1, 2]
    assert store.count() == 3

    hits = store.search("tiger", top_k=3)
    assert len(hits) >= 1
    assert hits[0]["text"] == "tiger"


def test_anthropic_missing_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Anthropic API key required"):
        AnthropicVectorStore(str(tmp_path / "anthropic_db"))
