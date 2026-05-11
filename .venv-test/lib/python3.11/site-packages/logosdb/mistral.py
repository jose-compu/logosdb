"""Mistral AI embeddings integration for LogosDB.

This module provides a lightweight vector-store wrapper that uses
Mistral embeddings and persists vectors in LogosDB.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ._core import DB, DIST_COSINE


class MistralVectorStore:
    """Mistral embeddings + LogosDB vector store.

    Example:
        >>> store = MistralVectorStore("/tmp/db", api_key="...", model="mistral-embed")
        >>> store.add_texts(["doc one", "doc two"])
        >>> store.search("doc", top_k=2)
    """

    def __init__(
        self,
        uri: str,
        api_key: Optional[str] = None,
        model: str = "mistral-embed",
        dim: int = 1024,
        distance: int = DIST_COSINE,
        max_elements: int = 1_000_000,
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50,
    ) -> None:
        self._api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self._api_key:
            raise ValueError("Mistral API key required (api_key or MISTRAL_API_KEY)")

        self._model = model
        self._dim = dim
        self._db = DB(
            path=uri,
            dim=dim,
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=M,
            ef_search=ef_search,
            distance=distance,
        )

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model(self) -> str:
        return self._model

    def _request_embeddings(self, texts: Sequence[str]) -> List[List[float]]:
        """Call Mistral embeddings endpoint and return vectors."""
        import requests

        resp = requests.post(
            "https://api.mistral.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self._model, "input": list(texts)},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", [])
        return [row["embedding"] for row in data]

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Embed and insert texts. Returns inserted row IDs."""
        if not texts:
            return []

        embeddings = self._request_embeddings(texts)
        if len(embeddings) != len(texts):
            raise RuntimeError("mistral embeddings count mismatch")

        ids: List[int] = []
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            vec = np.asarray(emb, dtype=np.float32)
            if vec.ndim != 1 or vec.shape[0] != self._dim:
                raise ValueError(f"expected embedding dim={self._dim}, got {vec.shape}")

            ts = ""
            if metadatas and i < len(metadatas):
                ts = str(metadatas[i].get("timestamp", "")) if metadatas[i] else ""
            rid = self._db.put(vec, text=text, timestamp=ts)
            ids.append(int(rid))
        return ids

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search by query text."""
        if top_k <= 0:
            return []
        q = self._request_embeddings([query])[0]
        qvec = np.asarray(q, dtype=np.float32)
        if qvec.shape[0] != self._dim:
            raise ValueError(f"expected query dim={self._dim}, got {qvec.shape}")

        hits = self._db.search(qvec, top_k=top_k)
        return [
            {"id": h.id, "text": h.text, "score": h.score, "timestamp": h.timestamp}
            for h in hits
        ]

    def count(self) -> int:
        return int(self._db.count())


class MistralEmbeddingProvider:
    """Standalone embedding provider wrapper for Mistral."""

    def __init__(self, api_key: Optional[str] = None, model: str = "mistral-embed") -> None:
        self._api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self._api_key:
            raise ValueError("Mistral API key required (api_key or MISTRAL_API_KEY)")
        self._model = model

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        import requests

        resp = requests.post(
            "https://api.mistral.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self._model, "input": list(texts)},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        return [row["embedding"] for row in payload.get("data", [])]
