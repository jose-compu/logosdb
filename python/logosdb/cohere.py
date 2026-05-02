"""Cohere embeddings integration for LogosDB.

Supports Cohere's embed-english-v3.0 and embed-multilingual-v3.0 models.
Automatically selects the correct input_type for indexing vs. querying.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ._core import DB, DIST_COSINE


_COHERE_MODEL_DIMS: Dict[str, int] = {
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
}

_COHERE_EMBED_URL = "https://api.cohere.com/v2/embed"


class CohereVectorStore:
    """Cohere embeddings + LogosDB vector store.

    Example:
        >>> store = CohereVectorStore("/tmp/db", api_key="...", model="embed-english-v3.0")
        >>> store.add_texts(["document one", "document two"])
        >>> store.search("document", top_k=2)
    """

    def __init__(
        self,
        uri: str,
        api_key: Optional[str] = None,
        model: str = "embed-english-v3.0",
        dim: Optional[int] = None,
        distance: int = DIST_COSINE,
        max_elements: int = 1_000_000,
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50,
    ) -> None:
        self._api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self._api_key:
            raise ValueError("Cohere API key required (api_key or COHERE_API_KEY env var)")

        self._model = model
        self._dim = dim or _COHERE_MODEL_DIMS.get(model, 1024)
        self._db = DB(
            path=uri,
            dim=self._dim,
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

    def _request_embeddings(
        self, texts: Sequence[str], input_type: str = "search_document"
    ) -> List[List[float]]:
        """Call Cohere v2 embed endpoint and return float vectors."""
        import requests

        resp = requests.post(
            _COHERE_EMBED_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "texts": list(texts),
                "input_type": input_type,
                "embedding_types": ["float"],
            },
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload["embeddings"]["float"]

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Embed and insert texts. Returns inserted row IDs."""
        if not texts:
            return []

        embeddings = self._request_embeddings(texts, input_type="search_document")
        if len(embeddings) != len(texts):
            raise RuntimeError("cohere embeddings count mismatch")

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
        """Search by query text. Query is embedded as 'search_query' input type."""
        if top_k <= 0:
            return []
        q = self._request_embeddings([query], input_type="search_query")[0]
        qvec = np.asarray(q, dtype=np.float32)
        if qvec.shape[0] != self._dim:
            raise ValueError(f"expected query dim={self._dim}, got {qvec.shape}")
        hits = self._db.search(qvec, top_k=top_k)
        return [
            {"id": h.id, "text": h.text, "score": h.score, "timestamp": h.timestamp}
            for h in hits
        ]

    def delete(self, row_id: int) -> None:
        """Delete a row by its ID."""
        self._db.delete(row_id)

    def count(self) -> int:
        return int(self._db.count())

    def count_live(self) -> int:
        return int(self._db.count_live())
