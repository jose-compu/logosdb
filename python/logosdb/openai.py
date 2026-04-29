"""OpenAI embeddings integration for LogosDB."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ._core import DB, DIST_COSINE


_OPENAI_MODEL_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


class OpenAIVectorStore:
    """OpenAI embeddings + LogosDB vector store."""

    def __init__(
        self,
        uri: str,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dim: Optional[int] = None,
        distance: int = DIST_COSINE,
        max_elements: int = 1_000_000,
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError("OpenAI API key required (api_key or OPENAI_API_KEY)")

        self._model = model
        self._dim = dim or _OPENAI_MODEL_DIMS.get(model, 1536)
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

    def _request_embeddings(self, texts: Sequence[str]) -> List[List[float]]:
        import requests

        resp = requests.post(
            "https://api.openai.com/v1/embeddings",
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

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[int]:
        if not texts:
            return []
        embeddings = self._request_embeddings(texts)
        if len(embeddings) != len(texts):
            raise RuntimeError("openai embeddings count mismatch")

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
