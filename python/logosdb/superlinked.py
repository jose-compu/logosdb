"""Superlinked integration for LogosDB.

This module provides a compact adapter that stores Superlinked vectors in
LogosDB and exposes query helpers for retrieval.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ._core import DB, DIST_COSINE


class LogosDBVectorIndex:
    """Superlinked-style vector index backed by LogosDB."""

    def __init__(
        self,
        uri: str,
        dim: int,
        index_name: str = "superlinked",
        distance: int = DIST_COSINE,
        max_elements: int = 1_000_000,
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50,
    ) -> None:
        self._uri = uri
        self._dim = dim
        self._index_name = index_name
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
    def index_name(self) -> str:
        return self._index_name

    def upsert(
        self,
        vectors: Sequence[Sequence[float]],
        payloads: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Insert vector payload rows and return LogosDB row ids."""
        if payloads is None:
            payloads = [{} for _ in vectors]
        if len(payloads) != len(vectors):
            raise ValueError("payloads and vectors must have the same length")

        ids: List[int] = []
        for vector, payload in zip(vectors, payloads):
            vec = np.asarray(vector, dtype=np.float32)
            if vec.ndim != 1 or vec.shape[0] != self._dim:
                raise ValueError(f"expected embedding dim={self._dim}, got {vec.shape}")
            text = str(payload.get("text", payload.get("content", "")))
            ts = str(payload.get("timestamp", ""))
            rid = self._db.put(vec, text=text, timestamp=ts)
            ids.append(int(rid))
        return ids

    def query(self, vector: Sequence[float], top_k: int = 5) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []
        qvec = np.asarray(vector, dtype=np.float32)
        if qvec.ndim != 1 or qvec.shape[0] != self._dim:
            raise ValueError(f"expected query dim={self._dim}, got {qvec.shape}")
        hits = self._db.search(qvec, top_k=top_k)
        return [
            {"id": h.id, "text": h.text, "score": h.score, "timestamp": h.timestamp}
            for h in hits
        ]

    def delete(self, row_id: int) -> None:
        self._db.delete(int(row_id))

    def count(self) -> int:
        return int(self._db.count())

