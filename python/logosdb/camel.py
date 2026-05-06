"""CAMEL-AI vector memory integration for LogosDB.

This adapter keeps CAMEL-style conversation memory vectors in a local LogosDB
index and exposes a small API that can be used from agent memory components.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ._core import DB, DIST_COSINE


class LogosDBVectorMemory:
    """CAMEL-friendly vector memory wrapper backed by LogosDB."""

    def __init__(
        self,
        uri: str,
        dim: int,
        collection: str = "agent_sessions",
        distance: int = DIST_COSINE,
        max_elements: int = 1_000_000,
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50,
    ) -> None:
        self._uri = uri
        self._dim = dim
        self._collection = collection
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
    def collection(self) -> str:
        return self._collection

    def add(
        self,
        vector: Sequence[float],
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert one memory vector and return LogosDB row id."""
        vec = np.asarray(vector, dtype=np.float32)
        if vec.ndim != 1 or vec.shape[0] != self._dim:
            raise ValueError(f"expected vector dim={self._dim}, got {vec.shape}")
        ts = str((metadata or {}).get("timestamp", ""))
        return int(self._db.put(vec, text=text, timestamp=ts))

    def add_many(
        self,
        vectors: Sequence[Sequence[float]],
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[int]:
        if len(vectors) != len(texts):
            raise ValueError("vectors and texts must have the same length")
        ids: List[int] = []
        for i, (vector, text) in enumerate(zip(vectors, texts)):
            md = metadatas[i] if metadatas and i < len(metadatas) else None
            ids.append(self.add(vector, text, md))
        return ids

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search by query vector."""
        if top_k <= 0:
            return []
        qvec = np.asarray(query_vector, dtype=np.float32)
        if qvec.ndim != 1 or qvec.shape[0] != self._dim:
            raise ValueError(f"expected query dim={self._dim}, got {qvec.shape}")
        hits = self._db.search(qvec, top_k=top_k)
        return [
            {"id": h.id, "text": h.text, "score": h.score, "timestamp": h.timestamp}
            for h in hits
        ]

    # Compatibility aliases commonly used by CAMEL memory wrappers
    def retrieve(self, query_vector: Sequence[float], top_k: int = 5) -> List[Dict[str, Any]]:
        return self.search(query_vector, top_k=top_k)

    def delete(self, row_id: int) -> None:
        self._db.delete(int(row_id))

    def count(self) -> int:
        return int(self._db.count())

    def count_live(self) -> int:
        return int(self._db.count_live())

