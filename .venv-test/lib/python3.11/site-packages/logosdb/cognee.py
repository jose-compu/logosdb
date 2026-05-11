"""Cognee knowledge-graph/RAG vector store integration for LogosDB."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ._core import DB, DIST_COSINE


class LogosDBVectorStore:
    """Cognee-style vector store adapter backed by LogosDB."""

    def __init__(
        self,
        uri: str,
        dim: int,
        collection: str = "cognee_graph",
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

    def upsert(
        self,
        vectors: Sequence[Sequence[float]],
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[int]:
        """Insert graph node/document vectors (simple upsert semantics)."""
        if len(vectors) != len(texts):
            raise ValueError("vectors and texts must have the same length")
        ids: List[int] = []
        for i, (vector, text) in enumerate(zip(vectors, texts)):
            vec = np.asarray(vector, dtype=np.float32)
            if vec.ndim != 1 or vec.shape[0] != self._dim:
                raise ValueError(f"expected embedding dim={self._dim}, got {vec.shape}")
            md = metadatas[i] if metadatas and i < len(metadatas) else {}
            ts = str(md.get("timestamp", ""))
            rid = self._db.put(vec, text=text, timestamp=ts)
            ids.append(int(rid))
        return ids

    def search(self, query_vector: Sequence[float], top_k: int = 5) -> List[Dict[str, Any]]:
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

    async def asearch(self, query_vector: Sequence[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Async helper for frameworks that expect awaitable search."""
        return self.search(query_vector=query_vector, top_k=top_k)

    def delete(self, row_id: int) -> None:
        self._db.delete(int(row_id))

    def count(self) -> int:
        return int(self._db.count())

