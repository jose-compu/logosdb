"""mem0 VectorStore adapter for LogosDB.

Implements the mem0 VectorStoreBase interface so LogosDB can be used as
the vector backend for mem0 memory layers.

Usage::

    from mem0 import Memory
    from logosdb.mem0 import LogosDBVectorStore

    config = {
        "vector_store": {
            "provider": "logosdb",
            "config": {
                "collection_name": "mem0",
                "uri": "/data/mem0_memory",
                "dim": 1536,
            },
        }
    }
    m = Memory.from_config(config)
    m.add("I prefer Python over JavaScript", user_id="alice")
    memories = m.get_all(user_id="alice")

Alternatively, instantiate directly and pass via Memory(vector_store=...)
for mem0 versions that support it::

    vector_store = LogosDBVectorStore(uri="/data/mem0_memory", dim=1536)
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from ._core import DB, DIST_COSINE


try:
    from mem0.vector_stores.base import VectorStoreBase

    _MEM0_AVAILABLE = True
except ImportError:
    _MEM0_AVAILABLE = False

    class VectorStoreBase:  # type: ignore[no-redef]
        """Fallback stub when mem0 is not installed."""


class _OutputHit:
    """Minimal output object matching mem0's expected search result shape."""

    __slots__ = ("id", "score", "payload")

    def __init__(self, id: str, score: float, payload: Dict[str, Any]) -> None:
        self.id = id
        self.score = score
        self.payload = payload


class LogosDBVectorStore(VectorStoreBase):
    """mem0 VectorStoreBase implementation backed by LogosDB.

    Each mem0 collection maps to a sub-directory of *uri*.

    Args:
        uri:             Root directory for all collections.
        collection_name: Default collection (mem0 may override per-call).
        dim:             Embedding dimension. Must match the configured
                         mem0 embedding model.
        distance:        LogosDB distance metric (default: DIST_COSINE).
        max_elements:    HNSW capacity per collection.
        ef_construction: HNSW build parameter.
        M:               HNSW graph out-degree.
        ef_search:       HNSW query parameter.
    """

    def __init__(
        self,
        uri: str,
        collection_name: str = "mem0",
        dim: int = 1536,
        distance: int = DIST_COSINE,
        max_elements: int = 1_000_000,
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50,
    ) -> None:
        if not _MEM0_AVAILABLE:
            raise ImportError(
                "mem0ai is required. Install with: pip install 'logosdb[mem0]'"
            )
        self._root = uri
        self._default_col = collection_name
        self._dim = dim
        self._distance = distance
        self._max_elements = max_elements
        self._ef_construction = ef_construction
        self._M = M
        self._ef_search = ef_search

        # In-process DB registry: collection_name → (DB, row_id → str_uuid)
        self._dbs: Dict[str, DB] = {}
        # row_id → external uuid per collection
        self._id_maps: Dict[str, Dict[int, str]] = {}

    # ── Internal helpers ───────────────────────────────────────────────────

    def _col_path(self, name: str) -> str:
        return os.path.join(self._root, name)

    def _open(self, name: str) -> DB:
        if name not in self._dbs:
            path = self._col_path(name)
            os.makedirs(path, exist_ok=True)
            self._dbs[name] = DB(
                path=path,
                dim=self._dim,
                max_elements=self._max_elements,
                ef_construction=self._ef_construction,
                M=self._M,
                ef_search=self._ef_search,
                distance=self._distance,
            )
            self._id_maps[name] = {}
        return self._dbs[name]

    def _resolve_name(self, name: Optional[str]) -> str:
        return name if name else self._default_col

    # ── VectorStoreBase interface ──────────────────────────────────────────

    def create_col(
        self,
        name: str,
        vector_size: int,
        distance: Any = None,
    ) -> None:
        """Create (or open) a collection."""
        self._open(name)

    def insert(
        self,
        name: str,
        vectors: List[List[float]],
        payloads: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Insert vectors into a collection."""
        db = self._open(name)
        id_map = self._id_maps[name]

        if payloads is None:
            payloads = [{} for _ in vectors]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]

        for ext_id, vec_list, payload in zip(ids, vectors, payloads):
            vec = np.asarray(vec_list, dtype=np.float32)
            text = str(payload.get("data", payload.get("text", "")))
            ts = str(payload.get("created_at", ""))
            row_id = int(db.put(vec, text=text, timestamp=ts))
            id_map[row_id] = ext_id

    def search(
        self,
        name: str,
        query: List[float],
        limit: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[_OutputHit]:
        """Search by vector. Returns list of hit objects with id/score/payload."""
        db = self._open(name)
        id_map = self._id_maps[name]
        qvec = np.asarray(query, dtype=np.float32)
        hits = db.search(qvec, top_k=limit)
        results: List[_OutputHit] = []
        for h in hits:
            ext_id = id_map.get(h.id, str(h.id))
            payload: Dict[str, Any] = {}
            if h.text:
                payload["data"] = h.text
            if h.timestamp:
                payload["created_at"] = h.timestamp
            results.append(_OutputHit(id=ext_id, score=float(h.score), payload=payload))
        return results

    def delete(self, name: str, vector_id: str) -> None:
        """Delete a vector by its external UUID."""
        db = self._open(name)
        id_map = self._id_maps[name]
        # Reverse lookup: find row_id for this UUID
        row_id = next((rid for rid, uid in id_map.items() if uid == vector_id), None)
        if row_id is not None:
            db.delete(row_id)
            del id_map[row_id]

    def update(
        self,
        name: str,
        vector_id: str,
        vector: Optional[List[float]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update a vector entry (tombstone + re-insert)."""
        db = self._open(name)
        id_map = self._id_maps[name]
        row_id = next((rid for rid, uid in id_map.items() if uid == vector_id), None)
        if row_id is None:
            return

        vec_list = vector if vector is not None else db.raw_vectors()[row_id].tolist()
        vec = np.asarray(vec_list, dtype=np.float32)
        text = str((payload or {}).get("data", ""))
        ts = str((payload or {}).get("created_at", ""))
        new_row_id = int(db.update(row_id, vec, text=text, timestamp=ts))
        del id_map[row_id]
        id_map[new_row_id] = vector_id

    def get(self, name: str, vector_id: str) -> Optional[_OutputHit]:
        """Retrieve a single entry by external UUID."""
        db = self._open(name)
        id_map = self._id_maps[name]
        row_id = next((rid for rid, uid in id_map.items() if uid == vector_id), None)
        if row_id is None:
            return None
        # We don't have a direct get-by-id in LogosDB; search for a placeholder
        # and scan metadata — approximate approach: return payload from id_map
        return _OutputHit(id=vector_id, score=1.0, payload={"row_id": row_id})

    def list_cols(self) -> List[str]:
        """List all collections (sub-directories under root)."""
        try:
            return [
                d for d in os.listdir(self._root)
                if os.path.isdir(os.path.join(self._root, d))
            ]
        except FileNotFoundError:
            return []

    def delete_col(self, name: str) -> None:
        """Close and remove a collection from the in-process registry."""
        if name in self._dbs:
            self._dbs[name].close()
            del self._dbs[name]
            del self._id_maps[name]

    def col_info(self, name: str) -> Dict[str, Any]:
        """Return stats for a collection."""
        db = self._open(name)
        return {
            "name": name,
            "count": int(db.count()),
            "count_live": int(db.count_live()),
            "dim": self._dim,
            "path": self._col_path(name),
        }
