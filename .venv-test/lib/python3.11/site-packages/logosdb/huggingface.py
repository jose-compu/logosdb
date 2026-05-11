"""Hugging Face local embeddings integration for LogosDB."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ._core import DB, DIST_COSINE


class HuggingFaceVectorStore:
    """Local Hugging Face embeddings + LogosDB vector store.

    Uses sentence-transformers by default and computes embeddings locally.
    """

    def __init__(
        self,
        uri: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dim: int = 384,
        device: str = "cpu",
        distance: int = DIST_COSINE,
        max_elements: int = 1_000_000,
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50,
    ) -> None:
        self._model_name = model_name
        self._dim = dim
        self._device = device
        self._embedder = self._build_embedder(model_name, device)
        self._db = DB(
            path=uri,
            dim=dim,
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=M,
            ef_search=ef_search,
            distance=distance,
        )

    @staticmethod
    def _build_embedder(model_name: str, device: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install 'logosdb[huggingface]'"
            ) from exc
        return SentenceTransformer(model_name, device=device)

    def _request_embeddings(self, texts: Sequence[str]) -> List[List[float]]:
        arr = self._embedder.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        out = np.asarray(arr, dtype=np.float32)
        if out.ndim == 1:
            out = out.reshape(1, -1)
        return [row.tolist() for row in out]

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[int]:
        if not texts:
            return []
        embeddings = self._request_embeddings(texts)
        if len(embeddings) != len(texts):
            raise RuntimeError("huggingface embeddings count mismatch")

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
