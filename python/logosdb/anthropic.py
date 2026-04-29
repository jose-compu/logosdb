"""Anthropic Claude-powered embedding integration for LogosDB.

Note:
Anthropic does not currently expose a first-party embeddings endpoint.
This adapter uses Claude messages to produce deterministic numeric vectors
via constrained JSON output (best-effort, experimental).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ._core import DB, DIST_COSINE


class AnthropicVectorStore:
    """Anthropic Claude prompt-based embeddings + LogosDB vector store (experimental)."""

    def __init__(
        self,
        uri: str,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-haiku-latest",
        embedding_mode: str = "response_embedding",
        dim: int = 1024,
        distance: int = DIST_COSINE,
        max_elements: int = 1_000_000,
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50,
    ) -> None:
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise ValueError("Anthropic API key required (api_key or ANTHROPIC_API_KEY)")
        self._model = model
        self._embedding_mode = embedding_mode
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

    def _embed_prompt(self, text: str) -> str:
        return (
            "Return ONLY valid JSON in this shape: "
            '{"embedding":[float,...]}. '
            f"The embedding must contain exactly {self._dim} floats. "
            "No prose, no markdown, no extra keys.\n"
            f"Text: {text}"
        )

    @staticmethod
    def _extract_text_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
            return "\n".join(parts)
        return str(content)

    def _request_embeddings(self, texts: Sequence[str]) -> List[List[float]]:
        import requests

        out: List[List[float]] = []
        for t in texts:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self._model,
                    "max_tokens": 4096,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": self._embed_prompt(t)}],
                },
                timeout=45,
            )
            resp.raise_for_status()
            payload = resp.json()
            raw_text = self._extract_text_content(payload.get("content", ""))

            # try strict json parse first
            emb: List[float]
            try:
                parsed = json.loads(raw_text)
                emb = parsed["embedding"]
            except Exception:
                # fallback: extract first {...} block
                start = raw_text.find("{")
                end = raw_text.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    raise RuntimeError("anthropic embedding parse failed")
                parsed = json.loads(raw_text[start : end + 1])
                emb = parsed["embedding"]

            vec = np.asarray(emb, dtype=np.float32)
            if vec.ndim != 1 or vec.shape[0] != self._dim:
                raise ValueError(f"expected embedding dim={self._dim}, got {vec.shape}")
            out.append(vec.tolist())
        return out

    def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[int]:
        if not texts:
            return []
        embeddings = self._request_embeddings(texts)
        if len(embeddings) != len(texts):
            raise RuntimeError("anthropic embeddings count mismatch")

        ids: List[int] = []
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            vec = np.asarray(emb, dtype=np.float32)
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
        hits = self._db.search(qvec, top_k=top_k)
        return [
            {"id": h.id, "text": h.text, "score": h.score, "timestamp": h.timestamp}
            for h in hits
        ]

    def count(self) -> int:
        return int(self._db.count())
