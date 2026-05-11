"""Mistral Vibe memory adapter for LogosDB.

Wraps :class:`logosdb.mistral.MistralVectorStore` with a chunking pipeline and
namespace management so that Mistral Vibe can index, search, and forget
project knowledge across sessions.

Quick start inside a Vibe session::

    from logosdb.vibe import VibeMemory

    mem = VibeMemory(uri="./.logosdb")       # reads MISTRAL_API_KEY from env

    # index a file or entire directory
    mem.index("./src", namespace="code")

    # semantic search
    results = mem.search("where is JWT validated?", namespace="code", top_k=5)

    # forget by ID or by closest semantic match
    mem.forget(namespace="code", memory_id=3)
    mem.forget(namespace="code", query="JWT validation code")

    # database statistics
    print(mem.info())

CLI entry point (``logosdb-vibe``)::

    logosdb-vibe index ./src --namespace code
    logosdb-vibe search "retry logic" --namespace code --top-k 5
    logosdb-vibe forget --namespace code --query "old feature"
    logosdb-vibe info
    logosdb-vibe list
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .mistral import MistralVectorStore


# ---------------------------------------------------------------------------
# File extensions considered indexable text
# ---------------------------------------------------------------------------

_DEFAULT_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".java", ".c", ".cpp",
    ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt", ".scala", ".sh",
    ".bash", ".zsh", ".fish", ".md", ".rst", ".txt", ".toml", ".yaml", ".yml",
    ".json", ".env.example", ".cfg", ".ini", ".sql",
}


# ---------------------------------------------------------------------------
# Paragraph-aware chunker (mirrors mcp/src/chunker.ts)
# ---------------------------------------------------------------------------

def _chunk_text(text: str, target_chars: int = 800, overlap_chars: int = 100) -> List[str]:
    """Split *text* into overlapping paragraph-aware chunks."""
    paragraphs = re.split(r"\n{2,}", text)
    chunks: List[str] = []
    buffer = ""

    for para in paragraphs:
        trimmed = para.strip()
        if not trimmed:
            continue
        if buffer and len(buffer) + len(trimmed) + 1 > target_chars:
            chunks.append(buffer.strip())
            overlap = buffer[-overlap_chars:] if len(buffer) > overlap_chars else buffer
            buffer = overlap + "\n" + trimmed
        else:
            buffer = (buffer + "\n\n" + trimmed) if buffer else trimmed

    if buffer.strip():
        chunks.append(buffer.strip())

    return chunks if chunks else [text.strip()]


# ---------------------------------------------------------------------------
# VibeMemory
# ---------------------------------------------------------------------------

class VibeMemory:
    """Mistral Vibe memory layer backed by LogosDB.

    Each namespace maps to a separate :class:`~logosdb.mistral.MistralVectorStore`
    stored under *uri/<namespace>/*.

    Args:
        uri:            Root directory for all namespace databases.
        api_key:        Mistral API key (falls back to ``MISTRAL_API_KEY`` env var).
        model:          Mistral embedding model (default ``"mistral-embed"``).
        chunk_size:     Target characters per chunk when indexing files.
        overlap_chars:  Overlap between consecutive chunks.
        dim:            Embedding dimension (1024 for mistral-embed).
    """

    def __init__(
        self,
        uri: str = "./.logosdb",
        api_key: Optional[str] = None,
        model: str = "mistral-embed",
        chunk_size: int = 800,
        overlap_chars: int = 100,
        dim: int = 1024,
    ) -> None:
        self._root = Path(uri)
        self._root.mkdir(parents=True, exist_ok=True)
        self._api_key = api_key or os.getenv("MISTRAL_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Mistral API key required. Set MISTRAL_API_KEY or pass api_key=."
            )
        self._model = model
        self._chunk_size = chunk_size
        self._overlap_chars = overlap_chars
        self._dim = dim
        self._stores: Dict[str, MistralVectorStore] = {}

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _store(self, namespace: str) -> MistralVectorStore:
        if namespace not in self._stores:
            ns_path = str(self._root / namespace)
            self._stores[namespace] = MistralVectorStore(
                uri=ns_path,
                api_key=self._api_key,
                model=self._model,
                dim=self._dim,
            )
        return self._stores[namespace]

    @staticmethod
    def _validate_namespace(namespace: str) -> None:
        if not re.fullmatch(r"[a-zA-Z0-9_\-\.]+", namespace):
            raise ValueError(
                f'Invalid namespace "{namespace}". Use [a-z A-Z 0-9 _ - .] only.'
            )

    # ── Public API ───────────────────────────────────────────────────────────

    def index(
        self,
        path: str,
        namespace: str = "code",
        chunk_size: Optional[int] = None,
        extensions: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Index a file or directory into *namespace*.

        Reads each matching file, splits it into overlapping chunks, embeds
        them with Mistral, and stores them in LogosDB.

        Args:
            path:       File or directory to index.
            namespace:  Collection name (e.g. ``"code"``, ``"docs"``).
            chunk_size: Override default chunk size (chars).
            extensions: Set of file extensions to include (e.g. ``{".py", ".md"}``).

        Returns:
            ``{"indexed": N, "files": M, "namespace": namespace, "path": path}``
        """
        self._validate_namespace(namespace)
        store = self._store(namespace)
        target = Path(path).expanduser()
        cs = chunk_size or self._chunk_size
        exts = set(extensions) if extensions else _DEFAULT_EXTENSIONS
        ts = datetime.now(timezone.utc).isoformat()

        files: List[Path] = []
        if target.is_file():
            files = [target]
        elif target.is_dir():
            files = [
                p for p in target.rglob("*")
                if p.is_file() and p.suffix in exts
            ]
        else:
            raise FileNotFoundError(f"Path not found: {target}")

        total_chunks = 0
        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            chunks = _chunk_text(content, target_chars=cs, overlap_chars=self._overlap_chars)
            texts = [
                f"[file:{file_path}][chunk:{i}/{len(chunks)}]\n{c}"
                for i, c in enumerate(chunks)
            ]
            metadatas = [{"timestamp": ts} for _ in texts]
            store.add_texts(texts, metadatas=metadatas)
            total_chunks += len(chunks)

        return {
            "indexed": total_chunks,
            "files": len(files),
            "namespace": namespace,
            "path": str(target),
        }

    def search(
        self,
        query: str,
        namespace: str = "code",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Semantic search over *namespace*.

        Args:
            query:     Natural-language query.
            namespace: Collection to search in.
            top_k:     Number of results.

        Returns:
            List of ``{"id", "text", "score", "timestamp", "file"}`` dicts,
            sorted by descending score.
        """
        self._validate_namespace(namespace)
        ns_path = self._root / namespace
        if not ns_path.exists():
            return []

        store = self._store(namespace)
        hits = store.search(query, top_k=top_k)

        results = []
        for h in hits:
            text = h.get("text") or ""
            # Extract file path from [file:...] prefix if present
            file_match = re.match(r"\[file:([^\]]+)\]", text)
            file_path = file_match.group(1) if file_match else None
            # Strip label prefix for display
            display = re.sub(r"^\[file:[^\]]+\]\[chunk:\d+/\d+\]\n?", "", text)
            results.append(
                {
                    "id": h["id"],
                    "score": round(h["score"], 4),
                    "text": display,
                    "timestamp": h.get("timestamp"),
                    "file": file_path,
                }
            )
        return results

    def forget(
        self,
        namespace: str = "code",
        memory_id: Optional[int] = None,
        query: Optional[str] = None,
        top_k: int = 1,
    ) -> Dict[str, Any]:
        """Delete one or more entries from *namespace*.

        Exactly one of *memory_id* or *query* must be provided.

        - ``memory_id``: delete the entry with that row ID.
        - ``query``: find the *top_k* closest matches and delete them.

        Returns:
            ``{"forgotten": N, "namespace": namespace}``
        """
        self._validate_namespace(namespace)
        if memory_id is None and query is None:
            raise ValueError("Provide either memory_id or query.")
        if memory_id is not None and query is not None:
            raise ValueError("Provide only one of memory_id or query, not both.")

        store = self._store(namespace)

        if memory_id is not None:
            store._db.delete(memory_id)
            return {"forgotten": 1, "namespace": namespace, "ids": [memory_id]}

        # Query-based: embed query, find closest matches, delete
        hits = self.search(query or "", namespace=namespace, top_k=top_k)
        ids = [h["id"] for h in hits]
        for row_id in ids:
            store._db.delete(row_id)
        return {"forgotten": len(ids), "namespace": namespace, "ids": ids}

    def info(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Return statistics for one or all namespaces.

        Args:
            namespace: If given, stats for that namespace only.
                       If ``None``, stats for every existing namespace.

        Returns:
            Dict (single namespace) or list of dicts (all namespaces).
        """
        if namespace is not None:
            self._validate_namespace(namespace)
            ns_path = self._root / namespace
            if not ns_path.exists():
                raise FileNotFoundError(f'Namespace "{namespace}" does not exist.')
            store = self._store(namespace)
            return {
                "namespace": namespace,
                "count": store.count(),
                "path": str(ns_path),
                "model": self._model,
                "dim": self._dim,
            }

        namespaces = self.list_namespaces()
        return {  # type: ignore[return-value]
            "namespaces": [self.info(ns) for ns in namespaces],
            "root": str(self._root),
        }

    def list_namespaces(self) -> List[str]:
        """Return names of all indexed namespaces under the root directory."""
        try:
            return [
                d.name for d in self._root.iterdir()
                if d.is_dir()
            ]
        except FileNotFoundError:
            return []
