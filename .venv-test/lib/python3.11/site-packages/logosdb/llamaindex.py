"""LlamaIndex VectorStore backend for LogosDB.

This module provides a LlamaIndex-compatible VectorStore implementation
that wraps the native LogosDB Python bindings.

Example:
    >>> from logosdb import LogosDBIndex
    >>> from llama_index.core import Document
    >>> import numpy as np
    >>>
    >>> # Create store
    >>> store = LogosDBIndex(uri="/tmp/mydb", dim=128)
    >>>
    >>> # Add nodes with embeddings
    >>> from llama_index.core.schema import TextNode
    >>> node = TextNode(text="hello world", metadata={"ts": "2025-01-01"})
    >>> node.embedding = np.random.randn(128).astype(np.float32).tolist()
    >>> store.add([node])
    >>>
    >>> # Query
    >>> from llama_index.core.vector_stores import VectorStoreQuery
    >>> query_embedding = np.random.randn(128).astype(np.float32).tolist()
    >>> query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=5)
    >>> results = store.query(query)
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._core import DB, DIST_COSINE

# LlamaIndex imports with graceful fallback
try:
    from llama_index.core.schema import BaseNode, MetadataMode, NodeRelationship, RelatedNodeInfo, TextNode
    from llama_index.core.vector_stores.types import (
        VectorStore,
        VectorStoreQuery,
        VectorStoreQueryResult,
        VectorStoreQueryMode,
    )
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False
    # Dummy classes for type checking
    class VectorStore:  # type: ignore
        pass
    class VectorStoreQuery:  # type: ignore  # noqa: F811
        pass
    class VectorStoreQueryResult:  # type: ignore  # noqa: F811
        pass
    class BaseNode:  # type: ignore
        pass
    class TextNode:  # type: ignore
        pass
    MetadataMode = None  # type: ignore
    NodeRelationship = None  # type: ignore
    RelatedNodeInfo = None  # type: ignore


class LogosDBIndex(VectorStore):
    """LlamaIndex VectorStore implementation backed by LogosDB.

    This adapter wraps LogosDB to provide a LlamaIndex-compatible interface
    for vector storage and similarity search.

    Args:
        uri: Directory path for the LogosDB database files.
        dim: Vector dimensionality. Must match embedding model output dimension.
        max_elements: Maximum capacity of the vector store (default: 1,000,000).
        ef_construction: HNSW build-time search width (default: 200).
        M: HNSW graph out-degree (default: 16).
        ef_search: HNSW query-time search width (default: 50).
        use_cosine: Use cosine similarity instead of inner product.
                     When True, vectors are automatically L2-normalized.
    """

    stores_text: bool = True
    is_embedding_query: bool = True

    def __init__(
        self,
        uri: str,
        dim: int,
        max_elements: int = 1_000_000,
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50,
        use_cosine: bool = True,
    ) -> None:
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "llama-index is required to use LogosDBIndex. "
                "Install with: pip install 'logosdb[llama-index]'"
            )

        self._uri = uri
        self._dim = dim

        # Use cosine distance if requested (automatically normalizes vectors)
        distance = DIST_COSINE if use_cosine else 0

        self._db = DB(
            path=uri,
            dim=dim,
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=M,
            ef_search=ef_search,
            distance=distance,
        )

    def client(self) -> DB:
        """Return the underlying LogosDB client."""
        return self._db

    def get(self, node_id: str) -> Optional[BaseNode]:
        """Get a single node by its ID (LogosDB row_id as string).

        Args:
            node_id: The node ID (LogosDB row_id as string).

        Returns:
            The node if found, None otherwise.
        """
        try:
            row_id = int(node_id)
        except ValueError:
            return None

        # Check if row exists and is not deleted
        if row_id >= self._db.count() or row_id < 0:
            return None

        # Get the vector and metadata via raw_vectors and manual lookup
        # Note: This is a best-effort retrieval; LlamaIndex typically
        # maintains its own docstore for full node retrieval
        return None  # Full node retrieval requires external docstore

    def add(
        self,
        nodes: List[BaseNode],
        **add_kwargs: Any,
    ) -> List[str]:
        """Add nodes to the vector store.

        Args:
            nodes: List of nodes with embeddings already set.
            **add_kwargs: Additional arguments (ignored).

        Returns:
            List of node IDs (the LogosDB row_id as string).

        Raises:
            ValueError: If a node does not have an embedding set.
        """
        ids = []
        for node in nodes:
            if node.embedding is None:
                raise ValueError(
                    f"Node {node.id_} does not have an embedding set. "
                    "Pre-compute embeddings before adding to LogosDBIndex."
                )

            # Convert embedding to numpy array
            embedding = np.array(node.embedding, dtype=np.float32)

            # Get text content
            text = node.get_content(metadata_mode=MetadataMode.NONE) if MetadataMode else str(node.text)

            # Extract timestamp from metadata if present
            timestamp = node.metadata.get("timestamp", "") if node.metadata else ""

            # Insert into LogosDB
            row_id = self._db.put(
                embedding=embedding,
                text=text,
                timestamp=timestamp,
            )

            # Store row_id as the node ID (for LlamaIndex compatibility)
            node_id = str(row_id)
            ids.append(node_id)

        return ids

    def delete(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete a node from the vector store.

        Args:
            node_id: The node ID to delete (LogosDB row_id as string).
            **delete_kwargs: Additional arguments (ignored).
        """
        try:
            row_id = int(node_id)
            self._db.delete(row_id)
        except (ValueError, RuntimeError):
            # Invalid node_id or already deleted - silently ignore
            pass

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query the vector store for similar nodes.

        Args:
            query: VectorStoreQuery containing query_embedding and similarity_top_k.
            **kwargs: Additional arguments:
                - ts_from: Optional timestamp filter (inclusive start).
                - ts_to: Optional timestamp filter (inclusive end).

        Returns:
            VectorStoreQueryResult with nodes and similarities.
        """
        if query.query_embedding is None:
            raise ValueError("LogosDBIndex.query() requires query_embedding")

        # Convert query embedding to numpy array
        query_embedding = np.array(query.query_embedding, dtype=np.float32)

        # Get top_k from query
        top_k = query.similarity_top_k or 4

        # Check for timestamp filters in query.filters or kwargs
        ts_from = kwargs.get("ts_from", "")
        ts_to = kwargs.get("ts_to", "")

        # Execute search
        if ts_from or ts_to:
            hits = self._db.search_ts_range(
                query=query_embedding,
                top_k=top_k,
                ts_from=ts_from,
                ts_to=ts_to,
            )
        else:
            hits = self._db.search(query_embedding, top_k=top_k)

        # Build result nodes
        nodes: List[TextNode] = []
        similarities: List[float] = []
        ids: List[str] = []

        for hit in hits:
            # Create a TextNode for each result
            metadata = {"row_id": hit.id, "score": hit.score}
            if hit.timestamp:
                metadata["timestamp"] = hit.timestamp

            node = TextNode(
                id_=str(hit.id),
                text=hit.text or "",
                metadata=metadata,
            )
            nodes.append(node)
            similarities.append(hit.score)
            ids.append(str(hit.id))

        return VectorStoreQueryResult(
            nodes=nodes,
            similarities=similarities,
            ids=ids,
        )

    def persist(self, persist_path: str, **persist_kwargs: Any) -> None:
        """Persist the vector store to disk.

        LogosDB is already persisted to disk. This method is a no-op
        but provided for LlamaIndex compatibility.

        Args:
            persist_path: Path to persist to (ignored, uses original uri).
            **persist_kwargs: Additional arguments (ignored).
        """
        # LogosDB is already persisted; sync to ensure durability
        pass

    @property
    def ref_doc_info(self) -> Dict[str, Any]:
        """Return reference document info (not implemented)."""
        return {}

    def clear(self) -> None:
        """Clear the vector store.

        Note: LogosDB does not support full database clearing via the
        public API. This method raises NotImplementedError.
        """
        raise NotImplementedError(
            "LogosDB does not support clearing via LlamaIndex adapter. "
            "Delete and recreate the database directory manually if needed."
        )

    def count(self) -> int:
        """Return the number of live (non-deleted) nodes in the store."""
        return self._db.count_live()

    def __len__(self) -> int:
        """Return the number of documents in the store."""
        return self._db.count_live()

    def __repr__(self) -> str:
        """Return a string representation of the store."""
        return (
            f"LogosDBIndex(uri={self._uri!r}, "
            f"dim={self._dim}, "
            f"count={len(self)})"
        )
