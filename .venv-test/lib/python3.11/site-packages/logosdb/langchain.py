"""LangChain VectorStore adapter for LogosDB.

This module provides a LangChain-compatible VectorStore implementation
that wraps the native LogosDB Python bindings.

Example:
    >>> from logosdb import LogosDBVectorStore
    >>> from langchain_core.documents import Document
    >>> import numpy as np
    >>>
    >>> # Create store
    >>> store = LogosDBVectorStore(path="/tmp/mydb", dim=128)
    >>>
    >>> # Add documents with embeddings
    >>> docs = [Document(page_content="hello world", metadata={"ts": "2025-01-01"})]
    >>> embeddings = [np.random.randn(128).astype(np.float32)]
    >>> store.add_documents(docs, embeddings=embeddings)
    >>>
    >>> # Search
    >>> query_embedding = np.random.randn(128).astype(np.float32)
    >>> results = store.similarity_search_by_vector(query_embedding, k=5)
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    # Dummy classes for type checking when langchain is not installed
    class VectorStore:  # type: ignore
        pass
    class Document:  # type: ignore
        pass
    class Embeddings:  # type: ignore
        pass

from ._core import DB, DIST_COSINE


class LogosDBVectorStore(VectorStore):
    """LangChain VectorStore implementation backed by LogosDB.

    This adapter wraps LogosDB to provide a LangChain-compatible interface
    for vector storage and similarity search.

    Args:
        path: Directory path for the LogosDB database files.
        dim: Vector dimensionality. Must match embedding model output dimension.
        max_elements: Maximum capacity of the vector store (default: 1,000,000).
        ef_construction: HNSW build-time search width (default: 200).
        M: HNSW graph out-degree (default: 16).
        ef_search: HNSW query-time search width (default: 50).
        use_cosine: Use cosine similarity instead of inner product.
                     When True, vectors are automatically L2-normalized.
    """

    def __init__(
        self,
        path: str,
        dim: int,
        max_elements: int = 1_000_000,
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50,
        use_cosine: bool = True,
    ) -> None:
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-core is required to use LogosDBVectorStore. "
                "Install with: pip install 'logosdb[langchain]'"
            )

        self._path = path
        self._dim = dim

        # Use cosine distance if requested (automatically normalizes vectors)
        distance = DIST_COSINE if use_cosine else 0

        self._db = DB(
            path=path,
            dim=dim,
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=M,
            ef_search=ef_search,
            distance=distance,
        )

    @property
    def embeddings(self) -> Optional[Embeddings]:
        """Return the embeddings object (None - user provides embeddings)."""
        return None

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        embeddings: Optional[List[NDArray[np.float32]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add texts with their embeddings to the vector store.

        Args:
            texts: Texts to add.
            metadatas: Optional metadata dicts for each text.
            embeddings: Required pre-computed embeddings for each text.
            **kwargs: Additional arguments (ignored).

        Returns:
            List of document IDs (auto-generated UUIDs).

        Raises:
            ValueError: If embeddings are not provided.
        """
        if embeddings is None:
            raise ValueError(
                "LogosDBVectorStore requires pre-computed embeddings. "
                "Use an Embeddings model to generate them, or use "
                "LogosDBVectorStore.from_documents() with an embedding model."
            )

        texts_list = list(texts)
        if metadatas is None:
            metadatas = [{} for _ in texts_list]

        if len(texts_list) != len(embeddings):
            raise ValueError(
                f"Number of texts ({len(texts_list)}) must match "
                f"number of embeddings ({len(embeddings)})"
            )

        ids = []
        for text, metadata, embedding in zip(texts_list, metadatas, embeddings):
            # Generate a unique ID
            doc_id = str(uuid.uuid4())

            # Extract timestamp from metadata if present
            timestamp = metadata.get("timestamp", "")

            # Insert into LogosDB
            row_id = self._db.put(
                embedding=embedding,
                text=text,
                timestamp=timestamp,
            )

            # Store the mapping from UUID to row_id in metadata
            # Note: We use the LogosDB row_id internally but return UUID to user
            ids.append(doc_id)

        return ids

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[NDArray[np.float32]]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add LangChain Document objects with embeddings.

        Args:
            documents: Documents to add.
            embeddings: Required pre-computed embeddings for each document.
            **kwargs: Additional arguments (ignored).

        Returns:
            List of document IDs.
        """
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas, embeddings, **kwargs)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        embeddings: Optional[Embeddings] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar documents by query text.

        Args:
            query: Query text to search for.
            k: Number of results to return.
            embeddings: Required embedding model to encode the query.
            **kwargs: Additional arguments (ignored).

        Returns:
            List of similar Documents.

        Raises:
            ValueError: If embeddings model is not provided.
        """
        if embeddings is None:
            raise ValueError(
                "LogosDBVectorStore.similarity_search() requires an embeddings model. "
                "Use similarity_search_by_vector() if you have pre-computed embeddings."
            )

        # Embed the query
        query_embedding = embeddings.embed_query(query)
        return self.similarity_search_by_vector(query_embedding, k, **kwargs)

    def similarity_search_by_vector(
        self,
        embedding: NDArray[np.float32],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        """Search for similar documents by vector.

        Args:
            embedding: Query embedding vector.
            k: Number of results to return.
            **kwargs: Additional arguments:
                - ts_from: Optional timestamp filter (inclusive start).
                - ts_to: Optional timestamp filter (inclusive end).

        Returns:
            List of similar Documents.
        """
        ts_from = kwargs.get("ts_from", "")
        ts_to = kwargs.get("ts_to", "")

        if ts_from or ts_to:
            # Use timestamp range search
            hits = self._db.search_ts_range(
                query=embedding,
                top_k=k,
                ts_from=ts_from,
                ts_to=ts_to,
            )
        else:
            # Regular search
            hits = self._db.search(embedding, top_k=k)

        documents = []
        for hit in hits:
            metadata = {"row_id": hit.id, "score": hit.score}
            if hit.timestamp:
                metadata["timestamp"] = hit.timestamp

            doc = Document(
                page_content=hit.text or "",
                metadata=metadata,
            )
            documents.append(doc)

        return documents

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        embeddings: Optional[Embeddings] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents with relevance scores.

        Args:
            query: Query text to search for.
            k: Number of results to return.
            embeddings: Required embedding model to encode the query.
            **kwargs: Additional arguments.

        Returns:
            List of (Document, score) tuples.
        """
        docs = self.similarity_search(query, k, embeddings, **kwargs)
        # Scores are already in metadata from similarity_search_by_vector
        return [(doc, doc.metadata.get("score", 0.0)) for doc in docs]

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Delete documents by ID.

        Note: LogosDB uses integer row IDs internally. This method requires
        the row_id to be stored in the document metadata.

        Args:
            ids: List of document IDs to delete (currently not supported via UUID).
            **kwargs: Can pass 'row_ids' with LogosDB integer row IDs.

        Raises:
            ValueError: If row_ids are not provided.
        """
        # Get row_ids from kwargs if provided
        row_ids = kwargs.get("row_ids")
        if row_ids is None:
            raise ValueError(
                "LogosDBVectorStore.delete() requires 'row_ids' kwarg with "
                "LogosDB integer row IDs (from document metadata)."
            )

        for row_id in row_ids:
            self._db.delete(int(row_id))

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        path: str,
        dim: int,
        **kwargs: Any,
    ) -> LogosDBVectorStore:
        """Create a LogosDBVectorStore from documents and an embedding model.

        This is the recommended way to create a store with initial documents.

        Args:
            documents: Documents to add.
            embedding: Embedding model to generate vectors.
            path: Database path.
            dim: Vector dimension (must match embedding model).
            **kwargs: Additional arguments passed to constructor.

        Returns:
            Initialized LogosDBVectorStore with documents added.
        """
        # Create the store
        store = cls(path=path, dim=dim, **kwargs)

        # Generate embeddings for all documents
        texts = [doc.page_content for doc in documents]
        embeddings_list = embedding.embed_documents(texts)

        # Convert to numpy arrays
        np_embeddings = [np.array(e, dtype=np.float32) for e in embeddings_list]

        # Add documents
        store.add_documents(documents, embeddings=np_embeddings)

        return store

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        path: str,
        dim: int,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> LogosDBVectorStore:
        """Create a LogosDBVectorStore from texts and an embedding model.

        Args:
            texts: Texts to add.
            embedding: Embedding model to generate vectors.
            path: Database path.
            dim: Vector dimension (must match embedding model).
            metadatas: Optional metadata for each text.
            **kwargs: Additional arguments passed to constructor.

        Returns:
            Initialized LogosDBVectorStore with texts added.
        """
        # Create documents from texts
        if metadatas is None:
            metadatas = [{} for _ in texts]

        documents = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]

        return cls.from_documents(documents, embedding, path, dim, **kwargs)

    def __len__(self) -> int:
        """Return the number of documents in the store."""
        return self._db.count_live()

    def __repr__(self) -> str:
        """Return a string representation of the store."""
        return (
            f"LogosDBVectorStore(path={self._path!r}, "
            f"dim={self._dim}, "
            f"count={len(self)})"
        )
