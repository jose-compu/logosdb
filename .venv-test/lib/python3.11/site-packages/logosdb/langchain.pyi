"""Type stubs for LogosDB LangChain adapter."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore
except ImportError:
    pass


class LogosDBVectorStore:
    """LangChain VectorStore implementation backed by LogosDB."""

    def __init__(
        self,
        path: str,
        dim: int,
        max_elements: int = 1_000_000,
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50,
        use_cosine: bool = True,
    ) -> None: ...

    @property
    def embeddings(self) -> Optional[Embeddings]: ...

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        embeddings: Optional[List[NDArray[np.float32]]] = None,
        **kwargs: Any,
    ) -> List[str]: ...

    def add_documents(
        self,
        documents: List[Document],
        embeddings: Optional[List[NDArray[np.float32]]] = None,
        **kwargs: Any,
    ) -> List[str]: ...

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        embeddings: Optional[Embeddings] = None,
        **kwargs: Any,
    ) -> List[Document]: ...

    def similarity_search_by_vector(
        self,
        embedding: NDArray[np.float32],
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]: ...

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        embeddings: Optional[Embeddings] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]: ...

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None: ...

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        path: str,
        dim: int,
        **kwargs: Any,
    ) -> LogosDBVectorStore: ...

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        path: str,
        dim: int,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> LogosDBVectorStore: ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
