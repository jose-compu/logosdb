"""Type stubs for the logosdb native extension module."""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
from numpy.typing import NDArray

__version__: str
LOGOSDB_VERSION: str

# Distance metric constants
DIST_IP: int
DIST_COSINE: int
DIST_L2: int


class SearchHit:
    id: int
    score: float
    text: str
    timestamp: str

    def __iter__(self) -> Iterable: ...
    def __repr__(self) -> str: ...


class DB:
    def __init__(
        self,
        path: str,
        dim: int,
        max_elements: int = 1_000_000,
        ef_construction: int = 200,
        M: int = 16,
        ef_search: int = 50,
        distance: int = 0,
    ) -> None: ...

    @property
    def dim(self) -> int: ...

    def put(
        self,
        embedding: NDArray[np.float32],
        text: str = "",
        timestamp: str = "",
    ) -> int: ...

    def delete(self, id: int) -> None: ...

    def update(
        self,
        id: int,
        embedding: NDArray[np.float32],
        text: str = "",
        timestamp: str = "",
    ) -> int: ...

    def search(
        self,
        query: NDArray[np.float32],
        top_k: int = 5,
    ) -> List[SearchHit]: ...

    def search_ts_range(
        self,
        query: NDArray[np.float32],
        top_k: int = 5,
        ts_from: str = "",
        ts_to: str = "",
        candidate_k: int = 0,
    ) -> List[SearchHit]: ...

    def count(self) -> int: ...
    def count_live(self) -> int: ...
    def raw_vectors(self) -> NDArray[np.float32]: ...

    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
