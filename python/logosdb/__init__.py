"""LogosDB — fast semantic vector database (HNSW + mmap).

Example:
    >>> import numpy as np
    >>> from logosdb import DB
    >>> db = DB("/tmp/mydb", dim=128)
    >>> v = np.random.randn(128).astype("float32")
    >>> v /= np.linalg.norm(v)
    >>> rid = db.put(v, text="hello", timestamp="2025-01-01T00:00:00Z")
    >>> hits = db.search(v, top_k=5)
    >>> hits[0].text
    'hello'
"""

from ._core import DB, SearchHit, LOGOSDB_VERSION, __version__

__all__ = ["DB", "SearchHit", "LOGOSDB_VERSION", "__version__"]
