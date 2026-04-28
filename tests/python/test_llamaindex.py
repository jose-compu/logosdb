"""Tests for LogosDB LlamaIndex VectorStore adapter."""

import shutil
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

# Skip all tests if llama-index is not installed
pytest.importorskip("llama_index")

from llama_index.core.schema import TextNode, MetadataMode
from llama_index.core.vector_stores.types import VectorStoreQuery

from logosdb import LogosDBIndex


def make_embedding(dim: int, seed: int) -> List[float]:
    """Generate a deterministic unit vector embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for the test database."""
    path = tempfile.mkdtemp(prefix="logosdb_llamaindex_test_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


class TestLogosDBIndex:
    """Test suite for LogosDBIndex."""

    def test_init(self, temp_db_path):
        """Test store initialization."""
        store = LogosDBIndex(uri=temp_db_path, dim=128)
        assert store._dim == 128
        assert len(store) == 0
        assert store.count() == 0

    def test_add_nodes(self, temp_db_path):
        """Test adding nodes with embeddings."""
        store = LogosDBIndex(uri=temp_db_path, dim=128)

        nodes = [
            TextNode(text="hello world", embedding=make_embedding(128, 1)),
            TextNode(text="foo bar", embedding=make_embedding(128, 2)),
            TextNode(text="baz qux", embedding=make_embedding(128, 3)),
        ]

        ids = store.add(nodes)

        assert len(ids) == 3
        assert all(isinstance(id, str) for id in ids)
        assert len(store) == 3
        assert store.count() == 3

    def test_add_nodes_with_metadata(self, temp_db_path):
        """Test adding nodes with metadata including timestamp."""
        store = LogosDBIndex(uri=temp_db_path, dim=128)

        nodes = [
            TextNode(
                text="early entry",
                embedding=make_embedding(128, 4),
                metadata={"timestamp": "2025-01-01T00:00:00Z", "category": "test"},
            ),
            TextNode(
                text="late entry",
                embedding=make_embedding(128, 5),
                metadata={"timestamp": "2025-01-15T00:00:00Z", "category": "test"},
            ),
        ]

        ids = store.add(nodes)
        assert len(ids) == 2

    def test_query_by_vector(self, temp_db_path):
        """Test querying by vector."""
        store = LogosDBIndex(uri=temp_db_path, dim=128)

        # Add nodes
        target_emb = make_embedding(128, 100)
        nodes = [
            TextNode(text="apple pie", embedding=target_emb),
            TextNode(text="banana bread", embedding=make_embedding(128, 101)),
            TextNode(text="cherry tart", embedding=make_embedding(128, 102)),
        ]
        store.add(nodes)

        # Query with the same embedding as "apple pie"
        query = VectorStoreQuery(
            query_embedding=target_emb,
            similarity_top_k=2,
        )
        results = store.query(query)

        assert len(results.nodes) == 2
        assert results.nodes[0].text == "apple pie"
        assert results.similarities[0] > 0.9  # High similarity for exact match
        assert results.ids[0] is not None

    def test_query_returns_scores(self, temp_db_path):
        """Test that query returns similarity scores."""
        store = LogosDBIndex(uri=temp_db_path, dim=128)

        nodes = [
            TextNode(text="cat", embedding=make_embedding(128, 200)),
            TextNode(text="dog", embedding=make_embedding(128, 201)),
            TextNode(text="fish", embedding=make_embedding(128, 202)),
        ]
        store.add(nodes)

        query = VectorStoreQuery(
            query_embedding=make_embedding(128, 200),  # Same as "cat"
            similarity_top_k=3,
        )
        results = store.query(query)

        assert len(results.similarities) == 3
        for score in results.similarities:
            assert 0.0 <= score <= 1.001

    def test_delete(self, temp_db_path):
        """Test deleting nodes."""
        store = LogosDBIndex(uri=temp_db_path, dim=128)

        nodes = [
            TextNode(text="one", embedding=make_embedding(128, 300)),
            TextNode(text="two", embedding=make_embedding(128, 301)),
            TextNode(text="three", embedding=make_embedding(128, 302)),
        ]
        ids = store.add(nodes)

        assert len(store) == 3

        # Delete the first node
        store.delete(ids[0])

        assert len(store) == 2
        assert store.count() == 2

    def test_timestamp_filter(self, temp_db_path):
        """Test timestamp range filtering."""
        store = LogosDBIndex(uri=temp_db_path, dim=128)

        # Add with timestamps
        nodes = [
            TextNode(
                text="early",
                embedding=make_embedding(128, 400),
                metadata={"timestamp": "2025-01-01T00:00:00Z"},
            ),
            TextNode(
                text="late",
                embedding=make_embedding(128, 401),
                metadata={"timestamp": "2025-01-15T00:00:00Z"},
            ),
        ]
        store.add(nodes)

        # Query with timestamp filter
        query = VectorStoreQuery(
            query_embedding=make_embedding(128, 400),  # Query similar to "early"
            similarity_top_k=10,
        )
        results = store.query(
            query,
            ts_from="2025-01-01T00:00:00Z",
            ts_to="2025-01-10T00:00:00Z",
        )

        assert len(results.nodes) == 1
        assert results.nodes[0].text == "early"

    def test_repr(self, temp_db_path):
        """Test string representation."""
        store = LogosDBIndex(uri=temp_db_path, dim=128)
        repr_str = repr(store)
        assert "LogosDBIndex" in repr_str
        assert temp_db_path in repr_str
        assert "dim=128" in repr_str

    def test_error_without_embedding(self, temp_db_path):
        """Test that add requires embeddings."""
        store = LogosDBIndex(uri=temp_db_path, dim=128)

        # Node without embedding
        node = TextNode(text="no embedding")

        with pytest.raises(ValueError, match="embedding"):
            store.add([node])

    def test_error_query_without_embedding(self, temp_db_path):
        """Test that query requires query_embedding."""
        store = LogosDBIndex(uri=temp_db_path, dim=128)

        # Add a document first
        node = TextNode(text="test", embedding=make_embedding(128, 500))
        store.add([node])

        # Query without embedding
        query = VectorStoreQuery(query_embedding=None, similarity_top_k=1)

        with pytest.raises(ValueError, match="query_embedding"):
            store.query(query)

    def test_cosine_normalization(self, temp_db_path):
        """Test that cosine mode handles unnormalized vectors."""
        store = LogosDBIndex(uri=temp_db_path, dim=128, use_cosine=True)

        # Add unnormalized vectors
        unnormalized = [np.random.randn(128).astype(np.float32) * 10]  # Large magnitude
        node = TextNode(text="test", embedding=unnormalized[0].tolist())
        ids = store.add([node])

        assert len(ids) == 1

        # Query should still work
        query = VectorStoreQuery(
            query_embedding=(np.random.randn(128).astype(np.float32) * 5).tolist(),
            similarity_top_k=1,
        )
        results = store.query(query)
        assert len(results.nodes) == 1

    def test_node_id_in_results(self, temp_db_path):
        """Test that returned nodes have proper IDs."""
        store = LogosDBIndex(uri=temp_db_path, dim=128)

        nodes = [
            TextNode(text="first", embedding=make_embedding(128, 600)),
            TextNode(text="second", embedding=make_embedding(128, 601)),
        ]
        store.add(nodes)

        query = VectorStoreQuery(
            query_embedding=make_embedding(128, 600),
            similarity_top_k=2,
        )
        results = store.query(query)

        assert len(results.ids) == 2
        # IDs should be string representations of row IDs (0, 1)
        assert results.ids[0] in ["0", "1"]
        assert results.ids[1] in ["0", "1"]

    def test_persist_is_noop(self, temp_db_path):
        """Test that persist() is a no-op (LogosDB is already persisted)."""
        store = LogosDBIndex(uri=temp_db_path, dim=128)

        # Should not raise
        store.persist("/some/path")

    def test_clear_raises(self, temp_db_path):
        """Test that clear() raises NotImplementedError."""
        store = LogosDBIndex(uri=temp_db_path, dim=128)

        with pytest.raises(NotImplementedError):
            store.clear()

    def test_get_returns_none(self, temp_db_path):
        """Test that get() returns None (requires external docstore)."""
        store = LogosDBIndex(uri=temp_db_path, dim=128)

        result = store.get("0")
        assert result is None

    def test_client_returns_db(self, temp_db_path):
        """Test that client() returns the underlying DB."""
        store = LogosDBIndex(uri=temp_db_path, dim=128)

        db = store.client()
        assert db is not None
        assert hasattr(db, 'put')
        assert hasattr(db, 'search')
