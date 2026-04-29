"""Tests for LogosDB Haystack 2.x DocumentStore and Retriever."""

import shutil
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

# Skip all tests if haystack is not installed
pytest.importorskip("haystack")

from haystack.dataclasses import Document

from logosdb import LogosDBDocumentStore, LogosDBRetriever


def make_embedding(dim: int, seed: int) -> List[float]:
    """Generate a deterministic unit vector embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec.tolist()


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for the test database."""
    path = tempfile.mkdtemp(prefix="logosdb_haystack_test_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


class TestLogosDBDocumentStore:
    """Test suite for LogosDBDocumentStore."""

    def test_init(self, temp_db_path):
        """Test document store initialization."""
        store = LogosDBDocumentStore(path=temp_db_path, dim=128)
        assert store._dim == 128
        assert store.count_documents() == 0

    def test_write_documents(self, temp_db_path):
        """Test writing documents with embeddings."""
        store = LogosDBDocumentStore(path=temp_db_path, dim=128)

        docs = [
            Document(content="hello world", embedding=make_embedding(128, 1)),
            Document(content="foo bar", embedding=make_embedding(128, 2)),
            Document(content="baz qux", embedding=make_embedding(128, 3)),
        ]

        count = store.write_documents(docs)

        assert count == 3
        assert store.count_documents() == 3

    def test_write_documents_with_metadata(self, temp_db_path):
        """Test writing documents with metadata including timestamp."""
        store = LogosDBDocumentStore(path=temp_db_path, dim=128)

        docs = [
            Document(
                content="early entry",
                embedding=make_embedding(128, 4),
                meta={"timestamp": "2025-01-01T00:00:00Z", "category": "test"},
            ),
            Document(
                content="late entry",
                embedding=make_embedding(128, 5),
                meta={"timestamp": "2025-01-15T00:00:00Z", "category": "test"},
            ),
        ]

        count = store.write_documents(docs)
        assert count == 2

    def test_error_without_embedding(self, temp_db_path):
        """Test that write_documents requires embeddings."""
        store = LogosDBDocumentStore(path=temp_db_path, dim=128)

        # Document without embedding
        doc = Document(content="no embedding")

        with pytest.raises(ValueError, match="embedding"):
            store.write_documents([doc])

    def test_delete_documents(self, temp_db_path):
        """Test deleting documents."""
        store = LogosDBDocumentStore(path=temp_db_path, dim=128)

        docs = [
            Document(content="one", embedding=make_embedding(128, 300)),
            Document(content="two", embedding=make_embedding(128, 301)),
            Document(content="three", embedding=make_embedding(128, 302)),
        ]
        store.write_documents(docs)

        assert store.count_documents() == 3

        # Delete the first document (row_id will be 0)
        store.delete_documents(["0"])

        assert store.count_documents() == 2

    def test_to_dict(self, temp_db_path):
        """Test serialization to dict."""
        store = LogosDBDocumentStore(path=temp_db_path, dim=128)

        data = store.to_dict()

        assert data["path"] == temp_db_path
        assert data["dim"] == 128
        assert "use_cosine" in data

    def test_from_dict(self, temp_db_path):
        """Test deserialization from dict."""
        data = {
            "path": temp_db_path,
            "dim": 256,
            "use_cosine": True,
        }

        store = LogosDBDocumentStore.from_dict(data)

        assert store._path == temp_db_path
        assert store._dim == 256

    def test_filter_documents_raises(self, temp_db_path):
        """Test that filter_documents with filters raises error."""
        store = LogosDBDocumentStore(path=temp_db_path, dim=128)

        with pytest.raises(NotImplementedError, match="filter"):
            store.filter_documents(filters={"key": "value"})


class TestLogosDBRetriever:
    """Test suite for LogosDBRetriever."""

    def test_init(self, temp_db_path):
        """Test retriever initialization."""
        store = LogosDBDocumentStore(path=temp_db_path, dim=128)
        retriever = LogosDBRetriever(store, top_k=5)

        assert retriever._top_k == 5
        assert retriever._document_store == store

    def test_run_basic(self, temp_db_path):
        """Test basic retrieval by vector."""
        store = LogosDBDocumentStore(path=temp_db_path, dim=128)

        # Add documents
        target_emb = make_embedding(128, 100)
        docs = [
            Document(content="apple pie", embedding=target_emb),
            Document(content="banana bread", embedding=make_embedding(128, 101)),
            Document(content="cherry tart", embedding=make_embedding(128, 102)),
        ]
        store.write_documents(docs)

        # Create retriever
        retriever = LogosDBRetriever(store, top_k=2)

        # Query with the same embedding as "apple pie"
        result = retriever.run(query_embedding=target_emb)

        assert "documents" in result
        assert len(result["documents"]) == 2
        assert result["documents"][0].content == "apple pie"
        assert result["documents"][0].meta["score"] > 0.9

    def test_run_with_top_k_override(self, temp_db_path):
        """Test retrieval with top_k override."""
        store = LogosDBDocumentStore(path=temp_db_path, dim=128)

        docs = [
            Document(content="one", embedding=make_embedding(128, 200)),
            Document(content="two", embedding=make_embedding(128, 201)),
            Document(content="three", embedding=make_embedding(128, 202)),
        ]
        store.write_documents(docs)

        retriever = LogosDBRetriever(store, top_k=1)

        # Override top_k at runtime
        result = retriever.run(query_embedding=make_embedding(128, 200), top_k=3)

        assert len(result["documents"]) == 3

    def test_run_timestamp_filter(self, temp_db_path):
        """Test retrieval with timestamp range filter."""
        store = LogosDBDocumentStore(path=temp_db_path, dim=128)

        # Add documents with timestamps
        docs = [
            Document(
                content="early",
                embedding=make_embedding(128, 400),
                meta={"timestamp": "2025-01-01T00:00:00Z"},
            ),
            Document(
                content="late",
                embedding=make_embedding(128, 401),
                meta={"timestamp": "2025-01-15T00:00:00Z"},
            ),
        ]
        store.write_documents(docs)

        retriever = LogosDBRetriever(store, top_k=10)

        # Query with timestamp filter
        result = retriever.run(
            query_embedding=make_embedding(128, 400),  # Similar to "early"
            ts_from="2025-01-01T00:00:00Z",
            ts_to="2025-01-10T00:00:00Z",
        )

        assert len(result["documents"]) == 1
        assert result["documents"][0].content == "early"

    def test_run_returns_scores(self, temp_db_path):
        """Test that retrieval returns similarity scores."""
        store = LogosDBDocumentStore(path=temp_db_path, dim=128)

        docs = [
            Document(content="cat", embedding=make_embedding(128, 500)),
            Document(content="dog", embedding=make_embedding(128, 501)),
        ]
        store.write_documents(docs)

        retriever = LogosDBRetriever(store, top_k=2)
        result = retriever.run(query_embedding=make_embedding(128, 500))

        for doc in result["documents"]:
            assert "score" in doc.meta
            assert 0.0 <= doc.meta["score"] <= 1.001

    def test_to_dict(self, temp_db_path):
        """Test serialization to dict."""
        store = LogosDBDocumentStore(path=temp_db_path, dim=128)
        retriever = LogosDBRetriever(store, top_k=5)

        data = retriever.to_dict()

        assert data["top_k"] == 5
        assert "document_store" in data

    def test_from_dict(self, temp_db_path):
        """Test deserialization from dict."""
        store = LogosDBDocumentStore(path=temp_db_path, dim=128)
        retriever = LogosDBRetriever(store, top_k=7)

        data = retriever.to_dict()
        new_retriever = LogosDBRetriever.from_dict(data)

        assert new_retriever._top_k == 7

    def test_cosine_normalization(self, temp_db_path):
        """Test that cosine mode handles unnormalized vectors."""
        store = LogosDBDocumentStore(path=temp_db_path, dim=128, use_cosine=True)

        # Add unnormalized vectors
        unnormalized = np.random.randn(128).astype(np.float32) * 10
        doc = Document(content="test", embedding=unnormalized.tolist())
        store.write_documents([doc])

        # Query should still work
        query = (np.random.randn(128).astype(np.float32) * 5).tolist()
        retriever = LogosDBRetriever(store, top_k=1)
        result = retriever.run(query_embedding=query)

        assert len(result["documents"]) == 1


class TestHaystackPipelineIntegration:
    """Test integration with Haystack pipelines."""

    def test_document_store_in_pipeline(self, temp_db_path):
        """Test that document store can be used in a Haystack pipeline context."""
        store = LogosDBDocumentStore(path=temp_db_path, dim=128)

        # Write some documents
        docs = [
            Document(content="machine learning", embedding=make_embedding(128, 600)),
            Document(content="deep learning", embedding=make_embedding(128, 601)),
        ]
        store.write_documents(docs)

        # Verify the store works as expected
        assert store.count_documents() == 2

        # Create a retriever and use it
        retriever = LogosDBRetriever(store, top_k=2)
        result = retriever.run(query_embedding=make_embedding(128, 600))

        assert len(result["documents"]) == 2
