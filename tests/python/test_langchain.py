"""Tests for LogosDB LangChain VectorStore adapter."""

import shutil
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

# Skip all tests if langchain is not installed
pytest.importorskip("langchain_core")

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from logosdb import LogosDBVectorStore


class DummyEmbeddings(Embeddings):
    """Dummy embedding model for testing."""

    def __init__(self, dim: int = 128):
        self.dim = dim
        self.embeddings = {}

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate deterministic embeddings for texts."""
        results = []
        for text in texts:
            if text not in self.embeddings:
                # Create a deterministic embedding based on text hash
                rng = np.random.RandomState(hash(text) % (2**31))
                vec = rng.randn(self.dim).astype(np.float32)
                vec /= np.linalg.norm(vec)
                self.embeddings[text] = vec.tolist()
            results.append(self.embeddings[text])
        return results

    def embed_query(self, text: str) -> List[float]:
        """Generate deterministic embedding for query."""
        return self.embed_documents([text])[0]


@pytest.fixture
def temp_db_path():
    """Create a temporary directory for the test database."""
    path = tempfile.mkdtemp(prefix="logosdb_langchain_test_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def embeddings():
    """Create a dummy embedding model."""
    return DummyEmbeddings(dim=128)


class TestLogosDBVectorStore:
    """Test suite for LogosDBVectorStore."""

    def test_init(self, temp_db_path):
        """Test store initialization."""
        store = LogosDBVectorStore(path=temp_db_path, dim=128)
        assert store._dim == 128
        assert len(store) == 0

    def test_add_texts_with_embeddings(self, temp_db_path, embeddings):
        """Test adding texts with pre-computed embeddings."""
        store = LogosDBVectorStore(path=temp_db_path, dim=128)

        texts = ["hello world", "foo bar", "baz qux"]
        embs = [np.array(e, dtype=np.float32) for e in embeddings.embed_documents(texts)]

        ids = store.add_texts(texts, embeddings=embs)

        assert len(ids) == 3
        assert len(store) == 3

    def test_add_documents(self, temp_db_path, embeddings):
        """Test adding Document objects."""
        store = LogosDBVectorStore(path=temp_db_path, dim=128)

        docs = [
            Document(page_content="first doc", metadata={"ts": "2025-01-01"}),
            Document(page_content="second doc", metadata={"ts": "2025-01-02"}),
        ]
        embs = [np.array(e, dtype=np.float32) for e in embeddings.embed_documents([d.page_content for d in docs])]

        ids = store.add_documents(docs, embeddings=embs)

        assert len(ids) == 2
        assert len(store) == 2

    def test_similarity_search_by_vector(self, temp_db_path, embeddings):
        """Test searching by vector."""
        store = LogosDBVectorStore(path=temp_db_path, dim=128)

        # Add documents
        texts = ["apple pie", "banana bread", "cherry tart"]
        embs = [np.array(e, dtype=np.float32) for e in embeddings.embed_documents(texts)]
        store.add_texts(texts, embeddings=embs)

        # Search with the same embedding as "apple pie"
        query_emb = np.array(embeddings.embed_query("apple pie"), dtype=np.float32)
        results = store.similarity_search_by_vector(query_emb, k=2)

        assert len(results) == 2
        assert results[0].page_content == "apple pie"
        assert "score" in results[0].metadata

    def test_similarity_search(self, temp_db_path, embeddings):
        """Test searching by text (requires embedding model)."""
        store = LogosDBVectorStore(path=temp_db_path, dim=128)

        # Add documents
        texts = ["machine learning", "deep learning", "neural networks"]
        embs = [np.array(e, dtype=np.float32) for e in embeddings.embed_documents(texts)]
        store.add_texts(texts, embeddings=embs)

        # Search with query text
        results = store.similarity_search("machine learning", k=2, embeddings=embeddings)

        assert len(results) == 2

    def test_similarity_search_with_score(self, temp_db_path, embeddings):
        """Test searching with scores."""
        store = LogosDBVectorStore(path=temp_db_path, dim=128)

        texts = ["cat", "dog", "fish"]
        embs = [np.array(e, dtype=np.float32) for e in embeddings.embed_documents(texts)]
        store.add_texts(texts, embeddings=embs)

        results = store.similarity_search_with_score("cat", k=2, embeddings=embeddings)

        assert len(results) == 2
        for doc, score in results:
            assert isinstance(doc, Document)
            # Allow small floating point tolerance
            assert 0.0 <= score <= 1.001

    def test_delete(self, temp_db_path, embeddings):
        """Test deleting documents."""
        store = LogosDBVectorStore(path=temp_db_path, dim=128)

        texts = ["one", "two", "three"]
        embs = [np.array(e, dtype=np.float32) for e in embeddings.embed_documents(texts)]
        store.add_texts(texts, embeddings=embs)

        assert len(store) == 3

        # Get row IDs from search results
        query_emb = np.array(embeddings.embed_query("one"), dtype=np.float32)
        results = store.similarity_search_by_vector(query_emb, k=1)
        row_id = results[0].metadata["row_id"]

        # Delete by row_id
        store.delete(row_ids=[row_id])

        assert len(store) == 2

    def test_from_documents(self, temp_db_path, embeddings):
        """Test creating store from documents."""
        docs = [
            Document(page_content="doc1", metadata={"key": "value1"}),
            Document(page_content="doc2", metadata={"key": "value2"}),
        ]

        store = LogosDBVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            path=temp_db_path,
            dim=128,
        )

        assert len(store) == 2

    def test_from_texts(self, temp_db_path, embeddings):
        """Test creating store from texts."""
        store = LogosDBVectorStore.from_texts(
            texts=["alpha", "beta", "gamma"],
            embedding=embeddings,
            path=temp_db_path,
            dim=128,
        )

        assert len(store) == 3

    def test_timestamp_filter(self, temp_db_path, embeddings):
        """Test timestamp range filtering."""
        store = LogosDBVectorStore(path=temp_db_path, dim=128)

        # Add with timestamps
        texts = ["early", "late"]
        embs = [np.array(e, dtype=np.float32) for e in embeddings.embed_documents(texts)]
        metadatas = [
            {"timestamp": "2025-01-01T00:00:00Z"},
            {"timestamp": "2025-01-15T00:00:00Z"},
        ]
        store.add_texts(texts, metadatas=metadatas, embeddings=embs)

        # Search with timestamp filter
        query_emb = np.array(embeddings.embed_query("early"), dtype=np.float32)
        results = store.similarity_search_by_vector(
            query_emb,
            k=10,
            ts_from="2025-01-01T00:00:00Z",
            ts_to="2025-01-10T00:00:00Z",
        )

        assert len(results) == 1
        assert results[0].page_content == "early"

    def test_repr(self, temp_db_path):
        """Test string representation."""
        store = LogosDBVectorStore(path=temp_db_path, dim=128)
        repr_str = repr(store)
        assert "LogosDBVectorStore" in repr_str
        assert temp_db_path in repr_str

    def test_error_without_embeddings(self, temp_db_path):
        """Test that add_texts requires embeddings."""
        store = LogosDBVectorStore(path=temp_db_path, dim=128)

        with pytest.raises(ValueError, match="pre-computed embeddings"):
            store.add_texts(["hello"])

    def test_error_without_embeddings_model(self, temp_db_path, embeddings):
        """Test that similarity_search requires embedding model."""
        store = LogosDBVectorStore(path=temp_db_path, dim=128)

        # Add a document first
        embs = [np.array(embeddings.embed_documents(["hello"])[0], dtype=np.float32)]
        store.add_texts(["hello"], embeddings=embs)

        with pytest.raises(ValueError, match="embeddings model"):
            store.similarity_search("hello")

    def test_cosine_normalization(self, temp_db_path, embeddings):
        """Test that cosine mode handles unnormalized vectors."""
        store = LogosDBVectorStore(path=temp_db_path, dim=128, use_cosine=True)

        # Add unnormalized vectors
        unnormalized = [np.random.randn(128).astype(np.float32) * 10]  # Large magnitude
        ids = store.add_texts(["test"], embeddings=unnormalized)

        assert len(ids) == 1

        # Search should still work
        query = np.random.randn(128).astype(np.float32) * 5
        results = store.similarity_search_by_vector(query, k=1)
        assert len(results) == 1
