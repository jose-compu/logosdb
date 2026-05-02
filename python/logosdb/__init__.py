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

from ._core import (
    DB, SearchHit, LOGOSDB_VERSION, __version__,
    DIST_IP, DIST_COSINE, DIST_L2,
)

# LangChain adapter is available as optional import
try:
    from .langchain import LogosDBVectorStore
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# LlamaIndex adapter is available as optional import
try:
    from .llamaindex import LogosDBIndex
    LLAMAINDEX_AVAILABLE = True
except ImportError:
    LLAMAINDEX_AVAILABLE = False

# Haystack adapter is available as optional import
try:
    from .haystack import LogosDBDocumentStore, LogosDBRetriever
    HAYSTACK_AVAILABLE = True
except ImportError:
    HAYSTACK_AVAILABLE = False

# Build __all__ based on available optional dependencies
__all__ = [
    "DB", "SearchHit", "LOGOSDB_VERSION", "__version__",
    "DIST_IP", "DIST_COSINE", "DIST_L2",
]

if LANGCHAIN_AVAILABLE:
    __all__.append("LogosDBVectorStore")
if LLAMAINDEX_AVAILABLE:
    __all__.append("LogosDBIndex")
if HAYSTACK_AVAILABLE:
    __all__.extend(["LogosDBDocumentStore", "LogosDBRetriever"])

# Mistral integration is available as optional import
try:
    from .mistral import MistralVectorStore, MistralEmbeddingProvider
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False

if MISTRAL_AVAILABLE:
    __all__.extend(["MistralVectorStore", "MistralEmbeddingProvider"])

# OpenAI integration is available as optional import
try:
    from .openai import OpenAIVectorStore
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

if OPENAI_AVAILABLE:
    __all__.append("OpenAIVectorStore")

# Anthropic integration is available as optional import
try:
    from .anthropic import AnthropicVectorStore
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

if ANTHROPIC_AVAILABLE:
    __all__.append("AnthropicVectorStore")

# Hugging Face integration is available as optional import
try:
    from .huggingface import HuggingFaceVectorStore
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

if HUGGINGFACE_AVAILABLE:
    __all__.append("HuggingFaceVectorStore")

# Cohere integration is available as optional import
try:
    from .cohere import CohereVectorStore
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False

if COHERE_AVAILABLE:
    __all__.append("CohereVectorStore")

# mem0 integration is available as optional import
try:
    from .mem0 import LogosDBVectorStore as Mem0VectorStore
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False

if MEM0_AVAILABLE:
    __all__.append("Mem0VectorStore")
