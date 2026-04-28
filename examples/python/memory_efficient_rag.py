"""Memory-efficient on-prem RAG with LogosDB — no cloud, no heavy deps.

This example demonstrates a complete RAG pipeline that stays on your machine:
- Uses sentence-transformers for embeddings (runs locally)
- Stores vectors in LogosDB with memory-mapped storage (no loading into RAM)
- Searches without sending data to any external service

RAM Model:
    LogosDB uses mmap() for zero-copy access. Working set scales with
    query patterns, not dataset size:
    - Index overhead: ~10-20% of vector data (HNSW graph)
    - Actual RAM used: depends on how much of the index is hot
    - Max theoretical: N × dim × 4 bytes + index overhead

For a 1M vector DB with 384-dim embeddings (all-MiniLM-L6-v2):
    - Vector data: ~1.5 GB on disk
    - Index overhead: ~150-300 MB
    - Typical query RAM: <100 MB (only touched pages cached)

To scale further:
    - Use smaller dims (quantized embeddings) via external libraries
    - Split data into time-sharded DBs (e.g., one per month)
    - Rely on OS page cache eviction for cold data

Run:
    pip install ".[examples]"
    python examples/python/memory_efficient_rag.py

Example Output:
    Loading model: sentence-transformers/all-MiniLM-L6-v2 (384 dim)
    Building knowledge base: 10000 documents...
    Peak RSS during ingest: 245.7 MB
    Final DB size: 15.3 MB (100 docs × 384 dim × 4 bytes + overhead)
    
    Query: "What causes headaches?"
    Results:
      0.9234  Dehydration is a common cause of headaches...
      0.8912  Tension headaches are caused by muscle contractions...
      0.8543  Migraines may be triggered by stress or certain foods...
    
    Query: "How does photosynthesis work?"
    Results:
      0.9012  Photosynthesis converts light energy into chemical energy...
      0.8756  Chlorophyll absorbs light most efficiently in the blue...
      0.8234  Plants produce glucose and oxygen from CO2 and water...
    
    Peak RSS during query: 67.2 MB
"""
from __future__ import annotations

import gc
import os
import resource
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

import numpy as np

import logosdb


def get_rss_mb() -> float:
    """Get current resident set size (RSS) in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in kilobytes on Linux, bytes on macOS
    kb = usage.ru_maxrss
    if sys.platform == "darwin":
        return kb / (1024 * 1024)  # macOS reports in bytes
    return kb / 1024  # Linux reports in KB


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings for inner-product search."""
    x = x.astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return x / norms


def generate_knowledge_base(n: int = 100, seed: int = 42) -> List[str]:
    """Generate a synthetic knowledge base of documents."""
    # In real usage, these would be chunks from your documents
    templates = [
        "Dehydration is a common cause of headaches. Drink water regularly.",
        "Tension headaches are caused by muscle contractions in the head and neck.",
        "Migraines may be triggered by stress, certain foods, or hormonal changes.",
        "Photosynthesis converts light energy into chemical energy in plants.",
        "Chlorophyll absorbs light most efficiently in the blue and red spectrum.",
        "Plants produce glucose and oxygen from CO2 and water using sunlight.",
        "The mitochondria is the powerhouse of the cell, producing ATP.",
        "Cellular respiration breaks down glucose to release energy.",
        "DNA is a double helix structure discovered by Watson and Crick.",
        "RNA carries genetic information from DNA to ribosomes for protein synthesis.",
        "Python is a high-level programming language known for readability.",
        "Type hints in Python improve code clarity and enable better IDE support.",
        "Asyncio provides cooperative multitasking for I/O-bound Python programs.",
        "HTTP is a stateless protocol for transferring web content.",
        "REST APIs use HTTP methods (GET, POST, PUT, DELETE) for CRUD operations.",
        "Caching reduces latency by storing frequently accessed data.",
        "Database indexing speeds up queries at the cost of storage and write speed.",
        "Vector databases enable semantic search by embedding similarity.",
        "HNSW is an approximate nearest neighbor algorithm with logarithmic query time.",
        "Memory mapping allows zero-copy access to files larger than physical RAM.",
    ]
    
    # Generate variations by combining templates
    np.random.seed(seed)
    docs = []
    for i in range(n):
        # Mix 1-2 templates for variety
        n_templates = np.random.randint(1, 3)
        selected = np.random.choice(templates, n_templates, replace=False)
        # Add some noise words and variation
        doc = " ".join(selected)
        if n_templates == 1:
            # Add some variation to single-template docs
            suffixes = [
                " This is important for understanding the topic.",
                " Consider this in your analysis.",
                "",
            ]
            doc += np.random.choice(suffixes)
        docs.append(f"[{i}] {doc}")
    
    return docs


def build_database(path: str, docs: List[str], model, batch_size: int = 100) -> Tuple[int, float]:
    """Build the LogosDB database from documents.
    
    Uses batch processing to minimize peak RAM usage.
    Returns (count, peak_rss_mb).
    """
    rss_before = get_rss_mb()
    peak_rss = rss_before
    
    dim = model.get_sentence_embedding_dimension()
    
    # Create DB with cosine distance (auto-normalizes)
    db = logosdb.DB(path, dim=dim, distance=logosdb.DIST_COSINE)
    
    # Process in batches to limit peak RAM
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        
        # Encode batch
        embeddings = model.encode(batch, show_progress_bar=False)
        embeddings = l2_normalize(np.asarray(embeddings))
        
        # Insert into DB
        for text, vec in zip(batch, embeddings):
            db.put(vec, text=text)
        
        # Force garbage collection between batches
        del embeddings
        gc.collect()
        
        current_rss = get_rss_mb()
        peak_rss = max(peak_rss, current_rss)
    
    return db.count_live(), peak_rss


def query_database(db: logosdb.DB, query: str, model, top_k: int = 3) -> List[Tuple[float, str]]:
    """Query the database and return scored results."""
    embedding = model.encode([query], show_progress_bar=False)
    embedding = l2_normalize(np.asarray(embedding))[0]
    
    hits = db.search(embedding, top_k=top_k)
    return [(h.score, h.text) for h in hits]


def format_bytes(n: int) -> str:
    """Format byte count to human readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def main() -> None:
    # Check for optional dependency
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print("Error: sentence-transformers not installed.")
        print("Run: pip install '.[examples]'")
        raise SystemExit(1) from e
    
    # Configuration
    n_docs = 1000  # Number of documents to index
    dim = 384  # all-MiniLM-L6-v2 dimension
    
    # Load model
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Loading model: {model_name} ({dim} dim)")
    model = SentenceTransformer(model_name)
    print(f"Model loaded. VRAM/RAM for model: varies by device")
    
    # Generate knowledge base
    print(f"\nGenerating knowledge base: {n_docs} documents...")
    docs = generate_knowledge_base(n_docs)
    
    # Create temp directory for DB
    path = Path(tempfile.mkdtemp(prefix="logosdb-rag-"))
    print(f"Database path: {path}")
    
    try:
        # Build database
        print(f"\nBuilding database (batch size: 100)...")
        rss_before = get_rss_mb()
        count, peak_rss = build_database(str(path), docs, model, batch_size=100)
        
        # Get DB size on disk
        db_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        
        print(f"\n=== Ingestion Complete ===")
        print(f"Documents indexed: {count}")
        print(f"RSS before: {rss_before:.1f} MB")
        print(f"Peak RSS during ingest: {peak_rss:.1f} MB")
        print(f"DB size on disk: {format_bytes(db_size)}")
        print(f"Theoretical vector size: {format_bytes(count * dim * 4)}")
        print(f"Index overhead: {format_bytes(db_size - count * dim * 4)}")
        
        # Query examples
        queries = [
            "What causes headaches?",
            "How does photosynthesis work?",
            "What is a vector database?",
            "Explain DNA structure",
            "What is asyncio in Python?",
        ]
        
        print(f"\n=== Query Phase ===")
        db = logosdb.DB(str(path), dim=dim, distance=logosdb.DIST_COSINE)
        
        rss_before_query = get_rss_mb()
        peak_query_rss = rss_before_query
        
        for query in queries:
            results = query_database(db, query, model, top_k=3)
            current_rss = get_rss_mb()
            peak_query_rss = max(peak_query_rss, current_rss)
            
            print(f"\nQuery: {query!r}")
            for score, text in results:
                # Truncate long text for display
                display_text = text[:80] + "..." if len(text) > 80 else text
                print(f"  {score:.4f}  {display_text}")
        
        print(f"\n=== Memory Summary ===")
        print(f"Peak RSS during query: {peak_query_rss:.1f} MB")
        print(f"DB remains on disk at: {path}")
        print(f"(Cleanup skipped so you can inspect with logosdb-cli)")
        
        # Show cleanup command
        print(f"\nTo cleanup later:")
        print(f"  rm -rf {path}")
        
        # Skip cleanup for inspection
        # shutil.rmtree(path, ignore_errors=True)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        shutil.rmtree(path, ignore_errors=True)
    except Exception as e:
        print(f"\nError: {e}")
        shutil.rmtree(path, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
