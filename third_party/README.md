# Third-Party Dependencies

This directory contains header-only libraries vendored for reproducible builds.

## hnswlib

Hierarchical Navigable Small World algorithm for approximate nearest neighbor search.
- Source: https://github.com/nmslib/hnswlib
- License: Apache-2.0
- Used by: `src/hnsw_index.cpp`

## nlohmann/json

JSON for Modern C++ - a header-only JSON library.
- Source: https://github.com/nlohmann/json
- Version: 3.11.3
- License: MIT
- Used by: `src/metadata.cpp` for JSONL parsing and generation
