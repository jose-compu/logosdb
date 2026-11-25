#!/usr/bin/env python3
"""Render SciFact benchmark markdown from benchmarks/breadbowl/results/scifact.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def render(data: dict) -> str:
    hw = data.get("hardware", {})
    hw_line = (
        f"CPU: {hw.get('cpu_model', '?')}, {hw.get('cpu_cores_physical', '?')} cores, "
        f"{hw.get('ram_gib', '?')} GiB RAM; torch {hw.get('torch_version', '?')} "
        f"({hw.get('torch_num_threads', '?')} threads); Python {hw.get('python_version', '?')}; "
        f"logosdb {hw.get('logosdb_version', '?')}; breadbowl_embed {hw.get('breadbowl_embed_version', '?')}."
    )
    title = data.get("title", "SciFact")
    subset = data.get("subset")
    if subset:
        title = f"{title} ({subset})"
    m = data["metrics"]
    nd = m["ndcg_at_10"]
    rc = m["recall_at_100"]
    sz = data["index_size_bytes"]
    lat = data["latency_seconds_median"]

    def f6(x):
        return "—" if x is None else f"{x:.6f}"

    return f"""### {title}

Corpus passages: {data.get('corpus_passages')}; test queries: {data.get('test_queries')}.

| Arm | nDCG@10 | recall@100 |
|-----|---------|------------|
| LogosDB routing (HNSW mean) | {f6(nd.get('logosdb_routing_hnsw'))} | {f6(rc.get('logosdb_routing_hnsw'))} |
| LogosDB routing + value rerank | {f6(nd.get('logosdb_routing_plus_value_rerank'))} | {f6(rc.get('logosdb_routing_plus_value_rerank'))} |
| Exact routing rescore (top 100) | {f6(nd.get('exact_routing_rescore_top100'))} | — |
| MiniLM-L6-v2 single vector | {f6(nd.get('minilm_l6_v2_single_vector'))} | {f6(rc.get('minilm_l6_v2_single_vector'))} |

Index size (bytes): LogosDB routing {sz.get('logosdb_routing_db')}, value store {sz.get('breadbowl_values_safetensors')}, combined {sz.get('combined_routing_plus_values')}; MiniLM {sz.get('minilm_db')}.

Latency median (s): 16-slot search {lat.get('routing_search_16_slots')}; search + score rerank {lat.get('routing_plus_score_rerank')}; encode one query {lat.get('encode_one_query')}.

{hw_line}
"""


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/breadbowl/results/scifact.json")
    data = json.loads(path.read_text())
    print(render(data))


if __name__ == "__main__":
    main()