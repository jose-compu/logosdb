#!/usr/bin/env python3
"""Render SciFact benchmark markdown from benchmarks/breadbowl/results/scifact.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def render(data: dict) -> str:
    hw = data.get("hardware", {})
    hw_line = (
        f"**Hardware:** {hw.get('cpu_model', '?')}, {hw.get('cpu_cores_physical', '?')} physical cores, "
        f"{hw.get('ram_gib', '?')} GiB RAM; `torch` {hw.get('torch_version', '?')} with "
        f"{hw.get('torch_num_threads', '?')} threads; Python {hw.get('python_version', '?')}; "
        f"`logosdb` {hw.get('logosdb_version', '?')}; `breadbowl_embed` {hw.get('breadbowl_embed_version', '?')}."
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
    st_pass = data.get("storage_bytes_per_passage", {})
    ram_pass = data.get("load_rss_delta_bytes_per_passage", {})

    def f6(x):
        return "—" if x is None else f"{x:.6f}"

    def f0(x):
        return "—" if x is None else f"{x:,.0f}"

    notes = data.get("notes", [])
    ram_note = next((n for n in notes if n.startswith("RAM per passage:")), "")
    storage_note = next((n for n in notes if n.startswith("Storage per passage:")), "")
    latency_note = next((n for n in notes if n.startswith("Search latencies")), "")

    return f"""### {title}

Corpus passages: {data.get('corpus_passages')}; test queries: {data.get('test_queries')}. Full test split (not a subset). Source: `benchmarks/breadbowl/results/scifact.json`.

| Arm | nDCG@10 | recall@100 | storage B/passage | load RSS B/passage | median search (s) |
|-----|---------|------------|-------------------|--------------------|-------------------|
| LogosDB routing (HNSW mean) | {f6(nd.get('logosdb_routing_hnsw'))} | {f6(rc.get('logosdb_routing_hnsw'))} | {f0(st_pass.get('routing_index_total_bytes'))} | {f0(ram_pass.get('routing_db_mmap'))} | {f6(lat.get('routing_search_16_slots'))} |
| LogosDB routing + value rerank | {f6(nd.get('logosdb_routing_plus_value_rerank'))} | {f6(rc.get('logosdb_routing_plus_value_rerank'))} | {f0(st_pass.get('routing_plus_value_bytes'))} | {f0(ram_pass.get('routing_plus_value'))} | {f6(lat.get('routing_plus_score_rerank'))} |
| Exact routing rescore (top 100) | {f6(nd.get('exact_routing_rescore_top100'))} | — | {f0(st_pass.get('routing_plus_value_bytes'))} | {f0(ram_pass.get('routing_plus_value'))} | — |
| MiniLM-L6-v2 single vector | {f6(nd.get('minilm_l6_v2_single_vector'))} | {f6(rc.get('minilm_l6_v2_single_vector'))} | {f0(st_pass.get('minilm_index_bytes'))} | {f0(ram_pass.get('minilm_db_mmap'))} | {f6(lat.get('minilm_hnsw_search'))} |

Storage B/passage splits routing into HNSW graph ({f0(st_pass.get('routing_hnsw_graph_bytes'))}), `vectors.bin` ({f0(st_pass.get('routing_vectors_bytes'))}), and value safetensors ({f0(st_pass.get('value_vectors_bytes'))}) where applicable. {storage_note}

{ram_note}

{latency_note} Encode one query median {f6(lat.get('encode_one_query'))} s.

| Index (total bytes) | |
|---------------------|---|
| LogosDB routing DB | {sz.get('logosdb_routing_db')} |
| BreadBowl values (safetensors) | {sz.get('breadbowl_values_safetensors')} |
| Combined routing + values | {sz.get('combined_routing_plus_values')} |
| MiniLM-L6-v2 DB | {sz.get('minilm_db')} |

{hw_line}
"""


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "benchmarks/breadbowl/results/scifact.json")
    data = json.loads(path.read_text())
    print(render(data))


if __name__ == "__main__":
    main()