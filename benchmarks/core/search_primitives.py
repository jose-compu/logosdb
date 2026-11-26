#!/usr/bin/env python3
"""Offline search-primitives benchmark (not run in CI)."""

from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path

import numpy as np

import logosdb

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "benchmarks/core/results/search_primitives.json"

N = 10_000
DIM = 128
SEED = 42
N_QUERY = 100
K = 10


def l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def brute_topk(db_vecs: np.ndarray, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    scores = db_vecs @ query
    idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.astype(np.uint64), scores[idx].astype(np.float32)


def main() -> None:
    rng = random.Random(SEED)
    np.random.seed(SEED)
    vecs = l2_normalize(np.random.randn(N, DIM).astype(np.float32))
    path = ROOT / "benchmarks/core/.tmp_search_primitives_db"
    if path.exists():
        import shutil

        shutil.rmtree(path)
    db = logosdb.DB(str(path), dim=DIM, max_elements=N + 100, ef_search=64)
    for i in range(N):
        db.put(vecs[i])
    queries = l2_normalize(np.random.randn(N_QUERY, DIM).astype(np.float32))

    lat = []
    recall_hits = 0
    for qi in range(N_QUERY):
        t0 = time.perf_counter()
        hits = db.search(queries[qi], K)
        lat.append(time.perf_counter() - t0)
        gt_ids, _ = brute_topk(vecs, queries[qi], K)
        got = {h.id for h in hits}
        recall_hits += len(got & set(int(x) for x in gt_ids)) / K

    allow_10 = np.array(rng.sample(range(N), N // 10), dtype=np.uint64)
    allow_1 = np.array(rng.sample(range(N), N // 100), dtype=np.uint64)

    def bench_allow(allow: np.ndarray) -> dict:
        rec = []
        for qi in range(min(20, N_QUERY)):
            hits = db.search(queries[qi], K, allow=allow)
            gt_ids, _ = brute_topk(vecs[allow], queries[qi], min(K, len(allow)))
            got = {h.id for h in hits}
            rec.append(len(got & set(int(x) for x in gt_ids)) / min(K, len(allow)))
        return {"recall_at_k_mean": float(np.mean(rec)) if rec else 0.0}

    flat_q = queries[:100].reshape(-1)
    t0 = time.perf_counter()
    ids_b, scores_b = db.search_batch(queries[:100], k=K)
    batch_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    for qi in range(100):
        db.search(queries[qi], K)
    serial_s = time.perf_counter() - t1

    sample_ids = np.arange(100, dtype=np.uint64)
    t2 = time.perf_counter()
    gm = db.get_many(sample_ids)
    gm_s = time.perf_counter() - t2
    t3 = time.perf_counter()
    for i in sample_ids:
        db.get(int(i))
    get_s = time.perf_counter() - t3

    f_min = db.search(queries[0], K, min_score=2.0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "fixture": {"n_vectors": N, "dim": DIM, "n_queries": N_QUERY, "seed": SEED},
                "unfiltered_recall_at_10_mean": recall_hits / N_QUERY,
                "search_latency_s": {"p50": statistics.median(lat), "p95": sorted(lat)[int(0.95 * len(lat))]},
                "allow_10pct": bench_allow(allow_10),
                "allow_1pct": bench_allow(allow_1),
                "search_batch_100_s": batch_s,
                "search_serial_100_s": serial_s,
                "get_many_100_s": gm_s,
                "get_serial_100_s": get_s,
                "min_score_above_all_hits_rows": len(f_min),
                "ids_batch_shape": list(ids_b.shape),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()