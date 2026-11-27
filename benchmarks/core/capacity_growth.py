#!/usr/bin/env python3
"""Offline HNSW capacity growth benchmark (not run in CI)."""

from __future__ import annotations

import json
import random
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import logosdb

ROOT = Path(__file__).resolve().parents[2]
DIM = 32
N = 2000
WINDOW = 128
EF = 64
SEED = 42
N_QUERY = 20
K = 10


def l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


def brute_topk(db_vecs: np.ndarray, query: np.ndarray, k: int) -> set[int]:
    scores = db_vecs @ query
    idx = np.argpartition(-scores, kth=min(k, len(scores) - 1))[:k]
    got = set(int(x) for x in idx)
    return got


def recall_at_k(db, vecs: np.ndarray, queries: np.ndarray, n_live: int) -> float:
    hits_total = 0.0
    for qi in range(N_QUERY):
        got = {h.id for h in db.search(queries[qi], K)}
        gt = brute_topk(vecs[:n_live], queries[qi], K)
        hits_total += len(got & gt) / K
    return hits_total / N_QUERY


def run_mode(label: str, max_elements: int) -> dict:
    path = ROOT / f"benchmarks/core/.tmp_capacity_{label}"
    if path.exists():
        shutil.rmtree(path)
    random.seed(SEED)
    np.random.seed(SEED)
    vecs = l2_normalize(np.random.randn(N, DIM).astype(np.float32))
    db = logosdb.DB(str(path), dim=DIM, max_elements=max_elements, ef_search=EF)
    queries = l2_normalize(np.random.randn(N_QUERY, DIM).astype(np.float32))

    windows: list[dict] = []
    recall_before_doubling: float | None = None
    i = 0
    while i < N:
        cap_before = db.index_capacity()
        t0 = time.perf_counter()
        end = min(i + WINDOW, N)
        for j in range(i, end):
            db.put(vecs[j])
        elapsed = time.perf_counter() - t0
        n_this = end - i
        cap_after = db.index_capacity()
        windows.append(
            {
                "inserts_per_sec": n_this / elapsed if elapsed > 0 else 0.0,
                "capacity_start": cap_before,
                "capacity_end": cap_after,
                "rows_end": end,
            }
        )
        if recall_before_doubling is None:
            if cap_after > cap_before:
                recall_before_doubling = recall_at_k(db, vecs, queries, end)
            elif end >= WINDOW:
                recall_before_doubling = recall_at_k(db, vecs, queries, WINDOW)
        i = end

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux ru_maxrss is kilobytes; macOS is bytes.
    peak_rss_mb = rss / 1024.0 if rss > 10_000_000 else rss / (1024.0 * 1024.0)

    if recall_before_doubling is None:
        n_probe = min(WINDOW, db.count())
        recall_before_doubling = recall_at_k(db, vecs, queries, n_probe)

    recall_after_growth = recall_at_k(db, vecs, queries, db.count())

    shutil.rmtree(path, ignore_errors=True)
    return {
        "mode": label,
        "max_elements_initial": max_elements,
        "windows": windows,
        "peak_rss_mb": peak_rss_mb,
        "recall_at_10_before_first_doubling": recall_before_doubling,
        "recall_at_10_after_last_doubling": recall_after_growth,
    }


def main() -> None:
    no_growth = run_mode("no_growth", max_elements=max(N, 2048))
    growth = run_mode("growth", max_elements=WINDOW)
    out = {"dim": DIM, "n_vectors": N, "window": WINDOW, "runs": [no_growth, growth]}
    print(json.dumps(out, indent=2))
    for run in out["runs"]:
        if run["recall_at_10_before_first_doubling"] < 0.90:
            sys.exit(1)
        if run["recall_at_10_after_last_doubling"] < 0.90:
            sys.exit(1)


if __name__ == "__main__":
    main()