#!/usr/bin/env python3
"""CPU SciFact benchmark: BreadBowl-Embed v1.3 routing in LogosDB + rerank."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import logosdb  # noqa: E402
from logosdb import DIST_COSINE, DIST_IP  # noqa: E402

from examples.python.breadbowl_logosdb import (  # noqa: E402
    ROUTING_DIM,
    SLOTS,
    aggregate_slot_hits,
    put_routing,
    rerank_passages,
    search_routing,
)

SCIFACT_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)
SCIFACT_QRELS_HF = (
    "https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/main/test.tsv"
)
MODEL_ID = "dotproductx/breadbowl-embed-v1.3"
SLOT_HITS = 512
RERANK_K = 100
RATE_GATE_DOCS = 30
RATE_GATE_MIN = 0.3


def lscpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def physical_cores() -> int:
    try:
        out = subprocess.check_output(["bash", "-c", "lscpu -p | grep -v '^#' | cut -d, -f2 | sort -u"], text=True)
        n = len([x for x in out.split() if x])
        return n if n > 0 else os.cpu_count() or 1
    except (subprocess.CalledProcessError, FileNotFoundError):
        return os.cpu_count() or 1


def download_scifact(work: Path) -> Path:
    zpath = work / "scifact.zip"
    root = work / "scifact"
    if root.is_dir() and any(root.rglob("corpus.jsonl")):
        return root
    work.mkdir(parents=True, exist_ok=True)
    if not zpath.is_file():
        print(f"downloading {SCIFACT_URL}", flush=True)
        for attempt in range(1, 6):
            try:
                subprocess.run(
                    ["curl", "-fL", "--retry", "5", "--retry-delay", "3", "-o", str(zpath), SCIFACT_URL],
                    check=True,
                )
                break
            except subprocess.CalledProcessError:
                if attempt == 5:
                    urlretrieve(SCIFACT_URL, zpath)
                time.sleep(attempt * 2)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(work)
    for child in work.iterdir():
        if child.is_dir() and (child / "corpus.jsonl").is_file():
            return child
    if (root / "corpus.jsonl").is_file():
        return root
    raise RuntimeError("scifact corpus.jsonl not found after extract")


def download_scifact_hf(work: Path) -> Path:
    """Fallback when the UKP zip host is unreachable."""
    from datasets import load_dataset

    root = work / "scifact"
    root.mkdir(parents=True, exist_ok=True)
    corpus_path = root / "corpus.jsonl"
    queries_path = root / "queries.jsonl"
    qrels_dir = root / "qrels"
    qrels_path = qrels_dir / "test.tsv"

    if corpus_path.is_file() and queries_path.is_file() and qrels_path.is_file():
        return root

    print("loading SciFact from Hugging Face datasets", flush=True)
    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    with corpus_path.open("w") as f:
        for row in corpus:
            f.write(
                json.dumps(
                    {"_id": row["_id"], "title": row.get("title", ""), "text": row["text"]},
                    ensure_ascii=False,
                )
                + "\n"
            )
    queries = load_dataset("BeIR/scifact", "queries", split="queries")
    with queries_path.open("w") as f:
        for row in queries:
            f.write(
                json.dumps({"_id": row["_id"], "text": row["text"]}, ensure_ascii=False)
                + "\n"
            )
    qrels_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-fL", "-o", str(qrels_path), SCIFACT_QRELS_HF],
        check=True,
    )
    return root


def ensure_scifact(work: Path) -> Path:
    root = work / "scifact"
    if root.is_dir() and (root / "corpus.jsonl").is_file() and (root / "qrels" / "test.tsv").is_file():
        return root
    zpath = work / "scifact.zip"
    if zpath.is_file() and zpath.stat().st_size > 1_000_000:
        try:
            return download_scifact(work)
        except Exception as exc:
            print(f"UKP zip present but unusable ({exc}); using Hugging Face", flush=True)
    return download_scifact_hf(work)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open() as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            qid, pid, rel = parts[0], parts[1], int(parts[2])
            qrels[qid][pid] = rel
    return qrels


def dcg_at_k(rels: list[int], k: int) -> float:
    s = 0.0
    for i, g in enumerate(rels[:k], start=1):
        if g > 0:
            s += (2**g - 1) / math.log2(i + 1)
    return s


def ndcg_at_k(ranked: list[str], qrels: dict[str, int], k: int = 10) -> float:
    gains = [qrels.get(pid, 0) for pid in ranked[:k]]
    ideal = sorted(qrels.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg <= 0:
        return 0.0
    return dcg_at_k(gains, k) / idcg


def recall_at_k(ranked: list[str], qrels: dict[str, int], k: int = 100) -> float:
    rel = {p for p, g in qrels.items() if g > 0}
    if not rel:
        return 0.0
    hit = sum(1 for p in ranked[:k] if p in rel)
    return hit / len(rel)


def build_subset(
    all_ids: list[str],
    qrels: dict[str, dict[str, int]],
    query_ids: list[str],
    *,
    n_queries: int = 50,
    corpus_size: int = 500,
) -> tuple[list[str], list[str]]:
    picked_q = query_ids[:n_queries]
    relevant: set[str] = set()
    for q in picked_q:
        for pid, g in qrels.get(q, {}).items():
            if g > 0:
                relevant.add(pid)
    filler = [p for p in all_ids if p not in relevant]
    corpus = list(relevant)
    need = corpus_size - len(corpus)
    corpus.extend(filler[: max(0, need)])
    return picked_q, corpus


def encode_corpus(
    model,
    passages: list[tuple[str, str]],
    cache_dir: Path,
    *,
    cores: int,
    check_rate_gate: bool,
) -> tuple[object, list[str], int]:
    """Return Embeddings, ordered ids, skipped_empty count."""
    import torch

    torch.set_num_threads(cores)
    manifest = cache_dir / "manifest.json"
    tensor_path = cache_dir / "embeddings.safetensors"
    ids_path = cache_dir / "passage_ids.json"
    ckpt_dir = cache_dir / "_checkpoint"
    ckpt_ids_path = cache_dir / "passage_ids.checkpoint.json"

    if manifest.is_file() and tensor_path.is_file() and ids_path.is_file():
        from breadbowl_embed import Embeddings

        ids = json.loads(ids_path.read_text())
        stored = Embeddings.load(cache_dir)
        print(f"loaded cached embeddings for {len(ids)} passages", flush=True)
        return stored, ids, 0

    all_ids: list[str] = []
    all_texts: list[str] = []
    skipped = 0
    for pid, text in passages:
        if not (text or "").strip():
            skipped += 1
            continue
        all_ids.append(pid)
        all_texts.append(text)

    from breadbowl_embed import Embeddings

    start_idx = 0
    parts: list = []
    if (ckpt_dir / "manifest.json").is_file() and ckpt_ids_path.is_file():
        done_ids = json.loads(ckpt_ids_path.read_text())
        start_idx = len(done_ids)
        if start_idx > 0:
            stored_ckpt = Embeddings.load(ckpt_dir)
            parts = [stored_ckpt]
            print(f"resuming encode from passage {start_idx}/{len(all_texts)}", flush=True)

    print(f"encoding {len(all_texts)} passages (batch_size=1)...", flush=True)
    enc_t0 = time.perf_counter()
    checkpoint_every = 250
    for idx in range(start_idx, len(all_texts)):
        text = all_texts[idx]
        parts.append(model.encode_documents([text], batch_size=1))
        if check_rate_gate and idx + 1 == RATE_GATE_DOCS:
            elapsed = time.perf_counter() - enc_t0
            rate = RATE_GATE_DOCS / elapsed
            print(f"encode rate after {RATE_GATE_DOCS} docs: {rate:.3f} doc/s", flush=True)
            if rate < RATE_GATE_MIN:
                raise RateGateTripped(rate)
        if (idx + 1) % checkpoint_every == 0:
            partial = Embeddings.concat(parts) if len(parts) > 1 else parts[0]
            if ckpt_dir.is_dir():
                shutil.rmtree(ckpt_dir)
            partial.without_tokens().save(ckpt_dir)
            ckpt_ids_path.write_text(json.dumps(all_ids[: idx + 1]))
            print(f"checkpoint at {idx + 1}/{len(all_texts)} passages", flush=True)
    emb = Embeddings.concat(parts) if len(parts) > 1 else parts[0]
    if cache_dir.is_dir() and not manifest.is_file():
        shutil.rmtree(cache_dir)
    emb.without_tokens().save(cache_dir)
    ids_path.write_text(json.dumps(all_ids))
    if ckpt_dir.is_dir():
        shutil.rmtree(ckpt_dir)
    ckpt_ids_path.unlink(missing_ok=True)
    print(f"saved embeddings to {cache_dir}", flush=True)
    stored = type(emb).load(cache_dir)
    return stored, all_ids, skipped


class RateGateTripped(Exception):
    def __init__(self, rate: float) -> None:
        self.rate = rate


def build_routing_db(db_path: Path, stored, ids: list[str]) -> logosdb.DB:
    n_rows = len(ids) * SLOTS
    if db_path.is_dir() and (db_path / "meta.json").exists():
        db = logosdb.DB(
            str(db_path),
            dim=ROUTING_DIM,
            distance=DIST_IP,
            max_elements=n_rows,
            ef_search=512,
            ef_construction=200,
            M=16,
        )
        if db.count_live() == n_rows:
            return db
        shutil.rmtree(db_path)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = logosdb.DB(
        str(db_path),
        dim=ROUTING_DIM,
        distance=DIST_IP,
        max_elements=n_rows,
        ef_search=512,
        ef_construction=200,
        M=16,
    )
    for i, pid in enumerate(ids):
        routing = np.asarray(stored.routing[i], dtype=np.float32)
        put_routing(db, pid, routing)
    return db


def run_benchmark(
    *,
    work_dir: Path,
    out_path: Path,
    subset: bool,
) -> None:
    import torch
    import breadbowl_embed
    from breadbowl_embed import BreadBowl
    from sentence_transformers import SentenceTransformer

    cores = physical_cores()
    torch.set_num_threads(cores)

    data_root = ensure_scifact(work_dir)
    corpus_rows = read_jsonl(data_root / "corpus.jsonl")
    query_rows = read_jsonl(data_root / "queries.jsonl")
    qrels_path = data_root / "qrels" / "test.tsv"
    if not qrels_path.is_file():
        raise SystemExit(f"missing {qrels_path}")

    qrels = load_qrels(qrels_path)
    corpus_by_id = {r["_id"]: r.get("title", "") + " " + r.get("text", "") for r in corpus_rows}
    queries = {r["_id"]: r["text"] for r in query_rows}
    eval_qids = [q for q in queries if q in qrels and any(g > 0 for g in qrels[q].values())]

    all_corpus_ids = list(corpus_by_id.keys())
    subset_label = None
    if subset:
        eval_qids, corpus_ids = build_subset(all_corpus_ids, qrels, eval_qids)
        subset_label = f"50 queries, {len(corpus_ids)} passages (labeled subset)"
    else:
        corpus_ids = all_corpus_ids

    passages = [(pid, corpus_by_id[pid]) for pid in corpus_ids]
    print(
        f"corpus passages: {len(corpus_ids)}, test queries with qrels: {len(eval_qids)}",
        flush=True,
    )

    model = BreadBowl.from_pretrained(MODEL_ID)
    enc_dir = work_dir / "embeddings"
    try:
        stored, id_order, skipped_empty = encode_corpus(
            model, passages, enc_dir, cores=cores, check_rate_gate=not subset
        )
    except RateGateTripped as e:
        print(f"rate gate tripped ({e.rate:.3f} doc/s); switching to labeled subset", flush=True)
        eval_qids, corpus_ids = build_subset(all_corpus_ids, qrels, eval_qids)
        subset_label = f"50 queries, {len(corpus_ids)} passages (labeled subset)"
        passages = [(pid, corpus_by_id[pid]) for pid in corpus_ids]
        enc_dir = work_dir / "embeddings_subset"
        stored, id_order, skipped_empty = encode_corpus(
            model, passages, enc_dir, cores=cores, check_rate_gate=False
        )

    id_to_pos = {pid: i for i, pid in enumerate(id_order)}
    routing_db_path = work_dir / ("routing_db_subset" if subset_label else "routing_db")
    db = build_routing_db(routing_db_path, stored, id_order)

    # MiniLM baseline
    minilm_path = work_dir / "minilm_db"
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    dim = st_model.get_sentence_embedding_dimension()
    if minilm_path.is_dir() and (minilm_path / "meta.json").exists():
        minilm_db = logosdb.DB(str(minilm_path), dim=dim, distance=DIST_COSINE, ef_search=128, max_elements=len(id_order))
    else:
        minilm_db = logosdb.DB(
            str(minilm_path),
            dim=dim,
            distance=DIST_COSINE,
            ef_search=128,
            max_elements=len(id_order),
        )
        texts = [corpus_by_id[pid] for pid in id_order]
        embs = st_model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
        embs = embs.astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embs /= norms
        minilm_db.put_batch(embs, texts=[str(pid) for pid in id_order])

    ndcg_routing: list[float] = []
    ndcg_rerank: list[float] = []
    ndcg_rescore: list[float] = []
    ndcg_minilm: list[float] = []
    recall_routing: list[float] = []
    recall_rerank: list[float] = []
    recall_minilm: list[float] = []

    lat_route: list[float] = []
    lat_rerank: list[float] = []
    lat_query_enc: list[float] = []

    warmup = True
    for qid in eval_qids:
        qtext = queries[qid]
        t_enc0 = time.perf_counter()
        q_emb = model.encode_queries([qtext], batch_size=1)
        t_enc = time.perf_counter() - t_enc0

        t_r0 = time.perf_counter()
        slot_lists = search_routing(db, np.asarray(q_emb.routing[0], dtype=np.float32), slot_hits=SLOT_HITS)
        ranked_route = [p for p, _ in aggregate_slot_hits(slot_lists)]
        t_route = time.perf_counter() - t_r0

        shortlist = ranked_route[:RERANK_K]
        positions = [id_to_pos[p] for p in shortlist if p in id_to_pos]

        t_rr0 = time.perf_counter()
        if positions:
            scores = rerank_passages(model, q_emb, stored, positions)
            comb = np.asarray(scores.combined)[0]
            rout = np.asarray(scores.routing)[0]
            order = np.argsort(-comb)
            ranked_rerank = [shortlist[i] for i in order]
            ranked_rescore = [shortlist[i] for i in np.argsort(-rout)]
        else:
            ranked_rerank = []
            ranked_rescore = []
        t_rerank = time.perf_counter() - t_rr0

        qv = st_model.encode([qtext], convert_to_numpy=True).astype(np.float32)[0]
        qv /= np.linalg.norm(qv) or 1.0
        minilm_hits = minilm_db.search(qv, top_k=max(100, RERANK_K))
        ranked_minilm = [h.text for h in minilm_hits]

        qrels_q = qrels[qid]
        if not any(g > 0 for g in qrels_q.values()):
            continue

        ndcg_routing.append(ndcg_at_k(ranked_route, qrels_q, 10))
        ndcg_rerank.append(ndcg_at_k(ranked_rerank, qrels_q, 10))
        ndcg_rescore.append(ndcg_at_k(ranked_rescore, qrels_q, 10))
        ndcg_minilm.append(ndcg_at_k(ranked_minilm, qrels_q, 10))
        recall_routing.append(recall_at_k(ranked_route, qrels_q, 100))
        recall_rerank.append(recall_at_k(ranked_rerank, qrels_q, 100))
        recall_minilm.append(recall_at_k(ranked_minilm, qrels_q, 100))

        if warmup:
            warmup = False
            continue
        lat_query_enc.append(t_enc)
        lat_route.append(t_route)
        lat_rerank.append(t_route + t_rerank)

    def du_sb(path: Path) -> int:
        try:
            out = subprocess.check_output(["du", "-sb", str(path)], text=True)
            return int(out.split()[0])
        except (subprocess.CalledProcessError, ValueError):
            return 0

    routing_bytes = du_sb(routing_db_path)
    value_bytes = du_sb(enc_dir)
    minilm_bytes = du_sb(minilm_path)

    title = "BEIR SciFact (test)" if not subset_label else f"SciFact subset — {subset_label}"

    result = {
        "title": title,
        "corpus_passages": len(id_order),
        "test_queries": len(eval_qids),
        "skipped_empty_passages": skipped_empty,
        "subset": subset_label,
        "metrics": {
            "ndcg_at_10": {
                "logosdb_routing_hnsw": statistics.mean(ndcg_routing) if ndcg_routing else None,
                "logosdb_routing_plus_value_rerank": statistics.mean(ndcg_rerank) if ndcg_rerank else None,
                "exact_routing_rescore_top100": statistics.mean(ndcg_rescore) if ndcg_rescore else None,
                "minilm_l6_v2_single_vector": statistics.mean(ndcg_minilm) if ndcg_minilm else None,
            },
            "recall_at_100": {
                "logosdb_routing_hnsw": statistics.mean(recall_routing) if recall_routing else None,
                "logosdb_routing_plus_value_rerank": statistics.mean(recall_rerank) if recall_rerank else None,
                "minilm_l6_v2_single_vector": statistics.mean(recall_minilm) if recall_minilm else None,
            },
        },
        "index_size_bytes": {
            "logosdb_routing_db": routing_bytes,
            "breadbowl_values_safetensors": value_bytes,
            "combined_routing_plus_values": routing_bytes + value_bytes,
            "minilm_db": minilm_bytes,
        },
        "latency_seconds_median": {
            "routing_search_16_slots": statistics.median(lat_route) if lat_route else None,
            "routing_plus_score_rerank": statistics.median(lat_rerank) if lat_rerank else None,
            "encode_one_query": statistics.median(lat_query_enc) if lat_query_enc else None,
        },
        "hardware": {
            "cpu_model": lscpu_model(),
            "cpu_cores_physical": cores,
            "ram_gib": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3), 1),
            "torch_num_threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
            "logosdb_version": logosdb.__version__,
            "breadbowl_embed_version": getattr(breadbowl_embed, "__version__", "unknown"),
            "python_version": sys.version.split()[0],
        },
        "notes": [
            "LogosDB routing arm is HNSW mean over slot searches (miss=-1), not log-mean-exp score_routing.",
            "Exact routing rescore is scores.routing on the top-100 shortlist only.",
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", type=Path, default=Path("benchmarks/breadbowl/work"))
    p.add_argument("--out", type=Path, default=Path("benchmarks/breadbowl/results/scifact.json"))
    p.add_argument("--subset", action="store_true", help="Force labeled subset corpus")
    args = p.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    run_benchmark(work_dir=args.work_dir, out_path=args.out, subset=args.subset)


if __name__ == "__main__":
    main()