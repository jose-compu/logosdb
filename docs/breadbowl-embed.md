# BreadBowl-Embed v1.3 + LogosDB

[BreadBowl-Embed v1.3](https://huggingface.co/dotproductx/breadbowl-embed-v1.3) encodes each passage into **16 routing vectors** (256-d, L2-normalized) and **16 raw value vectors** (256-d, not normalized). LogosDB stores **one row per routing slot** (inner product search). Value vectors live in a separate safetensors file from `Embeddings.save()`. Reranking calls `model.score()` on candidates; the passage text is not encoded again.

Upstream MS MARCO numbers are **BreadBowl’s own** runs (RTX 5090, IVF-PQ on the full 8.8M corpus): see their [v1.3 post](https://breadbowl.ai/blog/breadbowl-embed-v1-3/) and [model card](https://huggingface.co/dotproductx/breadbowl-embed-v1.3). They are not LogosDB measurements.

## Row layout

- `dim=256`, `distance=DIST_IP` (routing is already unit norm; do not use `DIST_COSINE`, which re-normalizes on insert).
- 16 rows per passage, slots `0..15`, via `put_batch`.
- Row `text`: compact JSON `{"passage_id":"<id>","slot":0}` (no spaces).
- `max_elements` = passages × 16; `ef_search` ≥ slot hits (benchmark uses 512).
- Values: `documents.without_tokens().save(dir)` then `Embeddings.load` + `select` for `model.score`.

Helpers: `examples/python/breadbowl_logosdb.py`. Demo: `examples/python/breadbowl_embed_demo.py`. Node insert layout: `examples/node/breadbowl_slots.js`.

## Routing search vs `score_routing`

LogosDB search runs **16 HNSW queries** (one per query slot), aggregates the best hit per slot with **miss = −1**, and averages over slots. That mean is **not** BreadBowl’s log-mean-exp `score_routing` (exact slot math). The benchmark also reports `scores.routing` on the top-100 shortlist as **exact routing rescore**.

## Install (optional, not a PyPI extra)

```bash
pip install "git+https://github.com/BreadBowlAI/breadbowl-embed.git"
pip install logosdb
```

Python **3.12+** and PyTorch are required for the demo and `benchmarks/breadbowl/scifact.py`.

## CPU SciFact benchmark

Results are copied verbatim from `benchmarks/breadbowl/results/scifact.json` (BEIR SciFact test split; macro-averaged nDCG@10 and recall@100).

<!-- SCIFACT_TABLE_START -->

### BEIR SciFact (test)

Corpus passages: 5183; test queries: 300. Full test split (not a subset). Source: `benchmarks/breadbowl/results/scifact.json`.

| Arm | nDCG@10 | recall@100 |
|-----|---------|------------|
| LogosDB routing (HNSW mean) | 0.7333720611924206 | 0.9733333333333334 |
| LogosDB routing + value rerank | 0.7444157064422888 | 0.9733333333333334 |
| Exact routing rescore (top 100) | 0.7415911594628716 | — |
| MiniLM-L6-v2 single vector | 0.6450816521455776 | 0.925 |

| Index | Size (bytes) |
|-------|----------------|
| LogosDB routing DB | 179706796 |
| BreadBowl values (safetensors) | 169897291 |
| Combined routing + values | 349604087 |
| MiniLM-L6-v2 DB | 16234603 |

| Latency (median, seconds) | |
|---------------------------|---|
| 16-slot routing search | 0.031503582000368624 |
| Search + `model.score` rerank | 0.03430102499987697 |
| Encode one query | 0.16775000099914905 |

**Hardware:** Intel(R) Xeon(R) Processor, 4 physical cores, 15.6 GiB RAM; `torch` 2.14.1+cu130 with 4 threads; Python 3.12.3; `logosdb` 1.6.0; `breadbowl_embed` 0.2.0.

<!-- SCIFACT_TABLE_END -->