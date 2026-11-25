# LogosDB roadmap to 2.0.0

Design for classical ML on LogosDB (`logos-ml`). Status: approved, with the version shift, the lightweight-benchmark rule, and the BreadBowl model rule below. This document does not change the C API.

`main` at the time of writing is 1.5.1 (`LOGOSDB_VERSION_STRING` in `include/logosdb/logosdb.h`, commit `1ba7717`). **1.6.0 is reserved for the in-progress BreadBowl-Embed release and is not part of this plan.** There is no BreadBowl tree on `main`. Core milestones here start at 1.7.0. `logos-ml` milestones stay `logos-ml 0.1.0` through `logos-ml 0.9.0 (Python)` and live in this repository for now.

Semver is the README rule. A minor may add functions and additive files. A major is a C API break, an on-disk format with no migration, or a binding contract that cannot stay compatible. Nothing in the ML work requires 2.0.0. 2.0.0 is only the default-flip listed in section 6.

## 1. Executive summary

LogosDB stays an embedded HNSW store: one directory, one mutex, a vector, a text string, and a timestamp string. Classical ML is a client of that store.

**C/C++ core, additive.** Id allow/deny inside the vendored hnswlib walk, with an exact scan when the allow-set is small. `search_batch`, `get_many`, and a `min_score` cut on the existing score. Already [#134](https://github.com/jose-compu/logosdb/issues/134), [#136](https://github.com/jose-compu/logosdb/issues/136), and [#144](https://github.com/jose-compu/logosdb/issues/144). Target milestone **1.7.0**.

**Bindings, 1.8.0.** Node promises via N-API `AsyncWorker` ([#137](https://github.com/jose-compu/logosdb/issues/137)). `Float32Array` in and out by one `memcpy` ([#140](https://github.com/jose-compu/logosdb/issues/140)). Python `search` / `search_batch` / `get_many` release the GIL. The C API stays synchronous. Today `put_batch`, NDJSON import, and NDJSON export release the GIL; `search` does not (`python/src/bindings.cpp`).

**`logos-ml` in this repo.** Envelopes, keymap, workspace lock, spherical k-means, kNN-graph clustering, UMAP, HDBSCAN, c-TF-IDF, MMR, Platt scaling, and the Q table. Q-values stay in a sidecar file, which [#144](https://github.com/jose-compu/logosdb/issues/144) already assumes.

**Models.**

- **Node `logos-ml` does not use BreadBowl** unless a verified ONNX export runs in Transformers.js / `onnxruntime-node` and matches Python outputs on a fixed fixture (section 5). There is no Python bridge and no subprocess backend.
- The v1.3 toolkit does not ship that export. `breadbowl-embed export` writes safetensors. [`@breadbowl/embed`](https://github.com/BreadBowlAI/breadbowl-embed/tree/main/sdks/typescript) is an HTTP client. Node is therefore not natively compatible, and its default is the best gated ONNX embedder and ONNX cross-encoder (MiniLM, bge-small, or e5-small, plus `ms-marco-MiniLM-L-6-v2` or `bge-reranker-base`).
- **Python `logos-ml` prefers BreadBowl-Embed v1.3** only if it passes the speed and quality gate. If it fails, the Python default switches to that same ONNX winner and BreadBowl stays an optional extra.

**Not in the core.** JSON parsed on the HNSW hot path, a recommend operator, UMAP, HDBSCAN, ONNX, and a Q-update. Qdrant’s recommend API is a query in front of the same index ([Explore](https://qdrant.tech/documentation/search/explore/)). BERTopic is a pipeline on embeddings, not a database ([algorithm](https://maartengr.github.io/BERTopic/algorithm/algorithm.html)). sqlite-vec hid in-extension k-means behind a compile flag ([sqlite-vec#277](https://github.com/asg017/sqlite-vec/pull/277)).

Shipped since the original plan was written against 1.2.0: `sync` / `stats` / `compact` (1.3.0), `dtype` (1.4.0), `live_ids` / `raw_vectors` / Node `exportNdjson` (1.5.0), score-range comments (1.5.1). Node `rawVectors()` still allocates, decodes, and `memcpy`s. `logosdb_raw_vectors` returns the float32 mmap pointer only; float16/int8 returns null and the decode API copies.

## 2. Placement

| Capability | Where | Format / ABI | Issue |
|---|---|---|---|
| Id allow/deny during HNSW, exact scan if the allow-set is small | Core, then both bindings | Additive. `vectors.bin` v2 unchanged | [#134](https://github.com/jose-compu/logosdb/issues/134), milestone 1.7.0 |
| `search_batch` (ids + scores, no text) | Core; threads under the existing mutex | Additive | [#136](https://github.com/jose-compu/logosdb/issues/136), 1.7.0 |
| `min_score`, `get_many` | Core | Additive | [#144](https://github.com/jose-compu/logosdb/issues/144), 1.7.0 |
| Node async search / putBatch / searchBatch | Node binding | Sync methods remain | [#137](https://github.com/jose-compu/logosdb/issues/137), 1.8.0 |
| `Float32Array` / NumPy, one copy | Bindings | No mmap view | [#140](https://github.com/jose-compu/logosdb/issues/140), 1.8.0 |
| GIL release on search | Python binding | None | New, 1.8.0 |
| Empty-timestamp skip, default off | Core flag | Default-on is a 2.0 behavior change | [#141](https://github.com/jose-compu/logosdb/issues/141), 1.8.0 |
| Caps on `k`, `candidate_k`, `ef_search` | New entry points first | Applying them to old `search` is 2.0 | New, 1.9.0 |
| Fuzz NDJSON, metadata lines, filters, truncated header | Tests | None | New, 1.9.0 |
| Optional `max_text` and exclusive `flock`, default off | Core | None until 2.0 flips the lock default | New, 1.9.0 |
| HNSW `random_seed`, default 100 | Core | Default unchanged | New, 1.9.0 |
| External string key | Experimental `keys.bin`; until then a library keymap | New file. Old readers ignore it | [#143](https://github.com/jose-compu/logosdb/issues/143), 1.10.0 |
| In-place replace, same id | New `logosdb_replace`. `update` stays delete+put | Slot overwrite | New, 1.10.0 |
| Typed label / `y` / tag bits | Optional `payload.bin` | New file | New, 1.10.0 |
| JSON envelope, c-TF-IDF, MMR, Platt, RRF | `logos-ml` | Beside the DBs | `logos-ml` 0.1–0.6 |
| Spherical k-means | `logos-ml` native addon, not `logosdb.h` | None in the DB | `logos-ml` 0.3.0 |
| kNN graph | `search_batch` in core; union-find in the library | None | [#136](https://github.com/jose-compu/logosdb/issues/136) |
| UMAP, HDBSCAN | `logos-ml`. Python extras may call umap-learn and hdbscan | None | `logos-ml` 0.4.0 |
| ONNX MiniLM / bge / e5 and a small cross-encoder | Node `logos-ml`, and Python when BreadBowl fails the gate | Manifest sha256 | `logos-ml` 0.1.0 |
| BreadBowl-Embed v1.3 | Python `logos-ml` only, and only if the gate passes and no verified ONNX exists | Optional extra. Not a Node dependency | `logos-ml` 0.9.0 |
| Q table | `logos-ml` `qtable.f32` | Sidecar | `logos-ml` 0.8.0 |
| HTTP | Out of scope | — | [#124](https://github.com/jose-compu/logosdb/issues/124) |

## 3. Designs

### 3.1 Filtered search

`HnswIndex::search` calls `searchKnn(query, k)` and drops the `BaseFilterFunctor*` the vendored algorithm already accepts (`third_party/hnswlib/hnswlib/hnswalg.h`). The filter runs when a candidate enters the result heap. Rejected nodes still expand the walk. Upper layers are unfiltered. USearch documents the same routing behavior, and a Python predicate re-enters the GIL ([filtered search](https://www.unum.cloud/docs/usearch/python/filtered-search.md)). hnswlib tells Python callers to set `num_threads=1` with a Python filter ([example](https://github.com/nmslib/hnswlib/blob/master/examples/python/example_filter.py)).

Qdrant’s planner exact-scans a small match set (default `full_scan_threshold` 10,000 KB) and otherwise walks the graph ([Filtering](https://qdrant.tech/documentation/search-design/vector-search-filtering/)). LanceDB can exact-scan a very selective HNSW mask ([vector search](https://docs.lancedb.com/oss/v0-41-0/search/vector-search)). FAISS checks an `IDSelector` while scanning; `IDSelectorBatch` is O(1) ([search parameters](https://github.com/facebookresearch/faiss/wiki/Setting-search-parameters-for-one-query)). LogosDB does not fork hnswlib to add payload edges. The functor is O(1) and does not call JS or Python.

```c
typedef struct logosdb_id_filter_t {
    const uint64_t* allow;   /* NULL = all live ids */
    size_t n_allow;
    const uint64_t* deny;
    size_t n_deny;
    float min_score;         /* NaN = no cut. Uses SearchHit.score */
    int skip_empty_ts;       /* 0 or 1 */
    const char* ts_from;     /* NULL or ISO-8601 */
    const char* ts_to;
} logosdb_id_filter_t;

logosdb_search_result_t* logosdb_search_filtered(
    logosdb_t* db, const float* query, int dim, int top_k,
    const logosdb_id_filter_t* f, char** errptr);
```

Build a bitmap of `max_elements` bits (default 1,000,000) from allow minus deny minus tombstones. If `n_allow` is at most 4,096 or the allow-set’s float32 bytes are at most 10,000 KB, score those rows exactly and skip HNSW. That byte budget is Qdrant’s published default, used here as the starting constant, then measured (section 5). Timestamp and `min_score` apply to both paths. `min_score` uses the public score: cosine and IP are `1 - hnsw_distance`, L2 is `1 / (1 + distance)`. Empty timestamps still pass unless `skip_empty_ts` is set. That matches today’s `logosdb_search_ts_range` loop.

`search` and `search_ts_range` stay. A selective filter may return fewer than `k`. Callers raise `ef_search`. There is no adaptive over-fetch loop in the core.

Node: `allow` / `deny` are `BigUint64Array`. Python: `np.uint64` arrays. The binding copies the ids, then releases the GIL or runs an `AsyncWorker`. No callable crosses the walk.

Security: reject a null `allow` with `n_allow > 0`, ids `>= count()`, and `k` above the new-API cap (4,096). The bitmap is `max_elements / 8` bytes, fixed at open. Do not parse JSON in the functor. MCP tag filters stay a post-filter over a `[tags:{…}]` text suffix ([#84](https://github.com/jose-compu/logosdb/issues/84)); they are the wrong tool for an exclude-set.

### 3.2 `search_batch`, async, and the GIL

Every public call takes `db->mu` (`src/logosdb.cpp`), including search. hnswlib allows concurrent `knn_query` with other `knn_query` and forbids it beside `add_items` (vendored README). The LogosDB mutex is stricter. An `AsyncWorker` unblocks libuv. It does not parallelize two operations on one `DB`.

```c
/* queries: n * dim floats. out_ids, out_scores: n * k, row-major.
   Missing slots: id UINT64_MAX, score NaN. No text. */
int logosdb_search_batch(logosdb_t* db, const float* queries, int n, int dim,
                         int k, const logosdb_id_filter_t* f,
                         uint64_t* out_ids, float* out_scores, char** errptr);
```

Hold `db->mu` for the whole batch. Inside it, `std::thread` may call const `searchKnn`. Cap `n` at 65,536 and `k` at 4,096 on this function only. `logosdb_get_many` copies float32 embeddings for a caller id list under one lock.

Node copies `Float32Array` bytes on the main thread, then `AsyncWorker::Execute` ([node-addon-api](https://github.com/nodejs/node-addon-api/blob/main/doc/async_worker.md)). Python copies under the GIL, releases it for the C call, and returns NumPy arrays the binding owns.

Security: reject non-finite queries. Do not release the mutex mid-batch. The caps are the DoS bound for the new APIs. Old `logosdb_search` stays uncapped until 2.0.0.

### 3.3 Buffers

`logosdb_raw_vectors` for float32 points at the mmap. Append can remap. A JS or NumPy view of that pointer is a use-after-unmap. [#140](https://github.com/jose-compu/logosdb/issues/140) requires views to be safe across close and compact. Ship one `memcpy` each way. Keep `number[]`. `rawVectors()` / `raw_vectors()` stay copies, including float16/int8 dequantize.

Node `SearchHit.id` stays a Number, exact through 2^53. Default `maxElements` is 1,000,000. Filter ids are `BigUint64Array`.

### 3.4 Keys, replace, payload

A row is a vector, text, and a timestamp (`src/metadata.h`). `meta.jsonl` is loaded entirely into memory. Keep document JSON in the library, capped at 8 KiB by default and 64 KiB hard, enforced in `logos-ml` so existing databases still open. Optional core `max_text` in 1.9.0 defaults to unlimited.

`logos-ml` keeps `keymap.<store>.json` (temp file, fsync, rename, high-water scan) until [#143](https://github.com/jose-compu/logosdb/issues/143) has a durable `keys.bin`. `logosdb_update` stays delete-then-put and returns a new id. New `logosdb_replace` overwrites the vector slot and the hnswlib element for the same label and does not consume `maxElements`. Create-time `allow_replace_deleted` is false today.

Optional `payload.bin` in 1.10.0: fixed records `(u32 label, f32 y, u64 tag_bits, u8 flags)` indexed by row id. The functor compares integers. Absent file means no payload predicate. 1.5 readers ignore the extra file.

### 3.5 k-means and the kNN graph

FAISS k-means is a C++ `Clustering` object in the library ([wiki](https://github.com/facebookresearch/faiss/wiki/K%E2%80%90means-clustering)). scikit-learn `KMeans` is Cython ([clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)). Neither lives in the point store. Spherical k-means belongs in a `logos-ml` addon, seeded, not in `logosdb.h`. A JS Lloyd loop is acceptable under the row cap the bench records. The kNN graph is `search_batch` with `k = 10` (11 for leave-one-out), then union-find in the library. Cluster ids go to `clusters/<run>.assign.json`, not through `update`.

### 3.6 HDBSCAN and UMAP

BERTopic’s default pipeline is embed, UMAP, HDBSCAN, then c-TF-IDF, and each stage is replaceable ([algorithm](https://maartengr.github.io/BERTopic/algorithm/algorithm.html)). The hdbscan maintainer had no C++ port to recommend ([hdbscan#403](https://github.com/scikit-learn-contrib/hdbscan/issues/403)). Keep UMAP and HDBSCAN in `logos-ml` for the small band (`umap-js`, `hdbscan-ts`). Above the measured cutoff, use the kNN graph. The Python package may depend on `umap-learn` and `hdbscan` as extras. Assignment after fit is search on the topics store. c-TF-IDF and phrase MMR stay in the library. The required bench does not install BERTopic (section 5).

### 3.7 Embeddings and the model gate

Inference stays outside the core. The README already says the caller supplies floats.

**Node path.** Transformers.js runs ONNX Runtime ([transformers.js](https://github.com/huggingface/transformers.js)). The Node library loads only ONNX. Candidates, skipped when no ONNX file is published:

- `Xenova/all-MiniLM-L6-v2` (384-d)
- `Xenova/bge-small-en-v1.5` (384-d)
- `Xenova/multilingual-e5-small` (384-d, with `query:` / `passage:` prefixes)
- Rerankers: `Xenova/ms-marco-MiniLM-L-6-v2`, and an ONNX build of `BAAI/bge-reranker-base` if one is published

Pin the Hub revision and the sha256 of each ONNX file in `manifest.json`. Open fails on mismatch. Quantized weights are the default. Batches of 32. Mean pool, L2 normalize. Stores opened for these models are `DIST_COSINE`, one row per passage. Chunk long text in the library. The cache key is sha256 of `model_revision || text`.

**Python path.** BreadBowl-Embed v1.3 is the preferred default only after the gate in section 5. Facts from the toolkit, not from a logos-ml run ([README](https://github.com/BreadBowlAI/breadbowl-embed), [model card](https://huggingface.co/dotproductx/breadbowl-embed-v1.3)):

- Weights: `dotproductx/breadbowl-embed-v1.3`, tag `v1.3.0`, step 5152. Safetensors. The loader checks manifest checksums. No pickle.
- Install: Python ≥ 3.12, `torch>=2.11,<2.15`, Transformers, Hugging Face Hub, safetensors, NumPy. `pip install breadbowl-embed` is documented as working once the package is on PyPI; until then the install is from the git repository. The toolkit’s own package version at the time of this writing is 0.2.0.
- Each passage is 16 routing slots and 16 value slots, each 256-d (`[items, 16, 256]`). Routing slots are L2-normalized. Value slots are raw. Normalizing values before the readout changes the model.
- `model.score` returns `routing`, `value`, and `combined = 0.75 * routing + 0.25 * value`. Routing temperature 0.02, value-attention temperature 0.1.
- Document token cap 384. Query token cap 128, including the instruction. Default `batch_size=1` so encodings do not depend on batch neighbors. CPU is the default device. CUDA is not validated by the toolkit.
- Float32 slot storage is 32,768 bytes per passage; float16 is 16,384; int8 is 8,256. A float32 copy of the routing slots is 16 KiB per passage.

The vendor’s GPU figures (a whole query at 43 ms, rerank 2.9 ms, on one RTX 5090 over 8.8 million passages) are in the [v1.3 post](https://breadbowl.ai/blog/breadbowl-embed-v1-3/). They are not the CPU gate and they are not logos-ml results.

If Python selects BreadBowl, the LogosDB store is dim 256, `DIST_COSINE`, **16 routing rows per passage**. Value slots live in a safetensors sidecar keyed by passage id. They are not written into HNSW and they are not L2-normalized. A query encodes to 16 routing slots, `search_batch` runs those 16 queries, and `model.score` reranks the shortlist. `maxElements` counts slots. One workspace binds one model. Opening a 384-d ONNX store with a 256-d BreadBowl manifest fails.

**ONNX compatibility test, which is the only way BreadBowl may enter Node.** A fixed 8-text fixture, Python float32, `batch_size=1`, against the ONNX session in `onnxruntime-node`. Max absolute difference on `routing` and on `value` ≤ 1e-4, and the argsort of `combined` matches. The measured max absolute difference is written by the test. Until that test exists and passes, Node must not load BreadBowl, must not spawn Python, and must not call the hosted HTTP SDK for local inference.

CLIP for images stays a separate workspace and a separate manifest. It is not BreadBowl.

### 3.8 Q-learning

State cells are append-only rows. Q and visit counts are `qtable.f32` (header, then `Q[id][a]` and `N[id][a]`), updated by temp file, fsync, and rename. They do not call `update()`. The online step uses `search_filtered` with `k = 8` and `min_score`. The Bellman update and ε / UCB stay in the library (MFEC, [arXiv:1606.04460](https://arxiv.org/abs/1606.04460); NEC, [arXiv:1703.01988](https://arxiv.org/abs/1703.01988)). `sweep()` is `search_batch` over the transition log. The policy RNG is seeded and stored in the agent manifest. The acceptance bench is a coordinate gridworld and does not load an embedding model (section 5).

### 3.9 What the libraries call

```ts
await db.search(vec, { k: 15, allow: ids, minScore: 0.4 });
await db.searchBatch(matrix, { k: 10 });
await db.getMany(ids);
```

```python
hits = db.search(q, k=15, allow=ids, min_score=0.4)  # GIL released
ids, scores = db.search_batch(Q, k=10)
vecs = db.get_many(ids)
```

Recommend is a decayed profile vector, a deny-set of seen ids, and MMR in the library. Qdrant’s `average_vector` strategy is that same shape ([Explore](https://qdrant.tech/documentation/search/explore/)). On the Python BreadBowl path the profile is 16 routing slots plus value slots, and the deny-set is passage ids expanded to their 16 row ids.

## 4. Security

`logosdb_open` does not confine paths. It creates the directory it is given. MCP confines `logosdb_index_file`; the core does not. `export_ndjson`, `import_ndjson`, and `compact` take caller paths. `logos-ml` `realpath`s the workspace, rejects a symlink that escapes it, and holds `flock` on `workspace.lock`. Core exclusive lock stays opt-in until 2.0.0, because two processes can open one directory today.

There is no inter-process lock in the core. Two writers can corrupt the WAL and the index. HNSW stores a copy of each vector and `vectors.bin` stores another. The existing fuzzer parses JSON with a 256 KiB cap (`tools/fuzz_jsonl_metadata.cpp`) and does not feed the WAL, `meta.jsonl`, or NDJSON import.

| Threat | Stop |
|---|---|
| Huge `k` or batch `n` | Caps on the new APIs only: `k ≤ 4096`, `n ≤ 65536`, `candidate_k ≤ 100000`, `ef_search ≤ 4096` |
| Id filter past `count()` | Reject before the bitmap. Bitmap size is `max_elements` |
| Host callback inside the walk | Not offered |
| JSON envelope in C | Parsed in the library into a known schema |
| Multi-GB text in `meta.jsonl` | Library cap. Optional core `max_text`, default unlimited |
| Path escape | Library realpath jail. Core paths are trusted |
| Two processes | Library `flock`. Core `logosdb_options_set_exclusive`, default off |
| Crash mid-write | WAL plus `sync` for the DB. Keymap, manifest, Q table, and value sidecar use temp + fsync + rename |
| `update` filling `maxElements` | Library avoids `update` for cluster ids and Q. `replace` and `compact` are the tools |
| Model supply chain | Manifest sha256. ONNX Runtime with custom ops disabled if the loaded API allows it. BreadBowl safetensors only. No pickle ([pickle](https://docs.python.org/3/library/pickle.html)) |
| Non-determinism | hnswlib seed stays 100. k-means and ε use a stored seed. BreadBowl encode bench uses `batch_size=1` |
| Mmap use-after-free | Bindings copy |
| Node loading BreadBowl by shelling out | Forbidden. The Node package has no Python dependency for inference |

Fuzz targets for 1.9.0: one NDJSON line, one `meta.jsonl` line, a `logosdb_id_filter_t` with random ids, and a truncated `vectors.bin` header. Keep the 256 KiB cap.

## 5. Benchmarks and the model-selection gate

Every classical ML capability ships with a lightweight CPU bench. Small public data. A few minutes, not a GPU suite. The bench writes a JSON file. **No number is checked in until that run exists.** Thresholds below are pass/fail rules, not measurements. Vendor GPU timings are not accepted as results.

The 100k run from the original plan is optional (`--full`) and is not required to close an issue.

### 5.1 Shared fixtures

| Bench | Data | Baseline | Metrics |
|---|---|---|---|
| Classifier | scikit-learn `fetch_20newsgroups`, categories `alt.atheism`, `comp.graphics`, `sci.med`, `soc.religion.christian`, `remove=('headers','footers','quotes')`, at most 400 documents per class, `train_test_split` `test_size=0.25`, `random_state=42` | `TfidfVectorizer(max_features=20000)` + `LogisticRegression`, and `NearestCentroid` | accuracy, macro-F1, wall time |
| Clustering | the same texts, `k=4` | `KMeans(n_init=10, random_state=42)` on L2-normalized TF-IDF | ARI, NMI, wall time |
| Topics | the same texts | `NMF(n_components=10, random_state=42)` on TF-IDF, top-10 terms; plus TF-IDF + `KMeans(k=10)` + c-TF-IDF as the lightweight BERTopic-style control. Full BERTopic is not required | document-level NPMI of the top-10 terms, topic diversity (unique terms / `10 * n_topics`), wall time |
| Recommender | GroupLens MovieLens 100K, official `u1.base` / `u1.test`. Item text is the title in `u.item` | popularity by training support; item-item cosine `NearestNeighbors` `k=20` | Recall@10, NDCG@10, wall time |
| Scoring | STS Benchmark **development** split (public labels, 1,500 pairs). Anomaly: hold out `sci.med`, equal samples of the other three categories | STS: TF-IDF cosine Spearman. Anomaly: `IsolationForest(random_state=42)` on TF-IDF | Spearman, AUROC, wall time |
| Q-learning | deterministic 5×5 grid in the bench. Start `(0,0)`, goal `(4,4)`, one blocked cell, step −1, goal +10, max 50 steps. No embedding model | tabular Q, `α=0.1`, `γ=0.95`, `ε=0.1`, seed 0, 200 train episodes | mean return over 20 greedy eval episodes; episodes until eval return is within 1.0 of tabular |
| Core search | synthetic: 10,000 L2-normalized float32 vectors, dim 128, seed 42, 100 queries. Ground truth is brute force | the same index, unfiltered vs filtered | recall@10, p50, p95 |

Each ML bench must finish in under 10 minutes on CPU after the model files are cached. The core bench must finish in under 5 minutes. The gridworld must finish in seconds. Download time is reported separately and is not part of the compute limit.

Core acceptance, from the run, not from this document:

- Unfiltered recall@10 and p50/p95.
- Allow-set at 10% of ids and at 1% of ids: recall@10 and p50/p95.
- Exact-scan branch versus brute force of the allow-set: recall 1.0 and scores within 1 ulp of the public score.
- `search_batch` of 100 queries versus 100 `search` calls: both wall times.
- `get_many` of 100 ids versus 100 `get` calls: both wall times.
- `min_score` above every hit returns zero rows.

### 5.2 Model-selection gate

Run once, on CPU, and write `bench/model-gate.json`. Candidates that have no ONNX file are skipped, not bridged through Python.

**Speed fixture.** 64 passages of about 128 tokens and 32 queries, one process.

- BreadBowl passage encode uses `batch_size=1` and float32. ONNX embedders use batch 32.
- Record p50, p95, and passages/sec for each embedder.
- **Speed fail:** BreadBowl passage-encode p50 is greater than 10× the fastest ONNX embedder’s p50 on that fixture.
- Rerank 100 candidate pairs. Record p50 for BreadBowl `model.score` and for each ONNX cross-encoder.
- **Rerank speed fail:** BreadBowl `model.score` p50 is greater than the fastest ONNX cross-encoder’s p50.

**Quality.** Run section 5.1 with each speed-eligible embedder. BreadBowl’s single-vector reduction, used only for classifier, clustering, topics, and STS, is the mean of the 16 L2-normalized routing slots, then L2-normalized again. Recommender and any retrieval number use the real 16-slot search plus `model.score`, not the mean. Record every metric. Do not invent one.

**Quality fail** if BreadBowl is worse than the best alternative by more than:

| Metric | Absolute margin |
|---|---|
| Classifier macro-F1 | 0.02 |
| Clustering ARI | 0.02 |
| Topics NPMI | 0.02 |
| Recommender NDCG@10 | 0.01 |
| STS-B Spearman | 0.02 |
| Anomaly AUROC | 0.02 |
| Reranked NDCG@10 vs best cross-encoder | 0.01, and the reranked list must be at least the unreranked list |

**Who becomes the default.**

1. Node always takes the ONNX embedder and the ONNX reranker. BreadBowl is not eligible on Node unless the ONNX match test in section 3.7 has passed. That test has not passed, because no ONNX export ships with v1.3.
2. Python takes BreadBowl as the embedder only when it is speed-eligible and inside every quality margin above. Otherwise Python takes the ONNX embedder with the best mean rank of macro-F1, ARI, and Spearman among models that finished the bench.
3. Python takes BreadBowl `model.score` as the reranker only when the rerank speed and NDCG rules pass. Otherwise the reranker is the ONNX cross-encoder with the higher NDCG@10, or none if both rerankers fail to beat the unreranked list.
4. A failed BreadBowl stays installable as an optional Python extra. It is not imported by the Node package.

The manifest records `runtime: "onnx"` or `runtime: "breadbowl"`, the Hub repo, the revision, and the sha256 copied from the downloaded manifest. The Node open path rejects `runtime: "breadbowl"`.

## 6. Versioned roadmap

Issue titles below are the filing list. Existing issues are linked and must not be duplicated. New issues get labels `core`, `nodejs`, `python`, `logos-ml`, `security`, or `breaking` as named, plus milestone.

### 1.7.0 — search primitives

**[#134](https://github.com/jose-compu/logosdb/issues/134) — id allow/deny bitmap plus exact scan for small allow-sets.**  
Pass a `BaseFilterFunctor` into `searchKnn`. Bitmap from allow minus deny. Exact scan when the allow-set is ≤ 4,096 ids or ≤ 10,000 KB of floats. Old `search` is unchanged.  
Benchmark: section 5.1 core rows. Milestone: 1.7.0. Labels: `core`, `api`.

**[#136](https://github.com/jose-compu/logosdb/issues/136) — `logosdb_search_batch` returns ids and scores only.**  
One lock around N const searches, optional internal threads. Caps `n` and `k` on this function. No text in the batch result.  
Benchmark: batch versus loop wall time on the 10,000 × 128 fixture. Milestone: 1.7.0.

**[#144](https://github.com/jose-compu/logosdb/issues/144) — `min_score` and `logosdb_get_many`.**  
Cut on `SearchHit.score` after the walk. `get_many` copies float32 embeddings under one lock.  
Benchmark: `get_many` versus N `get`; `min_score` above every hit returns zero rows. Milestone: 1.7.0.

### 1.8.0 — bindings

**[#137](https://github.com/jose-compu/logosdb/issues/137) — Promise `search` / `putBatch` / `searchBatch` beside the sync methods.**  
`AsyncWorker` copies buffers off the event loop. The C mutex still serializes one `DB`. No JS predicates. Milestone: 1.8.0. Label: `nodejs`.

**[#140](https://github.com/jose-compu/logosdb/issues/140) — `Float32Array` in and out by memcpy; keep `number[]`.**  
Do not return a view of the mmap. Milestone: 1.8.0. Label: `nodejs`.

**Python: release the GIL on `search`, `search_batch`, and `get_many`.**  
Copy inputs under the GIL, compute outside it, return NumPy arrays the binding owns. A second thread running pure Python during `search_batch` must make progress. Milestone: 1.8.0. Label: `python`.

**[#141](https://github.com/jose-compu/logosdb/issues/141) — `skip_empty_ts`, default off.**  
Empty timestamps still match a range unless the flag is set. Milestone: 1.8.0.

### 1.9.0 — bounds and fuzzing

**Security: cap `k`, `candidate_k`, and `ef_search` on the new search APIs.**  
4,096 / 100,000 / 4,096. Existing `logosdb_search` stays uncapped. Milestone: 1.9.0. Label: `security`.

**Security: fuzz NDJSON, `meta.jsonl` lines, filter structs, and a truncated storage header.**  
Extend the libFuzzer target past `nlohmann::json::parse`. Keep the 256 KiB cap. Milestone: 1.9.0. Label: `security`.

**API: optional `max_text` and optional exclusive `flock`, both default off.**  
Document that core paths are trusted and that two unlocked writers can corrupt the directory. Milestone: 1.9.0. Label: `security`.

**API: expose HNSW `random_seed`, default 100.**  
Do not change the constructor default. Milestone: 1.9.0. Label: `core`.

### 1.10.0 — experimental identity and payload

**[#143](https://github.com/jose-compu/logosdb/issues/143) — durable external-key sidecar `keys.bin`.**  
fsync’d log plus hash. `get_by_key` / `put_by_key`. Experimental until a crash test passes. Milestone: 1.10.0.

**API: `logosdb_replace` overwrites one id in place.**  
`logosdb_update` still returns a new id. `replace` does not consume `maxElements`. Milestone: 1.10.0. Label: `core`.

**Experimental: optional `payload.bin` of label, `y`, and tag bits.**  
Functor compares integers. Databases without the file behave as 1.5. Milestone: 1.10.0. Label: `experimental`.

### 2.0.0 — only these breaks

Ship 2.0.0 only if all four are accepted. Otherwise stop at 1.10.x.

**2.0.0: flip search defaults and make `update` keep the id.**  
`search` / `search_ts_range` reject `k` above the cap. Range search skips empty timestamps. `open` takes an exclusive lock. `update` is in-place and returns the same id. Migration: `logosdb_update_new_id` keeps today’s delete+put; `compact` rewrites a directory. No ML feature depends on this release. Milestone: 2.0.0. Label: `breaking`.

### `logos-ml` (this repository, label `logos-ml`)

**logos-ml 0.1.0: workspace, manifest, embedder, envelope, keymap, lock.**  
TypeScript ESM, Node ≥ 20, for the Node package. Node embedder is ONNX only. The model gate in section 5.2 writes the manifest the Node package will refuse if `runtime` is `breadbowl`. Workspace `flock`. JSON keymap until `keys.bin` exists. Six stores when the chosen model is single-vector; sixteen routing rows per passage only on the Python BreadBowl path.  
Benchmark: embed 64 sentences with the selected ONNX model, batch 32, record sentences/sec, confirm sha256. The gate JSON is the acceptance artifact. Milestone: `logos-ml 0.1.0`.

**logos-ml 0.2.0: classifier.**  
Zero-shot, nearest centroid, kNN vote, `unknown` on `min_score` and margin.  
Benchmark: section 5.1 classifier, plus the model-gate macro-F1 row. Milestone: `logos-ml 0.2.0`.

**logos-ml 0.3.0: clustering.**  
Spherical k-means with a seeded RNG. kNN graph via `searchBatch` once 1.7.0 is in. Online leader. Cluster ids in `clusters/<run>.assign.json`.  
Benchmark: section 5.1 clustering. Milestone: `logos-ml 0.3.0`.

**logos-ml 0.4.0: topics.**  
UMAP+HDBSCAN under the measured row cap; kNN graph above it. c-TF-IDF and phrase MMR in the library.  
Benchmark: section 5.1 topics. Milestone: `logos-ml 0.4.0`.

**logos-ml 0.5.0: recommender.**  
Decayed profile, deny-set of seen ids, user-user search, RRF, MMR. On Python BreadBowl, the deny-set expands to the 16 routing rows and `model.score` reranks only if the gate selected it.  
Benchmark: section 5.1 recommender, including the rerank NDCG row. Milestone: `logos-ml 0.5.0`.

**logos-ml 0.6.0: scoring.**  
Cosine, rubric, kNN regression, contrastive plus Platt, neighbor anomaly.  
Benchmark: section 5.1 scoring. Milestone: `logos-ml 0.6.0`.

**logos-ml 0.7.0: CLI and the lightweight bench.**  
`init`, `ingest`, `classify`, `cluster`, `topics`, `recommend`, `score`, `eval`. `eval` runs section 5.1 and writes JSON. Node publish depends only on ONNX. The 100k run is optional. Milestone: `logos-ml 0.7.0`.

**logos-ml 0.8.0: Q-learning agent.**  
Sidecar Q table, seeded ε-greedy and UCB, `sweep` via `searchBatch`.  
Benchmark: section 5.1 gridworld against tabular Q. The gridworld does not load BreadBowl or ONNX. Milestone: `logos-ml 0.8.0`.

**logos-ml 0.9.0 (Python): the same algorithms on NumPy.**  
Calls the 1.8.0 Python binding. Runs the model gate. BreadBowl is the preferred default only when section 5.2 says so; otherwise the ONNX winner is the default and BreadBowl is an extra. No pickle. No Node subprocess. `hdbscan` and `umap-learn` are extras.  
Benchmark: the same section 5.1 scripts, so Node and Python numbers are comparable, plus `bench/model-gate.json`. Milestone: `logos-ml 0.9.0 (Python)`.

[#124](https://github.com/jose-compu/logosdb/issues/124) stays out of this roadmap. Library and CLI come first.

## 7. Open questions only the owner can answer

1. Production languages. The benches are English. A multilingual corpus selects `multilingual-e5-small` as an ONNX candidate, not as an unmeasured default.
2. Corpus size and whether passages are long enough to hit BreadBowl’s 384-token cap. That sizes `maxElements` (×16 if Python selects BreadBowl).
3. Whether 2.0.0 should flip the four defaults in section 6.
4. Whether `logosdb_replace` and `keys.bin` should exist in core, or the library keymap is enough permanently.
5. Reward and action set for the first text Q agent. The gridworld acceptance test does not need that answer.
6. Whether a later BreadBowl ONNX export is in scope. Until the section 3.7 match test passes, Node does not wait on it.

## 8. Answers to the original plan’s questions

**MiniLM versus multilingual-e5.** Neither is the unmeasured default. The gate in section 5.2 picks the ONNX winner. MiniLM, bge-small, and e5-small are the candidates. English benches do not by themselves select e5.

**BreadBowl.** Preferred on Python only, and only after the gate. Not a Node backend. The published toolkit has no ONNX runtime, so the native-compatibility test is unmet.

**Corpus size.** Unknown. kNN-graph through `search_batch` is the large path. UMAP+HDBSCAN is the small path. The cutoff is whatever the bench records.

**Library and CLI, or HTTP.** Library and CLI. [#124](https://github.com/jose-compu/logosdb/issues/124) stays a later product decision.

**TypeScript or JS.** TypeScript for the Node library and the CLI, compiled to ESM.

**Native APIs first, or ship on workarounds.** Land 1.7.0 before `logos-ml` 0.3 (the kNN graph) and before Python BreadBowl retrieval (16 `search_batch` queries per passage). `logos-ml` 0.1 and 0.2 may ship on today’s synchronous `search`. The classifier does not need BreadBowl.

**First Q environment.** The 5×5 gridworld against tabular Q. MIND clicks are a later offline eval, not the acceptance bench.

## Sources

- Vendored hnswlib filter and batch search: `third_party/hnswlib/hnswlib/hnswalg.h`, [Python filter example](https://github.com/nmslib/hnswlib/blob/master/examples/python/example_filter.py)
- [USearch filtered search](https://www.unum.cloud/docs/usearch/python/filtered-search.md), [USearch](https://unum-cloud.github.io/USearch/)
- [Qdrant filtering](https://qdrant.tech/documentation/search-design/vector-search-filtering/), [Qdrant explore](https://qdrant.tech/documentation/search/explore/)
- [LanceDB vector search](https://docs.lancedb.com/oss/v0-41-0/search/vector-search), [Milvus filtered search](https://milvus.io/docs/filtered-search.md)
- [FAISS search parameters](https://github.com/facebookresearch/faiss/wiki/Setting-search-parameters-for-one-query), [FAISS k-means](https://github.com/facebookresearch/faiss/wiki/K%E2%80%90means-clustering)
- [scikit-learn k-means](https://scikit-learn.org/stable/modules/clustering.html#k-means), [20 newsgroups loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html)
- [BERTopic algorithm](https://maartengr.github.io/BERTopic/algorithm/algorithm.html), [hdbscan#403](https://github.com/scikit-learn-contrib/hdbscan/issues/403)
- [transformers.js](https://github.com/huggingface/transformers.js), [node-addon-api AsyncWorker](https://github.com/nodejs/node-addon-api/blob/main/doc/async_worker.md)
- [BreadBowl-Embed README](https://github.com/BreadBowlAI/breadbowl-embed), [v1.3 model card](https://huggingface.co/dotproductx/breadbowl-embed-v1.3), [v1.3 post](https://breadbowl.ai/blog/breadbowl-embed-v1-3/)
- [sqlite-vec#277](https://github.com/asg017/sqlite-vec/pull/277)
- MovieLens 100K: [GroupLens](https://grouplens.org/datasets/movielens/100k/)
- STS Benchmark development labels: [SemEval-2017 STS](https://aclanthology.org/S17-2001/)
- [Python pickle](https://docs.python.org/3/library/pickle.html)
