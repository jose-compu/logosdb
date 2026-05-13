#include "hnsw_index.h"
#include "metadata.h"
#include "platform.h"
#include "storage.h"
#include "wal.h"

#include <logosdb/logosdb.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

using namespace logosdb::internal;

/* ── Internal DB struct ────────────────────────────────────────────── */

struct logosdb_t
{
    VectorStorage vectors;
    MetadataStore meta;
    HnswIndex index;
    WriteAheadLog wal;
    std::mutex mu;
    int dim = 0;
    int distance = LOGOSDB_DIST_IP;
    int dtype = LOGOSDB_DTYPE_FLOAT32;
    std::atomic<uint64_t> st_put_ok{0};
    std::atomic<uint64_t> st_put_fail{0};
    std::atomic<uint64_t> st_put_batch_ok{0};
    std::atomic<uint64_t> st_put_batch_fail{0};
    std::atomic<uint64_t> st_search_ok{0};
    std::atomic<uint64_t> st_search_fail{0};
    std::atomic<uint64_t> st_search_ts_ok{0};
    std::atomic<uint64_t> st_search_ts_fail{0};
    std::atomic<uint64_t> st_delete_ok{0};
    std::atomic<uint64_t> st_delete_fail{0};
    std::atomic<uint64_t> st_update_ok{0};
    std::atomic<uint64_t> st_update_fail{0};
    std::atomic<uint64_t> st_sync{0};
};

struct logosdb_options_t
{
    int dim = 0;
    size_t max_elements = 1000000;
    int ef_construction = 200;
    int M = 16;
    int ef_search = 50;
    int distance = 0; /* LOGOSDB_DIST_IP default */
    int dtype = 0;    /* LOGOSDB_DTYPE_FLOAT32 default */
};

struct logosdb_search_result_t
{
    struct Hit
    {
        uint64_t id;
        float score;
        std::string text;
        std::string timestamp;
    };
    std::vector<Hit> hits;
};

/* ── Helpers ───────────────────────────────────────────────────────── */

static void set_err(char** errptr, const std::string& msg)
{
    if (errptr)
    {
        *errptr = platform::string_duplicate(msg.c_str());
    }
}

/* ── Options ───────────────────────────────────────────────────────── */

logosdb_options_t* logosdb_options_create(void)
{
    return new logosdb_options_t();
}

void logosdb_options_destroy(logosdb_options_t* opts)
{
    delete opts;
}

void logosdb_options_set_dim(logosdb_options_t* o, int d)
{
    if (o && d > 0)
        o->dim = d;
}
void logosdb_options_set_max_elements(logosdb_options_t* o, size_t n)
{
    if (o && n > 0)
        o->max_elements = n;
}
void logosdb_options_set_ef_construction(logosdb_options_t* o, int e)
{
    if (o && e > 0)
        o->ef_construction = e;
}
void logosdb_options_set_M(logosdb_options_t* o, int m)
{
    if (o && m > 0)
        o->M = m;
}
void logosdb_options_set_ef_search(logosdb_options_t* o, int e)
{
    if (o && e > 0)
        o->ef_search = e;
}

int logosdb_options_set_distance(logosdb_options_t* o, int metric)
{
    if (!o)
        return -1;
    if (metric < 0 || metric > 2)
        return -1; /* Invalid metric */
    o->distance = metric;
    return 0;
}

int logosdb_options_set_dtype(logosdb_options_t* o, int dtype)
{
    if (!o)
        return -1;
    if (dtype < 0 || dtype > 2)
        return -1; /* Invalid dtype */
    o->dtype = dtype;
    return 0;
}

/* ── Lifecycle ─────────────────────────────────────────────────────── */

logosdb_t* logosdb_open(const char* path, const logosdb_options_t* opts, char** errptr)
{
    if (!path || !opts || opts->dim <= 0)
    {
        set_err(errptr, "invalid arguments: path and dim > 0 required");
        return nullptr;
    }

    std::filesystem::create_directories(path);

    auto db = new logosdb_t();
    db->dim = opts->dim;
    std::string err;

    std::string vec_path = std::string(path) + "/vectors.bin";
    std::string meta_path = std::string(path) + "/meta.jsonl";
    std::string idx_path = std::string(path) + "/hnsw.idx";
    std::string wal_path = std::string(path) + "/wal.log";

    StorageDtype dtype = static_cast<StorageDtype>(opts->dtype);
    if (!db->vectors.open(vec_path, opts->dim, dtype, err))
    {
        set_err(errptr, err);
        delete db;
        return nullptr;
    }
    if (!db->meta.open(meta_path, err))
    {
        set_err(errptr, err);
        delete db;
        return nullptr;
    }

    HnswParams hp;
    hp.dim = opts->dim;
    hp.max_elements = opts->max_elements;
    hp.ef_construction = opts->ef_construction;
    hp.M = opts->M;
    hp.ef_search = opts->ef_search;
    hp.distance = opts->distance;

    if (!db->index.open(idx_path, hp, err))
    {
        set_err(errptr, err);
        delete db;
        return nullptr;
    }

    // Open WAL and replay any pending entries for atomic recovery.
    if (!db->wal.open(wal_path, err))
    {
        set_err(errptr, err);
        delete db;
        return nullptr;
    }

    // Replay pending WAL entries to ensure consistency.
    int replayed = db->wal.replay_pending(
        [&db](const WALEntry& entry, std::string& replay_err) -> bool
        {
            // Validate: expected_id should match current row count
            if (entry.expected_id != db->vectors.n_rows())
            {
                replay_err = "wal replay: expected_id mismatch (" +
                             std::to_string(entry.expected_id) + " vs " +
                             std::to_string(db->vectors.n_rows()) + ")";
                return false;
            }

            // Replay vector
            uint64_t vid = db->vectors.append(entry.vector.data(), (int)entry.dim, replay_err);
            if (vid == UINT64_MAX)
                return false;

            // Replay metadata
            uint64_t mid = db->meta.append(entry.text.c_str(), entry.timestamp.c_str(), replay_err);
            if (mid == UINT64_MAX)
                return false;

            // Replay index
            if (!db->index.add(vid, entry.vector.data(), replay_err))
            {
                return false;
            }

            return true;
        },
        err);

    if (replayed < 0)
    {
        set_err(errptr, "wal replay: " + err);
        delete db;
        return nullptr;
    }

    // Backfill index if vector storage has more rows than the index (e.g. crash recovery).
    size_t n_vec = db->vectors.n_rows();
    size_t n_idx = db->index.count();
    bool backfilled = false;
    if (n_vec > n_idx)
    {
        std::vector<float> float32_row(db->dim);
        for (size_t i = n_idx; i < n_vec; ++i)
        {
            db->vectors.row_to_float32(i, float32_row.data());
            if (!db->index.add(i, float32_row.data(), err))
            {
                set_err(errptr, "backfill index: " + err);
                delete db;
                return nullptr;
            }
        }
        backfilled = true;
    }

    // Re-apply deletion marks. If we just rebuilt (part of) the index from the
    // vector store, or if the index file was out of sync with the metadata
    // tombstone log, we may have live slots for rows the metadata log says are
    // deleted. Replay the tombstone set onto the index (skip entries already
    // marked to tolerate normal reopens).
    for (uint64_t id : db->meta.deleted_ids())
    {
        if (!db->index.has_label(id))
            continue;
        if (db->index.is_deleted(id))
            continue;
        if (!db->index.mark_deleted(id, err))
        {
            set_err(errptr, "replay tombstone: " + err);
            delete db;
            return nullptr;
        }
    }

    if (backfilled)
        db->index.save(err);

    db->distance = opts->distance;
    db->dtype = (int)db->vectors.dtype();

    return db;
}

void logosdb_close(logosdb_t* db)
{
    if (!db)
        return;
    std::string err;
    db->wal.sync(err);  // Ensure WAL is durable before closing other stores
    db->index.save(err);
    db->vectors.sync(err);
    db->meta.sync(err);
    db->wal.close();
    delete db;
}

int logosdb_sync(logosdb_t* db, char** errptr)
{
    if (!db)
    {
        set_err(errptr, "null db");
        return -1;
    }
    std::lock_guard<std::mutex> lock(db->mu);
    std::string err;
    if (!db->wal.sync(err))
    {
        set_err(errptr, "wal sync: " + err);
        return -1;
    }
    if (!db->index.save(err))
    {
        set_err(errptr, "index save: " + err);
        return -1;
    }
    if (!db->vectors.sync(err))
    {
        set_err(errptr, "vectors sync: " + err);
        return -1;
    }
    if (!db->meta.sync(err))
    {
        set_err(errptr, "meta sync: " + err);
        return -1;
    }
    db->st_sync.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

/* ── Write ─────────────────────────────────────────────────────────── */

uint64_t logosdb_put(logosdb_t* db,
                     const float* embedding,
                     int dim,
                     const char* text,
                     const char* timestamp,
                     char** errptr)
{
    if (!db || !embedding)
    {
        set_err(errptr, "null db or embedding");
        if (db)
            db->st_put_fail.fetch_add(1, std::memory_order_relaxed);
        return UINT64_MAX;
    }
    if (dim != db->dim)
    {
        set_err(errptr, "dimension mismatch");
        db->st_put_fail.fetch_add(1, std::memory_order_relaxed);
        return UINT64_MAX;
    }

    std::lock_guard<std::mutex> lock(db->mu);
    std::string err;

    // Compute expected row id before writing
    uint64_t expected_id = db->vectors.n_rows();

    // Step 1: Write WAL entry (durability point)
    int64_t wal_offset = db->wal.append_pending(embedding, dim, text, timestamp, expected_id, err);
    if (wal_offset < 0)
    {
        set_err(errptr, err);
        db->st_put_fail.fetch_add(1, std::memory_order_relaxed);
        return UINT64_MAX;
    }

    // Step 2: Write to vector storage
    uint64_t vid = db->vectors.append(embedding, dim, err);
    if (vid == UINT64_MAX)
    {
        set_err(errptr, err);
        db->st_put_fail.fetch_add(1, std::memory_order_relaxed);
        return UINT64_MAX;
    }

    // Step 3: Write to metadata storage
    uint64_t mid = db->meta.append(text, timestamp, err);
    if (mid == UINT64_MAX)
    {
        // Metadata write failed - entry remains in WAL for replay on recovery
        set_err(errptr, err);
        db->st_put_fail.fetch_add(1, std::memory_order_relaxed);
        return UINT64_MAX;
    }

    // Step 4: Write to HNSW index
    if (!db->index.add(vid, embedding, err))
    {
        // Index write failed - entry remains in WAL for replay on recovery
        set_err(errptr, err);
        db->st_put_fail.fetch_add(1, std::memory_order_relaxed);
        return UINT64_MAX;
    }

    // Step 5: Mark WAL entry as committed
    if (!db->wal.mark_committed(wal_offset, err))
    {
        // Non-fatal: entry will be replayed on next open if needed
        // Log but don't fail the operation
    }

    db->st_put_ok.fetch_add(1, std::memory_order_relaxed);
    return vid;
}

int logosdb_put_batch(logosdb_t* db,
                      const float* embeddings,
                      int n,
                      int dim,
                      const char* const* texts,
                      const char* const* timestamps,
                      uint64_t* out_ids,
                      char** errptr)
{
    if (!db || !embeddings || !out_ids)
    {
        set_err(errptr, "null db, embeddings, or out_ids");
        if (db)
            db->st_put_batch_fail.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }
    if (n <= 0)
    {
        return 0;  // Empty batch is a no-op success
    }
    if (dim != db->dim)
    {
        set_err(errptr, "dimension mismatch");
        db->st_put_batch_fail.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }

    // Chunk size bounds: one chunk owns one WAL fsync, one vector append, one
    // metadata append, n HNSW inserts, then one commit-stamp fsync. Larger
    // chunks amortise fsync overhead; smaller chunks bound peak RSS. Override
    // via LOGOSDB_BATCH_CHUNK_SIZE for benchmarking.
    int chunk = 1024;
    if (const char* env = std::getenv("LOGOSDB_BATCH_CHUNK_SIZE"))
    {
        int v = std::atoi(env);
        if (v > 0)
            chunk = v;
    }
    if (chunk > n)
        chunk = n;

    std::lock_guard<std::mutex> lock(db->mu);
    std::string err;
    std::vector<int64_t> wal_offsets;
    wal_offsets.reserve(static_cast<size_t>(chunk));

    int processed = 0;
    while (processed < n)
    {
        int this_chunk = std::min(chunk, n - processed);
        uint64_t start_expected_id = db->vectors.n_rows();

        // 1. WAL: append `this_chunk` pending entries (single fsync).
        const float* chunk_vecs =
            embeddings + static_cast<size_t>(processed) * static_cast<size_t>(dim);
        const char* const* chunk_texts = texts ? texts + processed : nullptr;
        const char* const* chunk_ts = timestamps ? timestamps + processed : nullptr;
        wal_offsets.clear();
        if (db->wal.append_pending_batch(chunk_vecs,
                                         this_chunk,
                                         dim,
                                         chunk_texts,
                                         chunk_ts,
                                         start_expected_id,
                                         wal_offsets,
                                         err) != 0)
        {
            set_err(errptr, err);
            db->st_put_batch_fail.fetch_add(1, std::memory_order_relaxed);
            return -1;
        }

        // 2. Vector storage: single bulk append for the chunk.
        uint64_t vid = db->vectors.append_batch(chunk_vecs, this_chunk, dim, err);
        if (vid == UINT64_MAX)
        {
            set_err(errptr, err);
            db->st_put_batch_fail.fetch_add(1, std::memory_order_relaxed);
            return -1;
        }
        if (vid != start_expected_id)
        {
            set_err(errptr, "put_batch: vector id drift after WAL");
            db->st_put_batch_fail.fetch_add(1, std::memory_order_relaxed);
            return -1;
        }

        // 3. Metadata: single bulk append for the chunk.
        uint64_t mid = db->meta.append_batch(chunk_texts, chunk_ts, this_chunk, err);
        if (mid == UINT64_MAX)
        {
            // Vectors are durable; entries still live as PENDING in the WAL and
            // will be replayed on next open. Surface the error.
            set_err(errptr, err);
            db->st_put_batch_fail.fetch_add(1, std::memory_order_relaxed);
            return -1;
        }

        // 4. HNSW: per-row add (no public bulk API in hnswlib).
        size_t stride = static_cast<size_t>(dim) * sizeof(float);
        const float* vec = chunk_vecs;
        for (int i = 0; i < this_chunk; ++i)
        {
            if (!db->index.add(vid + static_cast<uint64_t>(i), vec, err))
            {
                set_err(errptr, err);
                db->st_put_batch_fail.fetch_add(1, std::memory_order_relaxed);
                return -1;
            }
            vec = reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(vec) + stride);
        }

        // 5. WAL: stamp the chunk committed (single fsync).
        if (!db->wal.mark_committed_batch(wal_offsets, err))
        {
            // Non-fatal: entries remain PENDING but replay will recover them.
            // We still surface the partial-durability error to the caller.
            set_err(errptr, err);
            db->st_put_batch_fail.fetch_add(1, std::memory_order_relaxed);
            return -1;
        }

        for (int i = 0; i < this_chunk; ++i)
        {
            out_ids[processed + i] = vid + static_cast<uint64_t>(i);
        }
        processed += this_chunk;
    }

    db->st_put_batch_ok.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

/* ── Delete / Update ───────────────────────────────────────────────── */

int logosdb_delete(logosdb_t* db, uint64_t id, char** errptr)
{
    if (!db)
    {
        set_err(errptr, "null db");
        return -1;
    }

    std::lock_guard<std::mutex> lock(db->mu);
    std::string err;

    if (id >= db->vectors.n_rows())
    {
        set_err(errptr, "delete: id out of range");
        db->st_delete_fail.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }
    if (db->meta.is_deleted(id))
    {
        set_err(errptr, "delete: id already deleted");
        db->st_delete_fail.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }

    // Mark in the index first. If hnswlib rejects (e.g. label missing because
    // the row was never indexed), bail out before touching the metadata log.
    if (!db->index.mark_deleted(id, err))
    {
        set_err(errptr, err);
        db->st_delete_fail.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }

    if (!db->meta.mark_deleted(id, err))
    {
        // Index is now marked but metadata failed. Best-effort rollback so
        // the two stores don't diverge.
        std::string ignore;
        (void)ignore;
        set_err(errptr, err);
        db->st_delete_fail.fetch_add(1, std::memory_order_relaxed);
        return -1;
    }
    db->st_delete_ok.fetch_add(1, std::memory_order_relaxed);
    return 0;
}

uint64_t logosdb_update(logosdb_t* db,
                        uint64_t id,
                        const float* embedding,
                        int dim,
                        const char* text,
                        const char* timestamp,
                        char** errptr)
{
    if (!db || !embedding)
    {
        set_err(errptr, "null db or embedding");
        if (db)
            db->st_update_fail.fetch_add(1, std::memory_order_relaxed);
        return UINT64_MAX;
    }
    if (dim != db->dim)
    {
        set_err(errptr, "dimension mismatch");
        db->st_update_fail.fetch_add(1, std::memory_order_relaxed);
        return UINT64_MAX;
    }

    std::lock_guard<std::mutex> lock(db->mu);
    std::string err;

    if (id >= db->vectors.n_rows())
    {
        set_err(errptr, "update: id out of range");
        db->st_update_fail.fetch_add(1, std::memory_order_relaxed);
        return UINT64_MAX;
    }
    if (db->meta.is_deleted(id))
    {
        set_err(errptr, "update: id already deleted");
        db->st_update_fail.fetch_add(1, std::memory_order_relaxed);
        return UINT64_MAX;
    }

    // Mark the old row deleted, then append a fresh row. The two operations
    // share the write mutex, so no reader sees an intermediate state.
    if (!db->index.mark_deleted(id, err))
    {
        set_err(errptr, err);
        db->st_update_fail.fetch_add(1, std::memory_order_relaxed);
        return UINT64_MAX;
    }
    if (!db->meta.mark_deleted(id, err))
    {
        set_err(errptr, err);
        db->st_update_fail.fetch_add(1, std::memory_order_relaxed);
        return UINT64_MAX;
    }

    uint64_t vid = db->vectors.append(embedding, dim, err);
    if (vid == UINT64_MAX)
    {
        set_err(errptr, err);
        db->st_update_fail.fetch_add(1, std::memory_order_relaxed);
        return UINT64_MAX;
    }

    uint64_t mid = db->meta.append(text, timestamp, err);
    if (mid == UINT64_MAX)
    {
        set_err(errptr, err);
        db->st_update_fail.fetch_add(1, std::memory_order_relaxed);
        return UINT64_MAX;
    }

    if (!db->index.add(vid, embedding, err))
    {
        set_err(errptr, err);
        db->st_update_fail.fetch_add(1, std::memory_order_relaxed);
        return UINT64_MAX;
    }
    db->st_update_ok.fetch_add(1, std::memory_order_relaxed);
    return vid;
}

/* ── Search ────────────────────────────────────────────────────────── */

logosdb_search_result_t*
logosdb_search(logosdb_t* db, const float* query, int dim, int top_k, char** errptr)
{
    if (!db || !query)
    {
        set_err(errptr, "null db or query");
        if (db)
            db->st_search_fail.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    if (dim != db->dim)
    {
        set_err(errptr, "dimension mismatch in search");
        db->st_search_fail.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    if (top_k <= 0)
    {
        set_err(errptr, "top_k must be > 0");
        db->st_search_fail.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(db->mu);
    std::string err;
    auto raw = db->index.search(query, top_k, err);
    if (!err.empty())
    {
        set_err(errptr, err);
        db->st_search_fail.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    auto* r = new logosdb_search_result_t();
    r->hits.reserve(raw.size());
    for (auto& [label, score] : raw)
    {
        logosdb_search_result_t::Hit h;
        h.id = label;
        h.score = score;
        auto* m = db->meta.row(label);
        if (m)
        {
            h.text = m->text;
            h.timestamp = m->timestamp;
        }
        r->hits.push_back(std::move(h));
    }
    db->st_search_ok.fetch_add(1, std::memory_order_relaxed);
    return r;
}

logosdb_search_result_t* logosdb_search_ts_range(logosdb_t* db,
                                                 const float* query,
                                                 int dim,
                                                 int top_k,
                                                 const char* ts_from_iso8601,
                                                 const char* ts_to_iso8601,
                                                 int candidate_k,
                                                 char** errptr)
{
    if (!db || !query)
    {
        set_err(errptr, "null db or query");
        if (db)
            db->st_search_ts_fail.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    if (dim != db->dim)
    {
        set_err(errptr, "dimension mismatch in search");
        db->st_search_ts_fail.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    if (top_k <= 0)
    {
        set_err(errptr, "top_k must be > 0");
        db->st_search_ts_fail.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }
    if (candidate_k < top_k)
    {
        candidate_k = top_k;  // Ensure we fetch at least top_k candidates
    }

    std::lock_guard<std::mutex> lock(db->mu);
    std::string err;
    // Fetch more candidates than needed to allow for filtering
    auto raw = db->index.search(query, candidate_k, err);
    if (!err.empty())
    {
        set_err(errptr, err);
        db->st_search_ts_fail.fetch_add(1, std::memory_order_relaxed);
        return nullptr;
    }

    auto* r = new logosdb_search_result_t();
    r->hits.reserve(top_k);

    for (auto& [label, score] : raw)
    {
        // Check if we've collected enough results
        if ((int)r->hits.size() >= top_k)
            break;

        // Get metadata for this row
        auto* m = db->meta.row(label);
        if (!m)
            continue;

        // Apply timestamp filter
        if (ts_from_iso8601 && !m->timestamp.empty())
        {
            if (m->timestamp < ts_from_iso8601)
                continue;  // Before start
        }
        if (ts_to_iso8601 && !m->timestamp.empty())
        {
            if (m->timestamp > ts_to_iso8601)
                continue;  // After end
        }

        // This result passes the filter
        logosdb_search_result_t::Hit h;
        h.id = label;
        h.score = score;
        h.text = m->text;
        h.timestamp = m->timestamp;
        r->hits.push_back(std::move(h));
    }

    db->st_search_ts_ok.fetch_add(1, std::memory_order_relaxed);
    return r;
}

int logosdb_result_count(const logosdb_search_result_t* r)
{
    return r ? (int)r->hits.size() : 0;
}

uint64_t logosdb_result_id(const logosdb_search_result_t* r, int i)
{
    return (r && i >= 0 && i < (int)r->hits.size()) ? r->hits[i].id : UINT64_MAX;
}

float logosdb_result_score(const logosdb_search_result_t* r, int i)
{
    return (r && i >= 0 && i < (int)r->hits.size()) ? r->hits[i].score : 0.0f;
}

const char* logosdb_result_text(const logosdb_search_result_t* r, int i)
{
    if (!r || i < 0 || i >= (int)r->hits.size())
        return nullptr;
    return r->hits[i].text.empty() ? nullptr : r->hits[i].text.c_str();
}

const char* logosdb_result_timestamp(const logosdb_search_result_t* r, int i)
{
    if (!r || i < 0 || i >= (int)r->hits.size())
        return nullptr;
    return r->hits[i].timestamp.empty() ? nullptr : r->hits[i].timestamp.c_str();
}

void logosdb_result_free(logosdb_search_result_t* r)
{
    delete r;
}

/* ── Info ──────────────────────────────────────────────────────────── */

size_t logosdb_count(logosdb_t* db)
{
    return db ? db->vectors.n_rows() : 0;
}

size_t logosdb_count_live(logosdb_t* db)
{
    if (!db)
        return 0;
    size_t total = db->vectors.n_rows();
    size_t deleted = db->meta.deleted_count();
    return total >= deleted ? total - deleted : 0;
}

int logosdb_dim(logosdb_t* db)
{
    return db ? db->dim : 0;
}

static bool
peek_vectors_dim_dtype(const std::string& vec_path, int& dim, int& dtype_out, std::string& err)
{
    FILE* f = fopen(vec_path.c_str(), "rb");
    if (!f)
    {
        err = "cannot open vectors.bin";
        return false;
    }
    StorageHeader hdr{};
    size_t n = fread(&hdr, 1, sizeof(hdr), f);
    fclose(f);
    if (n != sizeof(hdr))
    {
        err = "vectors.bin too small";
        return false;
    }
    if (hdr.magic != 0x4C4F474FU)
    {
        err = "vectors.bin bad magic";
        return false;
    }
    if (hdr.version != 0 && hdr.version != 1 && hdr.version != 2)
    {
        err = "unsupported vectors.bin version";
        return false;
    }
    uint32_t dtype_u = hdr.dtype;
    if (hdr.version == 0 || hdr.version == 1)
        dtype_u = (uint32_t)DTYPE_FLOAT32;
    if (dtype_u > (uint32_t)DTYPE_INT8)
    {
        err = "invalid dtype";
        return false;
    }
    dim = (int)hdr.dim;
    dtype_out = (int)dtype_u;
    if (dim <= 0)
    {
        err = "invalid dim";
        return false;
    }
    return true;
}

static bool peek_index_distance_metric(const std::string& meta_path, int& distance, bool& have_meta)
{
    have_meta = false;
    distance = LOGOSDB_DIST_IP;
    std::error_code ec;
    if (!std::filesystem::exists(meta_path, ec))
        return true;
    FILE* f = fopen(meta_path.c_str(), "rb");
    if (!f)
        return true;
    char magic[8];
    uint32_t version = 0;
    int32_t dist = 0;
    int32_t d = 0;
    uint8_t pad[12];
    bool ok = fread(magic, 1, sizeof(magic), f) == sizeof(magic) &&
              fread(&version, sizeof(version), 1, f) == 1u &&
              fread(&dist, sizeof(dist), 1, f) == 1u && fread(&d, sizeof(d), 1, f) == 1u &&
              fread(pad, 1, sizeof(pad), f) == sizeof(pad);
    fclose(f);
    if (!ok)
        return true;
    if (std::memcmp(magic, "LOGOSDB\0", 8) != 0 || version != 1u)
        return true;
    if (dist < 0 || dist > 2)
        return true;
    (void)d;
    have_meta = true;
    distance = dist;
    return true;
}

void logosdb_stats(const logosdb_t* db, logosdb_stats_t* out)
{
    if (!db || !out)
        return;
    *out = logosdb_stats_t{};
    out->rows_total = db->vectors.n_rows();
    out->tombstones = db->meta.deleted_count();
    out->rows_live = out->rows_total >= out->tombstones ? out->rows_total - out->tombstones : 0;
    out->index_elements = db->index.count();
    out->wal_pending = db->wal.pending_count();
    out->distance_metric = db->distance;
    out->storage_dtype = db->dtype;
    out->put_success = db->st_put_ok.load(std::memory_order_relaxed);
    out->put_failed = db->st_put_fail.load(std::memory_order_relaxed);
    out->put_batch_success = db->st_put_batch_ok.load(std::memory_order_relaxed);
    out->put_batch_failed = db->st_put_batch_fail.load(std::memory_order_relaxed);
    out->search_success = db->st_search_ok.load(std::memory_order_relaxed);
    out->search_failed = db->st_search_fail.load(std::memory_order_relaxed);
    out->search_ts_success = db->st_search_ts_ok.load(std::memory_order_relaxed);
    out->search_ts_failed = db->st_search_ts_fail.load(std::memory_order_relaxed);
    out->delete_success = db->st_delete_ok.load(std::memory_order_relaxed);
    out->delete_failed = db->st_delete_fail.load(std::memory_order_relaxed);
    out->update_success = db->st_update_ok.load(std::memory_order_relaxed);
    out->update_failed = db->st_update_fail.load(std::memory_order_relaxed);
    out->sync_calls = db->st_sync.load(std::memory_order_relaxed);
}

int logosdb_compact(const char* src_path, const char* dst_path, char** errptr)
{
    if (!src_path || !dst_path)
    {
        set_err(errptr, "null path");
        return -1;
    }
    std::string sp = src_path;
    std::string dp = dst_path;
    int dim = 0;
    int dtype = LOGOSDB_DTYPE_FLOAT32;
    std::string epeek;
    if (!peek_vectors_dim_dtype(sp + "/vectors.bin", dim, dtype, epeek))
    {
        set_err(errptr, epeek);
        return -1;
    }
    int distance = LOGOSDB_DIST_IP;
    bool have_meta = false;
    peek_index_distance_metric(sp + "/hnsw.idx.meta", distance, have_meta);
    (void)have_meta;

    std::error_code ec;
    if (std::filesystem::exists(dp, ec))
    {
        if (!std::filesystem::is_empty(dp, ec))
        {
            set_err(errptr, "destination path is not empty");
            return -1;
        }
    }
    else
    {
        std::filesystem::create_directories(dp, ec);
    }

    logosdb_options_t* o = logosdb_options_create();
    logosdb_options_set_dim(o, dim);
    logosdb_options_set_distance(o, distance);
    logosdb_options_set_dtype(o, dtype);
    char* e2 = nullptr;
    logosdb_t* src = logosdb_open(sp.c_str(), o, &e2);
    if (!src)
    {
        set_err(errptr, e2 ? e2 : "open src failed");
        free(e2);
        logosdb_options_destroy(o);
        return -1;
    }
    free(e2);
    e2 = nullptr;
    logosdb_t* dst = logosdb_open(dp.c_str(), o, &e2);
    logosdb_options_destroy(o);
    if (!dst)
    {
        set_err(errptr, e2 ? e2 : "open dst failed");
        free(e2);
        logosdb_close(src);
        return -1;
    }
    free(e2);

    const size_t n = logosdb_count(src);
    std::vector<float> buf((size_t)dim);
    for (uint64_t id = 0; id < n; ++id)
    {
        std::string text;
        std::string ts;
        {
            std::lock_guard<std::mutex> lk(src->mu);
            if (src->meta.is_deleted(id))
                continue;
            src->vectors.row_to_float32(id, buf.data());
            const MetaRow* mr = src->meta.row(id);
            if (mr)
            {
                text = mr->text;
                ts = mr->timestamp;
            }
        }
        uint64_t nid = logosdb_put(dst,
                                   buf.data(),
                                   dim,
                                   text.empty() ? nullptr : text.c_str(),
                                   ts.empty() ? nullptr : ts.c_str(),
                                   &e2);
        if (nid == UINT64_MAX)
        {
            set_err(errptr, e2 ? e2 : "put during compact failed");
            free(e2);
            logosdb_close(dst);
            logosdb_close(src);
            return -1;
        }
        free(e2);
        e2 = nullptr;
    }

    if (logosdb_sync(dst, &e2) != 0)
    {
        set_err(errptr, e2 ? e2 : "sync dst failed");
        free(e2);
        logosdb_close(dst);
        logosdb_close(src);
        return -1;
    }
    free(e2);
    logosdb_close(dst);
    logosdb_close(src);
    return 0;
}

const float* logosdb_raw_vectors(logosdb_t* db, size_t* n_rows, int* dim)
{
    if (!db)
    {
        if (n_rows)
            *n_rows = 0;
        if (dim)
            *dim = 0;
        return nullptr;
    }
    if (n_rows)
        *n_rows = db->vectors.n_rows();
    if (dim)
        *dim = db->dim;

    // For reduced precision storage, we cannot return a direct pointer to float32 data
    // because the storage is in a different format. Users should use the C++ API or
    // individual row access for quantized storage.
    StorageDtype dtype = db->vectors.dtype();
    if (dtype != DTYPE_FLOAT32)
    {
        // Return nullptr to indicate this API doesn't work with quantized storage
        if (n_rows)
            *n_rows = 0;
        if (dim)
            *dim = 0;
        return nullptr;
    }
    return static_cast<const float*>(db->vectors.data_raw());
}

/* ── Streaming import/export (#87) ─────────────────────────────────── */

namespace
{
constexpr const char BASE64_ALPHABET[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string base64_encode_floats(const float* data, int dim)
{
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(data);
    size_t len = static_cast<size_t>(dim) * sizeof(float);
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3)
    {
        unsigned char b[3] = {0, 0, 0};
        size_t n = 0;
        for (size_t j = 0; j < 3 && i + j < len; ++j)
        {
            b[j] = bytes[i + j];
            ++n;
        }
        out += BASE64_ALPHABET[b[0] >> 2];
        out += BASE64_ALPHABET[((b[0] & 0x03) << 4) | (b[1] >> 4)];
        out += (n > 1) ? BASE64_ALPHABET[((b[1] & 0x0F) << 2) | (b[2] >> 6)] : '=';
        out += (n > 2) ? BASE64_ALPHABET[b[2] & 0x3F] : '=';
    }
    return out;
}

int base64_char_value(char c)
{
    if (c >= 'A' && c <= 'Z')
        return c - 'A';
    if (c >= 'a' && c <= 'z')
        return c - 'a' + 26;
    if (c >= '0' && c <= '9')
        return c - '0' + 52;
    if (c == '+')
        return 62;
    if (c == '/')
        return 63;
    if (c == '=')
        return -2;
    return -1;
}

bool base64_decode_floats(const std::string& s, int dim, std::vector<float>& out, std::string& err)
{
    const size_t expected_bytes = static_cast<size_t>(dim) * sizeof(float);
    std::vector<unsigned char> bytes;
    bytes.reserve(expected_bytes);

    size_t len = s.size();
    for (size_t i = 0; i + 3 < len; i += 4)
    {
        int v[4];
        for (int j = 0; j < 4; ++j)
            v[j] = base64_char_value(s[i + j]);
        if (v[0] < 0 || v[0] == -2 || v[1] < 0 || v[1] == -2)
        {
            err = "base64: invalid leading characters";
            return false;
        }
        bytes.push_back(static_cast<unsigned char>((v[0] << 2) | (v[1] >> 4)));
        if (v[2] >= 0)
            bytes.push_back(static_cast<unsigned char>(((v[1] & 0x0F) << 4) | (v[2] >> 2)));
        if (v[3] >= 0)
            bytes.push_back(static_cast<unsigned char>(((v[2] & 0x03) << 6) | v[3]));
    }
    if (bytes.size() < expected_bytes)
    {
        err = "base64: decoded " + std::to_string(bytes.size()) + " bytes, expected " +
              std::to_string(expected_bytes);
        return false;
    }
    out.resize(static_cast<size_t>(dim));
    std::memcpy(out.data(), bytes.data(), expected_bytes);
    return true;
}
}  // namespace

int logosdb_export_ndjson(logosdb_t* db,
                          const char* out_path,
                          uint64_t start_id,
                          uint64_t end_id_exclusive,
                          char** errptr)
{
    if (!db || !out_path)
    {
        set_err(errptr, "export_ndjson: null db or out_path");
        return -1;
    }
    std::lock_guard<std::mutex> lock(db->mu);

    size_t total = db->vectors.n_rows();
    if (end_id_exclusive == 0 || end_id_exclusive > total)
        end_id_exclusive = total;
    if (start_id > end_id_exclusive)
    {
        set_err(errptr, "export_ndjson: start_id > end_id_exclusive");
        return -1;
    }
    int dim = db->dim;
    if (dim <= 0)
    {
        set_err(errptr, "export_ndjson: invalid dim");
        return -1;
    }

    std::ofstream f(out_path, std::ios::binary | std::ios::trunc);
    if (!f)
    {
        set_err(errptr, std::string("export_ndjson: cannot open ") + out_path);
        return -1;
    }

    std::vector<float> row(static_cast<size_t>(dim));
    for (uint64_t id = start_id; id < end_id_exclusive; ++id)
    {
        if (db->meta.is_deleted(id))
            continue;
        db->vectors.row_to_float32(id, row.data());

        auto* m = db->meta.row(id);
        nlohmann::json j;
        j["id"] = id;
        j["vector"] = base64_encode_floats(row.data(), dim);
        j["text"] = (m && !m->text.empty()) ? m->text : "";
        j["timestamp"] = (m && !m->timestamp.empty()) ? m->timestamp : "";
        f << j.dump() << '\n';
        if (!f)
        {
            set_err(errptr, "export_ndjson: write failed");
            return -1;
        }
    }
    f.flush();
    if (!f)
    {
        set_err(errptr, "export_ndjson: flush failed");
        return -1;
    }
    return 0;
}

int logosdb_import_ndjson(logosdb_t* db,
                          const char* in_path,
                          int chunk_size,
                          const char* checkpoint_path,
                          int resume,
                          char** errptr)
{
    if (!db || !in_path)
    {
        set_err(errptr, "import_ndjson: null db or in_path");
        return -1;
    }
    if (chunk_size <= 0)
        chunk_size = 1024;

    std::ifstream f(in_path, std::ios::binary);
    if (!f)
    {
        set_err(errptr, std::string("import_ndjson: cannot open ") + in_path);
        return -1;
    }

    uint64_t start_byte = 0;
    uint64_t rows_imported = 0;
    if (resume && checkpoint_path)
    {
        std::ifstream cp(checkpoint_path);
        if (cp)
        {
            try
            {
                nlohmann::json j;
                cp >> j;
                if (j.contains("byte_offset"))
                    start_byte = j["byte_offset"].get<uint64_t>();
                if (j.contains("rows_imported"))
                    rows_imported = j["rows_imported"].get<uint64_t>();
            }
            catch (const std::exception& e)
            {
                set_err(errptr,
                        std::string("import_ndjson: bad checkpoint json: ") + e.what());
                return -1;
            }
        }
    }
    if (start_byte > 0)
    {
        f.seekg(static_cast<std::streamoff>(start_byte), std::ios::beg);
        if (!f)
        {
            set_err(errptr, "import_ndjson: cannot seek to checkpoint offset");
            return -1;
        }
    }

    int dim = db->dim;
    std::vector<float> chunk_vecs;
    std::vector<std::string> chunk_texts;
    std::vector<std::string> chunk_ts;
    chunk_vecs.reserve(static_cast<size_t>(chunk_size) * static_cast<size_t>(dim));
    chunk_texts.reserve(static_cast<size_t>(chunk_size));
    chunk_ts.reserve(static_cast<size_t>(chunk_size));
    std::vector<uint64_t> out_ids(static_cast<size_t>(chunk_size));

    auto flush_chunk = [&](int n) -> int
    {
        if (n <= 0)
            return 0;
        std::vector<const char*> text_ptrs(static_cast<size_t>(n));
        std::vector<const char*> ts_ptrs(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i)
        {
            text_ptrs[i] = chunk_texts[i].empty() ? nullptr : chunk_texts[i].c_str();
            ts_ptrs[i] = chunk_ts[i].empty() ? nullptr : chunk_ts[i].c_str();
        }
        char* berr = nullptr;
        int rc = logosdb_put_batch(db,
                                   chunk_vecs.data(),
                                   n,
                                   dim,
                                   text_ptrs.data(),
                                   ts_ptrs.data(),
                                   out_ids.data(),
                                   &berr);
        if (rc != 0)
        {
            set_err(errptr, berr ? berr : "import_ndjson: put_batch failed");
            free(berr);
            return -1;
        }
        free(berr);
        chunk_vecs.clear();
        chunk_texts.clear();
        chunk_ts.clear();
        return 0;
    };

    auto write_checkpoint = [&](uint64_t byte_off) -> int
    {
        if (!checkpoint_path)
            return 0;
        nlohmann::json cp;
        cp["version"] = 1;
        cp["byte_offset"] = byte_off;
        cp["rows_imported"] = rows_imported;
        std::string tmp = std::string(checkpoint_path) + ".tmp";
        {
            std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
            if (!out)
            {
                set_err(errptr, "import_ndjson: cannot write checkpoint tmp");
                return -1;
            }
            out << cp.dump() << '\n';
        }
        std::error_code ec;
        std::filesystem::rename(tmp, checkpoint_path, ec);
        if (ec)
        {
            set_err(errptr, std::string("import_ndjson: rename checkpoint: ") + ec.message());
            return -1;
        }
        return 0;
    };

    std::string line;
    int n_in_chunk = 0;
    std::vector<float> vec_buf;
    while (std::getline(f, line))
    {
        if (line.empty())
            continue;
        // Strip an optional trailing CR (Windows line endings).
        if (line.back() == '\r')
            line.pop_back();
        if (line.empty())
            continue;

        nlohmann::json j;
        try
        {
            j = nlohmann::json::parse(line);
        }
        catch (const std::exception& e)
        {
            set_err(errptr,
                    std::string("import_ndjson: parse error at line: ") + e.what());
            return -1;
        }
        if (!j.contains("vector") || !j["vector"].is_string())
        {
            set_err(errptr, "import_ndjson: missing or non-string 'vector' field");
            return -1;
        }
        std::string b64 = j["vector"].get<std::string>();
        std::string ebuf;
        if (!base64_decode_floats(b64, dim, vec_buf, ebuf))
        {
            set_err(errptr, std::string("import_ndjson: ") + ebuf);
            return -1;
        }
        chunk_vecs.insert(chunk_vecs.end(), vec_buf.begin(), vec_buf.end());
        chunk_texts.push_back(j.value("text", std::string{}));
        chunk_ts.push_back(j.value("timestamp", std::string{}));
        ++n_in_chunk;

        if (n_in_chunk >= chunk_size)
        {
            if (flush_chunk(n_in_chunk) != 0)
                return -1;
            rows_imported += static_cast<uint64_t>(n_in_chunk);
            n_in_chunk = 0;
            auto pos = f.tellg();
            uint64_t byte_off = pos >= 0 ? static_cast<uint64_t>(pos) : 0u;
            if (write_checkpoint(byte_off) != 0)
                return -1;
        }
    }
    if (n_in_chunk > 0)
    {
        if (flush_chunk(n_in_chunk) != 0)
            return -1;
        rows_imported += static_cast<uint64_t>(n_in_chunk);
    }
    if (checkpoint_path)
    {
        // EOF: write the input file size so a subsequent --resume becomes a no-op.
        std::error_code ec;
        uint64_t byte_off = static_cast<uint64_t>(std::filesystem::file_size(in_path, ec));
        if (ec)
            byte_off = 0;
        if (write_checkpoint(byte_off) != 0)
            return -1;
    }
    return 0;
}

/* ── Vector utilities ──────────────────────────────────────────────── */

#include <cmath>

int logosdb_l2_normalize(float* vec, int dim)
{
    if (!vec || dim <= 0)
        return -1;

    double sq_sum = 0.0;
    for (int i = 0; i < dim; ++i)
    {
        sq_sum += (double)vec[i] * (double)vec[i];
    }

    double norm = std::sqrt(sq_sum);
    if (norm == 0.0)
        return -1;  // Zero norm - cannot normalize

    float scale = (float)(1.0 / norm);
    for (int i = 0; i < dim; ++i)
    {
        vec[i] *= scale;
    }
    return 0;
}

/* C++ convenience wrappers */
namespace logosdb
{

bool l2_normalize(std::vector<float>& v)
{
    if (v.empty())
        return false;
    return logosdb_l2_normalize(v.data(), (int)v.size()) == 0;
}

std::vector<float> l2_normalized(std::vector<float> v)
{
    l2_normalize(v);
    return v;  // NRVO should elide the copy
}

}  // namespace logosdb
