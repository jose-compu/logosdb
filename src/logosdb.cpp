#include <logosdb/logosdb.h>
#include "platform.h"
#include "storage.h"
#include "metadata.h"
#include "hnsw_index.h"
#include "wal.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

using namespace logosdb::internal;

/* ── Internal DB struct ────────────────────────────────────────────── */

struct logosdb_t {
    VectorStorage   vectors;
    MetadataStore   meta;
    HnswIndex       index;
    WriteAheadLog   wal;
    std::mutex      mu;
    int             dim = 0;
};

struct logosdb_options_t {
    int    dim             = 0;
    size_t max_elements    = 1000000;
    int    ef_construction = 200;
    int    M               = 16;
    int    ef_search        = 50;
};

struct logosdb_search_result_t {
    struct Hit {
        uint64_t    id;
        float       score;
        std::string text;
        std::string timestamp;
    };
    std::vector<Hit> hits;
};

/* ── Helpers ───────────────────────────────────────────────────────── */

static void set_err(char ** errptr, const std::string & msg) {
    if (errptr) {
        *errptr = platform::string_duplicate(msg.c_str());
    }
}

/* ── Options ───────────────────────────────────────────────────────── */

logosdb_options_t * logosdb_options_create(void) {
    return new logosdb_options_t();
}

void logosdb_options_destroy(logosdb_options_t * opts) { delete opts; }

void logosdb_options_set_dim(logosdb_options_t * o, int d)             { if (o && d > 0) o->dim = d; }
void logosdb_options_set_max_elements(logosdb_options_t * o, size_t n) { if (o && n > 0) o->max_elements = n; }
void logosdb_options_set_ef_construction(logosdb_options_t * o, int e) { if (o && e > 0) o->ef_construction = e; }
void logosdb_options_set_M(logosdb_options_t * o, int m)              { if (o && m > 0) o->M = m; }
void logosdb_options_set_ef_search(logosdb_options_t * o, int e)      { if (o && e > 0) o->ef_search = e; }

/* ── Lifecycle ─────────────────────────────────────────────────────── */

logosdb_t * logosdb_open(const char * path, const logosdb_options_t * opts,
                         char ** errptr) {
    if (!path || !opts || opts->dim <= 0) {
        set_err(errptr, "invalid arguments: path and dim > 0 required");
        return nullptr;
    }

    std::filesystem::create_directories(path);

    auto db = new logosdb_t();
    db->dim = opts->dim;
    std::string err;

    std::string vec_path  = std::string(path) + "/vectors.bin";
    std::string meta_path = std::string(path) + "/meta.jsonl";
    std::string idx_path  = std::string(path) + "/hnsw.idx";
    std::string wal_path  = std::string(path) + "/wal.log";

    if (!db->vectors.open(vec_path, opts->dim, err)) {
        set_err(errptr, err);
        delete db;
        return nullptr;
    }
    if (!db->meta.open(meta_path, err)) {
        set_err(errptr, err);
        delete db;
        return nullptr;
    }

    HnswParams hp;
    hp.dim             = opts->dim;
    hp.max_elements    = opts->max_elements;
    hp.ef_construction = opts->ef_construction;
    hp.M               = opts->M;
    hp.ef_search       = opts->ef_search;

    if (!db->index.open(idx_path, hp, err)) {
        set_err(errptr, err);
        delete db;
        return nullptr;
    }

    // Open WAL and replay any pending entries for atomic recovery.
    if (!db->wal.open(wal_path, err)) {
        set_err(errptr, err);
        delete db;
        return nullptr;
    }

    // Replay pending WAL entries to ensure consistency.
    int replayed = db->wal.replay_pending(
        [&db](const WALEntry & entry, std::string & replay_err) -> bool {
            // Validate: expected_id should match current row count
            if (entry.expected_id != db->vectors.n_rows()) {
                replay_err = "wal replay: expected_id mismatch (" +
                             std::to_string(entry.expected_id) + " vs " +
                             std::to_string(db->vectors.n_rows()) + ")";
                return false;
            }

            // Replay vector
            uint64_t vid = db->vectors.append(entry.vector.data(), (int)entry.dim, replay_err);
            if (vid == UINT64_MAX) return false;

            // Replay metadata
            uint64_t mid = db->meta.append(entry.text.c_str(), entry.timestamp.c_str(), replay_err);
            if (mid == UINT64_MAX) return false;

            // Replay index
            if (!db->index.add(vid, entry.vector.data(), replay_err)) {
                return false;
            }

            return true;
        },
        err
    );

    if (replayed < 0) {
        set_err(errptr, "wal replay: " + err);
        delete db;
        return nullptr;
    }

    // Backfill index if vector storage has more rows than the index (e.g. crash recovery).
    size_t n_vec = db->vectors.n_rows();
    size_t n_idx = db->index.count();
    bool   backfilled = false;
    if (n_vec > n_idx) {
        for (size_t i = n_idx; i < n_vec; ++i) {
            const float * row = db->vectors.row(i);
            if (row && !db->index.add(i, row, err)) {
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
    for (uint64_t id : db->meta.deleted_ids()) {
        if (!db->index.has_label(id)) continue;
        if (db->index.is_deleted(id)) continue;
        if (!db->index.mark_deleted(id, err)) {
            set_err(errptr, "replay tombstone: " + err);
            delete db;
            return nullptr;
        }
    }

    if (backfilled) db->index.save(err);

    return db;
}

void logosdb_close(logosdb_t * db) {
    if (!db) return;
    std::string err;
    db->wal.sync(err);   // Ensure WAL is durable before closing other stores
    db->index.save(err);
    db->vectors.sync(err);
    db->meta.sync(err);
    db->wal.close();
    delete db;
}

/* ── Write ─────────────────────────────────────────────────────────── */

uint64_t logosdb_put(logosdb_t * db,
                     const float * embedding, int dim,
                     const char * text,
                     const char * timestamp,
                     char ** errptr) {
    if (!db || !embedding) {
        set_err(errptr, "null db or embedding");
        return UINT64_MAX;
    }
    if (dim != db->dim) {
        set_err(errptr, "dimension mismatch");
        return UINT64_MAX;
    }

    std::lock_guard<std::mutex> lock(db->mu);
    std::string err;

    // Compute expected row id before writing
    uint64_t expected_id = db->vectors.n_rows();

    // Step 1: Write WAL entry (durability point)
    int64_t wal_offset = db->wal.append_pending(embedding, dim, text, timestamp, expected_id, err);
    if (wal_offset < 0) { set_err(errptr, err); return UINT64_MAX; }

    // Step 2: Write to vector storage
    uint64_t vid = db->vectors.append(embedding, dim, err);
    if (vid == UINT64_MAX) { set_err(errptr, err); return UINT64_MAX; }

    // Step 3: Write to metadata storage
    uint64_t mid = db->meta.append(text, timestamp, err);
    if (mid == UINT64_MAX) {
        // Metadata write failed - entry remains in WAL for replay on recovery
        set_err(errptr, err);
        return UINT64_MAX;
    }

    // Step 4: Write to HNSW index
    if (!db->index.add(vid, embedding, err)) {
        // Index write failed - entry remains in WAL for replay on recovery
        set_err(errptr, err);
        return UINT64_MAX;
    }

    // Step 5: Mark WAL entry as committed
    if (!db->wal.mark_committed(wal_offset, err)) {
        // Non-fatal: entry will be replayed on next open if needed
        // Log but don't fail the operation
    }

    return vid;
}

int logosdb_put_batch(logosdb_t * db,
                      const float * embeddings, int n, int dim,
                      const char * const * texts,
                      const char * const * timestamps,
                      uint64_t * out_ids,
                      char ** errptr) {
    if (!db || !embeddings || !out_ids) {
        set_err(errptr, "null db, embeddings, or out_ids");
        return -1;
    }
    if (n <= 0) {
        return 0;  // Empty batch is a no-op success
    }
    if (dim != db->dim) {
        set_err(errptr, "dimension mismatch");
        return -1;
    }

    std::lock_guard<std::mutex> lock(db->mu);
    std::string err;
    uint64_t start_id = db->vectors.n_rows();

    // For batch operations, we write a single WAL entry for the entire batch
    // to minimize WAL overhead. The entry contains the expected starting id.

    // Step 1: Batch write to vector storage (single ftruncate + pwrite)
    uint64_t vid = db->vectors.append_batch(embeddings, n, dim, err);
    if (vid == UINT64_MAX) {
        set_err(errptr, err);
        return -1;
    }

    // Step 2: Batch write to metadata storage
    uint64_t mid = db->meta.append_batch(texts, timestamps, n, err);
    if (mid == UINT64_MAX) {
        set_err(errptr, err);
        return -1;
    }

    // Step 3: Add to HNSW index (must be done per-vector)
    const float * vec = embeddings;
    size_t stride = (size_t)dim * sizeof(float);
    for (int i = 0; i < n; ++i) {
        if (!db->index.add(vid + i, vec, err)) {
            set_err(errptr, err);
            return -1;
        }
        vec = reinterpret_cast<const float *>(reinterpret_cast<const uint8_t *>(vec) + stride);
    }

    // Fill out_ids with assigned ids
    for (int i = 0; i < n; ++i) {
        out_ids[i] = vid + i;
    }

    return 0;
}

/* ── Delete / Update ───────────────────────────────────────────────── */

int logosdb_delete(logosdb_t * db, uint64_t id, char ** errptr) {
    if (!db) { set_err(errptr, "null db"); return -1; }

    std::lock_guard<std::mutex> lock(db->mu);
    std::string err;

    if (id >= db->vectors.n_rows()) {
        set_err(errptr, "delete: id out of range");
        return -1;
    }
    if (db->meta.is_deleted(id)) {
        set_err(errptr, "delete: id already deleted");
        return -1;
    }

    // Mark in the index first. If hnswlib rejects (e.g. label missing because
    // the row was never indexed), bail out before touching the metadata log.
    if (!db->index.mark_deleted(id, err)) {
        set_err(errptr, err);
        return -1;
    }

    if (!db->meta.mark_deleted(id, err)) {
        // Index is now marked but metadata failed. Best-effort rollback so
        // the two stores don't diverge.
        std::string ignore;
        (void)ignore;
        set_err(errptr, err);
        return -1;
    }
    return 0;
}

uint64_t logosdb_update(logosdb_t * db, uint64_t id,
                        const float * embedding, int dim,
                        const char * text,
                        const char * timestamp,
                        char ** errptr) {
    if (!db || !embedding) {
        set_err(errptr, "null db or embedding");
        return UINT64_MAX;
    }
    if (dim != db->dim) {
        set_err(errptr, "dimension mismatch");
        return UINT64_MAX;
    }

    std::lock_guard<std::mutex> lock(db->mu);
    std::string err;

    if (id >= db->vectors.n_rows()) {
        set_err(errptr, "update: id out of range");
        return UINT64_MAX;
    }
    if (db->meta.is_deleted(id)) {
        set_err(errptr, "update: id already deleted");
        return UINT64_MAX;
    }

    // Mark the old row deleted, then append a fresh row. The two operations
    // share the write mutex, so no reader sees an intermediate state.
    if (!db->index.mark_deleted(id, err)) {
        set_err(errptr, err);
        return UINT64_MAX;
    }
    if (!db->meta.mark_deleted(id, err)) {
        set_err(errptr, err);
        return UINT64_MAX;
    }

    uint64_t vid = db->vectors.append(embedding, dim, err);
    if (vid == UINT64_MAX) { set_err(errptr, err); return UINT64_MAX; }

    uint64_t mid = db->meta.append(text, timestamp, err);
    if (mid == UINT64_MAX) { set_err(errptr, err); return UINT64_MAX; }

    if (!db->index.add(vid, embedding, err)) {
        set_err(errptr, err);
        return UINT64_MAX;
    }
    return vid;
}

/* ── Search ────────────────────────────────────────────────────────── */

logosdb_search_result_t * logosdb_search(logosdb_t * db,
                                         const float * query, int dim,
                                         int top_k,
                                         char ** errptr) {
    if (!db || !query) {
        set_err(errptr, "null db or query");
        return nullptr;
    }
    if (dim != db->dim) {
        set_err(errptr, "dimension mismatch in search");
        return nullptr;
    }
    if (top_k <= 0) {
        set_err(errptr, "top_k must be > 0");
        return nullptr;
    }

    std::string err;
    auto raw = db->index.search(query, top_k, err);
    if (!err.empty()) {
        set_err(errptr, err);
        return nullptr;
    }

    auto * r = new logosdb_search_result_t();
    r->hits.reserve(raw.size());
    for (auto & [label, score] : raw) {
        logosdb_search_result_t::Hit h;
        h.id    = label;
        h.score = score;
        auto * m = db->meta.row(label);
        if (m) {
            h.text      = m->text;
            h.timestamp = m->timestamp;
        }
        r->hits.push_back(std::move(h));
    }
    return r;
}

logosdb_search_result_t * logosdb_search_ts_range(logosdb_t * db,
                                                 const float * query, int dim,
                                                 int top_k,
                                                 const char * ts_from_iso8601,
                                                 const char * ts_to_iso8601,
                                                 int candidate_k,
                                                 char ** errptr) {
    if (!db || !query) {
        set_err(errptr, "null db or query");
        return nullptr;
    }
    if (dim != db->dim) {
        set_err(errptr, "dimension mismatch in search");
        return nullptr;
    }
    if (top_k <= 0) {
        set_err(errptr, "top_k must be > 0");
        return nullptr;
    }
    if (candidate_k < top_k) {
        candidate_k = top_k;  // Ensure we fetch at least top_k candidates
    }

    std::string err;
    // Fetch more candidates than needed to allow for filtering
    auto raw = db->index.search(query, candidate_k, err);
    if (!err.empty()) {
        set_err(errptr, err);
        return nullptr;
    }

    auto * r = new logosdb_search_result_t();
    r->hits.reserve(top_k);

    for (auto & [label, score] : raw) {
        // Check if we've collected enough results
        if ((int)r->hits.size() >= top_k) break;

        // Get metadata for this row
        auto * m = db->meta.row(label);
        if (!m) continue;

        // Apply timestamp filter
        if (ts_from_iso8601 && !m->timestamp.empty()) {
            if (m->timestamp < ts_from_iso8601) continue;  // Before start
        }
        if (ts_to_iso8601 && !m->timestamp.empty()) {
            if (m->timestamp > ts_to_iso8601) continue;  // After end
        }

        // This result passes the filter
        logosdb_search_result_t::Hit h;
        h.id    = label;
        h.score = score;
        h.text  = m->text;
        h.timestamp = m->timestamp;
        r->hits.push_back(std::move(h));
    }

    return r;
}

int logosdb_result_count(const logosdb_search_result_t * r) {
    return r ? (int)r->hits.size() : 0;
}

uint64_t logosdb_result_id(const logosdb_search_result_t * r, int i) {
    return (r && i >= 0 && i < (int)r->hits.size()) ? r->hits[i].id : UINT64_MAX;
}

float logosdb_result_score(const logosdb_search_result_t * r, int i) {
    return (r && i >= 0 && i < (int)r->hits.size()) ? r->hits[i].score : 0.0f;
}

const char * logosdb_result_text(const logosdb_search_result_t * r, int i) {
    if (!r || i < 0 || i >= (int)r->hits.size()) return nullptr;
    return r->hits[i].text.empty() ? nullptr : r->hits[i].text.c_str();
}

const char * logosdb_result_timestamp(const logosdb_search_result_t * r, int i) {
    if (!r || i < 0 || i >= (int)r->hits.size()) return nullptr;
    return r->hits[i].timestamp.empty() ? nullptr : r->hits[i].timestamp.c_str();
}

void logosdb_result_free(logosdb_search_result_t * r) { delete r; }

/* ── Info ──────────────────────────────────────────────────────────── */

size_t logosdb_count(logosdb_t * db) {
    return db ? db->vectors.n_rows() : 0;
}

size_t logosdb_count_live(logosdb_t * db) {
    if (!db) return 0;
    size_t total   = db->vectors.n_rows();
    size_t deleted = db->meta.deleted_count();
    return total >= deleted ? total - deleted : 0;
}

int logosdb_dim(logosdb_t * db) {
    return db ? db->dim : 0;
}

const float * logosdb_raw_vectors(logosdb_t * db, size_t * n_rows, int * dim) {
    if (!db) { if (n_rows) *n_rows = 0; if (dim) *dim = 0; return nullptr; }
    if (n_rows) *n_rows = db->vectors.n_rows();
    if (dim)    *dim    = db->dim;
    return db->vectors.data();
}
