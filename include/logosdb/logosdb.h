#ifndef LOGOSDB_H
#define LOGOSDB_H

#include <stddef.h>
#include <stdint.h>

#define LOGOSDB_VERSION_MAJOR 0
#define LOGOSDB_VERSION_MINOR 6
#define LOGOSDB_VERSION_PATCH 0
#define LOGOSDB_VERSION_STRING "0.6.0"

/* Distance metrics for vector similarity search */
#define LOGOSDB_DIST_IP      0  /* Inner product (default, requires L2-normalized vectors) */
#define LOGOSDB_DIST_COSINE  1  /* Cosine similarity (auto-normalizes vectors) */
#define LOGOSDB_DIST_L2      2  /* Euclidean distance (L2 space) */

#ifdef __cplusplus
extern "C" {
#endif

/* ── Opaque handles ────────────────────────────────────────────────── */

typedef struct logosdb_t              logosdb_t;
typedef struct logosdb_options_t      logosdb_options_t;
typedef struct logosdb_search_result_t logosdb_search_result_t;

/* ── Options ───────────────────────────────────────────────────────── */

logosdb_options_t * logosdb_options_create(void);
void logosdb_options_destroy(logosdb_options_t * opts);

void logosdb_options_set_dim(logosdb_options_t * opts, int dim);
void logosdb_options_set_max_elements(logosdb_options_t * opts, size_t n);
void logosdb_options_set_ef_construction(logosdb_options_t * opts, int ef);
void logosdb_options_set_M(logosdb_options_t * opts, int M);
void logosdb_options_set_ef_search(logosdb_options_t * opts, int ef);

/* Set distance metric for vector similarity.
 *   metric: one of LOGOSDB_DIST_IP, LOGOSDB_DIST_COSINE, or LOGOSDB_DIST_L2
 *           (default is LOGOSDB_DIST_IP)
 *
 * For LOGOSDB_DIST_IP: vectors must be L2-normalized by caller.
 * For LOGOSDB_DIST_COSINE: vectors are automatically L2-normalized on put/search.
 * For LOGOSDB_DIST_L2: Euclidean distance is used (lower is more similar).
 *
 * The distance metric is persisted in the index file and must match on reopen.
 * Returns 0 on success, -1 on invalid metric. */
int logosdb_options_set_distance(logosdb_options_t * opts, int metric);

/* ── Lifecycle ─────────────────────────────────────────────────────── */

logosdb_t * logosdb_open(const char * path, const logosdb_options_t * opts,
                         char ** errptr);
void        logosdb_close(logosdb_t * db);

/* ── Write ─────────────────────────────────────────────────────────── */

uint64_t logosdb_put(logosdb_t * db,
                     const float * embedding, int dim,
                     const char * text,
                     const char * timestamp,
                     char ** errptr);

/* Batch insert of N vectors. More efficient than N separate logosdb_put calls.
 *   embeddings: flattened (n x dim) float array
 *   texts, timestamps: arrays of n pointers (may be NULL)
 *   out_ids: pre-allocated array of size n to receive assigned row ids
 *
 * Returns 0 on success, -1 on error (partial insert may have occurred).
 * On error, *errptr is set to a newly-allocated message that the caller
 * must free(). */
int logosdb_put_batch(logosdb_t * db,
                      const float * embeddings, int n, int dim,
                      const char * const * texts,
                      const char * const * timestamps,
                      uint64_t * out_ids,
                      char ** errptr);

/* Mark the row with the given id as deleted. The vector bytes remain on
 * disk but the row is excluded from search results and future `raw_vectors`
 * bulk views (logosdb_count_live reflects live rows only).
 *
 * Returns 0 on success, -1 on error (id out of range, already deleted,
 * db not open). On error `*errptr` is set to a newly-allocated message
 * that the caller must free(). */
int logosdb_delete(logosdb_t * db, uint64_t id, char ** errptr);

/* Replace the row with the given id. This is implemented as a
 * delete-then-put, so the returned id is NEW (not the original `id`).
 * Callers must update any stored references.
 *
 * Returns the new id on success, UINT64_MAX on error. */
uint64_t logosdb_update(logosdb_t * db, uint64_t id,
                        const float * embedding, int dim,
                        const char * text,
                        const char * timestamp,
                        char ** errptr);

/* ── Search ────────────────────────────────────────────────────────── */

logosdb_search_result_t * logosdb_search(logosdb_t * db,
                                         const float * query, int dim,
                                         int top_k,
                                         char ** errptr);

/* Search with timestamp range filter.
 *   ts_from_iso8601: optional start timestamp (inclusive), NULL for no lower bound
 *   ts_to_iso8601:   optional end timestamp (inclusive), NULL for no upper bound
 *   candidate_k:     internal multiplier for post-filtering (e.g., 10x top_k)
 *
 * Returns top_k results that match both the vector similarity and timestamp range.
 * Uses post-filtering: fetches candidate_k results internally, filters by timestamp,
 * then returns the top_k that pass. Higher candidate_k improves recall but costs more.
 *
 * Timestamp format: ISO 8601 strings (e.g., "2025-01-01T00:00:00Z").
 * Comparison is lexicographic (string compare), which works correctly for ISO 8601.
 */
logosdb_search_result_t * logosdb_search_ts_range(logosdb_t * db,
                                                 const float * query, int dim,
                                                 int top_k,
                                                 const char * ts_from_iso8601,
                                                 const char * ts_to_iso8601,
                                                 int candidate_k,
                                                 char ** errptr);

int         logosdb_result_count    (const logosdb_search_result_t * r);
uint64_t    logosdb_result_id       (const logosdb_search_result_t * r, int i);
float       logosdb_result_score    (const logosdb_search_result_t * r, int i);
const char* logosdb_result_text     (const logosdb_search_result_t * r, int i);
const char* logosdb_result_timestamp(const logosdb_search_result_t * r, int i);
void        logosdb_result_free     (logosdb_search_result_t * r);

/* ── Info ──────────────────────────────────────────────────────────── */

/* Total row count, including rows that have been marked deleted. */
size_t logosdb_count     (logosdb_t * db);

/* Live row count: total rows minus rows marked deleted. */
size_t logosdb_count_live(logosdb_t * db);

int    logosdb_dim       (logosdb_t * db);

/* ── Bulk read (for tensor construction) ───────────────────────────── */

const float * logosdb_raw_vectors(logosdb_t * db, size_t * n_rows, int * dim);

/* ── Vector utilities ──────────────────────────────────────────────── */

/* Normalize `vec` in-place using L2 (Euclidean) norm.
 *   vec: pointer to float array (will be modified in place)
 *   dim: number of elements in vec
 *
 * Returns 0 on success, -1 if the input has zero norm (cannot normalize).
 * On zero norm, vec is left unchanged.
 *
 * Example:
 *   float vec[128] = { ... };
 *   if (logosdb_l2_normalize(vec, 128) == 0) {
 *       logosdb_put(db, vec, 128, "text", NULL, &err);
 *   }
 */
int logosdb_l2_normalize(float * vec, int dim);

#ifdef __cplusplus
} /* extern "C" */
#endif

/* ── C++ convenience wrapper ───────────────────────────────────────── */

#ifdef __cplusplus
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>

namespace logosdb {

struct Options {
    int    dim              = 0;
    size_t max_elements     = 1000000;
    int    ef_construction  = 200;
    int    M               = 16;
    int    ef_search        = 50;
    int    distance         = LOGOSDB_DIST_IP;  /* IP, COSINE, or L2 */
};

struct SearchHit {
    uint64_t    id;
    float       score;
    std::string text;
    std::string timestamp;
};

class DB {
public:
    explicit DB(const std::string & path, const Options & opts = {}) {
        logosdb_options_t * o = logosdb_options_create();
        if (opts.dim > 0)              logosdb_options_set_dim(o, opts.dim);
        if (opts.max_elements > 0)     logosdb_options_set_max_elements(o, opts.max_elements);
        if (opts.ef_construction > 0)  logosdb_options_set_ef_construction(o, opts.ef_construction);
        if (opts.M > 0)               logosdb_options_set_M(o, opts.M);
        if (opts.ef_search > 0)        logosdb_options_set_ef_search(o, opts.ef_search);
        logosdb_options_set_distance(o, opts.distance);
        char * err = nullptr;
        db_ = logosdb_open(path.c_str(), o, &err);
        logosdb_options_destroy(o);
        if (err) {
            std::string msg(err);
            free(err);
            throw std::runtime_error("logosdb_open: " + msg);
        }
    }

    ~DB() { if (db_) logosdb_close(db_); }

    DB(const DB &) = delete;
    DB & operator=(const DB &) = delete;
    DB(DB && o) noexcept : db_(o.db_) { o.db_ = nullptr; }
    DB & operator=(DB && o) noexcept {
        if (this != &o) { if (db_) logosdb_close(db_); db_ = o.db_; o.db_ = nullptr; }
        return *this;
    }

    uint64_t put(const std::vector<float> & embedding,
                 const std::string & text = {},
                 const std::string & timestamp = {}) {
        char * err = nullptr;
        uint64_t id = logosdb_put(db_, embedding.data(), (int)embedding.size(),
                                  text.empty() ? nullptr : text.c_str(),
                                  timestamp.empty() ? nullptr : timestamp.c_str(),
                                  &err);
        if (err) {
            std::string msg(err);
            free(err);
            throw std::runtime_error("logosdb_put: " + msg);
        }
        return id;
    }

    /* Batch insert of n vectors. 'embeddings' must contain n*dim floats.
     * Returns vector of assigned row ids. Throws on error. */
    std::vector<uint64_t> put_batch(const std::vector<float> & embeddings, int n,
                                    const std::vector<std::string> & texts = {},
                                    const std::vector<std::string> & timestamps = {}) {
        if (n <= 0) return {};
        if ((int)embeddings.size() < n * dim()) {
            throw std::runtime_error("put_batch: embeddings size too small");
        }
        std::vector<uint64_t> ids(n);
        std::vector<const char *> text_ptrs;
        std::vector<const char *> ts_ptrs;
        if (!texts.empty()) {
            text_ptrs.reserve(n);
            for (int i = 0; i < n; ++i) {
                text_ptrs.push_back(i < (int)texts.size() && !texts[i].empty() ? texts[i].c_str() : nullptr);
            }
        }
        if (!timestamps.empty()) {
            ts_ptrs.reserve(n);
            for (int i = 0; i < n; ++i) {
                ts_ptrs.push_back(i < (int)timestamps.size() && !timestamps[i].empty() ? timestamps[i].c_str() : nullptr);
            }
        }
        char * err = nullptr;
        int rc = logosdb_put_batch(db_, embeddings.data(), n, dim(),
                                   texts.empty() ? nullptr : text_ptrs.data(),
                                   timestamps.empty() ? nullptr : ts_ptrs.data(),
                                   ids.data(), &err);
        if (rc != 0) {
            std::string msg = err ? err : "unknown";
            free(err);
            throw std::runtime_error("logosdb_put_batch: " + msg);
        }
        return ids;
    }

    void del(uint64_t id) {
        char * err = nullptr;
        int rc = logosdb_delete(db_, id, &err);
        if (rc != 0) {
            std::string msg = err ? err : "unknown";
            free(err);
            throw std::runtime_error("logosdb_delete: " + msg);
        }
    }

    uint64_t update(uint64_t id,
                    const std::vector<float> & embedding,
                    const std::string & text = {},
                    const std::string & timestamp = {}) {
        char * err = nullptr;
        uint64_t new_id = logosdb_update(db_, id,
                                         embedding.data(), (int)embedding.size(),
                                         text.empty() ? nullptr : text.c_str(),
                                         timestamp.empty() ? nullptr : timestamp.c_str(),
                                         &err);
        if (err) {
            std::string msg(err);
            free(err);
            throw std::runtime_error("logosdb_update: " + msg);
        }
        return new_id;
    }

    std::vector<SearchHit> search(const std::vector<float> & query, int top_k) {
        char * err = nullptr;
        logosdb_search_result_t * r = logosdb_search(db_, query.data(),
                                                      (int)query.size(),
                                                      top_k, &err);
        if (err) {
            std::string msg(err);
            free(err);
            throw std::runtime_error("logosdb_search: " + msg);
        }
        std::vector<SearchHit> hits;
        int n = logosdb_result_count(r);
        hits.reserve(n);
        for (int i = 0; i < n; ++i) {
            SearchHit h;
            h.id    = logosdb_result_id(r, i);
            h.score = logosdb_result_score(r, i);
            const char * t = logosdb_result_text(r, i);
            if (t) h.text = t;
            const char * ts = logosdb_result_timestamp(r, i);
            if (ts) h.timestamp = ts;
            hits.push_back(std::move(h));
        }
        logosdb_result_free(r);
        return hits;
    }

    /* Search with timestamp range filter.
     *   ts_from: optional start timestamp (inclusive), empty for no lower bound
     *   ts_to:   optional end timestamp (inclusive), empty for no upper bound
     *   candidate_k: internal multiplier for post-filtering (default 10x top_k)
     *
     * Timestamp format: ISO 8601 strings (e.g., "2025-01-01T00:00:00Z").
     */
    std::vector<SearchHit> search_ts_range(const std::vector<float> & query,
                                              int top_k,
                                              const std::string & ts_from,
                                              const std::string & ts_to,
                                              int candidate_k = 0) {
        char * err = nullptr;
        int ck = candidate_k > 0 ? candidate_k : top_k * 10;
        logosdb_search_result_t * r = logosdb_search_ts_range(
            db_, query.data(), (int)query.size(), top_k,
            ts_from.empty() ? nullptr : ts_from.c_str(),
            ts_to.empty() ? nullptr : ts_to.c_str(),
            ck, &err);
        if (err) {
            std::string msg(err);
            free(err);
            throw std::runtime_error("logosdb_search_ts_range: " + msg);
        }
        std::vector<SearchHit> hits;
        int n = logosdb_result_count(r);
        hits.reserve(n);
        for (int i = 0; i < n; ++i) {
            SearchHit h;
            h.id    = logosdb_result_id(r, i);
            h.score = logosdb_result_score(r, i);
            const char * t = logosdb_result_text(r, i);
            if (t) h.text = t;
            const char * ts = logosdb_result_timestamp(r, i);
            if (ts) h.timestamp = ts;
            hits.push_back(std::move(h));
        }
        logosdb_result_free(r);
        return hits;
    }

    size_t count()      const { return logosdb_count(db_); }
    size_t count_live() const { return logosdb_count_live(db_); }
    int    dim()        const { return logosdb_dim(db_); }

    const float * raw_vectors(size_t & n_rows, int & d) const {
        return logosdb_raw_vectors(db_, &n_rows, &d);
    }

    logosdb_t * handle() { return db_; }

private:
    logosdb_t * db_ = nullptr;
};

} // namespace logosdb

/* L2-normalization helpers for C++
 *
 * These functions simplify preparing vectors for use with LOGOSDB_DIST_IP
 * (inner product distance), which requires L2-normalized vectors.
 *
 * Example:
 *   std::vector<float> vec = load_some_vector();  // unnormalized
 *   if (logosdb::l2_normalize(vec)) {
 *       db.put(vec, "text", "2025-04-28T10:00:00Z");
 *   }
 *
 * Or use l2_normalized() to get a normalized copy:
 *   auto normalized = logosdb::l2_normalized(vec);
 *   db.put(normalized, "text");
 */
namespace logosdb {

/* Normalize `v` in-place using L2 norm.
 * Returns true on success, false if the vector has zero norm.
 * On zero norm, v is left unchanged. */
bool l2_normalize(std::vector<float> & v);

/* Return an L2-normalized copy of `v`.
 * The input `v` is not modified.
 * Returns a zero vector if input has zero norm (caller should check). */
std::vector<float> l2_normalized(std::vector<float> v);

} // namespace logosdb

#endif

#endif // LOGOSDB_H
