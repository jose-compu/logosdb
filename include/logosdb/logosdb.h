#ifndef LOGOSDB_H
#define LOGOSDB_H

#include <stddef.h>
#include <stdint.h>

#define LOGOSDB_VERSION_MAJOR 0
#define LOGOSDB_VERSION_MINOR 3
#define LOGOSDB_VERSION_PATCH 2
#define LOGOSDB_VERSION_STRING "0.3.2"

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
#endif

#endif // LOGOSDB_H
