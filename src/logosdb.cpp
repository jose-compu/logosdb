#include <logosdb/logosdb.h>
#include "storage.h"
#include "metadata.h"
#include "hnsw_index.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

using namespace logosdb::internal;

/* ── Internal DB struct ────────────────────────────────────────────── */

struct logosdb_t {
    VectorStorage  vectors;
    MetadataStore  meta;
    HnswIndex      index;
    std::mutex     mu;
    int            dim = 0;
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
        *errptr = strdup(msg.c_str());
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

    // Backfill index if vector storage has more rows than the index (e.g. crash recovery).
    size_t n_vec = db->vectors.n_rows();
    size_t n_idx = db->index.count();
    if (n_vec > n_idx) {
        for (size_t i = n_idx; i < n_vec; ++i) {
            const float * row = db->vectors.row(i);
            if (row && !db->index.add(i, row, err)) {
                set_err(errptr, "backfill index: " + err);
                delete db;
                return nullptr;
            }
        }
        db->index.save(err);
    }

    return db;
}

void logosdb_close(logosdb_t * db) {
    if (!db) return;
    std::string err;
    db->index.save(err);
    db->vectors.sync(err);
    db->meta.sync(err);
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

    // NOTE: these three writes are not atomic. On partial failure the stores
    // may diverge (e.g. vectors written but metadata missing). The HNSW index
    // is backfilled from the vector store on next open, but metadata gaps are
    // not currently recoverable. A WAL would fix this; acceptable for now
    // given the single-process embedded use case.
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

int logosdb_dim(logosdb_t * db) {
    return db ? db->dim : 0;
}

const float * logosdb_raw_vectors(logosdb_t * db, size_t * n_rows, int * dim) {
    if (!db) { if (n_rows) *n_rows = 0; if (dim) *dim = 0; return nullptr; }
    if (n_rows) *n_rows = db->vectors.n_rows();
    if (dim)    *dim    = db->dim;
    return db->vectors.data();
}
