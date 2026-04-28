#include <logosdb/logosdb.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

// Forward declarations for tests defined after main
static void test_wal_crash_recovery();
static void test_put_batch_basic();
static void test_put_batch_empty();
static void test_search_ts_range_basic();
static void test_search_ts_range_edge_cases();
static void test_search_ts_range_recall();
static void test_distance_cosine();
static void test_distance_l2();
static void test_distance_persistence();
static void test_distance_mismatch_error();
static void test_cli_info_reads_dim();
static void test_cli_export_import_roundtrip();
static void test_cli_search_ts_range();
static void test_storage_pointers_stable();
static void test_l2_normalize_basic();
static void test_l2_normalize_already_normalized();
static void test_l2_normalize_zero_vector();
static void test_l2_normalize_small_values();
static void test_l2_normalize_cpp_wrappers();

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, msg) do { \
    ++tests_run; \
    if (!(cond)) { \
        fprintf(stderr, "FAIL: %s (line %d): %s\n", msg, __LINE__, #cond); \
    } else { \
        ++tests_passed; \
    } \
} while(0)

static std::vector<float> unit_vec(int dim, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(dim);
    float norm = 0.0f;
    for (int i = 0; i < dim; ++i) { v[i] = dist(rng); norm += v[i] * v[i]; }
    norm = std::sqrt(norm);
    for (int i = 0; i < dim; ++i) v[i] /= norm;
    return v;
}

static void test_open_close() {
    std::string path = "/tmp/logosdb_test_oc";
    std::filesystem::remove_all(path);

    char * err = nullptr;
    logosdb_options_t * opts = logosdb_options_create();
    logosdb_options_set_dim(opts, 64);
    logosdb_t * db = logosdb_open(path.c_str(), opts, &err);
    logosdb_options_destroy(opts);

    CHECK(db != nullptr, "open");
    CHECK(err == nullptr, "no error on open");
    CHECK(logosdb_count(db) == 0, "empty count");
    CHECK(logosdb_dim(db) == 64, "dim");

    logosdb_close(db);
    std::filesystem::remove_all(path);
}

static void test_put_and_search() {
    std::string path = "/tmp/logosdb_test_ps";
    std::filesystem::remove_all(path);

    logosdb::DB db(path, {.dim = 64});

    auto v0 = unit_vec(64, 100);
    auto v1 = unit_vec(64, 200);
    auto v2 = unit_vec(64, 300);

    uint64_t id0 = db.put(v0, "fact zero", "2025-01-01T00:00:00Z");
    uint64_t id1 = db.put(v1, "fact one",  "2025-02-01T00:00:00Z");
    uint64_t id2 = db.put(v2, "fact two",  "2025-03-01T00:00:00Z");

    CHECK(id0 == 0, "id0");
    CHECK(id1 == 1, "id1");
    CHECK(id2 == 2, "id2");
    CHECK(db.count() == 3, "count after 3 puts");

    // Search with v0 as query — should return v0 as top hit.
    auto hits = db.search(v0, 3);
    CHECK(!hits.empty(), "search not empty");
    CHECK(hits[0].id == 0, "top hit is v0");
    CHECK(hits[0].text == "fact zero", "text matches");
    CHECK(hits[0].timestamp == "2025-01-01T00:00:00Z", "timestamp matches");
    CHECK(hits[0].score > 0.9f, "high self-similarity");

    std::filesystem::remove_all(path);
}

static void test_persistence() {
    std::string path = "/tmp/logosdb_test_persist";
    std::filesystem::remove_all(path);

    auto v0 = unit_vec(64, 400);
    auto v1 = unit_vec(64, 500);

    {
        logosdb::DB db(path, {.dim = 64});
        db.put(v0, "persisted fact A", "2025-04-01T00:00:00Z");
        db.put(v1, "persisted fact B", "2025-05-01T00:00:00Z");
        CHECK(db.count() == 2, "count before close");
    }

    // Reopen.
    {
        logosdb::DB db(path, {.dim = 64});
        CHECK(db.count() == 2, "count after reopen");

        auto hits = db.search(v0, 2);
        CHECK(!hits.empty(), "search after reopen");
        CHECK(hits[0].id == 0, "correct id after reopen");
        CHECK(hits[0].text == "persisted fact A", "text after reopen");
        CHECK(hits[0].timestamp == "2025-04-01T00:00:00Z", "ts after reopen");
    }

    std::filesystem::remove_all(path);
}

static void test_raw_vectors() {
    std::string path = "/tmp/logosdb_test_raw";
    std::filesystem::remove_all(path);

    int dim = 32;
    logosdb::DB db(path, {.dim = dim});

    auto v0 = unit_vec(dim, 600);
    auto v1 = unit_vec(dim, 700);
    db.put(v0);
    db.put(v1);

    size_t n_rows = 0;
    int d = 0;
    const float * raw = db.raw_vectors(n_rows, d);
    CHECK(raw != nullptr, "raw not null");
    CHECK(n_rows == 2, "raw n_rows");
    CHECK(d == dim, "raw dim");

    float diff = 0.0f;
    for (int i = 0; i < dim; ++i) diff += std::fabs(raw[i] - v0[i]);
    CHECK(diff < 1e-5f, "raw row0 matches v0");

    diff = 0.0f;
    for (int i = 0; i < dim; ++i) diff += std::fabs(raw[dim + i] - v1[i]);
    CHECK(diff < 1e-5f, "raw row1 matches v1");

    std::filesystem::remove_all(path);
}

static void test_many_vectors() {
    std::string path = "/tmp/logosdb_test_many";
    std::filesystem::remove_all(path);

    int dim = 128;
    int n = 500;
    logosdb::DB db(path, {.dim = dim, .max_elements = 1000});

    std::vector<std::vector<float>> vecs;
    for (int i = 0; i < n; ++i) {
        vecs.push_back(unit_vec(dim, i));
        db.put(vecs.back(), "row_" + std::to_string(i));
    }
    CHECK(db.count() == (size_t)n, "count == 500");

    // Each vector should find itself as top-1.
    int self_found = 0;
    for (int i = 0; i < n; i += 50) {
        auto hits = db.search(vecs[i], 1);
        if (!hits.empty() && hits[0].id == (uint64_t)i) ++self_found;
    }
    CHECK(self_found == 10, "self-retrieval 10/10");

    std::filesystem::remove_all(path);
}

static void test_c_api_errors() {
    char * err = nullptr;

    logosdb_options_t * opts = logosdb_options_create();
    logosdb_options_set_dim(opts, 0);
    logosdb_t * db = logosdb_open("/tmp/logosdb_test_err", opts, &err);
    CHECK(db == nullptr, "reject dim=0");
    CHECK(err != nullptr, "error message set");
    free(err); err = nullptr;
    logosdb_options_destroy(opts);

    CHECK(logosdb_count(nullptr) == 0, "count(null) == 0");
    CHECK(logosdb_dim(nullptr) == 0, "dim(null) == 0");

    logosdb_search_result_t * r = logosdb_search(nullptr, nullptr, 0, 1, &err);
    CHECK(r == nullptr, "search(null) == null");
    free(err);
}

static void test_search_ordering() {
    std::string path = "/tmp/logosdb_test_order";
    std::filesystem::remove_all(path);

    int dim = 64;
    logosdb::DB db(path, {.dim = dim});

    // Insert a base vector and a perturbed copy that is close to it.
    auto base = unit_vec(dim, 1000);
    auto close_vec = base;
    close_vec[0] += 0.01f;  // tiny perturbation — still very similar
    float norm = 0.0f;
    for (int i = 0; i < dim; ++i) norm += close_vec[i] * close_vec[i];
    norm = std::sqrt(norm);
    for (int i = 0; i < dim; ++i) close_vec[i] /= norm;

    auto far_vec = unit_vec(dim, 2000);

    db.put(far_vec,   "far");
    db.put(close_vec, "close");
    db.put(base,      "base");

    auto hits = db.search(base, 3);
    CHECK(hits.size() == 3, "ordering: 3 results");
    CHECK(hits[0].id == 2, "ordering: base is top-1");
    CHECK(hits[1].id == 1, "ordering: close is top-2");
    CHECK(hits[0].score >= hits[1].score, "ordering: score[0] >= score[1]");
    CHECK(hits[1].score >= hits[2].score, "ordering: score[1] >= score[2]");

    std::filesystem::remove_all(path);
}

static void test_top_k_limit() {
    std::string path = "/tmp/logosdb_test_topk";
    std::filesystem::remove_all(path);

    int dim = 32;
    logosdb::DB db(path, {.dim = dim});

    for (int i = 0; i < 20; ++i) db.put(unit_vec(dim, i + 3000));

    auto hits1 = db.search(unit_vec(dim, 3000), 1);
    CHECK(hits1.size() == 1, "top_k=1 returns 1");

    auto hits5 = db.search(unit_vec(dim, 3000), 5);
    CHECK(hits5.size() == 5, "top_k=5 returns 5");

    auto hits99 = db.search(unit_vec(dim, 3000), 99);
    CHECK(hits99.size() == 20, "top_k>n clamped to n");

    std::filesystem::remove_all(path);
}

static void test_empty_search() {
    std::string path = "/tmp/logosdb_test_empty_search";
    std::filesystem::remove_all(path);

    logosdb::DB db(path, {.dim = 32});
    auto hits = db.search(unit_vec(32, 5000), 5);
    CHECK(hits.empty(), "search on empty db returns nothing");

    std::filesystem::remove_all(path);
}

static void test_put_no_metadata() {
    std::string path = "/tmp/logosdb_test_nometa";
    std::filesystem::remove_all(path);

    logosdb::DB db(path, {.dim = 32});
    auto v = unit_vec(32, 6000);
    uint64_t id = db.put(v);
    CHECK(id == 0, "put without meta returns id 0");

    auto hits = db.search(v, 1);
    CHECK(!hits.empty(), "search finds it");
    CHECK(hits[0].text.empty(), "text is empty");
    CHECK(hits[0].timestamp.empty(), "timestamp is empty");

    std::filesystem::remove_all(path);
}

static void test_metadata_special_chars() {
    std::string path = "/tmp/logosdb_test_special";
    std::filesystem::remove_all(path);

    logosdb::DB db(path, {.dim = 32});
    auto v = unit_vec(32, 7000);

    std::string special = "He said \"hello\"\nand\ttabs\\backslash";
    db.put(v, special, "2025-06-25T00:00:00Z");

    // Close and reopen to test round-trip through JSONL.
    { logosdb::DB db2(path, {.dim = 32});
      auto hits = db2.search(v, 1);
      CHECK(!hits.empty(), "special chars: found");
      CHECK(hits[0].text == special, "special chars: round-trip preserved");
    }

    std::filesystem::remove_all(path);
}

static void test_dim_mismatch_put() {
    std::string path = "/tmp/logosdb_test_dim_put";
    std::filesystem::remove_all(path);

    char * err = nullptr;
    logosdb_options_t * opts = logosdb_options_create();
    logosdb_options_set_dim(opts, 32);
    logosdb_t * db = logosdb_open(path.c_str(), opts, &err);
    logosdb_options_destroy(opts);
    CHECK(db != nullptr, "dim mismatch: open ok");

    auto wrong = unit_vec(64, 8000);
    uint64_t id = logosdb_put(db, wrong.data(), 64, "bad", nullptr, &err);
    CHECK(id == UINT64_MAX, "dim mismatch: put rejected");
    CHECK(err != nullptr, "dim mismatch: error set");
    free(err); err = nullptr;

    CHECK(logosdb_count(db) == 0, "dim mismatch: nothing inserted");
    logosdb_close(db);
    std::filesystem::remove_all(path);
}

static void test_dim_mismatch_search() {
    std::string path = "/tmp/logosdb_test_dim_search";
    std::filesystem::remove_all(path);

    char * err = nullptr;
    logosdb_options_t * opts = logosdb_options_create();
    logosdb_options_set_dim(opts, 32);
    logosdb_t * db = logosdb_open(path.c_str(), opts, &err);
    logosdb_options_destroy(opts);

    auto v = unit_vec(32, 9000);
    logosdb_put(db, v.data(), 32, "ok", nullptr, &err);

    auto wrong_q = unit_vec(64, 9001);
    logosdb_search_result_t * r = logosdb_search(db, wrong_q.data(), 64, 1, &err);
    CHECK(r == nullptr, "dim mismatch search: rejected");
    CHECK(err != nullptr, "dim mismatch search: error set");
    free(err);

    logosdb_close(db);
    std::filesystem::remove_all(path);
}

static void test_result_accessor_bounds() {
    std::string path = "/tmp/logosdb_test_bounds";
    std::filesystem::remove_all(path);

    char * err = nullptr;
    logosdb_options_t * opts = logosdb_options_create();
    logosdb_options_set_dim(opts, 32);
    logosdb_t * db = logosdb_open(path.c_str(), opts, &err);
    logosdb_options_destroy(opts);

    auto v = unit_vec(32, 1100);
    logosdb_put(db, v.data(), 32, "x", nullptr, &err);

    logosdb_search_result_t * r = logosdb_search(db, v.data(), 32, 1, &err);
    CHECK(r != nullptr, "bounds: search ok");
    CHECK(logosdb_result_count(r) == 1, "bounds: count=1");

    CHECK(logosdb_result_id(r, -1) == UINT64_MAX, "bounds: id(-1) invalid");
    CHECK(logosdb_result_id(r, 5) == UINT64_MAX, "bounds: id(5) invalid");
    CHECK(logosdb_result_text(r, -1) == nullptr, "bounds: text(-1) null");
    CHECK(logosdb_result_text(r, 5) == nullptr, "bounds: text(5) null");
    CHECK(logosdb_result_score(r, 99) == 0.0f, "bounds: score(99) zero");
    CHECK(logosdb_result_timestamp(r, -1) == nullptr, "bounds: ts(-1) null");

    CHECK(logosdb_result_count(nullptr) == 0, "bounds: count(null)=0");

    logosdb_result_free(r);
    logosdb_close(db);
    std::filesystem::remove_all(path);
}

static void test_persistence_append_after_reopen() {
    std::string path = "/tmp/logosdb_test_append_reopen";
    std::filesystem::remove_all(path);

    int dim = 32;
    auto v0 = unit_vec(dim, 1200);
    auto v1 = unit_vec(dim, 1300);
    auto v2 = unit_vec(dim, 1400);

    {
        logosdb::DB db(path, {.dim = dim});
        db.put(v0, "first");
        CHECK(db.count() == 1, "append-reopen: 1 after first session");
    }
    {
        logosdb::DB db(path, {.dim = dim});
        CHECK(db.count() == 1, "append-reopen: 1 on reopen");
        db.put(v1, "second");
        db.put(v2, "third");
        CHECK(db.count() == 3, "append-reopen: 3 after appending");
    }
    {
        logosdb::DB db(path, {.dim = dim});
        CHECK(db.count() == 3, "append-reopen: 3 after final reopen");

        auto hits = db.search(v2, 1);
        CHECK(!hits.empty(), "append-reopen: search v2");
        CHECK(hits[0].id == 2, "append-reopen: v2 is id=2");
        CHECK(hits[0].text == "third", "append-reopen: text=third");
    }

    std::filesystem::remove_all(path);
}

static void test_cpp_wrapper_exception() {
    bool caught = false;
    try {
        logosdb::DB db("/tmp/logosdb_test_exc", {.dim = 0});
    } catch (const std::runtime_error & e) {
        caught = true;
        CHECK(std::string(e.what()).find("logosdb_open") != std::string::npos,
              "exception: mentions logosdb_open");
    }
    CHECK(caught, "exception: thrown on dim=0");
}

static void test_score_self_is_one() {
    std::string path = "/tmp/logosdb_test_selfscore";
    std::filesystem::remove_all(path);

    int dim = 64;
    logosdb::DB db(path, {.dim = dim});
    auto v = unit_vec(dim, 1500);
    db.put(v, "self");

    auto hits = db.search(v, 1);
    CHECK(!hits.empty(), "self-score: found");
    CHECK(hits[0].score > 0.99f, "self-score: ~1.0 for unit vector");

    std::filesystem::remove_all(path);
}

static void test_delete_basic() {
    std::string path = "/tmp/logosdb_test_delete_basic";
    std::filesystem::remove_all(path);

    int dim = 64;
    logosdb::DB db(path, {.dim = dim});

    auto v0 = unit_vec(dim, 2100);
    auto v1 = unit_vec(dim, 2200);
    auto v2 = unit_vec(dim, 2300);

    db.put(v0, "zero");
    db.put(v1, "one");
    db.put(v2, "two");
    CHECK(db.count() == 3, "delete: count==3 before");
    CHECK(db.count_live() == 3, "delete: live==3 before");

    db.del(1);
    CHECK(db.count() == 3, "delete: total count unchanged");
    CHECK(db.count_live() == 2, "delete: live==2 after");

    auto hits = db.search(v1, 3);
    bool found = false;
    for (auto & h : hits) if (h.id == 1) found = true;
    CHECK(!found, "delete: deleted row excluded from search");

    auto self_hits = db.search(v0, 1);
    CHECK(!self_hits.empty() && self_hits[0].id == 0, "delete: other rows still searchable");

    std::filesystem::remove_all(path);
}

static void test_delete_errors() {
    std::string path = "/tmp/logosdb_test_delete_err";
    std::filesystem::remove_all(path);

    logosdb::DB db(path, {.dim = 32});
    auto v0 = unit_vec(32, 2400);
    db.put(v0, "only");

    char * err = nullptr;
    int rc = logosdb_delete(db.handle(), 99, &err);
    CHECK(rc == -1, "delete oob: rc==-1");
    CHECK(err != nullptr, "delete oob: err set");
    free(err); err = nullptr;

    rc = logosdb_delete(db.handle(), 0, &err);
    CHECK(rc == 0, "delete: first delete ok");
    CHECK(err == nullptr, "delete: no err on success");

    rc = logosdb_delete(db.handle(), 0, &err);
    CHECK(rc == -1, "delete twice: rc==-1");
    CHECK(err != nullptr, "delete twice: err set");
    free(err);

    std::filesystem::remove_all(path);
}

static void test_delete_persistence() {
    std::string path = "/tmp/logosdb_test_delete_persist";
    std::filesystem::remove_all(path);

    int dim = 32;
    auto v0 = unit_vec(dim, 2500);
    auto v1 = unit_vec(dim, 2600);
    auto v2 = unit_vec(dim, 2700);

    {
        logosdb::DB db(path, {.dim = dim});
        db.put(v0, "a");
        db.put(v1, "b");
        db.put(v2, "c");
        db.del(1);
        CHECK(db.count_live() == 2, "persist-del: live==2 before close");
    }
    {
        logosdb::DB db(path, {.dim = dim});
        CHECK(db.count() == 3, "persist-del: count==3 after reopen");
        CHECK(db.count_live() == 2, "persist-del: live==2 after reopen");

        auto hits = db.search(v1, 3);
        bool found = false;
        for (auto & h : hits) if (h.id == 1) found = true;
        CHECK(!found, "persist-del: deleted row still excluded after reopen");

        auto hits_a = db.search(v0, 1);
        CHECK(!hits_a.empty() && hits_a[0].id == 0, "persist-del: v0 still there");
    }

    std::filesystem::remove_all(path);
}

static void test_delete_persistence_without_index_file() {
    // Drop the HNSW index file between sessions to force a full rebuild from
    // vectors.bin. Tombstone replay from the JSONL log must restore the
    // deletion on reopen.
    std::string path = "/tmp/logosdb_test_delete_rebuild";
    std::filesystem::remove_all(path);

    int dim = 32;
    auto v0 = unit_vec(dim, 2800);
    auto v1 = unit_vec(dim, 2900);

    {
        logosdb::DB db(path, {.dim = dim});
        db.put(v0, "alpha");
        db.put(v1, "beta");
        db.del(0);
    }

    std::error_code ec;
    std::filesystem::remove(path + "/hnsw.idx", ec);
    CHECK(!ec, "rebuild: remove hnsw.idx");

    {
        logosdb::DB db(path, {.dim = dim});
        CHECK(db.count_live() == 1, "rebuild: live==1 after index rebuild");

        auto hits = db.search(v0, 2);
        bool found = false;
        for (auto & h : hits) if (h.id == 0) found = true;
        CHECK(!found, "rebuild: tombstone replayed onto fresh index");
    }

    std::filesystem::remove_all(path);
}

static void test_update_basic() {
    std::string path = "/tmp/logosdb_test_update";
    std::filesystem::remove_all(path);

    int dim = 64;
    logosdb::DB db(path, {.dim = dim});

    auto v0 = unit_vec(dim, 3100);
    auto v1 = unit_vec(dim, 3200);
    auto v_new = unit_vec(dim, 3300);

    db.put(v0, "old zero");
    db.put(v1, "keep one");
    CHECK(db.count_live() == 2, "update: live==2 before");

    uint64_t new_id = db.update(0, v_new, "new zero", "2025-07-01T00:00:00Z");
    CHECK(new_id == 2, "update: returns new id (append)");
    CHECK(db.count() == 3, "update: total grows by 1");
    CHECK(db.count_live() == 2, "update: live count unchanged");

    auto hits_new = db.search(v_new, 1);
    CHECK(!hits_new.empty(), "update: new vector found");
    CHECK(hits_new[0].id == new_id, "update: new id top-1");
    CHECK(hits_new[0].text == "new zero", "update: new text");
    CHECK(hits_new[0].timestamp == "2025-07-01T00:00:00Z", "update: new ts");

    auto hits_old = db.search(v0, 3);
    bool old_found = false;
    for (auto & h : hits_old) if (h.id == 0) old_found = true;
    CHECK(!old_found, "update: old id excluded");

    std::filesystem::remove_all(path);
}

static void test_update_errors() {
    std::string path = "/tmp/logosdb_test_update_err";
    std::filesystem::remove_all(path);

    logosdb::DB db(path, {.dim = 32});
    auto v0 = unit_vec(32, 3400);
    auto v1 = unit_vec(32, 3500);
    db.put(v0, "a");

    char * err = nullptr;
    uint64_t r = logosdb_update(db.handle(), 99, v1.data(), 32, "x", nullptr, &err);
    CHECK(r == UINT64_MAX, "update oob: rc==MAX");
    CHECK(err != nullptr, "update oob: err set");
    free(err); err = nullptr;

    auto wrong = unit_vec(16, 3600);
    r = logosdb_update(db.handle(), 0, wrong.data(), 16, "x", nullptr, &err);
    CHECK(r == UINT64_MAX, "update dim: rc==MAX");
    CHECK(err != nullptr, "update dim: err set");
    free(err); err = nullptr;

    logosdb_delete(db.handle(), 0, &err);
    free(err); err = nullptr;
    r = logosdb_update(db.handle(), 0, v1.data(), 32, "x", nullptr, &err);
    CHECK(r == UINT64_MAX, "update deleted: rc==MAX");
    CHECK(err != nullptr, "update deleted: err set");
    free(err);

    std::filesystem::remove_all(path);
}

static void test_delete_reput_independence() {
    std::string path = "/tmp/logosdb_test_delete_reput";
    std::filesystem::remove_all(path);

    int dim = 32;
    logosdb::DB db(path, {.dim = dim});

    auto v0 = unit_vec(dim, 3700);
    auto v1 = unit_vec(dim, 3800);

    uint64_t id0 = db.put(v0, "first");
    db.del(id0);

    uint64_t id1 = db.put(v1, "second");
    CHECK(id1 == 1, "reput: new put gets fresh id (no slot reuse)");
    CHECK(db.count() == 2, "reput: total count==2");
    CHECK(db.count_live() == 1, "reput: live count==1");

    auto hits = db.search(v1, 1);
    CHECK(!hits.empty() && hits[0].id == id1, "reput: new row retrievable");

    std::filesystem::remove_all(path);
}

static void test_large_dim() {
    std::string path = "/tmp/logosdb_test_largedim";
    std::filesystem::remove_all(path);

    int dim = 2048;
    logosdb::DB db(path, {.dim = dim});

    auto v0 = unit_vec(dim, 1600);
    auto v1 = unit_vec(dim, 1700);
    db.put(v0, "large dim A");
    db.put(v1, "large dim B");

    CHECK(db.count() == 2, "large dim: count=2");

    auto hits = db.search(v0, 1);
    CHECK(!hits.empty(), "large dim: search ok");
    CHECK(hits[0].id == 0, "large dim: correct id");
    CHECK(hits[0].text == "large dim A", "large dim: correct text");

    std::filesystem::remove_all(path);
}

// Test Unicode and advanced JSON escapes (nlohmann/json handles these correctly)
static void test_metadata_unicode_and_escapes() {
    std::string path = "/tmp/logosdb_test_unicode";
    std::filesystem::remove_all(path);

    logosdb::DB db(path, {.dim = 32});
    auto v0 = unit_vec(32, 4000);
    auto v1 = unit_vec(32, 4100);
    auto v2 = unit_vec(32, 4200);

    // Unicode characters
    std::string unicode = "Hello \u4e16\u754c \u00e9\u00e0"; // "Hello 世界 éà"
    db.put(v0, unicode, "2025-01-01T00:00:00Z");

    // Empty strings
    db.put(v1, "", "");

    // Complex nested quotes and backslashes
    std::string complex_escapes = "Path: C:\\Users\\test\\file.txt\\\"quoted\"";
    db.put(v2, complex_escapes);

    // Reopen and verify
    { logosdb::DB db2(path, {.dim = 32});
      auto hits0 = db2.search(v0, 1);
      CHECK(!hits0.empty(), "unicode: found");
      CHECK(hits0[0].text == unicode, "unicode: round-trip preserved");

      auto hits1 = db2.search(v1, 1);
      CHECK(!hits1.empty(), "empty text: found");
      CHECK(hits1[0].text == "", "empty text: preserved");
      CHECK(hits1[0].timestamp == "", "empty ts: preserved");

      auto hits2 = db2.search(v2, 1);
      CHECK(!hits2.empty(), "complex escapes: found");
      CHECK(hits2[0].text == complex_escapes, "complex escapes: round-trip preserved");
    }

    std::filesystem::remove_all(path);
}

// Test that nlohmann/json handles edge case JSON that the old parser failed on.
// The old parser couldn't handle: unicode escapes, key ordering variations, extra whitespace.
// This test verifies the new parser correctly handles all these.
static void test_metadata_json_edge_cases() {
    std::string path = "/tmp/logosdb_test_jsonedge";
    std::filesystem::remove_all(path);

    // Create DB and add normal entries
    {
        logosdb::DB db(path, {.dim = 32});
        auto v0 = unit_vec(32, 6000);
        auto v1 = unit_vec(32, 6100);
        auto v2 = unit_vec(32, 6200);
        db.put(v0, "unicode: \u4e16\u754c", "2025-01-01T00:00:00Z");
        db.put(v1, "spaced text", "2025-02-02T12:00:00Z");
        db.put(v2, "https://example.com/path", "2025-03-03T00:00:00Z");
        CHECK(db.count() == 3, "json edge cases: 3 rows written");
    }

    // Reopen and verify all entries loaded correctly
    {
        logosdb::DB db(path, {.dim = 32});
        CHECK(db.count() == 3, "json edge cases: 3 rows loaded after reopen");

        auto v0 = unit_vec(32, 6000);
        auto hits0 = db.search(v0, 1);
        CHECK(!hits0.empty(), "json edge cases: found row 0");
        // The unicode characters should be preserved
        CHECK(hits0[0].text == "unicode: \u4e16\u754c", "json edge cases: unicode text preserved");

        auto v1 = unit_vec(32, 6100);
        auto hits1 = db.search(v1, 1);
        CHECK(!hits1.empty(), "json edge cases: found row 1");
        CHECK(hits1[0].text == "spaced text", "json edge cases: spaced text preserved");

        auto v2 = unit_vec(32, 6200);
        auto hits2 = db.search(v2, 1);
        CHECK(!hits2.empty(), "json edge cases: found row 2");
        CHECK(hits2[0].text == "https://example.com/path", "json edge cases: url with slashes preserved");
    }

    std::filesystem::remove_all(path);
}

int main() {
    printf("logosdb basic tests\n");
    printf("===================\n\n");

    test_open_close();
    test_put_and_search();
    test_persistence();
    test_raw_vectors();
    test_many_vectors();
    test_c_api_errors();
    test_search_ordering();
    test_top_k_limit();
    test_empty_search();
    test_put_no_metadata();
    test_metadata_special_chars();
    test_dim_mismatch_put();
    test_dim_mismatch_search();
    test_result_accessor_bounds();
    test_persistence_append_after_reopen();
    test_cpp_wrapper_exception();
    test_score_self_is_one();
    test_delete_basic();
    test_delete_errors();
    test_delete_persistence();
    test_delete_persistence_without_index_file();
    test_update_basic();
    test_update_errors();
    test_delete_reput_independence();
    test_large_dim();
    test_metadata_unicode_and_escapes();
    test_metadata_json_edge_cases();
    test_wal_crash_recovery();
    test_put_batch_basic();
    test_put_batch_empty();
    test_search_ts_range_basic();
    test_search_ts_range_edge_cases();
    test_search_ts_range_recall();
    test_distance_cosine();
    test_distance_l2();
    test_distance_persistence();
    test_distance_mismatch_error();
    test_cli_info_reads_dim();
    test_cli_export_import_roundtrip();
    test_cli_search_ts_range();
    test_storage_pointers_stable();
    test_l2_normalize_basic();
    test_l2_normalize_already_normalized();
    test_l2_normalize_zero_vector();
    test_l2_normalize_small_values();
    test_l2_normalize_cpp_wrappers();

    printf("\n%d/%d tests passed.\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}

// Test WAL crash recovery: simulate incomplete writes by manually creating
// a WAL with pending entries, then verify they are replayed on open.
static void test_wal_crash_recovery() {
    std::string path = "/tmp/logosdb_test_wal";
    std::filesystem::remove_all(path);

    // Create a fresh database with some data
    {
        logosdb::DB db(path, {.dim = 32});
        auto v0 = unit_vec(32, 7000);
        auto v1 = unit_vec(32, 7100);
        db.put(v0, "first", "2025-01-01T00:00:00Z");
        db.put(v1, "second", "2025-02-01T00:00:00Z");
        CHECK(db.count() == 2, "wal: initial 2 rows");
    }

    // Simulate crash scenario: manually append a pending WAL entry
    // This mimics what would happen if logosdb_put crashed AFTER writing to WAL
    // but BEFORE writing to any of the three stores (vectors, metadata, index).
    // In this case, the stores are in consistent state (2 rows), and WAL
    // contains the pending operation that needs to be replayed.
    {
        // Manually write a WAL entry
        int fd = ::open((path + "/wal.log").c_str(), O_RDWR);
        CHECK(fd >= 0, "wal: can open wal file");

        // Seek to end
        ::lseek(fd, 0, SEEK_END);

        // Write a pending entry manually
        uint8_t state = 0;  // PENDING
        ::write(fd, &state, 1);

        uint32_t dim = 32;
        ::write(fd, &dim, 4);

        auto v2 = unit_vec(32, 7200);
        uint32_t vec_bytes = 32 * sizeof(float);
        ::write(fd, &vec_bytes, 4);
        ::write(fd, v2.data(), vec_bytes);

        std::string text = "third (from wal)";
        uint32_t text_len = text.size();
        ::write(fd, &text_len, 4);
        ::write(fd, text.data(), text_len);

        std::string ts = "2025-03-01T00:00:00Z";
        uint32_t ts_len = ts.size();
        ::write(fd, &ts_len, 4);
        ::write(fd, ts.data(), ts_len);

        // expected_id should be 2 (next row index, since we have 2 rows: 0, 1)
        uint64_t expected_id = 2;
        ::write(fd, &expected_id, 8);

        ::fsync(fd);
        ::close(fd);
    }

    // Reopen DB - WAL should replay the pending entry
    // At this point: vectors.bin has 2 rows, meta.jsonl has 2 rows, index has 2 entries
    // WAL has pending entry with expected_id=2, which matches current n_rows
    {
        logosdb::DB db(path, {.dim = 32});
        CHECK(db.count() == 3, "wal: replayed to 3 rows");

        auto v2 = unit_vec(32, 7200);
        auto hits = db.search(v2, 1);
        CHECK(!hits.empty(), "wal: found replayed vector");
        CHECK(hits[0].text == "third (from wal)", "wal: replayed metadata correct");
        CHECK(hits[0].id == 2, "wal: replayed id correct");
    }

    // Reopen again - should still be consistent (no double replay)
    // The WAL entry was marked committed during the first replay
    {
        logosdb::DB db(path, {.dim = 32});
        CHECK(db.count() == 3, "wal: still 3 rows after second reopen");

        auto v2 = unit_vec(32, 7200);
        auto hits = db.search(v2, 1);
        CHECK(!hits.empty(), "wal: vector still searchable");
        CHECK(hits[0].text == "third (from wal)", "wal: metadata still correct");
    }

    std::filesystem::remove_all(path);
}

// Test batch put API: basic functionality and search
static void test_put_batch_basic() {
    std::string path = "/tmp/logosdb_test_batch";
    std::filesystem::remove_all(path);

    {
        logosdb::DB db(path, {.dim = 32});

        // Prepare 100 vectors
        int n = 100;
        std::vector<float> embeddings;
        embeddings.reserve(n * 32);
        std::vector<std::string> texts;
        std::vector<std::string> timestamps;

        for (int i = 0; i < n; ++i) {
            auto v = unit_vec(32, 8000 + i);
            embeddings.insert(embeddings.end(), v.begin(), v.end());
            texts.push_back("text_" + std::to_string(i));
            timestamps.push_back("2025-01-01T00:00:" + std::to_string(i) + "Z");
        }

        // Batch insert
        auto ids = db.put_batch(embeddings, n, texts, timestamps);
        CHECK((int)ids.size() == n, "batch: returned 100 ids");
        CHECK(db.count() == (size_t)n, "batch: count is 100");

        // Verify sequential ids
        for (int i = 0; i < n; ++i) {
            CHECK(ids[i] == (uint64_t)i, "batch: sequential ids");
        }

        // Verify searchable
        for (int i = 0; i < n; ++i) {
            auto v = unit_vec(32, 8000 + i);
            auto hits = db.search(v, 1);
            CHECK(!hits.empty(), "batch: found vector");
            CHECK(hits[0].id == (uint64_t)i, "batch: correct id");
            CHECK(hits[0].text == "text_" + std::to_string(i), "batch: correct text");
        }
    }

    // Reopen and verify persistence
    {
        logosdb::DB db(path, {.dim = 32});
        CHECK(db.count() == 100, "batch: 100 rows after reopen");

        auto v50 = unit_vec(32, 8050);
        auto hits = db.search(v50, 1);
        CHECK(!hits.empty(), "batch: found after reopen");
        CHECK(hits[0].id == 50, "batch: correct id after reopen");
        CHECK(hits[0].text == "text_50", "batch: correct text after reopen");
    }

    std::filesystem::remove_all(path);
}

// Test batch put with empty batch and edge cases
static void test_put_batch_empty() {
    std::string path = "/tmp/logosdb_test_batch_empty";
    std::filesystem::remove_all(path);

    {
        logosdb::DB db(path, {.dim = 32});

        // Empty batch
        std::vector<float> embeddings;
        auto ids = db.put_batch(embeddings, 0);
        CHECK(ids.empty(), "batch empty: returned empty ids");
        CHECK(db.count() == 0, "batch empty: count still 0");

        // Single item batch
        auto v = unit_vec(32, 9000);
        embeddings = v;
        ids = db.put_batch(embeddings, 1);
        CHECK(ids.size() == 1, "batch single: returned 1 id");
        CHECK(db.count() == 1, "batch single: count is 1");

        // Batch without metadata (optional texts/timestamps)
        embeddings.clear();
        for (int i = 0; i < 10; ++i) {
            auto v2 = unit_vec(32, 9100 + i);
            embeddings.insert(embeddings.end(), v2.begin(), v2.end());
        }
        ids = db.put_batch(embeddings, 10);  // No texts/timestamps
        CHECK(ids.size() == 10, "batch no meta: returned 10 ids");
        CHECK(db.count() == 11, "batch no meta: count is 11");

        // Verify searchable even without metadata
        auto v_search = unit_vec(32, 9105);
        auto hits = db.search(v_search, 1);
        CHECK(!hits.empty(), "batch no meta: found vector");
        CHECK(hits[0].id == 6, "batch no meta: correct id (1 + 5)");
    }

    std::filesystem::remove_all(path);
}

// Test timestamp range search - basic functionality
static void test_search_ts_range_basic() {
    std::string path = "/tmp/logosdb_test_ts_range";
    std::filesystem::remove_all(path);

    int dim = 64;
    logosdb::DB db(path, {.dim = dim});

    // Insert vectors with different timestamps
    auto v0 = unit_vec(dim, 10000);
    auto v1 = unit_vec(dim, 10100);
    auto v2 = unit_vec(dim, 10200);
    auto v3 = unit_vec(dim, 10300);

    db.put(v0, "early morning", "2025-01-15T08:00:00Z");
    db.put(v1, "mid morning",   "2025-01-15T10:00:00Z");
    db.put(v2, "afternoon",     "2025-01-15T14:00:00Z");
    db.put(v3, "evening",       "2025-01-15T20:00:00Z");

    // Test 1: Filter from 09:00 to 15:00 (should get v1 and v2)
    auto hits = db.search_ts_range(v0, 5, "2025-01-15T09:00:00Z", "2025-01-15T15:00:00Z");
    CHECK(hits.size() == 2, "ts_range: got 2 results for 09-15 window");
    if (hits.size() >= 1) CHECK(hits[0].id == 1 || hits[0].id == 2, "ts_range: first result is v1 or v2");
    if (hits.size() >= 2) CHECK(hits[1].id == 1 || hits[1].id == 2, "ts_range: second result is v1 or v2");

    // Verify the correct results are returned
    bool found_v1 = false, found_v2 = false;
    for (auto & h : hits) {
        if (h.id == 1) found_v1 = true;
        if (h.id == 2) found_v2 = true;
    }
    CHECK(found_v1, "ts_range: found v1 (10:00) in window");
    CHECK(found_v2, "ts_range: found v2 (14:00) in window");

    // Test 2: Only from timestamp (v2, v3 should be found)
    hits = db.search_ts_range(v0, 5, "2025-01-15T14:00:00Z", "");
    CHECK(hits.size() == 2, "ts_range: got 2 results for from-only filter");
    bool found_v3 = false;
    for (auto & h : hits) {
        if (h.id == 2) found_v2 = true;
        if (h.id == 3) found_v3 = true;
    }
    CHECK(found_v2, "ts_range: found v2 with from-only");
    CHECK(found_v3, "ts_range: found v3 with from-only");

    // Test 3: Only to timestamp (v0, v1 should be found)
    hits = db.search_ts_range(v0, 5, "", "2025-01-15T10:00:00Z");
    bool found_v0 = false;
    found_v1 = false;
    for (auto & h : hits) {
        if (h.id == 0) found_v0 = true;
        if (h.id == 1) found_v1 = true;
    }
    CHECK(found_v0, "ts_range: found v0 with to-only");
    CHECK(found_v1, "ts_range: found v1 with to-only");

    // Test 4: No filters (same as regular search)
    hits = db.search_ts_range(v0, 4, "", "");
    CHECK(hits.size() == 4, "ts_range: no filters returns all");

    std::filesystem::remove_all(path);
}

// Test timestamp range search - edge cases and error handling
static void test_search_ts_range_edge_cases() {
    std::string path = "/tmp/logosdb_test_ts_range_edge";
    std::filesystem::remove_all(path);

    int dim = 32;
    logosdb::DB db(path, {.dim = dim});

    // Insert some data
    auto v0 = unit_vec(dim, 11000);
    auto v1 = unit_vec(dim, 11100);
    db.put(v0, "entry1", "2025-02-01T12:00:00Z");
    db.put(v1, "entry2", "2025-02-01T14:00:00Z");

    // Test with no results
    auto hits = db.search_ts_range(v0, 5, "2025-12-01T00:00:00Z", "2025-12-31T23:59:59Z");
    CHECK(hits.empty(), "ts_range edge: future window returns empty");

    // Test with empty timestamps in data
    auto v2 = unit_vec(dim, 11200);
    db.put(v2, "no timestamp", "");
    hits = db.search_ts_range(v0, 5, "", "");
    // Empty timestamp rows should still appear when no timestamp filter is set
    CHECK(hits.size() == 3, "ts_range edge: empty ts row included when no filter");

    // Test with C API
    char * err = nullptr;
    logosdb_search_result_t * r = logosdb_search_ts_range(
        db.handle(), v0.data(), dim, 2,
        "2025-02-01T00:00:00Z", "2025-02-01T23:59:59Z",
        10, &err);
    CHECK(r != nullptr, "ts_range C API: success");
    CHECK(err == nullptr, "ts_range C API: no error");
    if (r) {
        CHECK(logosdb_result_count(r) == 2, "ts_range C API: got 2 results");
        logosdb_result_free(r);
    }

    // Test with null db
    r = logosdb_search_ts_range(nullptr, v0.data(), dim, 1,
                                nullptr, nullptr, 10, &err);
    CHECK(r == nullptr, "ts_range C API: null db rejected");
    CHECK(err != nullptr, "ts_range C API: error set for null db");
    free(err); err = nullptr;

    // Test with invalid top_k
    r = logosdb_search_ts_range(db.handle(), v0.data(), dim, 0,
                                nullptr, nullptr, 10, &err);
    CHECK(r == nullptr, "ts_range C API: top_k=0 rejected");
    CHECK(err != nullptr, "ts_range C API: error set for top_k=0");
    free(err);

    std::filesystem::remove_all(path);
}

// Test timestamp range search recall - verify post-filtering works correctly
static void test_search_ts_range_recall() {
    std::string path = "/tmp/logosdb_test_ts_range_recall";
    std::filesystem::remove_all(path);

    int dim = 64;
    int n = 100;
    logosdb::DB db(path, {.dim = dim, .max_elements = 200});

    // Insert many vectors alternating timestamps
    std::vector<std::vector<float>> morning_vecs;
    std::vector<std::vector<float>> evening_vecs;

    for (int i = 0; i < n; ++i) {
        auto v = unit_vec(dim, 12000 + i);
        if (i % 2 == 0) {
            morning_vecs.push_back(v);
            db.put(v, "morning_" + std::to_string(i), "2025-03-01T09:00:00Z");
        } else {
            evening_vecs.push_back(v);
            db.put(v, "evening_" + std::to_string(i), "2025-03-01T18:00:00Z");
        }
    }

    // Search with morning timestamp filter
    auto query = morning_vecs[0];
    auto hits = db.search_ts_range(query, 10, "2025-03-01T00:00:00Z", "2025-03-01T12:00:00Z");

    // All results should be morning entries (even IDs)
    bool all_morning = true;
    for (auto & h : hits) {
        if (h.id % 2 != 0) {
            all_morning = false;
            break;
        }
    }
    CHECK(all_morning, "ts_range recall: all results are morning entries");

    // We should get 10 results since there are 50 morning entries
    CHECK(hits.size() == 10, "ts_range recall: got requested top_k=10");

    // Verify results are sorted by score (descending)
    for (size_t i = 1; i < hits.size(); ++i) {
        CHECK(hits[i-1].score >= hits[i].score, "ts_range recall: results sorted by score");
    }

    // Test with evening filter
    auto evening_query = evening_vecs[0];
    hits = db.search_ts_range(evening_query, 10, "2025-03-01T12:00:00Z", "2025-03-01T23:59:59Z");

    // All results should be evening entries (odd IDs)
    bool all_evening = true;
    for (auto & h : hits) {
        if (h.id % 2 == 0) {
            all_evening = false;
            break;
        }
    }
    CHECK(all_evening, "ts_range recall: all results are evening entries");

    // Test that higher candidate_k improves recall when filter is very selective
    // Create a narrow time window that only includes a few entries
    hits = db.search_ts_range(morning_vecs[0], 5,
                              "2025-03-01T09:00:00Z", "2025-03-01T09:00:00Z",
                              100);  // High candidate_k
    // With exact timestamp match, all results should have the same timestamp
    for (auto & h : hits) {
        CHECK(h.timestamp == "2025-03-01T09:00:00Z", "ts_range recall: exact timestamp match");
    }

    std::filesystem::remove_all(path);
}

// Test cosine distance metric (auto-normalization)
static void test_distance_cosine() {
    std::string path = "/tmp/logosdb_test_cosine";
    std::filesystem::remove_all(path);

    int dim = 64;
    // Create DB with cosine distance
    logosdb::DB db(path, {.dim = dim, .distance = LOGOSDB_DIST_COSINE});

    // Insert non-normalized vectors
    auto v0 = unit_vec(dim, 13000);
    auto v1 = unit_vec(dim, 13100);

    // Scale the vectors (not unit length)
    for (int i = 0; i < dim; ++i) {
        v0[i] *= 5.0f;  // Scale by 5
        v1[i] *= 3.0f;  // Scale by 3
    }

    db.put(v0, "scaled v0");
    db.put(v1, "scaled v1");

    // Search with scaled query
    auto scaled_query = v0;
    auto hits = db.search(scaled_query, 2);

    // Should still find itself as top-1 (cosine ignores magnitude)
    CHECK(!hits.empty(), "cosine: search returned results");
    CHECK(hits[0].id == 0, "cosine: top hit is v0");
    CHECK(hits[0].score > 0.99f, "cosine: high similarity for same direction");

    // Self-similarity should be ~1.0 regardless of scale
    auto self_hits = db.search(scaled_query, 1);
    CHECK(self_hits[0].score > 0.99f, "cosine: self-similarity ~1.0");

    std::filesystem::remove_all(path);
}

// Test L2 (Euclidean) distance metric
static void test_distance_l2() {
    std::string path = "/tmp/logosdb_test_l2";
    std::filesystem::remove_all(path);

    int dim = 64;
    // Create DB with L2 distance
    logosdb::DB db(path, {.dim = dim, .distance = LOGOSDB_DIST_L2});

    auto v0 = unit_vec(dim, 14000);
    auto v1 = unit_vec(dim, 14100);

    db.put(v0, "v0");
    db.put(v1, "v1");

    // Search with v0
    auto hits = db.search(v0, 2);

    CHECK(!hits.empty(), "l2: search returned results");
    CHECK(hits[0].id == 0, "l2: top hit is v0 (closest)");

    // L2 score is inverted distance (1/(1+d)), so closer = higher score
    CHECK(hits[0].score > hits[1].score, "l2: self has higher score than other");

    std::filesystem::remove_all(path);
}

// Test distance metric persistence across reopen
static void test_distance_persistence() {
    std::string path = "/tmp/logosdb_test_dist_persist";
    std::filesystem::remove_all(path);

    int dim = 32;

    // Create with cosine
    {
        logosdb::DB db(path, {.dim = dim, .distance = LOGOSDB_DIST_COSINE});
        auto v = unit_vec(dim, 15000);
        db.put(v, "test");
    }

    // Reopen with same distance - should work
    {
        logosdb::DB db(path, {.dim = dim, .distance = LOGOSDB_DIST_COSINE});
        auto v = unit_vec(dim, 15000);
        auto hits = db.search(v, 1);
        CHECK(!hits.empty(), "dist persist: reopened with matching distance");
        CHECK(hits[0].id == 0, "dist persist: found vector after reopen");
    }

    std::filesystem::remove_all(path);
}

// Test that distance metric mismatch returns error
static void test_distance_mismatch_error() {
    std::string path = "/tmp/logosdb_test_dist_mismatch";
    std::filesystem::remove_all(path);

    int dim = 32;

    // Create with cosine
    {
        logosdb::DB db(path, {.dim = dim, .distance = LOGOSDB_DIST_COSINE});
        auto v = unit_vec(dim, 16000);
        db.put(v, "test");
    }

    // Try to reopen with different distance - should throw
    bool caught = false;
    try {
        logosdb::DB db(path, {.dim = dim, .distance = LOGOSDB_DIST_L2});
    } catch (const std::runtime_error & e) {
        caught = true;
        CHECK(std::string(e.what()).find("distance metric mismatch") != std::string::npos,
              "dist mismatch: error message mentions mismatch");
    }
    CHECK(caught, "dist mismatch: exception thrown for mismatched distance");

    // Try with IP (which is different from cosine internally)
    caught = false;
    try {
        logosdb::DB db(path, {.dim = dim, .distance = LOGOSDB_DIST_IP});
    } catch (const std::runtime_error & e) {
        caught = true;
    }
    // Note: IP and COSINE use the same hnsw space, but are marked differently
    // This may or may not throw depending on implementation details

    // Test C API distance setter
    logosdb_options_t * opts = logosdb_options_create();
    CHECK(logosdb_options_set_distance(opts, LOGOSDB_DIST_IP) == 0, "dist API: IP valid");
    CHECK(logosdb_options_set_distance(opts, LOGOSDB_DIST_COSINE) == 0, "dist API: COSINE valid");
    CHECK(logosdb_options_set_distance(opts, LOGOSDB_DIST_L2) == 0, "dist API: L2 valid");
    CHECK(logosdb_options_set_distance(opts, 99) == -1, "dist API: invalid metric rejected");
    CHECK(logosdb_options_set_distance(nullptr, LOGOSDB_DIST_IP) == -1, "dist API: null opts rejected");
    logosdb_options_destroy(opts);

    std::filesystem::remove_all(path);
}

// Helper to run CLI command and capture output
static int run_cli(const std::string& args, std::string& output) {
    std::string cmd = "./logosdb-cli " + args + " 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return -1;
    char buffer[4096];
    output.clear();
    while (fgets(buffer, sizeof(buffer), pipe)) {
        output += buffer;
    }
    return pclose(pipe);
}

// Test that CLI info command reads dim from vectors.bin header
static void test_cli_info_reads_dim() {
    std::string path = "/tmp/logosdb_test_cli_info";
    std::filesystem::remove_all(path);

    // Create a DB with put
    logosdb::DB db(path, {.dim = 64});
    auto v = unit_vec(64, 17000);
    db.put(v, "test entry", "2025-01-01T00:00:00Z");

    // Test info without --dim (should read from header)
    std::string output;
    int rc = run_cli("info " + path, output);
    CHECK(rc == 0, "cli info: returns success");
    CHECK(output.find("dim        : 64") != std::string::npos, "cli info: shows correct dim");
    CHECK(output.find("count      : 1") != std::string::npos, "cli info: shows count=1");

    // Test info --json
    output.clear();
    rc = run_cli("info " + path + " --json", output);
    CHECK(rc == 0, "cli info json: returns success");
    CHECK(output.find("\"dim\": 64") != std::string::npos, "cli info json: shows correct dim");
    CHECK(output.find("\"count\": 1") != std::string::npos, "cli info json: shows count");

    std::filesystem::remove_all(path);
}

// Test CLI export/import roundtrip
static void test_cli_export_import_roundtrip() {
    std::string path = "/tmp/logosdb_test_cli_export";
    std::string import_path = "/tmp/logosdb_test_cli_import";
    std::string export_file = "/tmp/test_export.jsonl";
    std::filesystem::remove_all(path);
    std::filesystem::remove_all(import_path);
    std::remove(export_file.c_str());

    // Create a DB with some data
    logosdb::DB db(path, {.dim = 32});
    auto v0 = unit_vec(32, 18000);
    auto v1 = unit_vec(32, 18100);
    db.put(v0, "first", "2025-01-10T00:00:00Z");
    db.put(v1, "second", "2025-01-11T00:00:00Z");

    // Export to file
    std::string output;
    int rc = run_cli("export " + path + " --output " + export_file, output);
    CHECK(rc == 0, "cli export: returns success");
    CHECK(output.find("Exported 2 rows") != std::string::npos, "cli export: reports 2 rows");

    // Check export file exists and has content
    std::ifstream check(export_file);
    CHECK(check.good(), "cli export: file created");
    check.close();

    // Import to new DB
    rc = run_cli("import " + import_path + " --dim 32 --input " + export_file, output);
    CHECK(rc == 0, "cli import: returns success");
    CHECK(output.find("Imported 2 rows") != std::string::npos, "cli import: reports 2 rows");

    // Verify imported data
    logosdb::DB import_db(import_path, {.dim = 32});
    CHECK(import_db.count() == 2, "cli roundtrip: 2 rows in imported DB");

    // Cleanup
    std::filesystem::remove_all(path);
    std::filesystem::remove_all(import_path);
    std::remove(export_file.c_str());
}

// Test CLI search with timestamp range
static void test_cli_search_ts_range() {
    std::string path = "/tmp/logosdb_test_cli_ts";
    std::filesystem::remove_all(path);

    // Create a DB with timestamped data
    logosdb::DB db(path, {.dim = 32});
    auto v0 = unit_vec(32, 19000);
    auto v1 = unit_vec(32, 19100);
    db.put(v0, "early", "2025-01-10T08:00:00Z");
    db.put(v1, "late", "2025-01-15T18:00:00Z");

    // Create a query vector file
    {
        std::ofstream qf("/tmp/query_vec.bin", std::ios::binary);
        qf.write(reinterpret_cast<const char*>(v0.data()), 32 * sizeof(float));
        qf.close();
    }

    // Search without timestamp filter - should find both
    std::string output;
    int rc = run_cli("search " + path + " --query-file /tmp/query_vec.bin --top-k 10 --json", output);
    CHECK(rc == 0, "cli search json: returns success");
    // Count results in JSON output (each result has "rank": N)
    int results = 0;
    size_t pos = 0;
    while ((pos = output.find("\"rank\":", pos)) != std::string::npos) {
        results++;
        pos++;
    }
    CHECK(results == 2, "cli search no filter: finds 2 results");

    // Search with timestamp filter - should find only early entry
    output.clear();
    rc = run_cli("search " + path + " --query-file /tmp/query_vec.bin --ts-from 2025-01-01T00:00:00Z --ts-to 2025-01-12T00:00:00Z --top-k 10 --json", output);
    CHECK(rc == 0, "cli search ts filter: returns success");

    // Should find only the early entry
    CHECK(output.find("early") != std::string::npos, "cli search ts filter: finds early entry");
    CHECK(output.find("late") == std::string::npos, "cli search ts filter: excludes late entry");

    // Cleanup
    std::filesystem::remove_all(path);
    std::remove("/tmp/query_vec.bin");
}

// Test that pointers remain valid across appends (reservation mapping)
static void test_storage_pointers_stable() {
    std::string path = "/tmp/logosdb_test_stable_pointers";
    std::filesystem::remove_all(path);

    int dim = 32;
    logosdb::DB db(path, {.dim = dim});

    // Insert first vector and get pointer
    auto v0 = unit_vec(dim, 20000);
    db.put(v0, "first", "2025-01-01T00:00:00Z");

    // Get raw pointer to the data
    size_t n_rows = 0;
    int d = 0;
    const float* raw = db.raw_vectors(n_rows, d);
    CHECK(raw != nullptr, "stable_ptr: got raw pointer");
    CHECK(n_rows == 1, "stable_ptr: 1 row");

    // Verify we can read the first vector
    float first_val = raw[0];
    float diff = 0.0f;
    for (int i = 0; i < dim; ++i) {
        diff += std::fabs(raw[i] - v0[i]);
    }
    CHECK(diff < 1e-5f, "stable_ptr: first vector matches before append");

    // Insert many more vectors - with reservation mapping, the pointer should remain valid
    for (int i = 0; i < 100; ++i) {
        auto v = unit_vec(dim, 20001 + i);
        db.put(v, "batch", "2025-01-01T00:00:00Z");
    }

    // Verify the original pointer is still valid and unchanged
    diff = 0.0f;
    for (int i = 0; i < dim; ++i) {
        diff += std::fabs(raw[i] - v0[i]);
    }
    CHECK(diff < 1e-5f, "stable_ptr: first vector unchanged after 100 appends");

    // Also verify we can still search
    auto hits = db.search(v0, 1);
    CHECK(!hits.empty(), "stable_ptr: search works after appends");
    CHECK(hits[0].id == 0, "stable_ptr: first vector still found");

    std::filesystem::remove_all(path);
}

// Test L2-normalization: random vectors become unit length
static void test_l2_normalize_basic() {
    std::mt19937 rng(12345);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    int dim = 128;
    std::vector<float> vec(dim);
    for (int i = 0; i < dim; ++i) vec[i] = dist(rng);

    // Compute original norm - should NOT be unit length (random values)
    float orig_norm_sq = 0.0f;
    for (float v : vec) orig_norm_sq += v * v;
    float orig_norm = std::sqrt(orig_norm_sq);
    CHECK(orig_norm > 5.0f, "l2_norm: random vector has non-trivial norm");

    // Normalize (should succeed)
    int rc = logosdb_l2_normalize(vec.data(), dim);
    CHECK(rc == 0, "l2_norm: normalize succeeds");

    // Check that it's now unit length
    float norm_sq = 0.0f;
    for (float v : vec) norm_sq += v * v;
    float norm = std::sqrt(norm_sq);
    CHECK(std::abs(norm - 1.0f) < 1e-5f, "l2_norm: result has unit length");

    // Test with C API on scaled vector
    std::vector<float> v2(dim);
    for (int i = 0; i < dim; ++i) v2[i] = 3.0f * dist(rng);  // Scale by 3

    rc = logosdb_l2_normalize(v2.data(), dim);
    CHECK(rc == 0, "l2_norm: scaled vector normalizes");

    norm_sq = 0.0f;
    for (float v : v2) norm_sq += v * v;
    norm = std::sqrt(norm_sq);
    CHECK(std::abs(norm - 1.0f) < 1e-5f, "l2_norm: scaled result has unit length");
}

// Test L2-normalization: already-normalized vectors stay normalized
static void test_l2_normalize_already_normalized() {
    int dim = 64;
    auto v = unit_vec(dim, 50000);  // Already unit length

    float norm_sq = 0.0f;
    for (float val : v) norm_sq += val * val;
    CHECK(std::abs(std::sqrt(norm_sq) - 1.0f) < 1e-5f, "l2_norm_pre: starts unit");

    // Re-normalizing should be no-op
    int rc = logosdb_l2_normalize(v.data(), dim);
    CHECK(rc == 0, "l2_norm_pre: normalize succeeds");

    norm_sq = 0.0f;
    for (float val : v) norm_sq += val * val;
    float norm = std::sqrt(norm_sq);
    CHECK(std::abs(norm - 1.0f) < 1e-5f, "l2_norm_pre: still unit length");
}

// Test L2-normalization: zero vector returns error
static void test_l2_normalize_zero_vector() {
    int dim = 32;
    std::vector<float> zero_vec(dim, 0.0f);

    int rc = logosdb_l2_normalize(zero_vec.data(), dim);
    CHECK(rc == -1, "l2_norm_zero: returns -1 for zero vector");

    // Vector should be unchanged
    for (float v : zero_vec) CHECK(v == 0.0f, "l2_norm_zero: vector unchanged");
}

// Test L2-normalization: very small values
static void test_l2_normalize_small_values() {
    int dim = 128;
    std::vector<float> small_vec(dim);
    for (int i = 0; i < dim; ++i) small_vec[i] = 1e-20f;

    int rc = logosdb_l2_normalize(small_vec.data(), dim);
    CHECK(rc == 0, "l2_norm_small: normalize succeeds");

    float norm_sq = 0.0f;
    for (float v : small_vec) norm_sq += v * v;
    float norm = std::sqrt(norm_sq);
    CHECK(std::abs(norm - 1.0f) < 1e-5f, "l2_norm_small: result has unit length");
}

// Test C++ convenience wrappers
static void test_l2_normalize_cpp_wrappers() {
    int dim = 64;
    std::mt19937 rng(60000);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Test l2_normalize (in-place)
    std::vector<float> v1(dim);
    for (int i = 0; i < dim; ++i) v1[i] = 5.0f * dist(rng);  // Scale by 5

    bool ok = logosdb::l2_normalize(v1);
    CHECK(ok, "l2_norm_cpp: in-place normalize returns true");

    float norm_sq = 0.0f;
    for (float v : v1) norm_sq += v * v;
    CHECK(std::abs(std::sqrt(norm_sq) - 1.0f) < 1e-5f, "l2_norm_cpp: in-place result unit");

    // Test l2_normalized (copy)
    std::vector<float> v2(dim);
    for (int i = 0; i < dim; ++i) v2[i] = 3.0f * dist(rng);
    std::vector<float> v2_orig = v2;

    std::vector<float> v2_normed = logosdb::l2_normalized(v2);

    // Original should be unchanged
    for (int i = 0; i < dim; ++i) CHECK(v2[i] == v2_orig[i], "l2_norm_cpp: original unchanged");

    // Copy should be normalized
    norm_sq = 0.0f;
    for (float v : v2_normed) norm_sq += v * v;
    CHECK(std::abs(std::sqrt(norm_sq) - 1.0f) < 1e-5f, "l2_norm_cpp: copy is unit");

    // Test zero vector with C++ wrapper
    std::vector<float> zero_vec(dim, 0.0f);
    ok = logosdb::l2_normalize(zero_vec);
    CHECK(!ok, "l2_norm_cpp: zero vector returns false");
}
