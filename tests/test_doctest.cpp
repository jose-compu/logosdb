#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <logosdb/logosdb.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace std::string_literals;

// ============================================================================
// Test Helpers
// ============================================================================

// Returns a platform-specific temp directory prefix, e.g. "/tmp/" on POSIX
// and the system temp path (with trailing separator) on Windows.
static std::string tmp_path(const std::string& name)
{
    return (std::filesystem::temp_directory_path() / name).string();
}

static std::vector<float> unit_vec(int dim, int seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(dim);
    float norm = 0.0f;
    for (int i = 0; i < dim; ++i)
    {
        v[i] = dist(rng);
        norm += v[i] * v[i];
    }
    norm = std::sqrt(norm);
    for (int i = 0; i < dim; ++i)
        v[i] /= norm;
    return v;
}

static int run_cli(const std::string& args, std::string& output)
{
#ifdef _WIN32
    std::string cmd = "logosdb-cli.exe " + args + " 2>&1";
    FILE* pipe = _popen(cmd.c_str(), "r");
#else
    std::string cmd = "./logosdb-cli " + args + " 2>&1";
    FILE* pipe = popen(cmd.c_str(), "r");
#endif
    if (!pipe)
        return -1;
    char buffer[4096];
    output.clear();
    while (fgets(buffer, sizeof(buffer), pipe))
    {
        output += buffer;
    }
#ifdef _WIN32
    return _pclose(pipe);
#else
    return pclose(pipe);
#endif
}

// ============================================================================
// Test Suite: Core API (storage, db lifecycle)
// ============================================================================

TEST_SUITE("core")
{
    TEST_CASE("core: open and close")
    {
        std::string path = tmp_path("logosdb_test_oc");
        std::filesystem::remove_all(path);

        char* err = nullptr;
        logosdb_options_t* opts = logosdb_options_create();
        logosdb_options_set_dim(opts, 64);
        logosdb_t* db = logosdb_open(path.c_str(), opts, &err);
        logosdb_options_destroy(opts);

        CHECK(db != nullptr);
        CHECK(err == nullptr);
        CHECK(logosdb_count(db) == 0);
        CHECK(logosdb_dim(db) == 64);

        logosdb_close(db);
        std::filesystem::remove_all(path);
    }

    TEST_CASE("core: logosdb_sync")
    {
        std::string path = tmp_path("logosdb_test_sync_api");
        std::filesystem::remove_all(path);

        char* err = nullptr;
        logosdb_options_t* o = logosdb_options_create();
        logosdb_options_set_dim(o, 16);
        logosdb_t* db = logosdb_open(path.c_str(), o, &err);
        logosdb_options_destroy(o);
        CHECK(db != nullptr);
        CHECK(err == nullptr);

        auto v = unit_vec(16, 7777);
        uint64_t id = logosdb_put(db, v.data(), 16, "sync_row", nullptr, &err);
        CHECK(id != UINT64_MAX);
        free(err);
        err = nullptr;

        CHECK(logosdb_sync(db, &err) == 0);
        CHECK(err == nullptr);

        logosdb_close(db);

        {
            logosdb::DB db2(path, {.dim = 16});
            CHECK(db2.count() == 1);
            auto hits = db2.search(unit_vec(16, 7777), 1);
            CHECK(!hits.empty());
            CHECK(hits[0].text == "sync_row");
        }

        std::filesystem::remove_all(path);
    }

    TEST_CASE("core: logosdb_sync rejects null db")
    {
        char* err = nullptr;
        CHECK(logosdb_sync(nullptr, &err) == -1);
        CHECK(err != nullptr);
        free(err);
    }

    TEST_CASE("core: put and search")
    {
        std::string path = tmp_path("logosdb_test_ps");
        std::filesystem::remove_all(path);

        logosdb::DB db(path, {.dim = 64});

        auto v0 = unit_vec(64, 100);
        auto v1 = unit_vec(64, 200);
        auto v2 = unit_vec(64, 300);

        uint64_t id0 = db.put(v0, "fact zero", "2025-01-01T00:00:00Z");
        uint64_t id1 = db.put(v1, "fact one", "2025-02-01T00:00:00Z");
        uint64_t id2 = db.put(v2, "fact two", "2025-03-01T00:00:00Z");

        CHECK(id0 == 0);
        CHECK(id1 == 1);
        CHECK(id2 == 2);
        CHECK(db.count() == 3);

        auto hits = db.search(v0, 3);
        CHECK(!hits.empty());
        CHECK(hits[0].id == 0);
        CHECK(hits[0].text == "fact zero");
        CHECK(hits[0].timestamp == "2025-01-01T00:00:00Z");
        CHECK(hits[0].score > 0.9f);

        std::filesystem::remove_all(path);
    }

    TEST_CASE("core: persistence across reopen")
    {
        std::string path = tmp_path("logosdb_test_persist");
        std::filesystem::remove_all(path);

        auto v0 = unit_vec(64, 400);
        auto v1 = unit_vec(64, 500);

        {
            logosdb::DB db(path, {.dim = 64});
            db.put(v0, "persisted fact A", "2025-04-01T00:00:00Z");
            db.put(v1, "persisted fact B", "2025-05-01T00:00:00Z");
            CHECK(db.count() == 2);
        }

        {
            logosdb::DB db(path, {.dim = 64});
            CHECK(db.count() == 2);

            auto hits = db.search(v0, 2);
            CHECK(!hits.empty());
            CHECK(hits[0].id == 0);
            CHECK(hits[0].text == "persisted fact A");
            CHECK(hits[0].timestamp == "2025-04-01T00:00:00Z");
        }

        std::filesystem::remove_all(path);
    }

    TEST_CASE("core: raw vectors bulk access")
    {
        std::string path = tmp_path("logosdb_test_raw");
        std::filesystem::remove_all(path);

        int dim = 32;
        logosdb::DB db(path, {.dim = dim});

        auto v0 = unit_vec(dim, 600);
        auto v1 = unit_vec(dim, 700);
        db.put(v0);
        db.put(v1);

        size_t n_rows = 0;
        int d = 0;
        auto raw = db.raw_vectors(n_rows, d);
        CHECK(!raw.empty());
        CHECK(n_rows == 2);
        CHECK(d == dim);

        float diff = 0.0f;
        for (int i = 0; i < dim; ++i)
            diff += std::fabs(raw[i] - v0[i]);
        CHECK(diff < 1e-5f);

        diff = 0.0f;
        for (int i = 0; i < dim; ++i)
            diff += std::fabs(raw[dim + i] - v1[i]);
        CHECK(diff < 1e-5f);

        std::filesystem::remove_all(path);
    }

    TEST_CASE("core: many vectors self-retrieval")
    {
        std::string path = tmp_path("logosdb_test_many");
        std::filesystem::remove_all(path);

        int dim = 128;
        int n = 500;
        logosdb::DB db(path, {.dim = dim, .max_elements = 1000});

        std::vector<std::vector<float>> vecs;
        for (int i = 0; i < n; ++i)
        {
            vecs.push_back(unit_vec(dim, i));
            db.put(vecs.back(), "row_" + std::to_string(i));
        }
        CHECK(db.count() == (size_t)n);

        int self_found = 0;
        for (int i = 0; i < n; i += 50)
        {
            auto hits = db.search(vecs[i], 1);
            if (!hits.empty() && hits[0].id == (uint64_t)i)
                ++self_found;
        }
        CHECK(self_found == 10);

        std::filesystem::remove_all(path);
    }

    TEST_CASE("core: search ordering")
    {
        std::string path = tmp_path("logosdb_test_order");
        std::filesystem::remove_all(path);

        int dim = 64;
        logosdb::DB db(path, {.dim = dim});

        auto base = unit_vec(dim, 1000);
        auto close_vec = base;
        close_vec[0] += 0.01f;
        float norm = 0.0f;
        for (int i = 0; i < dim; ++i)
            norm += close_vec[i] * close_vec[i];
        norm = std::sqrt(norm);
        for (int i = 0; i < dim; ++i)
            close_vec[i] /= norm;

        auto far_vec = unit_vec(dim, 2000);

        db.put(far_vec, "far");
        db.put(close_vec, "close");
        db.put(base, "base");

        auto hits = db.search(base, 3);
        CHECK(hits.size() == 3);
        CHECK(hits[0].id == 2);
        CHECK(hits[1].id == 1);
        CHECK(hits[0].score >= hits[1].score);
        CHECK(hits[1].score >= hits[2].score);

        std::filesystem::remove_all(path);
    }

    TEST_CASE("core: top_k limit and clamping")
    {
        std::string path = tmp_path("logosdb_test_topk");
        std::filesystem::remove_all(path);

        int dim = 32;
        logosdb::DB db(path, {.dim = dim});

        for (int i = 0; i < 20; ++i)
            db.put(unit_vec(dim, i + 3000));

        auto hits1 = db.search(unit_vec(dim, 3000), 1);
        CHECK(hits1.size() == 1);

        auto hits5 = db.search(unit_vec(dim, 3000), 5);
        CHECK(hits5.size() == 5);

        auto hits99 = db.search(unit_vec(dim, 3000), 99);
        CHECK(hits99.size() == 20);

        std::filesystem::remove_all(path);
    }

    TEST_CASE("core: empty database search")
    {
        std::string path = tmp_path("logosdb_test_empty_search");
        std::filesystem::remove_all(path);

        logosdb::DB db(path, {.dim = 32});
        auto hits = db.search(unit_vec(32, 5000), 5);
        CHECK(hits.empty());

        std::filesystem::remove_all(path);
    }

    TEST_CASE("core: put without metadata")
    {
        std::string path = tmp_path("logosdb_test_nometa");
        std::filesystem::remove_all(path);

        logosdb::DB db(path, {.dim = 32});
        auto v = unit_vec(32, 6000);
        uint64_t id = db.put(v);
        CHECK(id == 0);

        auto hits = db.search(v, 1);
        CHECK(!hits.empty());
        CHECK(hits[0].text.empty());
        CHECK(hits[0].timestamp.empty());

        std::filesystem::remove_all(path);
    }

    TEST_CASE("core: persistence append after reopen")
    {
        std::string path = tmp_path("logosdb_test_append_reopen");
        std::filesystem::remove_all(path);

        int dim = 32;
        auto v0 = unit_vec(dim, 1200);
        auto v1 = unit_vec(dim, 1300);
        auto v2 = unit_vec(dim, 1400);

        {
            logosdb::DB db(path, {.dim = dim});
            db.put(v0, "first");
            CHECK(db.count() == 1);
        }
        {
            logosdb::DB db(path, {.dim = dim});
            CHECK(db.count() == 1);
            db.put(v1, "second");
            db.put(v2, "third");
            CHECK(db.count() == 3);
        }
        {
            logosdb::DB db(path, {.dim = dim});
            CHECK(db.count() == 3);

            auto hits = db.search(v2, 1);
            CHECK(!hits.empty());
            CHECK(hits[0].id == 2);
            CHECK(hits[0].text == "third");
        }

        std::filesystem::remove_all(path);
    }

    TEST_CASE("core: large dimension (2048)")
    {
        std::string path = tmp_path("logosdb_test_largedim");
        std::filesystem::remove_all(path);

        int dim = 2048;
        logosdb::DB db(path, {.dim = dim});

        auto v0 = unit_vec(dim, 1600);
        auto v1 = unit_vec(dim, 1700);
        db.put(v0, "large dim A");
        db.put(v1, "large dim B");

        CHECK(db.count() == 2);

        auto hits = db.search(v0, 1);
        CHECK(!hits.empty());
        CHECK(hits[0].id == 0);
        CHECK(hits[0].text == "large dim A");

        std::filesystem::remove_all(path);
    }

}  // TEST_SUITE("core")

// ============================================================================
// Test Suite: Metadata
// ============================================================================

TEST_SUITE("metadata")
{
    TEST_CASE("metadata: special characters round-trip")
    {
        std::string path = tmp_path("logosdb_test_special");
        std::filesystem::remove_all(path);

        logosdb::DB db(path, {.dim = 32});
        auto v = unit_vec(32, 7000);

        std::string special = "He said \"hello\"\nand\ttabs\\backslash";
        db.put(v, special, "2025-06-25T00:00:00Z");

        {
            logosdb::DB db2(path, {.dim = 32});
            auto hits = db2.search(v, 1);
            CHECK(!hits.empty());
            CHECK(hits[0].text == special);
        }

        std::filesystem::remove_all(path);
    }

    TEST_CASE("metadata: unicode and escapes")
    {
        std::string path = tmp_path("logosdb_test_unicode");
        std::filesystem::remove_all(path);

        logosdb::DB db(path, {.dim = 32});
        auto v0 = unit_vec(32, 4000);
        auto v1 = unit_vec(32, 4100);
        auto v2 = unit_vec(32, 4200);

        std::string unicode = "Hello \u4e16\u754c \u00e9\u00e0";
        db.put(v0, unicode, "2025-01-01T00:00:00Z");
        db.put(v1, "", "");

        std::string complex_escapes = "Path: C:\\Users\\test\\file.txt\\\"quoted\"";
        db.put(v2, complex_escapes);

        {
            logosdb::DB db2(path, {.dim = 32});
            auto hits0 = db2.search(v0, 1);
            CHECK(!hits0.empty());
            CHECK(hits0[0].text == unicode);

            auto hits1 = db2.search(v1, 1);
            CHECK(!hits1.empty());
            CHECK(hits1[0].text == "");
            CHECK(hits1[0].timestamp == "");

            auto hits2 = db2.search(v2, 1);
            CHECK(!hits2.empty());
            CHECK(hits2[0].text == complex_escapes);
        }

        std::filesystem::remove_all(path);
    }

    TEST_CASE("metadata: JSON edge cases")
    {
        std::string path = tmp_path("logosdb_test_jsonedge");
        std::filesystem::remove_all(path);

        {
            logosdb::DB db(path, {.dim = 32});
            auto v0 = unit_vec(32, 6000);
            auto v1 = unit_vec(32, 6100);
            auto v2 = unit_vec(32, 6200);
            db.put(v0, "unicode: \u4e16\u754c", "2025-01-01T00:00:00Z");
            db.put(v1, "spaced text", "2025-02-02T12:00:00Z");
            db.put(v2, "https://example.com/path", "2025-03-03T00:00:00Z");
            CHECK(db.count() == 3);
        }

        {
            logosdb::DB db(path, {.dim = 32});
            CHECK(db.count() == 3);

            auto v0 = unit_vec(32, 6000);
            auto v1 = unit_vec(32, 6100);
            auto v2 = unit_vec(32, 6200);

            auto hits0 = db.search(v0, 1);
            CHECK(!hits0.empty());
            CHECK(hits0[0].text == "unicode: \u4e16\u754c");

            auto hits1 = db.search(v1, 1);
            CHECK(!hits1.empty());
            CHECK(hits1[0].text == "spaced text");

            auto hits2 = db.search(v2, 1);
            CHECK(!hits2.empty());
            CHECK(hits2[0].text == "https://example.com/path");
        }

        std::filesystem::remove_all(path);
    }

}  // TEST_SUITE("metadata")

// ============================================================================
// Test Suite: Error Handling
// ============================================================================

TEST_SUITE("errors")
{
    TEST_CASE("errors: C API error handling")
    {
        char* err = nullptr;

        logosdb_options_t* opts = logosdb_options_create();
        logosdb_options_set_dim(opts, 0);
        logosdb_t* db = logosdb_open(tmp_path("logosdb_test_err").c_str(), opts, &err);
        CHECK(db == nullptr);
        CHECK(err != nullptr);
        free(err);
        err = nullptr;
        logosdb_options_destroy(opts);

        CHECK(logosdb_count(nullptr) == 0);
        CHECK(logosdb_dim(nullptr) == 0);

        logosdb_search_result_t* r = logosdb_search(nullptr, nullptr, 0, 1, &err);
        CHECK(r == nullptr);
        free(err);
    }

    TEST_CASE("errors: C++ wrapper exception")
    {
        bool caught = false;
        try
        {
            logosdb::DB db(tmp_path("logosdb_test_exc"), {.dim = 0});
        }
        catch (const std::runtime_error& e)
        {
            caught = true;
            CHECK(std::string(e.what()).find("logosdb_open") != std::string::npos);
        }
        CHECK(caught);
    }

    TEST_CASE("errors: dimension mismatch on put")
    {
        std::string path = tmp_path("logosdb_test_dim_put");
        std::filesystem::remove_all(path);

        char* err = nullptr;
        logosdb_options_t* opts = logosdb_options_create();
        logosdb_options_set_dim(opts, 32);
        logosdb_t* db = logosdb_open(path.c_str(), opts, &err);
        logosdb_options_destroy(opts);
        CHECK(db != nullptr);

        auto wrong = unit_vec(64, 8000);
        uint64_t id = logosdb_put(db, wrong.data(), 64, "bad", nullptr, &err);
        CHECK(id == UINT64_MAX);
        CHECK(err != nullptr);
        free(err);
        err = nullptr;

        CHECK(logosdb_count(db) == 0);
        logosdb_close(db);
        std::filesystem::remove_all(path);
    }

    TEST_CASE("errors: dimension mismatch on search")
    {
        std::string path = tmp_path("logosdb_test_dim_search");
        std::filesystem::remove_all(path);

        char* err = nullptr;
        logosdb_options_t* opts = logosdb_options_create();
        logosdb_options_set_dim(opts, 32);
        logosdb_t* db = logosdb_open(path.c_str(), opts, &err);
        logosdb_options_destroy(opts);

        auto v = unit_vec(32, 9000);
        logosdb_put(db, v.data(), 32, "ok", nullptr, &err);

        auto wrong_q = unit_vec(64, 9001);
        logosdb_search_result_t* r = logosdb_search(db, wrong_q.data(), 64, 1, &err);
        CHECK(r == nullptr);
        CHECK(err != nullptr);
        free(err);

        logosdb_close(db);
        std::filesystem::remove_all(path);
    }

    TEST_CASE("errors: result accessor bounds")
    {
        std::string path = tmp_path("logosdb_test_bounds");
        std::filesystem::remove_all(path);

        char* err = nullptr;
        logosdb_options_t* opts = logosdb_options_create();
        logosdb_options_set_dim(opts, 32);
        logosdb_t* db = logosdb_open(path.c_str(), opts, &err);
        logosdb_options_destroy(opts);

        auto v = unit_vec(32, 1100);
        logosdb_put(db, v.data(), 32, "x", nullptr, &err);

        logosdb_search_result_t* r = logosdb_search(db, v.data(), 32, 1, &err);
        CHECK(r != nullptr);
        CHECK(logosdb_result_count(r) == 1);

        CHECK(logosdb_result_id(r, -1) == UINT64_MAX);
        CHECK(logosdb_result_id(r, 5) == UINT64_MAX);
        CHECK(logosdb_result_text(r, -1) == nullptr);
        CHECK(logosdb_result_text(r, 5) == nullptr);
        CHECK(logosdb_result_score(r, 99) == 0.0f);
        CHECK(logosdb_result_timestamp(r, -1) == nullptr);

        CHECK(logosdb_result_count(nullptr) == 0);

        logosdb_result_free(r);
        logosdb_close(db);
        std::filesystem::remove_all(path);
    }

}  // TEST_SUITE("errors")

// ============================================================================
// Test Suite: Delete and Update
// ============================================================================

TEST_SUITE("delete_update")
{
    TEST_CASE("delete_update: basic delete")
    {
        std::string path = tmp_path("logosdb_test_delete_basic");
        std::filesystem::remove_all(path);

        int dim = 64;
        logosdb::DB db(path, {.dim = dim});

        auto v0 = unit_vec(dim, 2100);
        auto v1 = unit_vec(dim, 2200);
        auto v2 = unit_vec(dim, 2300);

        db.put(v0, "zero");
        db.put(v1, "one");
        db.put(v2, "two");
        CHECK(db.count() == 3);
        CHECK(db.count_live() == 3);

        db.del(1);
        CHECK(db.count() == 3);
        CHECK(db.count_live() == 2);

        auto hits = db.search(v1, 3);
        bool found = false;
        for (auto& h : hits)
            if (h.id == 1)
                found = true;
        CHECK(!found);

        auto self_hits = db.search(v0, 1);
        CHECK(!self_hits.empty());
        CHECK(self_hits[0].id == 0);

        std::filesystem::remove_all(path);
    }

    TEST_CASE("delete_update: delete errors")
    {
        std::string path = tmp_path("logosdb_test_delete_err");
        std::filesystem::remove_all(path);

        logosdb::DB db(path, {.dim = 32});
        auto v0 = unit_vec(32, 2400);
        db.put(v0, "only");

        char* err = nullptr;
        int rc = logosdb_delete(db.handle(), 99, &err);
        CHECK(rc == -1);
        CHECK(err != nullptr);
        free(err);
        err = nullptr;

        rc = logosdb_delete(db.handle(), 0, &err);
        CHECK(rc == 0);
        CHECK(err == nullptr);

        rc = logosdb_delete(db.handle(), 0, &err);
        CHECK(rc == -1);
        CHECK(err != nullptr);
        free(err);

        std::filesystem::remove_all(path);
    }

    TEST_CASE("delete_update: delete persistence")
    {
        std::string path = tmp_path("logosdb_test_delete_persist");
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
            CHECK(db.count_live() == 2);
        }
        {
            logosdb::DB db(path, {.dim = dim});
            CHECK(db.count() == 3);
            CHECK(db.count_live() == 2);

            auto hits = db.search(v1, 3);
            bool found = false;
            for (auto& h : hits)
                if (h.id == 1)
                    found = true;
            CHECK(!found);

            auto hits_a = db.search(v0, 1);
            CHECK(!hits_a.empty());
            CHECK(hits_a[0].id == 0);
        }

        std::filesystem::remove_all(path);
    }

    TEST_CASE("delete_update: tombstone replay on index rebuild")
    {
        std::string path = tmp_path("logosdb_test_delete_rebuild");
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
        CHECK(!ec);

        {
            logosdb::DB db(path, {.dim = dim});
            CHECK(db.count_live() == 1);

            auto hits = db.search(v0, 2);
            bool found = false;
            for (auto& h : hits)
                if (h.id == 0)
                    found = true;
            CHECK(!found);
        }

        std::filesystem::remove_all(path);
    }

    TEST_CASE("delete_update: basic update")
    {
        std::string path = tmp_path("logosdb_test_update");
        std::filesystem::remove_all(path);

        int dim = 64;
        logosdb::DB db(path, {.dim = dim});

        auto v0 = unit_vec(dim, 3100);
        auto v1 = unit_vec(dim, 3200);
        auto v_new = unit_vec(dim, 3300);

        db.put(v0, "old zero");
        db.put(v1, "keep one");
        CHECK(db.count_live() == 2);

        uint64_t new_id = db.update(0, v_new, "new zero", "2025-07-01T00:00:00Z");
        CHECK(new_id == 2);
        CHECK(db.count() == 3);
        CHECK(db.count_live() == 2);

        auto hits_new = db.search(v_new, 1);
        CHECK(!hits_new.empty());
        CHECK(hits_new[0].id == new_id);
        CHECK(hits_new[0].text == "new zero");
        CHECK(hits_new[0].timestamp == "2025-07-01T00:00:00Z");

        auto hits_old = db.search(v0, 3);
        bool old_found = false;
        for (auto& h : hits_old)
            if (h.id == 0)
                old_found = true;
        CHECK(!old_found);

        std::filesystem::remove_all(path);
    }

    TEST_CASE("delete_update: update errors")
    {
        std::string path = tmp_path("logosdb_test_update_err");
        std::filesystem::remove_all(path);

        logosdb::DB db(path, {.dim = 32});
        auto v0 = unit_vec(32, 3400);
        auto v1 = unit_vec(32, 3500);
        db.put(v0, "a");

        char* err = nullptr;
        uint64_t r = logosdb_update(db.handle(), 99, v1.data(), 32, "x", nullptr, &err);
        CHECK(r == UINT64_MAX);
        CHECK(err != nullptr);
        free(err);
        err = nullptr;

        auto wrong = unit_vec(16, 3600);
        r = logosdb_update(db.handle(), 0, wrong.data(), 16, "x", nullptr, &err);
        CHECK(r == UINT64_MAX);
        CHECK(err != nullptr);
        free(err);
        err = nullptr;

        logosdb_delete(db.handle(), 0, &err);
        free(err);
        err = nullptr;
        r = logosdb_update(db.handle(), 0, v1.data(), 32, "x", nullptr, &err);
        CHECK(r == UINT64_MAX);
        CHECK(err != nullptr);
        free(err);

        std::filesystem::remove_all(path);
    }

    TEST_CASE("delete_update: delete and reput independence")
    {
        std::string path = tmp_path("logosdb_test_delete_reput");
        std::filesystem::remove_all(path);

        int dim = 32;
        logosdb::DB db(path, {.dim = dim});

        auto v0 = unit_vec(dim, 3700);
        auto v1 = unit_vec(dim, 3800);

        uint64_t id0 = db.put(v0, "first");
        db.del(id0);

        uint64_t id1 = db.put(v1, "second");
        CHECK(id1 == 1);
        CHECK(db.count() == 2);
        CHECK(db.count_live() == 1);

        auto hits = db.search(v1, 1);
        CHECK(!hits.empty());
        CHECK(hits[0].id == id1);

        std::filesystem::remove_all(path);
    }

}  // TEST_SUITE("delete_update")

// ============================================================================
// Test Suite: WAL and Crash Recovery
// ============================================================================

TEST_SUITE("wal")
{
    TEST_CASE("wal: crash recovery replay")
    {
        std::string path = tmp_path("logosdb_test_wal");
        std::filesystem::remove_all(path);

        {
            logosdb::DB db(path, {.dim = 32});
            auto v0 = unit_vec(32, 7000);
            auto v1 = unit_vec(32, 7100);
            db.put(v0, "first", "2025-01-01T00:00:00Z");
            db.put(v1, "second", "2025-02-01T00:00:00Z");
            CHECK(db.count() == 2);
        }

        {
#ifdef _WIN32
            int fd = _open((path + "/wal.log").c_str(), O_RDWR | O_BINARY);
#else
            int fd = ::open((path + "/wal.log").c_str(), O_RDWR);
#endif
            CHECK(fd >= 0);

#ifdef _WIN32
            _lseeki64(fd, 0, SEEK_END);
#else
            ::lseek(fd, 0, SEEK_END);
#endif

            auto raw_write = [&](const void* buf, unsigned int len)
            {
#ifdef _WIN32
                return _write(fd, buf, len);
#else
                return static_cast<int>(::write(fd, buf, len));
#endif
            };

            uint8_t state = 0;
            raw_write(&state, 1);

            uint32_t dim = 32;
            raw_write(&dim, 4);

            auto v2 = unit_vec(32, 7200);
            uint32_t vec_bytes = 32 * sizeof(float);
            raw_write(&vec_bytes, 4);
            raw_write(v2.data(), vec_bytes);

            std::string text = "third (from wal)";
            uint32_t text_len = static_cast<uint32_t>(text.size());
            raw_write(&text_len, 4);
            raw_write(text.data(), text_len);

            std::string ts = "2025-03-01T00:00:00Z";
            uint32_t ts_len = static_cast<uint32_t>(ts.size());
            raw_write(&ts_len, 4);
            raw_write(ts.data(), ts_len);

            uint64_t expected_id = 2;
            raw_write(&expected_id, 8);

#ifdef _WIN32
            _commit(fd);
            _close(fd);
#else
            ::fsync(fd);
            ::close(fd);
#endif
        }

        {
            logosdb::DB db(path, {.dim = 32});
            CHECK(db.count() == 3);

            auto v2 = unit_vec(32, 7200);
            auto hits = db.search(v2, 1);
            CHECK(!hits.empty());
            CHECK(hits[0].text == "third (from wal)");
            CHECK(hits[0].id == 2);
        }

        {
            logosdb::DB db(path, {.dim = 32});
            CHECK(db.count() == 3);

            auto v2 = unit_vec(32, 7200);
            auto hits = db.search(v2, 1);
            CHECK(!hits.empty());
            CHECK(hits[0].text == "third (from wal)");
        }

        std::filesystem::remove_all(path);
    }

}  // TEST_SUITE("wal")

// ============================================================================
// Test Suite: Batch Operations
// ============================================================================

TEST_SUITE("batch")
{
    TEST_CASE("batch: basic put_batch")
    {
        std::string path = tmp_path("logosdb_test_batch");
        std::filesystem::remove_all(path);

        {
            logosdb::DB db(path, {.dim = 32});

            int n = 100;
            std::vector<float> embeddings;
            embeddings.reserve(n * 32);
            std::vector<std::string> texts;
            std::vector<std::string> timestamps;

            for (int i = 0; i < n; ++i)
            {
                auto v = unit_vec(32, 8000 + i);
                embeddings.insert(embeddings.end(), v.begin(), v.end());
                texts.push_back("text_" + std::to_string(i));
                timestamps.push_back("2025-01-01T00:00:" + std::to_string(i) + "Z");
            }

            auto ids = db.put_batch(embeddings, n, texts, timestamps);
            CHECK((int)ids.size() == n);
            CHECK(db.count() == (size_t)n);

            for (int i = 0; i < n; ++i)
            {
                CHECK(ids[i] == (uint64_t)i);
            }

            for (int i = 0; i < n; ++i)
            {
                auto v = unit_vec(32, 8000 + i);
                auto hits = db.search(v, 1);
                CHECK(!hits.empty());
                CHECK(hits[0].id == (uint64_t)i);
                CHECK(hits[0].text == "text_" + std::to_string(i));
            }
        }

        {
            logosdb::DB db(path, {.dim = 32});
            CHECK(db.count() == 100);

            auto v50 = unit_vec(32, 8050);
            auto hits = db.search(v50, 1);
            CHECK(!hits.empty());
            CHECK(hits[0].id == 50);
            CHECK(hits[0].text == "text_50");
        }

        std::filesystem::remove_all(path);
    }

    TEST_CASE("batch: empty and edge cases")
    {
        std::string path = tmp_path("logosdb_test_batch_empty");
        std::filesystem::remove_all(path);

        {
            logosdb::DB db(path, {.dim = 32});

            std::vector<float> embeddings;
            auto ids = db.put_batch(embeddings, 0);
            CHECK(ids.empty());
            CHECK(db.count() == 0);

            auto v = unit_vec(32, 9000);
            embeddings = v;
            ids = db.put_batch(embeddings, 1);
            CHECK(ids.size() == 1);
            CHECK(db.count() == 1);

            embeddings.clear();
            for (int i = 0; i < 10; ++i)
            {
                auto v2 = unit_vec(32, 9100 + i);
                embeddings.insert(embeddings.end(), v2.begin(), v2.end());
            }
            ids = db.put_batch(embeddings, 10);
            CHECK(ids.size() == 10);
            CHECK(db.count() == 11);

            auto v_search = unit_vec(32, 9105);
            auto hits = db.search(v_search, 1);
            CHECK(!hits.empty());
            CHECK(hits[0].id == 6);
        }

        std::filesystem::remove_all(path);
    }

    TEST_CASE("batch: chunked WAL-aware put_batch crosses chunk boundary")
    {
        std::string path = tmp_path("logosdb_test_batch_chunked");
        std::filesystem::remove_all(path);
        // Force several small chunks to exercise the chunking path.
        setenv("LOGOSDB_BATCH_CHUNK_SIZE", "64", 1);
        {
            logosdb::DB db(path, {.dim = 16});
            const int n = 257;  // > 4 chunks of 64
            std::vector<float> embeddings;
            embeddings.reserve(n * 16);
            std::vector<std::string> texts;
            texts.reserve(n);
            for (int i = 0; i < n; ++i)
            {
                auto v = unit_vec(16, 20000 + i);
                embeddings.insert(embeddings.end(), v.begin(), v.end());
                texts.push_back("row_" + std::to_string(i));
            }
            auto ids = db.put_batch(embeddings, n, texts);
            CHECK((int)ids.size() == n);
            for (int i = 0; i < n; ++i)
                CHECK(ids[i] == (uint64_t)i);
            CHECK(db.count() == (size_t)n);
            auto s = db.get_stats();
            // No PENDING entries should remain once put_batch returns success.
            CHECK(s.wal_pending == 0u);
        }
        // Reopen — sanity-check WAL replay path did nothing on a clean batch.
        {
            logosdb::DB db(path, {.dim = 16});
            CHECK(db.count() == 257u);
            auto v = unit_vec(16, 20000 + 200);
            auto hits = db.search(v, 1);
            REQUIRE(!hits.empty());
            CHECK(hits[0].id == 200u);
            CHECK(hits[0].text == "row_200");
        }
        unsetenv("LOGOSDB_BATCH_CHUNK_SIZE");
        std::filesystem::remove_all(path);
    }

}  // TEST_SUITE("batch")

// ============================================================================
// Test Suite: Streaming NDJSON import/export (#87)
// ============================================================================

TEST_SUITE("streaming")
{
    TEST_CASE("streaming: ndjson export then import round-trip")
    {
        std::string src = tmp_path("logosdb_test_stream_src");
        std::string dst = tmp_path("logosdb_test_stream_dst");
        std::string ndjson = tmp_path("logosdb_test_stream.ndjson");
        std::filesystem::remove_all(src);
        std::filesystem::remove_all(dst);
        std::filesystem::remove(ndjson);

        const int dim = 16;
        const int n = 50;
        {
            logosdb::DB db(src, {.dim = dim});
            for (int i = 0; i < n; ++i)
            {
                auto v = unit_vec(dim, 30000 + i);
                db.put(
                    v, "row_" + std::to_string(i), "2025-02-01T00:00:" + std::to_string(i) + "Z");
            }
            db.export_ndjson(ndjson);
        }
        // The exported file should have exactly n non-empty lines.
        {
            std::ifstream f(ndjson);
            int lines = 0;
            std::string line;
            while (std::getline(f, line))
                if (!line.empty())
                    ++lines;
            CHECK(lines == n);
        }
        {
            logosdb::DB db(dst, {.dim = dim});
            db.import_ndjson(ndjson, /*chunk_size=*/8);
            CHECK(db.count() == (size_t)n);
            auto v = unit_vec(dim, 30000 + 25);
            auto hits = db.search(v, 1);
            REQUIRE(!hits.empty());
            CHECK(hits[0].id == 25u);
            CHECK(hits[0].text == "row_25");
        }

        std::filesystem::remove_all(src);
        std::filesystem::remove_all(dst);
        std::filesystem::remove(ndjson);
    }

    TEST_CASE("streaming: import resume from checkpoint after partial run")
    {
        std::string src = tmp_path("logosdb_test_stream_src2");
        std::string dst = tmp_path("logosdb_test_stream_dst2");
        std::string ndjson = tmp_path("logosdb_test_stream2.ndjson");
        std::string checkpoint = tmp_path("logosdb_test_stream2.checkpoint");
        std::filesystem::remove_all(src);
        std::filesystem::remove_all(dst);
        std::filesystem::remove(ndjson);
        std::filesystem::remove(checkpoint);

        const int dim = 8;
        const int n = 30;
        {
            logosdb::DB db(src, {.dim = dim});
            for (int i = 0; i < n; ++i)
            {
                auto v = unit_vec(dim, 40000 + i);
                db.put(v, "row_" + std::to_string(i));
            }
            db.export_ndjson(ndjson);
        }
        // Truncate the ndjson file so the first import sees half the rows and
        // writes a checkpoint at its byte offset.
        size_t total_size = std::filesystem::file_size(ndjson);
        std::string truncated = ndjson + ".half";
        {
            std::ifstream in(ndjson, std::ios::binary);
            std::ofstream out(truncated, std::ios::binary | std::ios::trunc);
            std::vector<char> buf(total_size);
            in.read(buf.data(), total_size);
            // Write only the first 10 NDJSON rows by line count.
            int written_lines = 0;
            size_t offset = 0;
            for (size_t i = 0; i < total_size; ++i)
            {
                if (buf[i] == '\n')
                {
                    ++written_lines;
                    if (written_lines == 10)
                    {
                        offset = i + 1;
                        break;
                    }
                }
            }
            out.write(buf.data(), static_cast<std::streamsize>(offset));
        }
        {
            logosdb::DB db(dst, {.dim = dim});
            db.import_ndjson(truncated, /*chunk_size=*/5, checkpoint, /*resume=*/false);
            CHECK(db.count() == 10u);
        }
        // Now resume against the full file: importer should skip past the
        // already-imported prefix using the checkpoint byte offset and only
        // ingest the remaining rows.
        {
            logosdb::DB db(dst, {.dim = dim});
            db.import_ndjson(ndjson, /*chunk_size=*/5, checkpoint, /*resume=*/true);
            CHECK(db.count() == (size_t)n);
        }

        std::filesystem::remove_all(src);
        std::filesystem::remove_all(dst);
        std::filesystem::remove(ndjson);
        std::filesystem::remove(truncated);
        std::filesystem::remove(checkpoint);
    }
}  // TEST_SUITE("streaming")

// ============================================================================
// Test Suite: Timestamp Range Search
// ============================================================================

TEST_SUITE("ts_range")
{
    TEST_CASE("ts_range: basic functionality")
    {
        std::string path = tmp_path("logosdb_test_ts_range");
        std::filesystem::remove_all(path);

        int dim = 64;
        logosdb::DB db(path, {.dim = dim});

        auto v0 = unit_vec(dim, 10000);
        auto v1 = unit_vec(dim, 10100);
        auto v2 = unit_vec(dim, 10200);
        auto v3 = unit_vec(dim, 10300);

        db.put(v0, "early morning", "2025-01-15T08:00:00Z");
        db.put(v1, "mid morning", "2025-01-15T10:00:00Z");
        db.put(v2, "afternoon", "2025-01-15T14:00:00Z");
        db.put(v3, "evening", "2025-01-15T20:00:00Z");

        auto hits = db.search_ts_range(v0, 5, "2025-01-15T09:00:00Z", "2025-01-15T15:00:00Z");
        CHECK(hits.size() == 2);

        bool found_v1 = false, found_v2 = false;
        for (auto& h : hits)
        {
            if (h.id == 1)
                found_v1 = true;
            if (h.id == 2)
                found_v2 = true;
        }
        CHECK(found_v1);
        CHECK(found_v2);

        hits = db.search_ts_range(v0, 5, "2025-01-15T14:00:00Z", "");
        CHECK(hits.size() == 2);

        hits = db.search_ts_range(v0, 4, "", "");
        CHECK(hits.size() == 4);

        std::filesystem::remove_all(path);
    }

    TEST_CASE("ts_range: edge cases and C API")
    {
        std::string path = tmp_path("logosdb_test_ts_range_edge");
        std::filesystem::remove_all(path);

        int dim = 32;
        logosdb::DB db(path, {.dim = dim});

        auto v0 = unit_vec(dim, 11000);
        auto v1 = unit_vec(dim, 11100);
        db.put(v0, "entry1", "2025-02-01T12:00:00Z");
        db.put(v1, "entry2", "2025-02-01T14:00:00Z");

        auto hits = db.search_ts_range(v0, 5, "2025-12-01T00:00:00Z", "2025-12-31T23:59:59Z");
        CHECK(hits.empty());

        auto v2 = unit_vec(dim, 11200);
        db.put(v2, "no timestamp", "");
        hits = db.search_ts_range(v0, 5, "", "");
        CHECK(hits.size() == 3);

        char* err = nullptr;
        logosdb_search_result_t* r = logosdb_search_ts_range(db.handle(),
                                                             v0.data(),
                                                             dim,
                                                             2,
                                                             "2025-02-01T00:00:00Z",
                                                             "2025-02-01T23:59:59Z",
                                                             10,
                                                             &err);
        CHECK(r != nullptr);
        CHECK(err == nullptr);
        if (r)
        {
            CHECK(logosdb_result_count(r) == 2);
            logosdb_result_free(r);
        }

        r = logosdb_search_ts_range(nullptr, v0.data(), dim, 1, nullptr, nullptr, 10, &err);
        CHECK(r == nullptr);
        CHECK(err != nullptr);
        free(err);
        err = nullptr;

        r = logosdb_search_ts_range(db.handle(), v0.data(), dim, 0, nullptr, nullptr, 10, &err);
        CHECK(r == nullptr);
        CHECK(err != nullptr);
        free(err);

        std::filesystem::remove_all(path);
    }

    TEST_CASE("ts_range: recall with alternating timestamps")
    {
        std::string path = tmp_path("logosdb_test_ts_range_recall");
        std::filesystem::remove_all(path);

        int dim = 64;
        int n = 100;
        logosdb::DB db(path, {.dim = dim, .max_elements = 200});

        std::vector<std::vector<float>> morning_vecs;
        std::vector<std::vector<float>> evening_vecs;

        for (int i = 0; i < n; ++i)
        {
            auto v = unit_vec(dim, 12000 + i);
            if (i % 2 == 0)
            {
                morning_vecs.push_back(v);
                db.put(v, "morning_" + std::to_string(i), "2025-03-01T09:00:00Z");
            }
            else
            {
                evening_vecs.push_back(v);
                db.put(v, "evening_" + std::to_string(i), "2025-03-01T18:00:00Z");
            }
        }

        auto query = morning_vecs[0];
        auto hits = db.search_ts_range(query, 10, "2025-03-01T00:00:00Z", "2025-03-01T12:00:00Z");

        bool all_morning = true;
        for (auto& h : hits)
        {
            if (h.id % 2 != 0)
            {
                all_morning = false;
                break;
            }
        }
        CHECK(all_morning);
        CHECK(hits.size() == 10);

        for (size_t i = 1; i < hits.size(); ++i)
        {
            CHECK(hits[i - 1].score >= hits[i].score);
        }

        std::filesystem::remove_all(path);
    }

}  // TEST_SUITE("ts_range")

// ============================================================================
// Test Suite: Distance Metrics
// ============================================================================

TEST_SUITE("distance")
{
    TEST_CASE("distance: cosine auto-normalization")
    {
        std::string path = tmp_path("logosdb_test_cosine");
        std::filesystem::remove_all(path);

        int dim = 64;
        logosdb::DB db(path, {.dim = dim, .distance = LOGOSDB_DIST_COSINE});

        auto v0 = unit_vec(dim, 13000);
        auto v1 = unit_vec(dim, 13100);

        for (int i = 0; i < dim; ++i)
        {
            v0[i] *= 5.0f;
            v1[i] *= 3.0f;
        }

        db.put(v0, "scaled v0");
        db.put(v1, "scaled v1");

        auto scaled_query = v0;
        auto hits = db.search(scaled_query, 2);

        CHECK(!hits.empty());
        CHECK(hits[0].id == 0);
        CHECK(hits[0].score > 0.99f);

        auto self_hits = db.search(scaled_query, 1);
        CHECK(!self_hits.empty());
        CHECK(self_hits[0].score > 0.99f);

        std::filesystem::remove_all(path);
    }

    TEST_CASE("distance: L2 (Euclidean)")
    {
        std::string path = tmp_path("logosdb_test_l2");
        std::filesystem::remove_all(path);

        int dim = 64;
        logosdb::DB db(path, {.dim = dim, .distance = LOGOSDB_DIST_L2});

        auto v0 = unit_vec(dim, 14000);
        auto v1 = unit_vec(dim, 14100);

        db.put(v0, "v0");
        db.put(v1, "v1");

        auto hits = db.search(v0, 2);

        CHECK(!hits.empty());
        CHECK(hits[0].id == 0);
        CHECK(hits[0].score > hits[1].score);

        std::filesystem::remove_all(path);
    }

    TEST_CASE("distance: persistence across reopen")
    {
        std::string path = tmp_path("logosdb_test_dist_persist");
        std::filesystem::remove_all(path);

        int dim = 32;

        {
            logosdb::DB db(path, {.dim = dim, .distance = LOGOSDB_DIST_COSINE});
            auto v = unit_vec(dim, 15000);
            db.put(v, "test");
        }

        {
            logosdb::DB db(path, {.dim = dim, .distance = LOGOSDB_DIST_COSINE});
            auto v = unit_vec(dim, 15000);
            auto hits = db.search(v, 1);
            CHECK(!hits.empty());
            CHECK(hits[0].id == 0);
        }

        std::filesystem::remove_all(path);
    }

    TEST_CASE("distance: mismatch error")
    {
        std::string path = tmp_path("logosdb_test_dist_mismatch");
        std::filesystem::remove_all(path);

        int dim = 32;

        {
            logosdb::DB db(path, {.dim = dim, .distance = LOGOSDB_DIST_COSINE});
            auto v = unit_vec(dim, 16000);
            db.put(v, "test");
        }

        bool caught = false;
        try
        {
            logosdb::DB db(path, {.dim = dim, .distance = LOGOSDB_DIST_L2});
        }
        catch (const std::runtime_error& e)
        {
            caught = true;
            CHECK(std::string(e.what()).find("distance metric mismatch") != std::string::npos);
        }
        CHECK(caught);

        logosdb_options_t* opts = logosdb_options_create();
        CHECK(logosdb_options_set_distance(opts, LOGOSDB_DIST_IP) == 0);
        CHECK(logosdb_options_set_distance(opts, LOGOSDB_DIST_COSINE) == 0);
        CHECK(logosdb_options_set_distance(opts, LOGOSDB_DIST_L2) == 0);
        CHECK(logosdb_options_set_distance(opts, 99) == -1);
        CHECK(logosdb_options_set_distance(nullptr, LOGOSDB_DIST_IP) == -1);
        logosdb_options_destroy(opts);

        std::filesystem::remove_all(path);
    }

}  // TEST_SUITE("distance")

// ============================================================================
// Test Suite: CLI
// ============================================================================

TEST_SUITE("cli")
{
    TEST_CASE("cli: info reads dim from header")
    {
        std::string path = tmp_path("logosdb_test_cli_info");
        std::filesystem::remove_all(path);

        logosdb::DB db(path, {.dim = 64});
        auto v = unit_vec(64, 17000);
        db.put(v, "test entry", "2025-01-01T00:00:00Z");

        std::string output;
        int rc = run_cli("info " + path, output);
        CHECK(rc == 0);
        CHECK(output.find("dim        : 64") != std::string::npos);
        CHECK(output.find("count      : 1") != std::string::npos);

        output.clear();
        rc = run_cli("info " + path + " --json", output);
        CHECK(rc == 0);
        CHECK(output.find("\"dim\": 64") != std::string::npos);
        CHECK(output.find("\"count\": 1") != std::string::npos);

        std::filesystem::remove_all(path);
    }

    TEST_CASE("cli: export/import roundtrip")
    {
        std::string path = tmp_path("logosdb_test_cli_export");
        std::string import_path = tmp_path("logosdb_test_cli_import");
        std::string export_file = tmp_path("test_export.jsonl");
        std::filesystem::remove_all(path);
        std::filesystem::remove_all(import_path);
        std::remove(export_file.c_str());

        logosdb::DB db(path, {.dim = 32});
        auto v0 = unit_vec(32, 18000);
        auto v1 = unit_vec(32, 18100);
        db.put(v0, "first", "2025-01-10T00:00:00Z");
        db.put(v1, "second", "2025-01-11T00:00:00Z");

        std::string output;
        int rc = run_cli("export " + path + " --output " + export_file, output);
        CHECK(rc == 0);
        CHECK(output.find("Exported rows to") != std::string::npos);

        std::ifstream check(export_file);
        CHECK(check.good());
        check.close();

        rc = run_cli("import " + import_path + " --dim 32 --input " + export_file, output);
        CHECK(rc == 0);
        CHECK(output.find("Imported 2 rows") != std::string::npos);

        logosdb::DB import_db(import_path, {.dim = 32});
        CHECK(import_db.count() == 2);

        std::filesystem::remove_all(path);
        std::filesystem::remove_all(import_path);
        std::remove(export_file.c_str());
    }

    TEST_CASE("cli: search with timestamp range")
    {
        std::string path = tmp_path("logosdb_test_cli_ts");
        std::filesystem::remove_all(path);

        logosdb::DB db(path, {.dim = 32});
        auto v0 = unit_vec(32, 19000);
        auto v1 = unit_vec(32, 19100);
        db.put(v0, "early", "2025-01-10T08:00:00Z");
        db.put(v1, "late", "2025-01-15T18:00:00Z");

        {
            std::ofstream qf(tmp_path("query_vec.bin"), std::ios::binary);
            qf.write(reinterpret_cast<const char*>(v0.data()), 32 * sizeof(float));
            qf.close();
        }

        std::string output;
        int rc = run_cli("search " + path + " --query-file " + tmp_path("query_vec.bin") +
                             " --top-k 10 --json",
                         output);
        CHECK(rc == 0);
        int results = 0;
        size_t pos = 0;
        while ((pos = output.find("\"rank\":", pos)) != std::string::npos)
        {
            results++;
            pos++;
        }
        CHECK(results == 2);

        output.clear();
        rc = run_cli("search " + path + " --query-file " + tmp_path("query_vec.bin") +
                         " --ts-from 2025-01-01T00:00:00Z --ts-to "
                         "2025-01-12T00:00:00Z --top-k 10 --json",
                     output);
        CHECK(rc == 0);
        CHECK(output.find("early") != std::string::npos);
        CHECK(output.find("late") == std::string::npos);

        std::filesystem::remove_all(path);
        std::remove(tmp_path("query_vec.bin").c_str());
    }

    TEST_CASE("cli: doctor reports healthy database")
    {
        std::string path = tmp_path("logosdb_test_cli_doctor");
        std::filesystem::remove_all(path);

        logosdb::DB db(path, {.dim = 32});
        db.put(unit_vec(32, 42000), "x", "2025-01-01T00:00:00Z");

        std::string output;
        int rc = run_cli("doctor " + path, output);
        CHECK(rc == 0);
        CHECK(output.find("probe logosdb_open") != std::string::npos);
        CHECK(output.find("vectors layout      : OK") != std::string::npos);

        output.clear();
        rc = run_cli("doctor " + path + " --json", output);
        CHECK(rc == 0);
        CHECK(output.find("\"library_version\":") != std::string::npos);
        CHECK(output.find(LOGOSDB_VERSION_STRING) != std::string::npos);
        CHECK(output.find("\"layout_ok\": true") != std::string::npos);

        std::filesystem::remove_all(path);
    }

    TEST_CASE("cli: upgrade dry-run and apply on legacy vectors header")
    {
        std::string path = tmp_path("logosdb_test_cli_upgrade_hdr");
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
        const std::string vec = path + "/vectors.bin";

        {
            std::ofstream out(vec, std::ios::binary);
            uint32_t hdr32[4] = {0x4C4F474FU, 1u, 4u, 0u};
            uint64_t n_rows = 1ull;
            uint64_t reserved2 = 0ull;
            out.write(reinterpret_cast<const char*>(hdr32), sizeof(hdr32));
            out.write(reinterpret_cast<const char*>(&n_rows), sizeof(n_rows));
            out.write(reinterpret_cast<const char*>(&reserved2), sizeof(reserved2));
            float row[4] = {0.25f, 0.25f, 0.25f, 0.25f};
            out.write(reinterpret_cast<const char*>(row), sizeof(row));
        }

        std::string o;
        CHECK(run_cli("upgrade " + path, o) == 0);
        CHECK(o.find("dry run") != std::string::npos);

        o.clear();
        CHECK(run_cli("upgrade " + path + " --apply", o) == 0);
        CHECK(o.find("Wrote storage header v2") != std::string::npos);

        std::ifstream in(vec, std::ios::binary);
        uint32_t ver = 0;
        in.seekg(4);
        in.read(reinterpret_cast<char*>(&ver), sizeof(ver));
        CHECK(ver == 2u);

        o.clear();
        CHECK(run_cli("upgrade " + path, o) == 0);
        CHECK(o.find("already storage format v2") != std::string::npos);

        std::filesystem::remove_all(path);
    }

    TEST_CASE("cli: snapshot and restore roundtrip")
    {
        std::string src = tmp_path("logosdb_test_snap_src");
        std::string snap = tmp_path("logosdb_test_snap_out");
        std::string dst = tmp_path("logosdb_test_snap_dst");
        std::filesystem::remove_all(src);
        std::filesystem::remove_all(snap);
        std::filesystem::remove_all(dst);

        {
            logosdb::DB db(src, {.dim = 32});
            db.put(unit_vec(32, 50111), "hello", "2025-01-01T00:00:00Z");
            db.put(unit_vec(32, 50222), "world", "2025-02-01T00:00:00Z");
        }

        std::string output;
        CHECK(run_cli("snapshot " + src + " " + snap + " --overwrite", output) == 0);
        CHECK(output.find("snapshot written") != std::string::npos);
        CHECK(std::filesystem::exists(snap + "/snapshot.json"));

        output.clear();
        CHECK(run_cli("restore " + snap + " " + dst + " --force", output) == 0);
        CHECK(output.find("restored snapshot") != std::string::npos);

        {
            logosdb::DB db(dst, {.dim = 32});
            CHECK(db.count() == 2);
            auto hits = db.search(unit_vec(32, 50111), 1);
            CHECK(!hits.empty());
            CHECK(hits[0].text == "hello");
        }

        std::filesystem::remove_all(src);
        std::filesystem::remove_all(snap);
        std::filesystem::remove_all(dst);
    }

    TEST_CASE("cli: restore rejects missing manifest")
    {
        std::string snap = tmp_path("logosdb_test_restore_bad");
        std::string dst = tmp_path("logosdb_test_restore_dst");
        std::filesystem::remove_all(snap);
        std::filesystem::remove_all(dst);
        std::filesystem::create_directories(snap);
        std::string o;
        CHECK(run_cli("restore " + snap + " " + dst + " --force", o) != 0);
        std::filesystem::remove_all(snap);
    }

    TEST_CASE("cli: stats and compact")
    {
        std::string src = tmp_path("logosdb_test_cli_stats_src");
        std::string dst = tmp_path("logosdb_test_cli_stats_dst");
        std::filesystem::remove_all(src);
        std::filesystem::remove_all(dst);

        {
            logosdb::DB db(src, {.dim = 16});
            db.put(unit_vec(16, 88001), "a");
            db.put(unit_vec(16, 88002), "b");
            db.del(0);
            auto hits = db.search(unit_vec(16, 88002), 1);
            CHECK(!hits.empty());
        }

        std::string o;
        CHECK(run_cli("stats " + src, o) == 0);
        CHECK(o.find("put ok / fail") != std::string::npos);
        CHECK(o.find("search ok / fail") != std::string::npos);

        o.clear();
        CHECK(run_cli("compact " + src + " " + dst + " --force", o) == 0);
        CHECK(o.find("compact completed") != std::string::npos);

        {
            logosdb::DB db2(dst, {.dim = 16});
            CHECK(db2.count() == 1);
            CHECK(db2.count_live() == 1);
            auto h = db2.search(unit_vec(16, 88002), 1);
            CHECK(!h.empty());
            CHECK(h[0].text == "b");
        }

        std::filesystem::remove_all(src);
        std::filesystem::remove_all(dst);
    }

}  // TEST_SUITE("cli")

// ============================================================================
// Test Suite: Storage (reduced precision)
// ============================================================================

TEST_SUITE("storage")
{
    TEST_CASE("storage: float16 basic")
    {
        std::string path = tmp_path("logosdb_test_float16");
        std::filesystem::remove_all(path);

        int dim = 64;
        logosdb::DB db(path, {.dim = dim, .dtype = LOGOSDB_DTYPE_FLOAT16});

        auto v0 = unit_vec(dim, 8000);
        auto v1 = unit_vec(dim, 8100);
        auto v2 = unit_vec(dim, 8200);

        db.put(v0, "first", "2025-01-01T00:00:00Z");
        db.put(v1, "second", "2025-01-02T00:00:00Z");
        db.put(v2, "third", "2025-01-03T00:00:00Z");

        CHECK(db.count() == 3);

        auto hits = db.search(v0, 3);
        CHECK(!hits.empty());
        CHECK(hits[0].id == 0);
        CHECK(hits[0].score > 0.99f);

        {
            logosdb::DB db2(path, {.dim = dim});
            CHECK(db2.count() == 3);

            auto hits2 = db2.search(v1, 3);
            CHECK(!hits2.empty());
            CHECK(hits2[0].id == 1);
        }

        std::filesystem::remove_all(path);
    }

    TEST_CASE("storage: int8 basic")
    {
        std::string path = tmp_path("logosdb_test_int8");
        std::filesystem::remove_all(path);

        int dim = 64;
        logosdb::DB db(path, {.dim = dim, .dtype = LOGOSDB_DTYPE_INT8});

        auto v0 = unit_vec(dim, 9000);
        auto v1 = unit_vec(dim, 9100);
        auto v2 = unit_vec(dim, 9200);

        db.put(v0, "first", "2025-01-01T00:00:00Z");
        db.put(v1, "second", "2025-01-02T00:00:00Z");
        db.put(v2, "third", "2025-01-03T00:00:00Z");

        CHECK(db.count() == 3);

        auto hits = db.search(v0, 3);
        CHECK(!hits.empty());
        CHECK(hits[0].id == 0);
        CHECK(hits[0].score > 0.90f);

        {
            logosdb::DB db2(path, {.dim = dim});
            CHECK(db2.count() == 3);

            auto hits2 = db2.search(v1, 3);
            CHECK(!hits2.empty());
            CHECK(hits2[0].id == 1);
        }

        std::filesystem::remove_all(path);
    }

    TEST_CASE("storage: dtype persistence")
    {
        std::string path = tmp_path("logosdb_test_dtype_persist");
        std::filesystem::remove_all(path);

        int dim = 32;

        {
            logosdb::DB db(path, {.dim = dim, .dtype = LOGOSDB_DTYPE_FLOAT16});
            auto v = unit_vec(dim, 10000);
            db.put(v, "test");
        }

        {
            logosdb::DB db(path, {.dim = dim});
            CHECK(db.count() == 1);

            auto v = unit_vec(dim, 10000);
            auto hits = db.search(v, 1);
            CHECK(!hits.empty());
        }

        std::filesystem::remove_all(path);
    }

    TEST_CASE("storage: pointers stable across appends")
    {
        std::string path = tmp_path("logosdb_test_stable_pointers");
        std::filesystem::remove_all(path);

        int dim = 32;
        logosdb::DB db(path, {.dim = dim});

        auto v0 = unit_vec(dim, 20000);
        db.put(v0, "first", "2025-01-01T00:00:00Z");

        size_t n_rows = 0;
        int d = 0;
        auto raw = db.raw_vectors(n_rows, d);
        CHECK(!raw.empty());
        CHECK(n_rows == 1);

        float diff = 0.0f;
        for (int i = 0; i < dim; ++i)
            diff += std::fabs(raw[i] - v0[i]);
        CHECK(diff < 1e-5f);

        for (int i = 0; i < 100; ++i)
        {
            auto v = unit_vec(dim, 20001 + i);
            db.put(v, "batch", "2025-01-01T00:00:00Z");
        }

        raw = db.raw_vectors(n_rows, d);
        CHECK(n_rows == 101);

        diff = 0.0f;
        for (int i = 0; i < dim; ++i)
            diff += std::fabs(raw[i] - v0[i]);
        CHECK(diff < 1e-5f);

        auto hits = db.search(v0, 1);
        CHECK(!hits.empty());
        CHECK(hits[0].id == 0);

        std::filesystem::remove_all(path);
    }

}  // TEST_SUITE("storage")

// ============================================================================
// Test Suite: L2 Normalization
// ============================================================================

TEST_SUITE("l2_normalize")
{
    TEST_CASE("l2_normalize: basic random vectors")
    {
        std::mt19937 rng(12345);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        int dim = 128;
        std::vector<float> vec(dim);
        for (int i = 0; i < dim; ++i)
            vec[i] = dist(rng);

        float orig_norm_sq = 0.0f;
        for (float v : vec)
            orig_norm_sq += v * v;
        float orig_norm = std::sqrt(orig_norm_sq);
        CHECK(orig_norm > 5.0f);

        int rc = logosdb_l2_normalize(vec.data(), dim);
        CHECK(rc == 0);

        float norm_sq = 0.0f;
        for (float v : vec)
            norm_sq += v * v;
        float norm = std::sqrt(norm_sq);
        CHECK(std::abs(norm - 1.0f) < 1e-5f);

        std::vector<float> v2(dim);
        for (int i = 0; i < dim; ++i)
            v2[i] = 3.0f * dist(rng);

        rc = logosdb_l2_normalize(v2.data(), dim);
        CHECK(rc == 0);

        norm_sq = 0.0f;
        for (float v : v2)
            norm_sq += v * v;
        norm = std::sqrt(norm_sq);
        CHECK(std::abs(norm - 1.0f) < 1e-5f);
    }

    TEST_CASE("l2_normalize: already normalized stays unit")
    {
        int dim = 64;
        auto v = unit_vec(dim, 50000);

        float norm_sq = 0.0f;
        for (float val : v)
            norm_sq += val * val;
        CHECK(std::abs(std::sqrt(norm_sq) - 1.0f) < 1e-5f);

        int rc = logosdb_l2_normalize(v.data(), dim);
        CHECK(rc == 0);

        norm_sq = 0.0f;
        for (float val : v)
            norm_sq += val * val;
        float norm = std::sqrt(norm_sq);
        CHECK(std::abs(norm - 1.0f) < 1e-5f);
    }

    TEST_CASE("l2_normalize: zero vector returns error")
    {
        int dim = 32;
        std::vector<float> zero_vec(dim, 0.0f);

        int rc = logosdb_l2_normalize(zero_vec.data(), dim);
        CHECK(rc == -1);

        for (float v : zero_vec)
            CHECK(v == 0.0f);
    }

    TEST_CASE("l2_normalize: very small values")
    {
        int dim = 128;
        std::vector<float> small_vec(dim);
        for (int i = 0; i < dim; ++i)
            small_vec[i] = 1e-20f;

        int rc = logosdb_l2_normalize(small_vec.data(), dim);
        CHECK(rc == 0);

        float norm_sq = 0.0f;
        for (float v : small_vec)
            norm_sq += v * v;
        float norm = std::sqrt(norm_sq);
        CHECK(std::abs(norm - 1.0f) < 1e-5f);
    }

    TEST_CASE("l2_normalize: C++ wrappers")
    {
        int dim = 64;
        std::mt19937 rng(60000);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        std::vector<float> v1(dim);
        for (int i = 0; i < dim; ++i)
            v1[i] = 5.0f * dist(rng);

        bool ok = logosdb::l2_normalize(v1);
        CHECK(ok);

        float norm_sq = 0.0f;
        for (float v : v1)
            norm_sq += v * v;
        CHECK(std::abs(std::sqrt(norm_sq) - 1.0f) < 1e-5f);

        std::vector<float> v2(dim);
        for (int i = 0; i < dim; ++i)
            v2[i] = 3.0f * dist(rng);
        std::vector<float> v2_orig = v2;

        std::vector<float> v2_normed = logosdb::l2_normalized(v2);

        for (int i = 0; i < dim; ++i)
            CHECK(v2[i] == v2_orig[i]);

        norm_sq = 0.0f;
        for (float v : v2_normed)
            norm_sq += v * v;
        CHECK(std::abs(std::sqrt(norm_sq) - 1.0f) < 1e-5f);

        std::vector<float> zero_vec(dim, 0.0f);
        ok = logosdb::l2_normalize(zero_vec);
        CHECK(!ok);
    }

}  // TEST_SUITE("l2_normalize")

// ============================================================================
// Test Suite: Scoring
// ============================================================================

TEST_SUITE("scoring")
{
    TEST_CASE("scoring: self-similarity is ~1.0")
    {
        std::string path = tmp_path("logosdb_test_selfscore");
        std::filesystem::remove_all(path);

        int dim = 64;
        logosdb::DB db(path, {.dim = dim});
        auto v = unit_vec(dim, 1500);
        db.put(v, "self");

        auto hits = db.search(v, 1);
        CHECK(!hits.empty());
        CHECK(hits[0].score > 0.99f);

        std::filesystem::remove_all(path);
    }

}  // TEST_SUITE("scoring")
