// Opt-in stress tests: registered under doctest suite "stress".
// Default runs exclude this suite (CMake add_test + CI). To run locally:
//   ./logosdb-test --test-suite=stress
// Or run everything including stress:
//   ./logosdb-test

#include "doctest.h"

#include <logosdb/logosdb.h>

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace
{

std::vector<float> unit_vec(int dim, int seed)
{
    std::mt19937 rng(static_cast<unsigned>(seed));
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(static_cast<size_t>(dim));
    float norm = 0.0f;
    for (int i = 0; i < dim; ++i)
    {
        v[static_cast<size_t>(i)] = dist(rng);
        norm += v[static_cast<size_t>(i)] * v[static_cast<size_t>(i)];
    }
    norm = std::sqrt(norm);
    for (int i = 0; i < dim; ++i)
        v[static_cast<size_t>(i)] /= norm;
    return v;
}

}  // namespace

TEST_SUITE("stress")
{
    TEST_CASE("stress: many sequential puts")
    {
        const std::string path = "/tmp/logosdb_stress_puts";
        std::filesystem::remove_all(path);

        char* err = nullptr;
        logosdb_options_t* opts = logosdb_options_create();
        logosdb_options_set_dim(opts, 64);
        logosdb_t* db = logosdb_open(path.c_str(), opts, &err);
        logosdb_options_destroy(opts);
        REQUIRE(db != nullptr);
        REQUIRE(err == nullptr);

        constexpr int n = 8000;
        for (int i = 0; i < n; ++i)
        {
            auto v = unit_vec(64, 10000 + i);
            uint64_t id = logosdb_put(db, v.data(), 64, "row", nullptr, &err);
            REQUIRE(id != UINT64_MAX);
            REQUIRE(err == nullptr);
        }
        CHECK(logosdb_count(db) == n);

        err = nullptr;
        CHECK(logosdb_sync(db, &err) == 0);
        CHECK(err == nullptr);

        logosdb_close(db);
        std::filesystem::remove_all(path);
    }

    TEST_CASE("stress: search under load")
    {
        const std::string path = "/tmp/logosdb_stress_search";
        std::filesystem::remove_all(path);

        char* err = nullptr;
        logosdb_options_t* opts = logosdb_options_create();
        logosdb_options_set_dim(opts, 32);
        logosdb_t* db = logosdb_open(path.c_str(), opts, &err);
        logosdb_options_destroy(opts);
        REQUIRE(db != nullptr);
        REQUIRE(err == nullptr);

        constexpr int rows = 2500;
        for (int i = 0; i < rows; ++i)
        {
            auto v = unit_vec(32, 20000 + i);
            uint64_t id = logosdb_put(db, v.data(), 32, "s", nullptr, &err);
            REQUIRE(id != UINT64_MAX);
        }

        constexpr int queries = 2000;
        for (int q = 0; q < queries; ++q)
        {
            auto qv = unit_vec(32, 50000 + q);
            logosdb_search_result_t* r = logosdb_search(db, qv.data(), 32, 8, &err);
            REQUIRE(r != nullptr);
            REQUIRE(err == nullptr);
            CHECK(logosdb_result_count(r) >= 1);
            logosdb_result_free(r);
        }

        logosdb_close(db);
        std::filesystem::remove_all(path);
    }

    TEST_CASE("stress: large put_batch")
    {
        const std::string path = "/tmp/logosdb_stress_batch";
        std::filesystem::remove_all(path);

        char* err = nullptr;
        logosdb_options_t* opts = logosdb_options_create();
        constexpr int dim = 48;
        constexpr int n = 4000;
        logosdb_options_set_dim(opts, dim);
        logosdb_t* db = logosdb_open(path.c_str(), opts, &err);
        logosdb_options_destroy(opts);
        REQUIRE(db != nullptr);
        REQUIRE(err == nullptr);

        std::vector<float> emb(static_cast<size_t>(n) * static_cast<size_t>(dim));
        for (int i = 0; i < n; ++i)
        {
            auto v = unit_vec(dim, 30000 + i);
            for (int d = 0; d < dim; ++d)
                emb[static_cast<size_t>(i) * static_cast<size_t>(dim) + static_cast<size_t>(d)] =
                    v[static_cast<size_t>(d)];
        }

        std::vector<const char*> texts(static_cast<size_t>(n), "b");
        std::vector<uint64_t> out_ids(static_cast<size_t>(n));
        int rc =
            logosdb_put_batch(db, emb.data(), n, dim, texts.data(), nullptr, out_ids.data(), &err);
        CHECK(rc == 0);
        CHECK(err == nullptr);
        CHECK(logosdb_count(db) == n);

        logosdb_close(db);
        std::filesystem::remove_all(path);
    }

    TEST_CASE("stress: interleaved put search sync")
    {
        const std::string path = "/tmp/logosdb_stress_mixed";
        std::filesystem::remove_all(path);

        char* err = nullptr;
        logosdb_options_t* opts = logosdb_options_create();
        logosdb_options_set_dim(opts, 16);
        logosdb_t* db = logosdb_open(path.c_str(), opts, &err);
        logosdb_options_destroy(opts);
        REQUIRE(db != nullptr);
        REQUIRE(err == nullptr);

        for (int round = 0; round < 40; ++round)
        {
            for (int i = 0; i < 200; ++i)
            {
                int idx = round * 200 + i;
                auto v = unit_vec(16, 40000 + idx);
                REQUIRE(logosdb_put(db, v.data(), 16, "m", nullptr, &err) != UINT64_MAX);
            }
            auto q = unit_vec(16, 90000 + round);
            logosdb_search_result_t* r = logosdb_search(db, q.data(), 16, 5, &err);
            REQUIRE(r != nullptr);
            logosdb_result_free(r);
            err = nullptr;
            CHECK(logosdb_sync(db, &err) == 0);
            CHECK(err == nullptr);
        }
        CHECK(logosdb_count(db) == 8000);

        logosdb_close(db);
        std::filesystem::remove_all(path);
    }

    TEST_CASE("stress: concurrent puts (shared handle)")
    {
        const std::string path = "/tmp/logosdb_stress_par_puts";
        std::filesystem::remove_all(path);

        char* err = nullptr;
        logosdb_options_t* opts = logosdb_options_create();
        logosdb_options_set_dim(opts, 32);
        logosdb_t* db = logosdb_open(path.c_str(), opts, &err);
        logosdb_options_destroy(opts);
        REQUIRE(db != nullptr);
        REQUIRE(err == nullptr);

        constexpr int n_threads = 8;
        constexpr int per_thread = 400;
        std::atomic<int> fail_count{0};

        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(n_threads));
        for (int t = 0; t < n_threads; ++t)
        {
            workers.emplace_back(
                [db, t, &fail_count]()
                {
                    for (int i = 0; i < per_thread; ++i)
                    {
                        char* e = nullptr;
                        auto v = unit_vec(32, t * 200000 + i);
                        uint64_t id = logosdb_put(db, v.data(), 32, "par", nullptr, &e);
                        if (id == UINT64_MAX || e != nullptr)
                        {
                            if (e)
                                free(e);
                            fail_count.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                });
        }
        for (auto& w : workers)
            w.join();

        CHECK(fail_count.load(std::memory_order_relaxed) == 0);
        CHECK(logosdb_count(db) == n_threads * per_thread);

        err = nullptr;
        CHECK(logosdb_sync(db, &err) == 0);
        CHECK(err == nullptr);

        logosdb_close(db);
        std::filesystem::remove_all(path);
    }

    TEST_CASE("stress: concurrent searches (shared handle)")
    {
        const std::string path = "/tmp/logosdb_stress_par_search";
        std::filesystem::remove_all(path);

        char* err = nullptr;
        logosdb_options_t* opts = logosdb_options_create();
        logosdb_options_set_dim(opts, 24);
        logosdb_t* db = logosdb_open(path.c_str(), opts, &err);
        logosdb_options_destroy(opts);
        REQUIRE(db != nullptr);
        REQUIRE(err == nullptr);

        constexpr int seed_rows = 1200;
        for (int i = 0; i < seed_rows; ++i)
        {
            auto v = unit_vec(24, 70000 + i);
            REQUIRE(logosdb_put(db, v.data(), 24, "seed", nullptr, &err) != UINT64_MAX);
        }

        constexpr int n_threads = 8;
        constexpr int queries_per_thread = 400;
        std::atomic<int> fail_count{0};

        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(n_threads));
        for (int t = 0; t < n_threads; ++t)
        {
            workers.emplace_back(
                [db, t, &fail_count]()
                {
                    for (int q = 0; q < queries_per_thread; ++q)
                    {
                        char* e = nullptr;
                        auto qv = unit_vec(24, 800000 + t * 100000 + q);
                        logosdb_search_result_t* r = logosdb_search(db, qv.data(), 24, 10, &e);
                        if (!r || e != nullptr)
                        {
                            if (e)
                                free(e);
                            fail_count.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                        if (logosdb_result_count(r) < 1)
                            fail_count.fetch_add(1, std::memory_order_relaxed);
                        logosdb_result_free(r);
                    }
                });
        }
        for (auto& w : workers)
            w.join();

        CHECK(fail_count.load(std::memory_order_relaxed) == 0);
        CHECK(logosdb_count(db) == seed_rows);

        logosdb_close(db);
        std::filesystem::remove_all(path);
    }

    TEST_CASE("stress: concurrent mixed put and search")
    {
        const std::string path = "/tmp/logosdb_stress_par_mixed";
        std::filesystem::remove_all(path);

        char* err = nullptr;
        logosdb_options_t* opts = logosdb_options_create();
        logosdb_options_set_dim(opts, 16);
        logosdb_t* db = logosdb_open(path.c_str(), opts, &err);
        logosdb_options_destroy(opts);
        REQUIRE(db != nullptr);
        REQUIRE(err == nullptr);

        for (int i = 0; i < 300; ++i)
        {
            auto v = unit_vec(16, 60000 + i);
            REQUIRE(logosdb_put(db, v.data(), 16, "base", nullptr, &err) != UINT64_MAX);
        }

        constexpr int n_putters = 4;
        constexpr int n_searchers = 4;
        constexpr int put_iters = 250;
        constexpr int search_iters = 300;
        std::atomic<int> fail_count{0};

        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(n_putters + n_searchers));

        for (int t = 0; t < n_putters; ++t)
        {
            workers.emplace_back(
                [db, t, &fail_count]()
                {
                    for (int i = 0; i < put_iters; ++i)
                    {
                        char* e = nullptr;
                        auto v = unit_vec(16, 900000 + t * 50000 + i);
                        uint64_t id = logosdb_put(db, v.data(), 16, "mix", nullptr, &e);
                        if (id == UINT64_MAX || e != nullptr)
                        {
                            if (e)
                                free(e);
                            fail_count.fetch_add(1, std::memory_order_relaxed);
                        }
                    }
                });
        }
        for (int t = 0; t < n_searchers; ++t)
        {
            workers.emplace_back(
                [db, t, &fail_count]()
                {
                    for (int q = 0; q < search_iters; ++q)
                    {
                        char* e = nullptr;
                        auto qv = unit_vec(16, 1200000 + t * 70000 + q);
                        logosdb_search_result_t* r = logosdb_search(db, qv.data(), 16, 6, &e);
                        if (!r || e != nullptr)
                        {
                            if (e)
                                free(e);
                            fail_count.fetch_add(1, std::memory_order_relaxed);
                            continue;
                        }
                        logosdb_result_free(r);
                    }
                });
        }

        for (auto& w : workers)
            w.join();

        CHECK(fail_count.load(std::memory_order_relaxed) == 0);
        CHECK(logosdb_count(db) == 300 + n_putters * put_iters);

        err = nullptr;
        CHECK(logosdb_sync(db, &err) == 0);
        CHECK(err == nullptr);

        logosdb_close(db);
        std::filesystem::remove_all(path);
    }
}
