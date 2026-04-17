#include <logosdb/logosdb.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::vector<float> random_unit_vec(int dim, std::mt19937 & rng) {
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(dim);
    float norm = 0.0f;
    for (int i = 0; i < dim; ++i) { v[i] = dist(rng); norm += v[i] * v[i]; }
    norm = std::sqrt(norm);
    if (norm > 0.0f) for (int i = 0; i < dim; ++i) v[i] /= norm;
    return v;
}

static float dot(const float * a, const float * b, int dim) {
    float s = 0.0f;
    for (int i = 0; i < dim; ++i) s += a[i] * b[i];
    return s;
}

struct BenchResult {
    int    n;
    double put_ms;
    double hnsw_search_ms;
    double brute_search_ms;
    double recall_at_k;
};

static BenchResult run_bench(int dim, int n, int n_queries, int top_k) {
    BenchResult result = {};
    result.n = n;

    std::mt19937 rng(42);
    std::string tmp = "/tmp/logosdb_bench_" + std::to_string(n);
    std::filesystem::remove_all(tmp);

    logosdb::Options opts;
    opts.dim = dim;
    opts.max_elements = (size_t)(n * 1.2);
    logosdb::DB db(tmp, opts);

    // Generate and insert vectors.
    std::vector<std::vector<float>> vecs(n);
    double t0 = now_ms();
    for (int i = 0; i < n; ++i) {
        vecs[i] = random_unit_vec(dim, rng);
        db.put(vecs[i], "row_" + std::to_string(i));
    }
    result.put_ms = now_ms() - t0;

    // Generate queries.
    std::vector<std::vector<float>> queries(n_queries);
    for (int q = 0; q < n_queries; ++q) {
        queries[q] = random_unit_vec(dim, rng);
    }

    // HNSW search.
    t0 = now_ms();
    std::vector<std::vector<uint64_t>> hnsw_results(n_queries);
    for (int q = 0; q < n_queries; ++q) {
        auto hits = db.search(queries[q], top_k);
        for (auto & h : hits) hnsw_results[q].push_back(h.id);
    }
    result.hnsw_search_ms = (now_ms() - t0) / n_queries;

    // Brute-force search for comparison.
    t0 = now_ms();
    std::vector<std::vector<uint64_t>> brute_results(n_queries);
    for (int q = 0; q < n_queries; ++q) {
        std::vector<std::pair<float, uint64_t>> scores(n);
        for (int i = 0; i < n; ++i) {
            scores[i] = {dot(queries[q].data(), vecs[i].data(), dim), (uint64_t)i};
        }
        std::partial_sort(scores.begin(), scores.begin() + top_k, scores.end(),
            [](auto & a, auto & b) { return a.first > b.first; });
        for (int k = 0; k < top_k && k < n; ++k) {
            brute_results[q].push_back(scores[k].second);
        }
    }
    result.brute_search_ms = (now_ms() - t0) / n_queries;

    // Recall@k: fraction of true top-k found by HNSW.
    double total_recall = 0.0;
    for (int q = 0; q < n_queries; ++q) {
        int found = 0;
        for (auto hid : hnsw_results[q]) {
            for (auto bid : brute_results[q]) {
                if (hid == bid) { ++found; break; }
            }
        }
        total_recall += (double)found / std::min(top_k, n);
    }
    result.recall_at_k = total_recall / n_queries;

    std::filesystem::remove_all(tmp);
    return result;
}

int main(int argc, char ** argv) {
    int dim = 2048;
    std::vector<int> counts = {1000, 10000, 100000};
    int n_queries = 50;
    int top_k = 10;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--dim") && i + 1 < argc) dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--queries") && i + 1 < argc) n_queries = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--counts") && i + 1 < argc) {
            counts.clear();
            char * tok = strtok(argv[++i], ",");
            while (tok) { counts.push_back(atoi(tok)); tok = strtok(nullptr, ","); }
        }
    }

    printf("LogosDB Benchmark — dim=%d top_k=%d queries=%d\n", dim, top_k, n_queries);
    printf("%-10s %12s %14s %14s %10s\n",
           "N", "put(ms)", "HNSW(ms/q)", "brute(ms/q)", "recall@k");
    printf("---------- ------------ -------------- -------------- ----------\n");

    for (int n : counts) {
        auto r = run_bench(dim, n, n_queries, top_k);
        printf("%-10d %12.1f %14.3f %14.3f %9.1f%%\n",
               r.n, r.put_ms, r.hnsw_search_ms, r.brute_search_ms,
               r.recall_at_k * 100.0);
    }

    printf("\n--- ChromaDB comparison (published estimates, dim=2048) ---\n");
    printf("%-20s %14s %14s\n", "Metric", "ChromaDB", "LogosDB");
    printf("%-20s %14s %14s\n", "Language", "Python+C", "C/C++");
    printf("%-20s %14s %14s\n", "Search algo", "HNSW", "HNSW");
    printf("%-20s %14s %14s\n", "Storage", "SQLite+Parquet", "mmap+JSONL");
    printf("%-20s %14s %14s\n", "100K search (ms)", "~5-10", "(see above)");
    printf("%-20s %14s %14s\n", "Startup overhead", "Python RT", "zero");

    return 0;
}
