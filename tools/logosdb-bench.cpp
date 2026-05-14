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

static double now_ms()
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

static std::vector<float> random_unit_vec(int dim, std::mt19937& rng)
{
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> v(dim);
    float norm = 0.0f;
    for (int i = 0; i < dim; ++i)
    {
        v[i] = dist(rng);
        norm += v[i] * v[i];
    }
    norm = std::sqrt(norm);
    if (norm > 0.0f)
        for (int i = 0; i < dim; ++i)
            v[i] /= norm;
    return v;
}

static float dot(const float* a, const float* b, int dim)
{
    float s = 0.0f;
    for (int i = 0; i < dim; ++i)
        s += a[i] * b[i];
    return s;
}

struct BenchResult
{
    int n;
    double put_ms;
    double hnsw_search_ms;
    double brute_search_ms;
    double recall_at_k;
};

static BenchResult run_bench(int dim, int n, int n_queries, int top_k)
{
    BenchResult result = {};
    result.n = n;

    std::mt19937 rng(42);
    std::string tmp = "/tmp/logosdb_bench_" + std::to_string(n);
    std::filesystem::remove_all(tmp);

    // Scale efSearch with corpus size so HNSW recall stays representative.
    // Floor is 50 (5× the default top_k=10) — using top_k as the floor
    // guarantees terrible recall because HNSW must explore >> top_k candidates.
    int ef_search = std::max(50, std::min(500, n / 50));

    logosdb::Options opts;
    opts.dim = dim;
    opts.max_elements = (size_t)(n * 1.2);
    opts.ef_search = ef_search;
    // Higher construction params give better graph quality at the cost of build time;
    // these are ceiling values for a recall benchmark — production defaults are M=16, efc=200.
    opts.M = 32;
    opts.ef_construction = 400;
    logosdb::DB db(tmp, opts);

    // Generate and insert vectors; capture the WAL row IDs returned by put().
    // The benchmark recall check compares HNSW hit IDs against these real IDs —
    // using the loop index directly was wrong when WAL IDs have any offset.
    std::vector<std::vector<float>> vecs(n);
    std::vector<uint64_t> inserted_ids(n);
    double t0 = now_ms();
    for (int i = 0; i < n; ++i)
    {
        vecs[i] = random_unit_vec(dim, rng);
        inserted_ids[i] = (uint64_t)db.put(vecs[i], "row_" + std::to_string(i));
    }
    result.put_ms = now_ms() - t0;

    // Generate queries.
    std::vector<std::vector<float>> queries(n_queries);
    for (int q = 0; q < n_queries; ++q)
    {
        queries[q] = random_unit_vec(dim, rng);
    }

    // HNSW search.
    t0 = now_ms();
    std::vector<std::vector<uint64_t>> hnsw_results(n_queries);
    for (int q = 0; q < n_queries; ++q)
    {
        auto hits = db.search(queries[q], top_k);
        for (auto& h : hits)
            hnsw_results[q].push_back(h.id);
    }
    result.hnsw_search_ms = (now_ms() - t0) / n_queries;

    // Brute-force search for comparison.
    t0 = now_ms();
    std::vector<std::vector<uint64_t>> brute_results(n_queries);
    for (int q = 0; q < n_queries; ++q)
    {
        std::vector<std::pair<float, uint64_t>> scores(n);
        for (int i = 0; i < n; ++i)
        {
            // Use the actual WAL row ID, not the loop index.
            scores[i] = {dot(queries[q].data(), vecs[i].data(), dim), inserted_ids[i]};
        }
        std::partial_sort(scores.begin(),
                          scores.begin() + top_k,
                          scores.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
        for (int k = 0; k < top_k && k < n; ++k)
        {
            brute_results[q].push_back(scores[k].second);
        }
    }
    result.brute_search_ms = (now_ms() - t0) / n_queries;

    // Recall@k: fraction of true top-k found by HNSW.
    double total_recall = 0.0;
    for (int q = 0; q < n_queries; ++q)
    {
        int found = 0;
        for (auto hid : hnsw_results[q])
        {
            for (auto bid : brute_results[q])
            {
                if (hid == bid)
                {
                    ++found;
                    break;
                }
            }
        }
        total_recall += (double)found / std::min(top_k, n);
    }
    result.recall_at_k = total_recall / n_queries;

    std::filesystem::remove_all(tmp);
    return result;
}

// ── Recall sweep ─────────────────────────────────────────────────────────────
//
// Builds one HNSW index and sweeps efSearch to show the recall / latency
// trade-off.  Useful for tuning HNSW parameters for a specific workload.
//
// Usage:  logosdb-bench --recall-sweep [options]
//   --sweep-n N              Corpus size            (default 10000)
//   --sweep-M N              HNSW M                 (default 16)
//   --sweep-efc N            HNSW efConstruction    (default 200)
//   --sweep-ef 10,20,...     efSearch values        (default 10,20,50,100,200,500)
//   --sweep-queries N        Queries per ef value   (default 200)
//   --dim D                  Vector dimension       (default 384)
//   --top-k K                top-k                  (default 10)
// ---------------------------------------------------------------------------

static void run_recall_sweep(
    int dim, int n, int M, int efc, const std::vector<int>& ef_values, int n_queries, int top_k)
{
    std::mt19937 rng(42);
    std::string tmp = "/tmp/logosdb_bench_sweep";
    std::filesystem::remove_all(tmp);

    // ── 1. Build index once ──────────────────────────────────────────────────
    printf("Building HNSW index: dim=%d N=%d M=%d efConstruction=%d …\n", dim, n, M, efc);
    fflush(stdout);

    std::vector<std::vector<float>> corpus(n);
    std::vector<uint64_t> corpus_ids(n);
    {
        logosdb::Options opts;
        opts.dim = dim;
        opts.max_elements = (size_t)(n * 1.2);
        opts.M = M;
        opts.ef_construction = efc;
        opts.ef_search = ef_values.back();  // irrelevant for build
        logosdb::DB db(tmp, opts);

        double t0 = now_ms();
        for (int i = 0; i < n; ++i)
        {
            corpus[i] = random_unit_vec(dim, rng);
            corpus_ids[i] = db.put(corpus[i]);
        }
        printf("  Insert: %.0f ms  (%.0f vec/s)\n", now_ms() - t0, n / ((now_ms() - t0) / 1000.0));
    }

    // ── 2. Query vectors ─────────────────────────────────────────────────────
    std::vector<std::vector<float>> queries(n_queries);
    for (int q = 0; q < n_queries; ++q)
        queries[q] = random_unit_vec(dim, rng);

    // ── 3. Brute-force ground truth ──────────────────────────────────────────
    std::vector<std::vector<uint64_t>> gt(n_queries);
    double t0_brute = now_ms();
    for (int q = 0; q < n_queries; ++q)
    {
        std::vector<std::pair<float, uint64_t>> scores(n);
        for (int i = 0; i < n; ++i)
            scores[i] = {dot(queries[q].data(), corpus[i].data(), dim), corpus_ids[i]};
        std::partial_sort(scores.begin(),
                          scores.begin() + top_k,
                          scores.end(),
                          [](auto& a, auto& b) { return a.first > b.first; });
        for (int k = 0; k < top_k; ++k)
            gt[q].push_back(scores[k].second);
    }
    double brute_ms_per_q = (now_ms() - t0_brute) / n_queries;

    // ── 4. Sweep ─────────────────────────────────────────────────────────────
    printf("\nHNSW Recall Sweep — dim=%d N=%d M=%d efConstruction=%d top_k=%d queries=%d\n",
           dim,
           n,
           M,
           efc,
           top_k,
           n_queries);
    printf("%-10s %10s %14s %12s\n", "efSearch", "recall@k", "HNSW(ms/q)", "vs brute");
    printf("---------- ---------- -------------- ------------\n");

    for (int ef : ef_values)
    {
        // Reopen the same DB directory with the new efSearch value.
        // No rebuild — only the search parameter changes.
        logosdb::Options opts;
        opts.dim = dim;
        opts.max_elements = (size_t)(n * 1.2);
        opts.M = M;
        opts.ef_construction = efc;
        opts.ef_search = ef;
        logosdb::DB db(tmp, opts);

        // Warm-up pass
        for (int q = 0; q < std::min(5, n_queries); ++q)
            db.search(queries[q], top_k);

        // Timed search
        double t0 = now_ms();
        std::vector<std::vector<uint64_t>> hnsw_results(n_queries);
        for (int q = 0; q < n_queries; ++q)
        {
            auto hits = db.search(queries[q], top_k);
            for (auto& h : hits)
                hnsw_results[q].push_back(h.id);
        }
        double hnsw_ms = (now_ms() - t0) / n_queries;

        // Recall@k
        double total_recall = 0.0;
        for (int q = 0; q < n_queries; ++q)
        {
            int found = 0;
            for (uint64_t hid : hnsw_results[q])
                for (uint64_t bid : gt[q])
                    if (hid == bid)
                    {
                        ++found;
                        break;
                    }
            total_recall += (double)found / top_k;
        }
        double recall = total_recall / n_queries;

        printf("%-10d %9.1f%% %13.3f %10.1fx\n",
               ef,
               recall * 100.0,
               hnsw_ms,
               brute_ms_per_q / hnsw_ms);
    }

    printf("\nBrute-force reference: %.3f ms/query\n", brute_ms_per_q);
    std::filesystem::remove_all(tmp);
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    int dim = 2048;
    std::vector<int> counts = {1000, 10000, 100000};
    int n_queries = 50;
    int top_k = 10;
    int batch_n = 100000;
    int batch_dim = 256;

    // Recall-sweep options
    bool recall_sweep = false;
    int sweep_n = 10000;
    int sweep_M = 16;
    int sweep_efc = 200;
    int sweep_queries = 200;
    std::vector<int> sweep_ef_values = {10, 20, 50, 100, 200, 500};

    for (int i = 1; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--dim") && i + 1 < argc)
            dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc)
            top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--queries") && i + 1 < argc)
            n_queries = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch-n") && i + 1 < argc)
            batch_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--batch-dim") && i + 1 < argc)
            batch_dim = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--counts") && i + 1 < argc)
        {
            counts.clear();
            char* tok = strtok(argv[++i], ",");
            while (tok)
            {
                counts.push_back(atoi(tok));
                tok = strtok(nullptr, ",");
            }
        }
        // ── recall-sweep flags ──────────────────────────────────────────────
        else if (!strcmp(argv[i], "--recall-sweep"))
            recall_sweep = true;
        else if (!strcmp(argv[i], "--sweep-n") && i + 1 < argc)
            sweep_n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sweep-M") && i + 1 < argc)
            sweep_M = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sweep-efc") && i + 1 < argc)
            sweep_efc = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sweep-queries") && i + 1 < argc)
            sweep_queries = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sweep-ef") && i + 1 < argc)
        {
            sweep_ef_values.clear();
            char* tok = strtok(argv[++i], ",");
            while (tok)
            {
                sweep_ef_values.push_back(atoi(tok));
                tok = strtok(nullptr, ",");
            }
        }
    }

    // ── Recall-sweep mode: run sweep only and exit ───────────────────────────
    if (recall_sweep)
    {
        run_recall_sweep(dim, sweep_n, sweep_M, sweep_efc, sweep_ef_values, sweep_queries, top_k);
        return 0;
    }

    printf("LogosDB Benchmark — dim=%d top_k=%d queries=%d  M=32 efConstruction=400\n",
           dim,
           top_k,
           n_queries);
    printf("  efSearch scales with N: max(50, min(500, N/50))\n");
    printf("%-10s %8s %12s %14s %14s %10s\n",
           "N",
           "efSearch",
           "put(ms)",
           "HNSW(ms/q)",
           "brute(ms/q)",
           "recall@k");
    printf("---------- -------- ------------ -------------- -------------- ----------\n");

    for (int n : counts)
    {
        auto r = run_bench(dim, n, n_queries, top_k);
        int ef = std::max(50, std::min(500, n / 50));
        printf("%-10d %8d %12.1f %14.3f %14.3f %9.1f%%\n",
               r.n,
               ef,
               r.put_ms,
               r.hnsw_search_ms,
               r.brute_search_ms,
               r.recall_at_k * 100.0);
    }

    printf("\n--- ChromaDB comparison (published estimates, dim=2048) ---\n");
    printf("%-20s %14s %14s\n", "Metric", "ChromaDB", "LogosDB");
    printf("%-20s %14s %14s\n", "Language", "Python+C", "C/C++");
    printf("%-20s %14s %14s\n", "Search algo", "HNSW", "HNSW");
    printf("%-20s %14s %14s\n", "Storage", "SQLite+Parquet", "mmap+JSONL");
    printf("%-20s %14s %14s\n", "100K search (ms)", "~5-10", "(see above)");
    printf("%-20s %14s %14s\n", "Startup overhead", "Python RT", "zero");

    // Batch benchmark: compare individual puts vs batch puts
    printf("\n--- Batch Put Benchmark (%d vectors, dim=%d) ---\n", batch_n, batch_dim);
    {
        double individual_ms = 0.0;
        double batch_ms = 0.0;

        // Individual puts
        std::string tmp1 = "/tmp/logosdb_bench_individual";
        std::filesystem::remove_all(tmp1);
        {
            logosdb::Options opts;
            opts.dim = batch_dim;
            opts.max_elements = (size_t)(batch_n * 1.2);
            logosdb::DB db(tmp1, opts);

            std::mt19937 rng(42);
            double t0 = now_ms();
            for (int i = 0; i < batch_n; ++i)
            {
                auto v = random_unit_vec(batch_dim, rng);
                db.put(v, "row_" + std::to_string(i));
            }
            individual_ms = now_ms() - t0;

            printf("Individual puts:  %.1f ms (%.1f vectors/sec)\n",
                   individual_ms,
                   batch_n / (individual_ms / 1000.0));
        }
        std::filesystem::remove_all(tmp1);

        // Batch puts
        std::string tmp2 = "/tmp/logosdb_bench_batch";
        std::filesystem::remove_all(tmp2);
        {
            logosdb::Options opts;
            opts.dim = batch_dim;
            opts.max_elements = (size_t)(batch_n * 1.2);
            logosdb::DB db(tmp2, opts);

            std::mt19937 rng(42);
            std::vector<float> embeddings;
            embeddings.reserve(batch_n * batch_dim);
            std::vector<std::string> texts;
            texts.reserve(batch_n);

            for (int i = 0; i < batch_n; ++i)
            {
                auto v = random_unit_vec(batch_dim, rng);
                embeddings.insert(embeddings.end(), v.begin(), v.end());
                texts.push_back("row_" + std::to_string(i));
            }

            double t0 = now_ms();
            auto ids = db.put_batch(embeddings, batch_n, texts);
            batch_ms = now_ms() - t0;

            printf("Batch put:        %.1f ms (%.1f vectors/sec)\n",
                   batch_ms,
                   batch_n / (batch_ms / 1000.0));
        }
        std::filesystem::remove_all(tmp2);

        double speedup = individual_ms / batch_ms;
        printf("Speedup:          %.2fx faster\n", speedup);
    }

    return 0;
}
