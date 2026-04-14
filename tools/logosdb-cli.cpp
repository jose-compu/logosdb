#include <logosdb/logosdb.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void usage(const char * prog) {
    fprintf(stderr,
        "Usage:\n"
        "  %s info   <db-path> --dim <D>\n"
        "  %s put    <db-path> --dim <D> --text <TEXT> [--ts <TS>] --embedding-file <FILE>\n"
        "  %s search <db-path> --dim <D> --query-file <FILE> --top-k <K>\n"
        , prog, prog, prog);
}

static std::vector<float> read_binary_vec(const char * path, int dim) {
    std::vector<float> v(dim);
    FILE * f = fopen(path, "rb");
    if (!f) return {};
    size_t n = fread(v.data(), sizeof(float), dim, f);
    fclose(f);
    if ((int)n != dim) return {};
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 3) { usage(argv[0]); return 1; }

    std::string cmd     = argv[1];
    std::string db_path = argv[2];
    int dim = 0, top_k = 5;
    const char * text = nullptr;
    const char * ts = nullptr;
    const char * emb_file = nullptr;
    const char * query_file = nullptr;

    for (int i = 3; i < argc; ++i) {
        if (!strcmp(argv[i], "--dim") && i + 1 < argc) {
            long v = strtol(argv[++i], nullptr, 10);
            dim = (v > 0 && v <= 65536) ? (int)v : 0;
        }
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) {
            long v = strtol(argv[++i], nullptr, 10);
            top_k = (v > 0 && v <= 10000) ? (int)v : 5;
        }
        else if (!strcmp(argv[i], "--text") && i + 1 < argc) text = argv[++i];
        else if (!strcmp(argv[i], "--ts") && i + 1 < argc) ts = argv[++i];
        else if (!strcmp(argv[i], "--embedding-file") && i + 1 < argc) emb_file = argv[++i];
        else if (!strcmp(argv[i], "--query-file") && i + 1 < argc) query_file = argv[++i];
    }

    if (dim <= 0) { fprintf(stderr, "error: --dim required (1..65536)\n"); return 1; }

    char * err = nullptr;
    logosdb_options_t * opts = logosdb_options_create();
    logosdb_options_set_dim(opts, dim);
    logosdb_t * db = logosdb_open(db_path.c_str(), opts, &err);
    logosdb_options_destroy(opts);
    if (!db) {
        fprintf(stderr, "error: %s\n", err ? err : "unknown");
        free(err);
        return 1;
    }

    int rc = 0;

    if (cmd == "info") {
        printf("path   : %s\n", db_path.c_str());
        printf("dim    : %d\n", logosdb_dim(db));
        printf("count  : %zu\n", logosdb_count(db));
    } else if (cmd == "put") {
        if (!emb_file) { fprintf(stderr, "error: --embedding-file required\n"); rc = 1; goto done; }
        auto vec = read_binary_vec(emb_file, dim);
        if ((int)vec.size() != dim) { fprintf(stderr, "error: could not read embedding\n"); rc = 1; goto done; }
        uint64_t id = logosdb_put(db, vec.data(), dim, text, ts, &err);
        if (id == UINT64_MAX) {
            fprintf(stderr, "error: %s\n", err ? err : "unknown");
            free(err);
            rc = 1;
        } else {
            printf("put id=%llu\n", (unsigned long long)id);
        }
    } else if (cmd == "search") {
        if (!query_file) { fprintf(stderr, "error: --query-file required\n"); rc = 1; goto done; }
        auto qvec = read_binary_vec(query_file, dim);
        if ((int)qvec.size() != dim) { fprintf(stderr, "error: could not read query\n"); rc = 1; goto done; }
        logosdb_search_result_t * res = logosdb_search(db, qvec.data(), dim, top_k, &err);
        if (!res) {
            fprintf(stderr, "error: %s\n", err ? err : "unknown");
            free(err);
            rc = 1;
        } else {
            int n = logosdb_result_count(res);
            printf("results: %d\n", n);
            for (int i = 0; i < n; ++i) {
                printf("  #%d id=%llu score=%.6f", i,
                       (unsigned long long)logosdb_result_id(res, i),
                       logosdb_result_score(res, i));
                const char * t = logosdb_result_text(res, i);
                if (t) printf(" text=\"%s\"", t);
                const char * tts = logosdb_result_timestamp(res, i);
                if (tts) printf(" ts=%s", tts);
                printf("\n");
            }
            logosdb_result_free(res);
        }
    } else {
        fprintf(stderr, "unknown command: %s\n", cmd.c_str());
        rc = 1;
    }

done:
    logosdb_close(db);
    return rc;
}
