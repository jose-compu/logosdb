#include <logosdb/logosdb.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <memory>

// Base64 encoding/decoding for import/export
static const char* BASE64_CHARS =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string base64_encode(const float* data, size_t n_floats) {
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(data);
    size_t len = n_floats * sizeof(float);
    std::string ret;
    ret.reserve(((len + 2) / 3) * 4);

    for (size_t i = 0; i < len; i += 3) {
        unsigned char b[3] = {0, 0, 0};
        size_t n = 0;
        for (size_t j = 0; j < 3 && i + j < len; ++j) {
            b[j] = bytes[i + j];
            n++;
        }
        unsigned char idx0 = b[0] >> 2;
        unsigned char idx1 = ((b[0] & 0x03) << 4) | (b[1] >> 4);
        unsigned char idx2 = ((b[1] & 0x0F) << 2) | (b[2] >> 6);
        unsigned char idx3 = b[2] & 0x3F;

        ret += BASE64_CHARS[idx0];
        ret += BASE64_CHARS[idx1];
        ret += (n > 1) ? BASE64_CHARS[idx2] : '=';
        ret += (n > 2) ? BASE64_CHARS[idx3] : '=';
    }
    return ret;
}

static std::vector<float> base64_decode(const std::string& encoded, int dim) {
    std::vector<float> result;
    std::vector<unsigned char> bytes;

    // Pre-calculate expected bytes (dim * sizeof(float))
    size_t expected_bytes = dim * sizeof(float);
    bytes.reserve(expected_bytes + 8);  // extra for padding

    size_t len = encoded.size();

    auto find_char = [](char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        if (c == '=') return -2;  // padding
        return -1;
    };

    for (size_t i = 0; i < len; i += 4) {
        if (i + 3 >= len) break;

        int b[4] = {-1, -1, -1, -1};
        for (int j = 0; j < 4 && i + j < len; ++j) {
            b[j] = find_char(encoded[i + j]);
        }

        // Skip if we hit invalid characters
        if (b[0] < 0 || b[0] == -2) break;
        if (b[1] < 0 || b[1] == -2) break;

        unsigned char o0 = (b[0] << 2) | (b[1] >> 4);
        bytes.push_back(o0);

        if (b[2] >= 0) {
            unsigned char o1 = ((b[1] & 0x0F) << 4) | (b[2] >> 2);
            bytes.push_back(o1);
        }
        if (b[3] >= 0) {
            unsigned char o2 = ((b[2] & 0x03) << 6) | b[3];
            bytes.push_back(o2);
        }
    }

    // Convert bytes to floats - ensure we have enough bytes
    if (bytes.size() < expected_bytes) {
        return result;  // Return empty if not enough data
    }

    result.reserve(dim);
    for (int i = 0; i < dim; i++) {
        float f;
        std::memcpy(&f, &bytes[i * sizeof(float)], sizeof(float));
        result.push_back(f);
    }

    return result;
}

// Storage header structure (from storage.h)
struct StorageHeader {
    uint32_t magic    = 0x4C4F474F;  // "LOGO"
    uint32_t version  = 1;
    uint32_t dim      = 0;
    uint32_t reserved = 0;
    uint64_t n_rows   = 0;
    uint64_t reserved2 = 0;
};

static bool read_dim_from_header(const char* path, int& dim) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    StorageHeader hdr;
    size_t n = fread(&hdr, 1, sizeof(hdr), f);
    fclose(f);
    if (n != sizeof(hdr)) return false;
    if (hdr.magic != 0x4C4F474FU) return false;
    dim = (int)hdr.dim;
    return dim > 0;
}

static std::vector<float> read_binary_vec(const char* path, int dim) {
    std::vector<float> v(dim);
    FILE* f = fopen(path, "rb");
    if (!f) return {};
    size_t n = fread(v.data(), sizeof(float), dim, f);
    fclose(f);
    if ((int)n != dim) return {};
    return v;
}

static void print_version() {
    printf("logosdb-cli version %s\n", LOGOSDB_VERSION_STRING);
}

static void print_usage(const char* prog) {
    printf(
        "Usage: %s <command> [options]\n"
        "\n"
        "Commands:\n"
        "  info <db-path> [--json]           Show database info (dim read from file)\n"
        "  put <db-path> [options]           Insert a vector\n"
        "  search <db-path> [options]        Search for similar vectors\n"
        "  export <db-path> [--output FILE]  Export DB to JSONL with base64 vectors\n"
        "  import <db-path> [--input FILE]   Import from JSONL\n"
        "\n"
        "Global Options:\n"
        "  --version                         Show version and exit\n"
        "  --help, -h                        Show this help and exit\n"
        "\n"
        "Put Options:\n"
        "  --dim N                           Vector dimension (required for new DB)\n"
        "  --text TEXT                       Text metadata\n"
        "  --ts TIMESTAMP                    ISO 8601 timestamp\n"
        "  --embedding-file FILE             Binary float32 vector file\n"
        "\n"
        "Search Options:\n"
        "  --dim N                           Vector dimension (required for new DB)\n"
        "  --query-file FILE                 Binary float32 query vector\n"
        "  --query-id ID                     Use existing vector as query\n"
        "  --top-k N                         Number of results (default: 5)\n"
        "  --ts-from TIMESTAMP               Filter from timestamp (inclusive)\n"
        "  --ts-to TIMESTAMP                 Filter to timestamp (inclusive)\n"
        "  --json                            Output results as JSON\n"
        "\n"
        "Export Options:\n"
        "  --output FILE                     Output file (default: stdout)\n"
        "  --json                            Same as default (JSONL output)\n"
        "\n"
        "Import Options:\n"
        "  --dim N                           Vector dimension (required for new DB)\n"
        "  --input FILE                      Input JSONL file (default: stdin)\n"
        "\n"
        "Examples:\n"
        "  %s info /tmp/mydb                 Show database info\n"
        "  %s info /tmp/mydb --json          Show info as JSON\n"
        "  %s put /tmp/mydb --dim 128 --text \"hello\" --embedding-file vec.bin\n"
        "  %s search /tmp/mydb --dim 128 --query-file query.bin --top-k 10\n"
        "  %s search /tmp/mydb --dim 128 --query-id 0 --ts-from 2025-01-01T00:00:00Z\n"
        "  %s export /tmp/mydb --output backup.jsonl\n"
        "  %s import /tmp/newdb --dim 128 --input backup.jsonl\n"
        , prog, prog, prog, prog, prog, prog, prog, prog
    );
}

static void print_cmd_help(const char* cmd) {
    if (strcmp(cmd, "info") == 0) {
        printf("Usage: logosdb-cli info <db-path> [--json]\n\nShow database information.\n");
    } else if (strcmp(cmd, "put") == 0) {
        printf("Usage: logosdb-cli put <db-path> [options]\n\nOptions:\n  --dim N\n  --text TEXT\n  --ts TIMESTAMP\n  --embedding-file FILE\n");
    } else if (strcmp(cmd, "search") == 0) {
        printf("Usage: logosdb-cli search <db-path> [options]\n\nOptions:\n  --dim N\n  --query-file FILE\n  --query-id ID\n  --top-k N\n  --ts-from TIMESTAMP\n  --ts-to TIMESTAMP\n  --json\n");
    } else if (strcmp(cmd, "export") == 0) {
        printf("Usage: logosdb-cli export <db-path> [--output FILE]\n\nExport database to JSONL.\n");
    } else if (strcmp(cmd, "import") == 0) {
        printf("Usage: logosdb-cli import <db-path> [options]\n\nOptions:\n  --dim N (required for new DB)\n  --input FILE\n");
    } else {
        printf("Unknown command: %s\n", cmd);
    }
}

// Command-line argument structure
struct Args {
    std::string cmd;
    std::string db_path;
    int dim = 0;
    int top_k = 5;
    const char* text = nullptr;
    const char* ts = nullptr;
    const char* emb_file = nullptr;
    const char* query_file = nullptr;
    uint64_t query_id = UINT64_MAX;
    const char* ts_from = nullptr;
    const char* ts_to = nullptr;
    const char* output_file = nullptr;
    const char* input_file = nullptr;
    bool json = false;
    bool help = false;
    bool version = false;
};

static Args parse_args(int argc, char** argv) {
    Args args;
    if (argc < 2) return args;

    // Check for global --version or --help first
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--version") == 0) {
            args.version = true;
            return args;
        }
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            args.help = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args.cmd = argv[i + 1];  // --help <cmd>
            }
            return args;
        }
    }

    if (argc < 3) return args;

    args.cmd = argv[1];
    args.db_path = argv[2];

    for (int i = 3; i < argc; ++i) {
        if (!strcmp(argv[i], "--dim") && i + 1 < argc) {
            long v = strtol(argv[++i], nullptr, 10);
            args.dim = (v > 0 && v <= 65536) ? (int)v : 0;
        }
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc) {
            long v = strtol(argv[++i], nullptr, 10);
            args.top_k = (v > 0 && v <= 10000) ? (int)v : 5;
        }
        else if (!strcmp(argv[i], "--text") && i + 1 < argc) args.text = argv[++i];
        else if (!strcmp(argv[i], "--ts") && i + 1 < argc) args.ts = argv[++i];
        else if (!strcmp(argv[i], "--embedding-file") && i + 1 < argc) args.emb_file = argv[++i];
        else if (!strcmp(argv[i], "--query-file") && i + 1 < argc) args.query_file = argv[++i];
        else if (!strcmp(argv[i], "--query-id") && i + 1 < argc) {
            long long v = strtoll(argv[++i], nullptr, 10);
            args.query_id = (v >= 0) ? (uint64_t)v : UINT64_MAX;
        }
        else if (!strcmp(argv[i], "--ts-from") && i + 1 < argc) args.ts_from = argv[++i];
        else if (!strcmp(argv[i], "--ts-to") && i + 1 < argc) args.ts_to = argv[++i];
        else if (!strcmp(argv[i], "--output") && i + 1 < argc) args.output_file = argv[++i];
        else if (!strcmp(argv[i], "--input") && i + 1 < argc) args.input_file = argv[++i];
        else if (!strcmp(argv[i], "--json")) args.json = true;
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) args.help = true;
    }

    return args;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    if (args.version) {
        print_version();
        return 0;
    }

    if (args.help) {
        if (!args.cmd.empty()) {
            print_cmd_help(args.cmd.c_str());
        } else {
            print_usage(argv[0]);
        }
        return 0;
    }

    if (args.cmd.empty() || args.db_path.empty()) {
        print_usage(argv[0]);
        return 1;
    }

    // Try to read dim from existing database for info command
    if (args.cmd == "info" && args.dim == 0) {
        std::string vec_path = args.db_path + "/vectors.bin";
        if (!read_dim_from_header(vec_path.c_str(), args.dim)) {
            fprintf(stderr, "error: cannot read dim from %s (use --dim for new DB)\n", vec_path.c_str());
            return 1;
        }
    }

    // For import, dim may be optional if DB exists
    if (args.cmd == "import" && args.dim == 0) {
        std::string vec_path = args.db_path + "/vectors.bin";
        if (!read_dim_from_header(vec_path.c_str(), args.dim)) {
            fprintf(stderr, "error: --dim required for new database\n");
            return 1;
        }
    }

    // For other commands, dim is required if DB doesn't exist
    // For export, we also need to read the dim from the file
    if (args.dim == 0) {
        std::string vec_path = args.db_path + "/vectors.bin";
        if (!read_dim_from_header(vec_path.c_str(), args.dim)) {
            if (args.cmd != "export") {
                fprintf(stderr, "error: --dim required (1..65536)\n");
                return 1;
            }
            // For export, if we can't read dim, the DB doesn't exist
            fprintf(stderr, "error: cannot read database at %s\n", args.db_path.c_str());
            return 1;
        }
    }

    char* err = nullptr;
    logosdb_options_t* opts = logosdb_options_create();
    if (args.dim > 0) logosdb_options_set_dim(opts, args.dim);
    logosdb_t* db = logosdb_open(args.db_path.c_str(), opts, &err);
    logosdb_options_destroy(opts);

    if (!db) {
        fprintf(stderr, "error: %s\n", err ? err : "unknown");
        free(err);
        return 1;
    }

    int rc = 0;

    if (args.cmd == "info") {
        if (args.json) {
            printf("{\n");
            printf("  \"path\": \"%s\",\n", args.db_path.c_str());
            printf("  \"dim\": %d,\n", logosdb_dim(db));
            printf("  \"count\": %zu,\n", logosdb_count(db));
            printf("  \"count_live\": %zu\n", logosdb_count_live(db));
            printf("}\n");
        } else {
            printf("path       : %s\n", args.db_path.c_str());
            printf("dim        : %d\n", logosdb_dim(db));
            printf("count      : %zu\n", logosdb_count(db));
            printf("count_live : %zu\n", logosdb_count_live(db));
        }
    }
    else if (args.cmd == "put") {
        if (!args.emb_file) {
            fprintf(stderr, "error: --embedding-file required\n");
            rc = 1;
            goto done;
        }
        auto vec = read_binary_vec(args.emb_file, logosdb_dim(db));
        if ((int)vec.size() != logosdb_dim(db)) {
            fprintf(stderr, "error: could not read embedding (expected %d floats)\n", logosdb_dim(db));
            rc = 1;
            goto done;
        }
        uint64_t id = logosdb_put(db, vec.data(), logosdb_dim(db), args.text, args.ts, &err);
        if (id == UINT64_MAX) {
            fprintf(stderr, "error: %s\n", err ? err : "unknown");
            free(err);
            rc = 1;
        } else {
            if (args.json) {
                printf("{\"id\": %llu}\n", (unsigned long long)id);
            } else {
                printf("put id=%llu\n", (unsigned long long)id);
            }
        }
    }
    else if (args.cmd == "search") {
        std::vector<float> qvec;

        if (args.query_file) {
            qvec = read_binary_vec(args.query_file, logosdb_dim(db));
            if ((int)qvec.size() != logosdb_dim(db)) {
                fprintf(stderr, "error: could not read query (expected %d floats)\n", logosdb_dim(db));
                rc = 1;
                goto done;
            }
        } else if (args.query_id != UINT64_MAX) {
            // Use existing vector as query
            const float* raw = logosdb_raw_vectors(db, nullptr, nullptr);
            if (!raw || args.query_id >= logosdb_count(db)) {
                fprintf(stderr, "error: invalid query-id %llu\n", (unsigned long long)args.query_id);
                rc = 1;
                goto done;
            }
            qvec.resize(logosdb_dim(db));
            std::memcpy(qvec.data(), raw + args.query_id * logosdb_dim(db), logosdb_dim(db) * sizeof(float));
        } else {
            fprintf(stderr, "error: --query-file or --query-id required\n");
            rc = 1;
            goto done;
        }

        logosdb_search_result_t* res;
        if (args.ts_from || args.ts_to) {
            int candidate_k = args.top_k * 10;
            res = logosdb_search_ts_range(db, qvec.data(), logosdb_dim(db), args.top_k,
                                         args.ts_from, args.ts_to, candidate_k, &err);
        } else {
            res = logosdb_search(db, qvec.data(), logosdb_dim(db), args.top_k, &err);
        }

        if (!res) {
            fprintf(stderr, "error: %s\n", err ? err : "unknown");
            free(err);
            rc = 1;
        } else {
            int n = logosdb_result_count(res);
            if (args.json) {
                printf("[\n");
                for (int i = 0; i < n; ++i) {
                    printf("  {\n");
                    printf("    \"rank\": %d,\n", i);
                    printf("    \"id\": %llu,\n", (unsigned long long)logosdb_result_id(res, i));
                    printf("    \"score\": %.6f", logosdb_result_score(res, i));
                    const char* t = logosdb_result_text(res, i);
                    const char* tts = logosdb_result_timestamp(res, i);
                    if (t) {
                        printf(",\n    \"text\": \"%s\"", t);
                        if (tts) {
                            printf(",\n    \"timestamp\": \"%s\"", tts);
                        }
                    } else if (tts) {
                        printf(",\n    \"timestamp\": \"%s\"", tts);
                    }
                    printf("\n  }%s\n", (i < n - 1) ? "," : "");
                }
                printf("]\n");
            } else {
                printf("results: %d\n", n);
                for (int i = 0; i < n; ++i) {
                    printf("  #%d id=%llu score=%.6f", i,
                           (unsigned long long)logosdb_result_id(res, i),
                           logosdb_result_score(res, i));
                    const char* t = logosdb_result_text(res, i);
                    if (t) printf(" text=\"%s\"", t);
                    const char* tts = logosdb_result_timestamp(res, i);
                    if (tts) printf(" ts=%s", tts);
                    printf("\n");
                }
            }
            logosdb_result_free(res);
        }
    }
    else if (args.cmd == "export") {
        FILE* out = stdout;
        if (args.output_file) {
            out = fopen(args.output_file, "w");
            if (!out) {
                fprintf(stderr, "error: cannot open output file %s\n", args.output_file);
                rc = 1;
                goto done;
            }
        }

        size_t n_rows = 0;
        int dim = 0;
        const float* raw = logosdb_raw_vectors(db, &n_rows, &dim);

        for (size_t i = 0; i < n_rows; ++i) {
            // Get metadata
            char tmp_err[256];
            logosdb_search_result_t* tmp_res = logosdb_search(db, raw + i * dim, dim, 1, nullptr);
            const char* text = "";
            const char* ts = "";
            if (tmp_res) {
                text = logosdb_result_text(tmp_res, 0);
                if (!text) text = "";
                ts = logosdb_result_timestamp(tmp_res, 0);
                if (!ts) ts = "";
            }

            // Encode vector as base64
            std::string b64 = base64_encode(raw + i * dim, dim);

            // Output JSONL
            fprintf(out, "{\"id\": %zu, \"vector\": \"%s\", \"text\": \"%s\", \"timestamp\": \"%s\"}\n",
                    i, b64.c_str(), text, ts);

            if (tmp_res) logosdb_result_free(tmp_res);
        }

        if (args.output_file) {
            fclose(out);
            printf("Exported %zu rows to %s\n", n_rows, args.output_file);
        }
    }
    else if (args.cmd == "import") {
        FILE* in = stdin;
        if (args.input_file) {
            in = fopen(args.input_file, "r");
            if (!in) {
                fprintf(stderr, "error: cannot open input file %s\n", args.input_file);
                rc = 1;
                goto done;
            }
        }

        size_t imported = 0;
        size_t lines_read = 0;
        char line[65536];
        while (fgets(line, sizeof(line), in)) {
            lines_read++;

            // Extract vector (base64)
            char* vec_field = strstr(line, "\"vector\"");
            if (!vec_field) continue;

            // Find the value after "vector":
            char* vec_start = strchr(vec_field, ':');
            if (!vec_start) continue;
            vec_start++;
            while (*vec_start == ' ') vec_start++;
            if (*vec_start != '"') continue;
            vec_start++;  // skip opening quote

            char* vec_end = strchr(vec_start, '"');
            if (!vec_end) continue;

            // Temporarily null-terminate to extract
            *vec_end = '\0';
            std::string b64(vec_start);
            *vec_end = '"';

            auto vec = base64_decode(b64, logosdb_dim(db));
            if ((int)vec.size() != logosdb_dim(db)) {
                fprintf(stderr, "warning: skipping row with wrong dimension\n");
                continue;
            }

            // Extract text
            std::string text;
            char* text_field = strstr(line, "\"text\"");
            if (text_field) {
                char* text_start = strchr(text_field, ':');
                if (text_start) {
                    text_start++;
                    while (*text_start == ' ') text_start++;
                    if (*text_start == '"') {
                        text_start++;
                        char* text_end = strchr(text_start, '"');
                        if (text_end) {
                            *text_end = '\0';
                            text = text_start;
                            *text_end = '"';
                        }
                    }
                }
            }

            // Extract timestamp
            std::string ts;
            char* ts_field = strstr(line, "\"timestamp\"");
            if (ts_field) {
                char* ts_start = strchr(ts_field, ':');
                if (ts_start) {
                    ts_start++;
                    while (*ts_start == ' ') ts_start++;
                    if (*ts_start == '"') {
                        ts_start++;
                        char* ts_end = strchr(ts_start, '"');
                        if (ts_end) {
                            *ts_end = '\0';
                            ts = ts_start;
                            *ts_end = '"';
                        }
                    }
                }
            }

            // Insert
            uint64_t id = logosdb_put(db, vec.data(), logosdb_dim(db),
                                     text.empty() ? nullptr : text.c_str(),
                                     ts.empty() ? nullptr : ts.c_str(), &err);
            if (id != UINT64_MAX) {
                imported++;
            } else {
                if (err) {
                    fprintf(stderr, "warning: insert failed: %s\n", err);
                    free(err);
                    err = nullptr;
                }
            }
        }

        if (args.input_file) fclose(in);
        printf("Imported %zu rows\n", imported);
    }
    else {
        fprintf(stderr, "error: unknown command: %s\n", args.cmd.c_str());
        print_usage(argv[0]);
        rc = 1;
    }

done:
    logosdb_close(db);
    return rc;
}
