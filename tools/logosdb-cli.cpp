#include <logosdb/logosdb.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// Base64 encoding/decoding for import/export
static const char* BASE64_CHARS =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string base64_encode(const float* data, size_t n_floats)
{
    const unsigned char* bytes = reinterpret_cast<const unsigned char*>(data);
    size_t len = n_floats * sizeof(float);
    std::string ret;
    ret.reserve(((len + 2) / 3) * 4);

    for (size_t i = 0; i < len; i += 3)
    {
        unsigned char b[3] = {0, 0, 0};
        size_t n = 0;
        for (size_t j = 0; j < 3 && i + j < len; ++j)
        {
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

static std::vector<float> base64_decode(const std::string& encoded, int dim)
{
    std::vector<float> result;
    std::vector<unsigned char> bytes;

    // Pre-calculate expected bytes (dim * sizeof(float))
    size_t expected_bytes = dim * sizeof(float);
    bytes.reserve(expected_bytes + 8);  // extra for padding

    size_t len = encoded.size();

    auto find_char = [](char c) -> int
    {
        if (c >= 'A' && c <= 'Z')
            return c - 'A';
        if (c >= 'a' && c <= 'z')
            return c - 'a' + 26;
        if (c >= '0' && c <= '9')
            return c - '0' + 52;
        if (c == '+')
            return 62;
        if (c == '/')
            return 63;
        if (c == '=')
            return -2;  // padding
        return -1;
    };

    for (size_t i = 0; i < len; i += 4)
    {
        if (i + 3 >= len)
            break;

        int b[4] = {-1, -1, -1, -1};
        for (int j = 0; j < 4 && i + j < len; ++j)
        {
            b[j] = find_char(encoded[i + j]);
        }

        // Skip if we hit invalid characters
        if (b[0] < 0 || b[0] == -2)
            break;
        if (b[1] < 0 || b[1] == -2)
            break;

        unsigned char o0 = (b[0] << 2) | (b[1] >> 4);
        bytes.push_back(o0);

        if (b[2] >= 0)
        {
            unsigned char o1 = ((b[1] & 0x0F) << 4) | (b[2] >> 2);
            bytes.push_back(o1);
        }
        if (b[3] >= 0)
        {
            unsigned char o2 = ((b[2] & 0x03) << 6) | b[3];
            bytes.push_back(o2);
        }
    }

    // Convert bytes to floats - ensure we have enough bytes
    if (bytes.size() < expected_bytes)
    {
        return result;  // Return empty if not enough data
    }

    result.reserve(dim);
    for (int i = 0; i < dim; i++)
    {
        float f;
        std::memcpy(&f, &bytes[i * sizeof(float)], sizeof(float));
        result.push_back(f);
    }

    return result;
}

// On-disk vector storage header (must match `src/storage.h` StorageHeader, 32 bytes).
struct StorageHeaderOnDisk
{
    uint32_t magic = 0x4C4F474F;
    uint32_t version = 2;
    uint32_t dim = 0;
    uint32_t dtype = 0;
    uint64_t n_rows = 0;
    float scale = 1.0f;
    float reserved = 0.0f;
};

static constexpr uint32_t kVecMagic = 0x4C4F474FU;

static size_t storage_row_stride_bytes(int dim, uint32_t dtype)
{
    if (dim <= 0)
        return 0;
    switch (dtype)
    {
    case 1:
        return (size_t)dim * 2u;
    case 2:
        return (size_t)dim * 1u;
    case 0:
    default:
        return (size_t)dim * 4u;
    }
}

static const char* dtype_label(uint32_t dtype)
{
    switch (dtype)
    {
    case 1:
        return "float16";
    case 2:
        return "int8";
    case 0:
    default:
        return "float32";
    }
}

static const char* distance_label(int d)
{
    switch (d)
    {
    case LOGOSDB_DIST_COSINE:
        return "cosine";
    case LOGOSDB_DIST_L2:
        return "l2";
    case LOGOSDB_DIST_IP:
    default:
        return "ip";
    }
}

static bool read_vectors_probe(const std::string& path,
                               StorageHeaderOnDisk& out_hdr,
                               size_t file_size,
                               std::string& err)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
    {
        err = "cannot open file";
        return false;
    }
    StorageHeaderOnDisk raw{};
    size_t n = fread(&raw, 1, sizeof(raw), f);
    fclose(f);
    if (n != sizeof(raw))
    {
        err = "file too small for header";
        return false;
    }
    if (raw.magic != kVecMagic)
    {
        err = "bad magic (expected LOGO)";
        return false;
    }
    const uint32_t file_ver = raw.version;
    if (file_ver != 0 && file_ver != 1 && file_ver != 2)
    {
        err = "unsupported vectors.bin version: " + std::to_string(file_ver);
        return false;
    }
    if (raw.dim == 0 || raw.dim > 65536u)
    {
        err = "invalid dim in header";
        return false;
    }

    uint32_t row_dtype = raw.dtype;
    if (file_ver == 0 || file_ver == 1)
        row_dtype = 0;
    if (file_ver == 2 && raw.dtype > 2u)
    {
        err = "invalid dtype in header";
        return false;
    }

    size_t stride = storage_row_stride_bytes((int)raw.dim, row_dtype);
    if (stride == 0)
    {
        err = "invalid row stride";
        return false;
    }
    if (file_size < sizeof(StorageHeaderOnDisk))
    {
        err = "file smaller than header";
        return false;
    }
    uint64_t body = file_size - sizeof(StorageHeaderOnDisk);
    if (body % stride != 0)
    {
        err = "file size is not header + whole rows (possible corruption)";
        return false;
    }
    uint64_t inferred_rows = body / stride;
    if (inferred_rows != raw.n_rows)
    {
        err = "n_rows mismatch: header=" + std::to_string(raw.n_rows) +
              " inferred from size=" + std::to_string(inferred_rows);
        return false;
    }

    out_hdr = raw;
    if (file_ver == 0 || file_ver == 1)
    {
        out_hdr.version = 2;
        out_hdr.dtype = 0;
        out_hdr.scale = 1.0f;
        out_hdr.reserved = 0.0f;
    }
    err.clear();
    return true;
}

struct IndexMetaProbe
{
    bool file_exists = false;
    bool valid = false;
    int distance = LOGOSDB_DIST_IP;
    int dim = 0;
    std::string parse_error;
};

static IndexMetaProbe read_index_meta(const std::string& meta_path)
{
    IndexMetaProbe p;
    std::error_code ec;
    p.file_exists = std::filesystem::exists(meta_path, ec);
    if (!p.file_exists)
        return p;

    FILE* f = fopen(meta_path.c_str(), "rb");
    if (!f)
    {
        p.parse_error = "cannot open";
        return p;
    }
    char magic[8];
    uint32_t version = 0;
    int32_t distance = 0;
    int32_t dim = 0;
    uint8_t pad[12];
    size_t n1 = fread(magic, 1, sizeof(magic), f);
    size_t n2 = fread(&version, sizeof(version), 1, f);
    size_t n3 = fread(&distance, sizeof(distance), 1, f);
    size_t n4 = fread(&dim, sizeof(dim), 1, f);
    size_t n5 = fread(pad, 1, sizeof(pad), f);
    fclose(f);
    if (n1 != sizeof(magic) || n2 != 1u || n3 != 1u || n4 != 1u || n5 != sizeof(pad))
    {
        p.parse_error = "short read";
        return p;
    }
    if (std::memcmp(magic, "LOGOSDB\0", 8) != 0)
    {
        p.parse_error = "bad magic";
        return p;
    }
    if (version != 1u)
    {
        p.parse_error = "unsupported index meta version";
        return p;
    }
    if (distance < 0 || distance > 2 || dim <= 0)
    {
        p.parse_error = "invalid distance or dim";
        return p;
    }
    p.valid = true;
    p.distance = distance;
    p.dim = dim;
    return p;
}

static bool
read_wal_probe(const std::string& path, bool& exists, uint32_t& version, std::string& err)
{
    exists = false;
    version = 0;
    std::error_code ec;
    if (!std::filesystem::exists(path, ec))
    {
        err.clear();
        return true;
    }
    exists = true;
    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
    {
        err = "cannot open wal.log";
        return false;
    }
    uint32_t magic = 0;
    uint32_t ver = 0;
    size_t n = fread(&magic, sizeof(magic), 1, f);
    n += fread(&ver, sizeof(ver), 1, f);
    fclose(f);
    if (n != 2u)
    {
        err = "wal.log too small";
        return false;
    }
    if (magic != 0x57474F4CU)
    {
        err = "wal: bad magic";
        return false;
    }
    version = ver;
    if (ver != 1u)
        err = "wal: unsupported version (only 1 is supported)";
    else
        err.clear();
    return err.empty();
}

static bool read_dim_from_header(const char* path, int& dim)
{
    std::error_code ec;
    auto sz = std::filesystem::file_size(path, ec);
    if (ec)
        return false;
    StorageHeaderOnDisk hdr{};
    std::string err;
    if (!read_vectors_probe(path, hdr, (size_t)sz, err))
        return false;
    dim = (int)hdr.dim;
    return dim > 0;
}

static std::vector<float> read_binary_vec(const char* path, int dim)
{
    std::vector<float> v(dim);
    FILE* f = fopen(path, "rb");
    if (!f)
        return {};
    size_t n = fread(v.data(), sizeof(float), dim, f);
    fclose(f);
    if ((int)n != dim)
        return {};
    return v;
}

/** Reject NUL and C0 controls (except TAB/LF/CR) in CLI text metadata; cap length. */
static bool validate_cli_text_metadata(const char* label, const char* text)
{
    if (!text)
        return true;
    const unsigned char* p = reinterpret_cast<const unsigned char*>(text);
    size_t len = std::strlen(text);
    const size_t kMax = 512u * 1024u;
    if (len > kMax)
    {
        fprintf(stderr, "error: %s exceeds maximum length (%zu bytes)\n", label, kMax);
        return false;
    }
    for (size_t i = 0; i < len; ++i)
    {
        unsigned char c = p[i];
        if (c == 0)
        {
            fprintf(stderr, "error: %s contains NUL byte\n", label);
            return false;
        }
        if (c < 32 && c != '\t' && c != '\n' && c != '\r')
        {
            fprintf(stderr, "error: %s contains disallowed control character\n", label);
            return false;
        }
    }
    return true;
}

static int run_upgrade_cmd(const std::string& db_path, bool apply)
{
    const std::string vec = db_path + "/vectors.bin";
    std::error_code ec;
    if (!std::filesystem::exists(vec, ec))
    {
        fprintf(stderr, "error: %s not found\n", vec.c_str());
        return 1;
    }
    const std::uintmax_t fsz_um = std::filesystem::file_size(vec, ec);
    if (ec)
    {
        fprintf(stderr, "error: cannot stat vectors.bin\n");
        return 1;
    }
    const size_t fsz = (size_t)fsz_um;

    StorageHeaderOnDisk raw{};
    FILE* rf = fopen(vec.c_str(), "rb");
    if (!rf)
    {
        fprintf(stderr, "error: cannot open vectors.bin\n");
        return 1;
    }
    size_t n = fread(&raw, 1, sizeof(raw), rf);
    fclose(rf);
    if (n != sizeof(raw))
    {
        fprintf(stderr, "error: vectors.bin too small for header\n");
        return 1;
    }
    if (raw.magic != kVecMagic)
    {
        fprintf(stderr, "error: vectors.bin bad magic\n");
        return 1;
    }

    if (raw.version == 2u)
    {
        StorageHeaderOnDisk tmp{};
        std::string e;
        if (!read_vectors_probe(vec, tmp, fsz, e))
        {
            fprintf(stderr, "error: v2 header but layout check failed: %s\n", e.c_str());
            return 1;
        }
        printf("vectors.bin: already storage format v2; layout check OK.\n");
        return 0;
    }

    if (raw.version != 0u && raw.version != 1u)
    {
        fprintf(stderr, "error: unsupported on-disk vectors.bin version %u\n", raw.version);
        return 1;
    }
    if (raw.dim == 0 || raw.dim > 65536u)
    {
        fprintf(stderr, "error: invalid dim in legacy header\n");
        return 1;
    }
    const size_t stride = (size_t)raw.dim * sizeof(float);
    if (fsz < sizeof(StorageHeaderOnDisk))
    {
        fprintf(stderr, "error: file smaller than header\n");
        return 1;
    }
    const uint64_t body = (uint64_t)(fsz - sizeof(StorageHeaderOnDisk));
    if (stride == 0 || body % stride != 0)
    {
        fprintf(stderr, "error: legacy file size does not match float32 rows\n");
        return 1;
    }
    const uint64_t inferred = body / stride;
    if (inferred != raw.n_rows)
    {
        fprintf(stderr,
                "error: legacy n_rows mismatch (header=%llu size implies %llu)\n",
                (unsigned long long)raw.n_rows,
                (unsigned long long)inferred);
        return 1;
    }

    printf("Planned: rewrite vectors.bin header from v%u to v2 (dim=%u n_rows=%llu, float32 "
           "rows).\n",
           raw.version,
           raw.dim,
           (unsigned long long)raw.n_rows);
    if (!apply)
    {
        printf("This was a dry run. Re-run with --apply or --yes to write the new header.\n");
        return 0;
    }

    FILE* wf = fopen(vec.c_str(), "r+b");
    if (!wf)
    {
        fprintf(stderr, "error: cannot open vectors.bin for write\n");
        return 1;
    }
    StorageHeaderOnDisk out = raw;
    out.version = 2u;
    out.dtype = 0u;
    out.scale = 1.0f;
    out.reserved = 0.0f;
    if (fseek(wf, 0, SEEK_SET) != 0 || fwrite(&out, 1, sizeof(out), wf) != sizeof(out) ||
        fflush(wf) != 0)
    {
        fprintf(stderr, "error: failed to write header\n");
        fclose(wf);
        return 1;
    }
    fclose(wf);
    printf("Wrote storage header v2 to %s\n", vec.c_str());
    return 0;
}

static void print_json_escaped_value(const std::string& s)
{
    for (char c : s)
    {
        if (c == '"' || c == '\\')
            printf("\\%c", c);
        else if (c == '\n')
            printf("\\n");
        else
            putchar(c);
    }
}

static int run_doctor(const std::string& db_path, bool json, int distance_override)
{
    const std::string p_vec = db_path + "/vectors.bin";
    const std::string p_meta = db_path + "/meta.jsonl";
    const std::string p_idx = db_path + "/hnsw.idx";
    const std::string p_idx_meta = db_path + "/hnsw.idx.meta";
    const std::string p_wal = db_path + "/wal.log";

    auto exists_size = [](const std::string& p, bool& ex, size_t& sz)
    {
        std::error_code ec;
        ex = std::filesystem::exists(p, ec);
        sz = 0;
        if (ex)
        {
            auto u = std::filesystem::file_size(p, ec);
            if (!ec)
                sz = (size_t)u;
        }
    };

    bool e_vec = false, e_meta = false, e_idx = false, e_imeta = false, e_wal = false;
    size_t s_vec = 0, s_meta = 0, s_idx = 0, s_imeta = 0, s_wal = 0;
    exists_size(p_vec, e_vec, s_vec);
    exists_size(p_meta, e_meta, s_meta);
    exists_size(p_idx, e_idx, s_idx);
    exists_size(p_idx_meta, e_imeta, s_imeta);
    exists_size(p_wal, e_wal, s_wal);

    StorageHeaderOnDisk disk_raw{};
    uint32_t disk_vec_version = 0;
    bool have_disk_header = false;
    if (e_vec && s_vec >= sizeof(StorageHeaderOnDisk))
    {
        FILE* f = fopen(p_vec.c_str(), "rb");
        if (f)
        {
            if (fread(&disk_raw, 1, sizeof(disk_raw), f) == sizeof(disk_raw))
            {
                disk_vec_version = disk_raw.version;
                have_disk_header = true;
            }
            fclose(f);
        }
    }

    StorageHeaderOnDisk vhdr{};
    std::string vec_err;
    bool vec_ok = e_vec && read_vectors_probe(p_vec, vhdr, s_vec, vec_err);

    IndexMetaProbe im = read_index_meta(p_idx_meta);
    bool wal_ex = false;
    uint32_t wal_ver = 0;
    std::string wal_err;
    bool wal_ok = read_wal_probe(p_wal, wal_ex, wal_ver, wal_err);

    std::vector<std::string> recs;
    if (!e_vec)
        recs.push_back("vectors.bin is missing; this directory is not a logosdb data path.");
    else if (!vec_ok)
        recs.push_back(std::string("vectors.bin: ") + vec_err);
    if (e_idx && !e_imeta)
        recs.push_back("hnsw.idx exists without hnsw.idx.meta; restore metadata or rebuild the "
                       "index.");
    if (im.valid && vec_ok && im.dim != (int)vhdr.dim)
        recs.push_back("index meta dimension does not match vectors.bin dim.");
    if (wal_ex && !wal_ok)
        recs.push_back(std::string("wal.log: ") + wal_err);
    if (have_disk_header && disk_vec_version > 0u && disk_vec_version < 2u)
        recs.push_back("vectors.bin on-disk header is legacy v0/v1; run: logosdb-cli upgrade " +
                       db_path + " --apply");

    int probe_distance = LOGOSDB_DIST_IP;
    if (distance_override >= 0)
        probe_distance = distance_override;
    else if (im.valid)
        probe_distance = im.distance;

    if (im.valid && distance_override >= 0 && distance_override != im.distance)
        recs.push_back("doctor: using --distance override; index meta reports a different metric.");

    bool open_ok = false;
    std::string open_err;
    if (vec_ok && vhdr.dim > 0u)
    {
        logosdb_options_t* opts = logosdb_options_create();
        logosdb_options_set_dim(opts, (int)vhdr.dim);
        logosdb_options_set_distance(opts, probe_distance);
        logosdb_options_set_dtype(opts, (int)vhdr.dtype);
        char* err = nullptr;
        logosdb_t* db = logosdb_open(db_path.c_str(), opts, &err);
        logosdb_options_destroy(opts);
        if (db)
        {
            open_ok = true;
            logosdb_close(db);
        }
        else
        {
            open_err = err ? err : "unknown";
            free(err);
            recs.push_back(std::string("logosdb_open failed: ") + open_err);
        }
    }

    if (json)
    {
        printf("{\n");
        printf("  \"library_version\": \"%s\",\n", LOGOSDB_VERSION_STRING);
        printf("  \"path\": \"");
        print_json_escaped_value(db_path);
        printf("\",\n");
        printf("  \"files\": {\n");
        printf("    \"vectors.bin\": { \"exists\": %s, \"size\": %zu },\n",
               e_vec ? "true" : "false",
               s_vec);
        printf("    \"meta.jsonl\": { \"exists\": %s, \"size\": %zu },\n",
               e_meta ? "true" : "false",
               s_meta);
        printf("    \"hnsw.idx\": { \"exists\": %s, \"size\": %zu },\n",
               e_idx ? "true" : "false",
               s_idx);
        printf("    \"hnsw.idx.meta\": { \"exists\": %s, \"size\": %zu },\n",
               e_imeta ? "true" : "false",
               s_imeta);
        printf("    \"wal.log\": { \"exists\": %s, \"size\": %zu }\n",
               e_wal ? "true" : "false",
               s_wal);
        printf("  },\n");
        printf("  \"vectors\": {\n");
        if (have_disk_header)
        {
            printf("    \"on_disk_version\": %u,\n", disk_vec_version);
            printf("    \"dim\": %u,\n", disk_raw.dim);
            printf("    \"n_rows\": %llu,\n", (unsigned long long)disk_raw.n_rows);
        }
        else
        {
            printf("    \"on_disk_version\": null,\n");
            printf("    \"dim\": null,\n");
            printf("    \"n_rows\": null,\n");
        }
        printf("    \"layout_ok\": %s,\n", vec_ok ? "true" : "false");
        if (!vec_ok && !vec_err.empty())
        {
            printf("    \"layout_error\": \"");
            print_json_escaped_value(vec_err);
            printf("\",\n");
        }
        if (vec_ok)
        {
            printf("    \"logical_dtype\": \"%s\",\n", dtype_label(vhdr.dtype));
            printf("    \"logical_version\": %u\n", vhdr.version);
        }
        else
        {
            printf("    \"logical_dtype\": null,\n");
            printf("    \"logical_version\": null\n");
        }
        printf("  },\n");
        printf("  \"index_meta\": {\n");
        printf("    \"present\": %s,\n", im.file_exists ? "true" : "false");
        printf("    \"valid\": %s", im.valid ? "true" : "false");
        if (im.valid)
        {
            printf(",\n    \"distance\": \"%s\",\n", distance_label(im.distance));
            printf("    \"dim\": %d\n", im.dim);
        }
        else if (!im.parse_error.empty())
        {
            printf(",\n    \"error\": \"");
            print_json_escaped_value(im.parse_error);
            printf("\"\n");
        }
        else
        {
            printf(",\n    \"distance\": null,\n");
            printf("    \"dim\": null\n");
        }
        printf("  },\n");
        printf("  \"wal\": {\n");
        printf("    \"exists\": %s,\n", wal_ex ? "true" : "false");
        if (wal_ex)
            printf("    \"version\": %u,\n", wal_ver);
        else
            printf("    \"version\": null,\n");
        printf("    \"header_ok\": %s", wal_ok ? "true" : "false");
        if (wal_ex && !wal_ok && !wal_err.empty())
        {
            printf(",\n    \"error\": \"");
            print_json_escaped_value(wal_err);
            printf("\"");
        }
        printf("\n  },\n");
        printf("  \"probe_open\": {\n");
        printf("    \"distance\": \"%s\",\n", distance_label(probe_distance));
        printf("    \"ok\": %s", open_ok ? "true" : "false");
        if (!open_ok && !open_err.empty())
        {
            printf(",\n    \"error\": \"");
            print_json_escaped_value(open_err);
            printf("\"");
        }
        printf("\n  },\n");
        printf("  \"recommendations\": [\n");
        for (size_t i = 0; i < recs.size(); ++i)
        {
            printf("    \"");
            print_json_escaped_value(recs[i]);
            printf("\"%s\n", (i + 1 < recs.size()) ? "," : "");
        }
        printf("  ]\n}\n");
        return (vec_ok && open_ok && wal_ok) ? 0 : 1;
    }

    printf("logosdb doctor — %s\n", db_path.c_str());
    printf("library version     : %s\n", LOGOSDB_VERSION_STRING);
    printf("vectors.bin         : %s  (%zu bytes)\n", e_vec ? "yes" : "no", s_vec);
    printf("meta.jsonl          : %s  (%zu bytes)\n", e_meta ? "yes" : "no", s_meta);
    printf("hnsw.idx            : %s  (%zu bytes)\n", e_idx ? "yes" : "no", s_idx);
    printf("hnsw.idx.meta       : %s  (%zu bytes)\n", e_imeta ? "yes" : "no", s_imeta);
    printf("wal.log             : %s  (%zu bytes)\n", e_wal ? "yes" : "no", s_wal);
    if (have_disk_header)
    {
        printf("vectors on-disk hdr : version=%u dim=%u n_rows=%llu\n",
               disk_vec_version,
               disk_raw.dim,
               (unsigned long long)disk_raw.n_rows);
    }
    if (vec_ok)
    {
        printf("vectors layout      : OK (logical v%u dtype=%s)\n",
               vhdr.version,
               dtype_label(vhdr.dtype));
    }
    else if (e_vec)
        printf("vectors layout      : FAILED (%s)\n", vec_err.c_str());

    if (im.valid)
        printf(
            "index meta          : OK (distance=%s dim=%d)\n", distance_label(im.distance), im.dim);
    else if (e_imeta)
        printf("index meta          : INVALID (%s)\n", im.parse_error.c_str());
    else
        printf("index meta          : (missing)\n");

    if (!wal_ex)
        printf("wal.log             : (missing; normal for a new directory before first write)\n");
    else if (wal_ok)
        printf("wal.log header      : OK (version %u)\n", wal_ver);
    else
        printf("wal.log header      : FAILED (%s)\n", wal_err.c_str());

    printf("probe logosdb_open  : distance=%s %s\n",
           distance_label(probe_distance),
           open_ok ? "OK" : "FAILED");
    if (!open_ok && !open_err.empty())
        printf("                      (%s)\n", open_err.c_str());

    if (!recs.empty())
    {
        printf("\nRecommendations:\n");
        for (const auto& r : recs)
            printf("  - %s\n", r.c_str());
    }

    return (vec_ok && open_ok && wal_ok) ? 0 : 1;
}

static bool snapshot_manifest_ok(const std::string& json_path)
{
    std::ifstream in(json_path);
    if (!in.good())
        return false;
    std::string s((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return s.find("\"format\":\"logosdb.snapshot\"") != std::string::npos &&
           s.find("\"format_version\":1") != std::string::npos;
}

static int cmd_restore(const std::string& snap_root, const std::string& target_root, bool force)
{
    const std::string man = snap_root + "/snapshot.json";
    if (!snapshot_manifest_ok(man))
    {
        fprintf(stderr, "error: missing or invalid %s\n", man.c_str());
        return 1;
    }
    const std::string vec_snap = snap_root + "/vectors.bin";
    std::error_code ec;
    if (!std::filesystem::exists(vec_snap, ec))
    {
        fprintf(stderr, "error: snapshot has no vectors.bin\n");
        return 1;
    }

    bool tgt_exists = std::filesystem::exists(target_root, ec);
    if (tgt_exists)
    {
        if (!std::filesystem::is_empty(target_root, ec))
        {
            if (!force)
            {
                fprintf(stderr, "error: restore target is not empty (use --force to replace it)\n");
                return 1;
            }
            std::filesystem::remove_all(target_root, ec);
        }
    }
    std::filesystem::create_directories(target_root, ec);

    static const char* kParts[] = {
        "vectors.bin", "meta.jsonl", "hnsw.idx", "hnsw.idx.meta", "wal.log", nullptr};
    for (int i = 0; kParts[i]; ++i)
    {
        std::string src = snap_root + "/" + kParts[i];
        if (!std::filesystem::exists(src, ec))
            continue;
        std::filesystem::copy_file(src,
                                   target_root + "/" + kParts[i],
                                   std::filesystem::copy_options::overwrite_existing,
                                   ec);
        if (ec)
        {
            fprintf(stderr, "error: copy %s: %s\n", kParts[i], ec.message().c_str());
            return 1;
        }
    }
    printf("restored snapshot to %s\n", target_root.c_str());
    return 0;
}

static int cmd_snapshot(const std::string& db_root, const std::string& dest_root, bool overwrite)
{
    std::error_code ec;
    if (std::filesystem::exists(dest_root, ec))
    {
        if (!std::filesystem::is_empty(dest_root, ec))
        {
            if (!overwrite)
            {
                fprintf(stderr, "error: snapshot directory is not empty (use --overwrite)\n");
                return 1;
            }
            std::filesystem::remove_all(dest_root, ec);
        }
    }
    std::filesystem::create_directories(dest_root, ec);

    int dim = 0;
    std::string vec_path = db_root + "/vectors.bin";
    if (!read_dim_from_header(vec_path.c_str(), dim))
    {
        fprintf(stderr, "error: cannot read dim from %s\n", vec_path.c_str());
        return 1;
    }

    logosdb_options_t* opts = logosdb_options_create();
    logosdb_options_set_dim(opts, dim);
    char* err = nullptr;
    logosdb_t* db = logosdb_open(db_root.c_str(), opts, &err);
    logosdb_options_destroy(opts);
    if (!db)
    {
        fprintf(stderr, "error: %s\n", err ? err : "open failed");
        free(err);
        return 1;
    }

    const size_t n_total = logosdb_count(db);
    const size_t n_live = logosdb_count_live(db);
    const int dim_chk = logosdb_dim(db);

    if (logosdb_sync(db, &err) != 0)
    {
        fprintf(stderr, "error: logosdb_sync: %s\n", err ? err : "unknown");
        free(err);
        logosdb_close(db);
        return 1;
    }
    logosdb_close(db);

    static const char* kParts[] = {
        "vectors.bin", "meta.jsonl", "hnsw.idx", "hnsw.idx.meta", "wal.log", nullptr};
    for (int i = 0; kParts[i]; ++i)
    {
        std::string src = db_root + "/" + kParts[i];
        if (!std::filesystem::exists(src, ec))
            continue;
        std::filesystem::copy_file(src,
                                   dest_root + "/" + kParts[i],
                                   std::filesystem::copy_options::overwrite_existing,
                                   ec);
        if (ec)
        {
            fprintf(stderr, "error: copy %s: %s\n", kParts[i], ec.message().c_str());
            return 1;
        }
    }

    const std::string man = dest_root + "/snapshot.json";
    FILE* mf = fopen(man.c_str(), "w");
    if (!mf)
    {
        fprintf(stderr, "error: cannot write snapshot.json\n");
        return 1;
    }
    std::time_t now = std::time(nullptr);
    fprintf(mf,
            "{\"format\":\"logosdb.snapshot\",\"format_version\":1,\"library_version\":\"%s\","
            "\"created_unix\":%lld,\"dim\":%d,\"count\":%zu,\"count_live\":%zu}\n",
            LOGOSDB_VERSION_STRING,
            (long long)now,
            dim_chk,
            n_total,
            n_live);
    fclose(mf);
    printf("snapshot written to %s (%zu rows, %zu live, dim=%d)\n",
           dest_root.c_str(),
           n_total,
           n_live,
           dim_chk);
    return 0;
}

static int cmd_stats(const std::string& db_path, bool json)
{
    int dim = 0;
    std::string vec_path = db_path + "/vectors.bin";
    if (!read_dim_from_header(vec_path.c_str(), dim))
    {
        fprintf(stderr, "error: cannot read dim from %s\n", vec_path.c_str());
        return 1;
    }
    logosdb_options_t* opts = logosdb_options_create();
    logosdb_options_set_dim(opts, dim);
    char* err = nullptr;
    logosdb_t* db = logosdb_open(db_path.c_str(), opts, &err);
    logosdb_options_destroy(opts);
    if (!db)
    {
        fprintf(stderr, "error: %s\n", err ? err : "open failed");
        free(err);
        return 1;
    }
    logosdb_stats_t s{};
    logosdb_stats(db, &s);
    logosdb_close(db);

    if (json)
    {
        printf("{\n");
        printf("  \"rows_total\": %llu,\n", (unsigned long long)s.rows_total);
        printf("  \"rows_live\": %llu,\n", (unsigned long long)s.rows_live);
        printf("  \"tombstones\": %llu,\n", (unsigned long long)s.tombstones);
        printf("  \"index_elements\": %llu,\n", (unsigned long long)s.index_elements);
        printf("  \"wal_pending\": %llu,\n", (unsigned long long)s.wal_pending);
        printf("  \"distance_metric\": %d,\n", s.distance_metric);
        printf("  \"storage_dtype\": %d,\n", s.storage_dtype);
        printf("  \"put_success\": %llu,\n", (unsigned long long)s.put_success);
        printf("  \"put_failed\": %llu,\n", (unsigned long long)s.put_failed);
        printf("  \"put_batch_success\": %llu,\n", (unsigned long long)s.put_batch_success);
        printf("  \"put_batch_failed\": %llu,\n", (unsigned long long)s.put_batch_failed);
        printf("  \"search_success\": %llu,\n", (unsigned long long)s.search_success);
        printf("  \"search_failed\": %llu,\n", (unsigned long long)s.search_failed);
        printf("  \"search_ts_success\": %llu,\n", (unsigned long long)s.search_ts_success);
        printf("  \"search_ts_failed\": %llu,\n", (unsigned long long)s.search_ts_failed);
        printf("  \"delete_success\": %llu,\n", (unsigned long long)s.delete_success);
        printf("  \"delete_failed\": %llu,\n", (unsigned long long)s.delete_failed);
        printf("  \"update_success\": %llu,\n", (unsigned long long)s.update_success);
        printf("  \"update_failed\": %llu,\n", (unsigned long long)s.update_failed);
        printf("  \"sync_calls\": %llu\n", (unsigned long long)s.sync_calls);
        printf("}\n");
        return 0;
    }

    printf("stats — %s\n", db_path.c_str());
    printf("rows_total / live / tombstones : %llu / %llu / %llu\n",
           (unsigned long long)s.rows_total,
           (unsigned long long)s.rows_live,
           (unsigned long long)s.tombstones);
    printf("index_elements                 : %llu\n", (unsigned long long)s.index_elements);
    printf("wal_pending                    : %llu\n", (unsigned long long)s.wal_pending);
    printf("distance / storage_dtype       : %d / %d\n", s.distance_metric, s.storage_dtype);
    printf("put ok / fail                  : %llu / %llu\n",
           (unsigned long long)s.put_success,
           (unsigned long long)s.put_failed);
    printf("put_batch ok / fail            : %llu / %llu\n",
           (unsigned long long)s.put_batch_success,
           (unsigned long long)s.put_batch_failed);
    printf("search ok / fail               : %llu / %llu\n",
           (unsigned long long)s.search_success,
           (unsigned long long)s.search_failed);
    printf("search_ts ok / fail            : %llu / %llu\n",
           (unsigned long long)s.search_ts_success,
           (unsigned long long)s.search_ts_failed);
    printf("delete ok / fail               : %llu / %llu\n",
           (unsigned long long)s.delete_success,
           (unsigned long long)s.delete_failed);
    printf("update ok / fail               : %llu / %llu\n",
           (unsigned long long)s.update_success,
           (unsigned long long)s.update_failed);
    printf("sync_calls                     : %llu\n", (unsigned long long)s.sync_calls);
    return 0;
}

static int cmd_compact_cli(const std::string& src, const std::string& dst, bool force_replace)
{
    std::error_code ec;
    if (std::filesystem::exists(dst, ec) && !std::filesystem::is_empty(dst, ec))
    {
        if (!force_replace)
        {
            fprintf(stderr, "error: compact destination is not empty (use --force)\n");
            return 1;
        }
        std::filesystem::remove_all(dst, ec);
    }
    char* err = nullptr;
    if (logosdb_compact(src.c_str(), dst.c_str(), &err) != 0)
    {
        fprintf(stderr, "error: %s\n", err ? err : "compact failed");
        free(err);
        return 1;
    }
    free(err);
    printf("compact completed: %s -> %s\n", src.c_str(), dst.c_str());
    return 0;
}

static void print_version()
{
    printf("logosdb-cli version %s\n", LOGOSDB_VERSION_STRING);
}

static void print_usage(const char* prog)
{
    printf(
        "Usage: %s <command> [options]\n"
        "\n"
        "Commands:\n"
        "  info <db-path> [--json]           Show database info (dim read from file)\n"
        "  put <db-path> [options]           Insert a vector\n"
        "  search <db-path> [options]        Search for similar vectors\n"
        "  export <db-path> [--output FILE]  Export DB to JSONL with base64 vectors\n"
        "  import <db-path> [--input FILE]   Import from JSONL\n"
        "  doctor <db-path> [--json] [--distance ip|cosine|l2]\n"
        "                                    Compatibility checks and open probe\n"
        "  upgrade <db-path> [--apply|--yes] Rewrite legacy vectors.bin header to v2\n"
        "  snapshot <db-path> <dir> [--overwrite]\n"
        "                                    Point-in-time copy after logosdb_sync\n"
        "  restore <snapshot-dir> <db-path> [--force]\n"
        "                                    Restore files from snapshot to new DB dir\n"
        "  stats <db-path> [--json]          Ingest/query counters and store health\n"
        "  compact <src-db> <dst-db> [--force]\n"
        "                                    Dense copy (drops tombstones); dst empty or --force\n"
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
        "  %s doctor /tmp/mydb --json\n"
        "  %s upgrade /tmp/mydb --apply\n"
        "  %s snapshot /tmp/mydb /tmp/mydb.snap --overwrite\n"
        "  %s restore /tmp/mydb.snap /tmp/mydb_restored --force\n"
        "  %s stats /tmp/mydb --json\n"
        "  %s compact /tmp/mydb /tmp/mydb_dense --force\n",
        prog,
        prog,
        prog,
        prog,
        prog,
        prog,
        prog,
        prog,
        prog,
        prog,
        prog,
        prog,
        prog,
        prog);
}

static void print_cmd_help(const char* cmd)
{
    if (strcmp(cmd, "info") == 0)
    {
        printf("Usage: logosdb-cli info <db-path> [--json]\n\nShow database information.\n");
    }
    else if (strcmp(cmd, "put") == 0)
    {
        printf("Usage: logosdb-cli put <db-path> [options]\n\nOptions:\n  --dim N\n  --text TEXT\n "
               " --ts TIMESTAMP\n  --embedding-file FILE\n");
    }
    else if (strcmp(cmd, "search") == 0)
    {
        printf("Usage: logosdb-cli search <db-path> [options]\n\nOptions:\n  --dim N\n  "
               "--query-file FILE\n  --query-id ID\n  --top-k N\n  --ts-from TIMESTAMP\n  --ts-to "
               "TIMESTAMP\n  --json\n");
    }
    else if (strcmp(cmd, "export") == 0)
    {
        printf(
            "Usage: logosdb-cli export <db-path> [--output FILE]\n\nExport database to JSONL.\n");
    }
    else if (strcmp(cmd, "import") == 0)
    {
        printf("Usage: logosdb-cli import <db-path> [options]\n\nOptions:\n  --dim N (required for "
               "new DB)\n  --input FILE\n");
    }
    else if (strcmp(cmd, "doctor") == 0)
    {
        printf("Usage: logosdb-cli doctor <db-path> [--json] [--distance ip|cosine|l2]\n\n"
               "Inspects expected files, validates vectors.bin layout, reads hnsw.idx.meta and "
               "wal.log headers,\n"
               "and probes logosdb_open using inferred dim/dtype/distance.\n"
               "Exit status is non-zero if layout check fails, WAL header is invalid, or open "
               "fails.\n");
    }
    else if (strcmp(cmd, "upgrade") == 0)
    {
        printf("Usage: logosdb-cli upgrade <db-path> [--apply|--yes]\n\n"
               "Rewrites legacy vectors.bin storage headers (v0/v1) to v2 on disk.\n"
               "Without --apply/--yes, only prints the planned change (dry run).\n");
    }
    else if (strcmp(cmd, "snapshot") == 0)
    {
        printf("Usage: logosdb-cli snapshot <db-path> <snapshot-dir> [--overwrite]\n\n"
               "Calls logosdb_sync (WAL + index + vectors + metadata), then copies data files\n"
               "and writes snapshot.json (format logosdb.snapshot v1).\n");
    }
    else if (strcmp(cmd, "restore") == 0)
    {
        printf("Usage: logosdb-cli restore <snapshot-dir> <db-path> [--force]\n\n"
               "Validates snapshot.json, then copies snapshot data files into the target path.\n"
               "Target must be missing or empty unless --force is given.\n");
    }
    else if (strcmp(cmd, "stats") == 0)
    {
        printf("Usage: logosdb-cli stats <db-path> [--json]\n\n"
               "Prints operation counters (puts, searches, deletes, sync) and store health.\n");
    }
    else if (strcmp(cmd, "compact") == 0)
    {
        printf("Usage: logosdb-cli compact <src-db> <dst-db> [--force]\n\n"
               "Copies live rows into a new dense database (same dim/distance/dtype).\n"
               "Destination must be empty unless --force.\n");
    }
    else
    {
        printf("Unknown command: %s\n", cmd);
    }
}

// Command-line argument structure
struct Args
{
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
    int distance_override = -1;
    bool upgrade_apply = false;
    std::string second_path;
    bool snapshot_overwrite = false;
    bool force_replace = false;
};

static Args parse_args(int argc, char** argv)
{
    Args args;
    if (argc < 2)
        return args;

    // Check for global --version or --help first
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--version") == 0)
        {
            args.version = true;
            return args;
        }
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0)
        {
            args.help = true;
            if (i + 1 < argc && argv[i + 1][0] != '-')
            {
                args.cmd = argv[i + 1];  // --help <cmd>
            }
            return args;
        }
    }

    if (argc < 3)
        return args;

    args.cmd = argv[1];
    args.db_path = argv[2];
    int flag_i = 3;
    if ((args.cmd == "snapshot" || args.cmd == "restore" || args.cmd == "compact") && argc >= 4)
    {
        args.second_path = argv[3];
        flag_i = 4;
    }

    for (int i = flag_i; i < argc; ++i)
    {
        if (!strcmp(argv[i], "--dim") && i + 1 < argc)
        {
            long v = strtol(argv[++i], nullptr, 10);
            args.dim = (v > 0 && v <= 65536) ? (int)v : 0;
        }
        else if (!strcmp(argv[i], "--top-k") && i + 1 < argc)
        {
            long v = strtol(argv[++i], nullptr, 10);
            args.top_k = (v > 0 && v <= 10000) ? (int)v : 5;
        }
        else if (!strcmp(argv[i], "--text") && i + 1 < argc)
            args.text = argv[++i];
        else if (!strcmp(argv[i], "--ts") && i + 1 < argc)
            args.ts = argv[++i];
        else if (!strcmp(argv[i], "--embedding-file") && i + 1 < argc)
            args.emb_file = argv[++i];
        else if (!strcmp(argv[i], "--query-file") && i + 1 < argc)
            args.query_file = argv[++i];
        else if (!strcmp(argv[i], "--query-id") && i + 1 < argc)
        {
            long long v = strtoll(argv[++i], nullptr, 10);
            args.query_id = (v >= 0) ? (uint64_t)v : UINT64_MAX;
        }
        else if (!strcmp(argv[i], "--ts-from") && i + 1 < argc)
            args.ts_from = argv[++i];
        else if (!strcmp(argv[i], "--ts-to") && i + 1 < argc)
            args.ts_to = argv[++i];
        else if (!strcmp(argv[i], "--output") && i + 1 < argc)
            args.output_file = argv[++i];
        else if (!strcmp(argv[i], "--input") && i + 1 < argc)
            args.input_file = argv[++i];
        else if (!strcmp(argv[i], "--json"))
            args.json = true;
        else if (!strcmp(argv[i], "--apply") || !strcmp(argv[i], "--yes"))
            args.upgrade_apply = true;
        else if (!strcmp(argv[i], "--overwrite"))
            args.snapshot_overwrite = true;
        else if (!strcmp(argv[i], "--force"))
            args.force_replace = true;
        else if (!strcmp(argv[i], "--distance") && i + 1 < argc)
        {
            const char* d = argv[++i];
            if (!strcmp(d, "ip"))
                args.distance_override = LOGOSDB_DIST_IP;
            else if (!strcmp(d, "cosine"))
                args.distance_override = LOGOSDB_DIST_COSINE;
            else if (!strcmp(d, "l2"))
                args.distance_override = LOGOSDB_DIST_L2;
        }
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
            args.help = true;
    }

    return args;
}

int main(int argc, char** argv)
{
    Args args = parse_args(argc, argv);

    if (args.version)
    {
        print_version();
        return 0;
    }

    if (args.help)
    {
        if (!args.cmd.empty())
        {
            print_cmd_help(args.cmd.c_str());
        }
        else
        {
            print_usage(argv[0]);
        }
        return 0;
    }

    if (args.cmd.empty() || args.db_path.empty())
    {
        print_usage(argv[0]);
        return 1;
    }

    if (args.cmd == "doctor")
        return run_doctor(args.db_path, args.json, args.distance_override);
    if (args.cmd == "upgrade")
        return run_upgrade_cmd(args.db_path, args.upgrade_apply);
    if (args.cmd == "snapshot" || args.cmd == "restore" || args.cmd == "compact")
    {
        if (args.second_path.empty())
        {
            fprintf(stderr, "error: %s requires a second path argument\n", args.cmd.c_str());
            print_usage(argv[0]);
            return 1;
        }
        if (args.cmd == "snapshot")
            return cmd_snapshot(args.db_path, args.second_path, args.snapshot_overwrite);
        if (args.cmd == "restore")
            return cmd_restore(args.db_path, args.second_path, args.force_replace);
        return cmd_compact_cli(args.db_path, args.second_path, args.force_replace);
    }

    if (args.cmd == "stats")
        return cmd_stats(args.db_path, args.json);

    // Try to read dim from existing database for info command
    if (args.cmd == "info" && args.dim == 0)
    {
        std::string vec_path = args.db_path + "/vectors.bin";
        if (!read_dim_from_header(vec_path.c_str(), args.dim))
        {
            fprintf(stderr,
                    "error: cannot read dim from %s (use --dim for new DB)\n",
                    vec_path.c_str());
            return 1;
        }
    }

    // For import, dim may be optional if DB exists
    if (args.cmd == "import" && args.dim == 0)
    {
        std::string vec_path = args.db_path + "/vectors.bin";
        if (!read_dim_from_header(vec_path.c_str(), args.dim))
        {
            fprintf(stderr, "error: --dim required for new database\n");
            return 1;
        }
    }

    // For other commands, dim is required if DB doesn't exist
    // For export, we also need to read the dim from the file
    if (args.dim == 0)
    {
        std::string vec_path = args.db_path + "/vectors.bin";
        if (!read_dim_from_header(vec_path.c_str(), args.dim))
        {
            if (args.cmd != "export")
            {
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
    if (args.dim > 0)
        logosdb_options_set_dim(opts, args.dim);
    logosdb_t* db = logosdb_open(args.db_path.c_str(), opts, &err);
    logosdb_options_destroy(opts);

    if (!db)
    {
        fprintf(stderr, "error: %s\n", err ? err : "unknown");
        free(err);
        return 1;
    }

    int rc = 0;

    if (args.cmd == "info")
    {
        if (args.json)
        {
            printf("{\n");
            printf("  \"path\": \"%s\",\n", args.db_path.c_str());
            printf("  \"dim\": %d,\n", logosdb_dim(db));
            printf("  \"count\": %zu,\n", logosdb_count(db));
            printf("  \"count_live\": %zu\n", logosdb_count_live(db));
            printf("}\n");
        }
        else
        {
            printf("path       : %s\n", args.db_path.c_str());
            printf("dim        : %d\n", logosdb_dim(db));
            printf("count      : %zu\n", logosdb_count(db));
            printf("count_live : %zu\n", logosdb_count_live(db));
        }
    }
    else if (args.cmd == "put")
    {
        if (!args.emb_file)
        {
            fprintf(stderr, "error: --embedding-file required\n");
            rc = 1;
            goto done;
        }
        auto vec = read_binary_vec(args.emb_file, logosdb_dim(db));
        if ((int)vec.size() != logosdb_dim(db))
        {
            fprintf(
                stderr, "error: could not read embedding (expected %d floats)\n", logosdb_dim(db));
            rc = 1;
            goto done;
        }
        if (!validate_cli_text_metadata("--text", args.text))
        {
            rc = 1;
            goto done;
        }
        uint64_t id = logosdb_put(db, vec.data(), logosdb_dim(db), args.text, args.ts, &err);
        if (id == UINT64_MAX)
        {
            fprintf(stderr, "error: %s\n", err ? err : "unknown");
            free(err);
            rc = 1;
        }
        else
        {
            if (args.json)
            {
                printf("{\"id\": %llu}\n", (unsigned long long)id);
            }
            else
            {
                printf("put id=%llu\n", (unsigned long long)id);
            }
        }
    }
    else if (args.cmd == "search")
    {
        std::vector<float> qvec;

        if (args.query_file)
        {
            qvec = read_binary_vec(args.query_file, logosdb_dim(db));
            if ((int)qvec.size() != logosdb_dim(db))
            {
                fprintf(
                    stderr, "error: could not read query (expected %d floats)\n", logosdb_dim(db));
                rc = 1;
                goto done;
            }
        }
        else if (args.query_id != UINT64_MAX)
        {
            // Use existing vector as query
            const float* raw = logosdb_raw_vectors(db, nullptr, nullptr);
            if (!raw || args.query_id >= logosdb_count(db))
            {
                fprintf(
                    stderr, "error: invalid query-id %llu\n", (unsigned long long)args.query_id);
                rc = 1;
                goto done;
            }
            qvec.resize(logosdb_dim(db));
            std::memcpy(qvec.data(),
                        raw + args.query_id * logosdb_dim(db),
                        logosdb_dim(db) * sizeof(float));
        }
        else
        {
            fprintf(stderr, "error: --query-file or --query-id required\n");
            rc = 1;
            goto done;
        }

        logosdb_search_result_t* res;
        if (args.ts_from || args.ts_to)
        {
            int candidate_k = args.top_k * 10;
            res = logosdb_search_ts_range(db,
                                          qvec.data(),
                                          logosdb_dim(db),
                                          args.top_k,
                                          args.ts_from,
                                          args.ts_to,
                                          candidate_k,
                                          &err);
        }
        else
        {
            res = logosdb_search(db, qvec.data(), logosdb_dim(db), args.top_k, &err);
        }

        if (!res)
        {
            fprintf(stderr, "error: %s\n", err ? err : "unknown");
            free(err);
            rc = 1;
        }
        else
        {
            int n = logosdb_result_count(res);
            if (args.json)
            {
                printf("[\n");
                for (int i = 0; i < n; ++i)
                {
                    printf("  {\n");
                    printf("    \"rank\": %d,\n", i);
                    printf("    \"id\": %llu,\n", (unsigned long long)logosdb_result_id(res, i));
                    printf("    \"score\": %.6f", logosdb_result_score(res, i));
                    const char* t = logosdb_result_text(res, i);
                    const char* tts = logosdb_result_timestamp(res, i);
                    if (t)
                    {
                        printf(",\n    \"text\": \"%s\"", t);
                        if (tts)
                        {
                            printf(",\n    \"timestamp\": \"%s\"", tts);
                        }
                    }
                    else if (tts)
                    {
                        printf(",\n    \"timestamp\": \"%s\"", tts);
                    }
                    printf("\n  }%s\n", (i < n - 1) ? "," : "");
                }
                printf("]\n");
            }
            else
            {
                printf("results: %d\n", n);
                for (int i = 0; i < n; ++i)
                {
                    printf("  #%d id=%llu score=%.6f",
                           i,
                           (unsigned long long)logosdb_result_id(res, i),
                           logosdb_result_score(res, i));
                    const char* t = logosdb_result_text(res, i);
                    if (t)
                        printf(" text=\"%s\"", t);
                    const char* tts = logosdb_result_timestamp(res, i);
                    if (tts)
                        printf(" ts=%s", tts);
                    printf("\n");
                }
            }
            logosdb_result_free(res);
        }
    }
    else if (args.cmd == "export")
    {
        FILE* out = stdout;
        if (args.output_file)
        {
            out = fopen(args.output_file, "w");
            if (!out)
            {
                fprintf(stderr, "error: cannot open output file %s\n", args.output_file);
                rc = 1;
                goto done;
            }
        }

        size_t n_rows = 0;
        int dim = 0;
        const float* raw = logosdb_raw_vectors(db, &n_rows, &dim);

        for (size_t i = 0; i < n_rows; ++i)
        {
            // Get metadata
            char tmp_err[256];
            logosdb_search_result_t* tmp_res = logosdb_search(db, raw + i * dim, dim, 1, nullptr);
            const char* text = "";
            const char* ts = "";
            if (tmp_res)
            {
                text = logosdb_result_text(tmp_res, 0);
                if (!text)
                    text = "";
                ts = logosdb_result_timestamp(tmp_res, 0);
                if (!ts)
                    ts = "";
            }

            // Encode vector as base64
            std::string b64 = base64_encode(raw + i * dim, dim);

            // Output JSONL
            fprintf(out,
                    "{\"id\": %zu, \"vector\": \"%s\", \"text\": \"%s\", \"timestamp\": \"%s\"}\n",
                    i,
                    b64.c_str(),
                    text,
                    ts);

            if (tmp_res)
                logosdb_result_free(tmp_res);
        }

        if (args.output_file)
        {
            fclose(out);
            printf("Exported %zu rows to %s\n", n_rows, args.output_file);
        }
    }
    else if (args.cmd == "import")
    {
        FILE* in = stdin;
        if (args.input_file)
        {
            in = fopen(args.input_file, "r");
            if (!in)
            {
                fprintf(stderr, "error: cannot open input file %s\n", args.input_file);
                rc = 1;
                goto done;
            }
        }

        size_t imported = 0;
        size_t lines_read = 0;
        char line[65536];
        while (fgets(line, sizeof(line), in))
        {
            lines_read++;

            // Extract vector (base64)
            char* vec_field = strstr(line, "\"vector\"");
            if (!vec_field)
                continue;

            // Find the value after "vector":
            char* vec_start = strchr(vec_field, ':');
            if (!vec_start)
                continue;
            vec_start++;
            while (*vec_start == ' ')
                vec_start++;
            if (*vec_start != '"')
                continue;
            vec_start++;  // skip opening quote

            char* vec_end = strchr(vec_start, '"');
            if (!vec_end)
                continue;

            // Temporarily null-terminate to extract
            *vec_end = '\0';
            std::string b64(vec_start);
            *vec_end = '"';

            auto vec = base64_decode(b64, logosdb_dim(db));
            if ((int)vec.size() != logosdb_dim(db))
            {
                fprintf(stderr, "warning: skipping row with wrong dimension\n");
                continue;
            }

            // Extract text
            std::string text;
            char* text_field = strstr(line, "\"text\"");
            if (text_field)
            {
                char* text_start = strchr(text_field, ':');
                if (text_start)
                {
                    text_start++;
                    while (*text_start == ' ')
                        text_start++;
                    if (*text_start == '"')
                    {
                        text_start++;
                        char* text_end = strchr(text_start, '"');
                        if (text_end)
                        {
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
            if (ts_field)
            {
                char* ts_start = strchr(ts_field, ':');
                if (ts_start)
                {
                    ts_start++;
                    while (*ts_start == ' ')
                        ts_start++;
                    if (*ts_start == '"')
                    {
                        ts_start++;
                        char* ts_end = strchr(ts_start, '"');
                        if (ts_end)
                        {
                            *ts_end = '\0';
                            ts = ts_start;
                            *ts_end = '"';
                        }
                    }
                }
            }

            // Insert
            uint64_t id = logosdb_put(db,
                                      vec.data(),
                                      logosdb_dim(db),
                                      text.empty() ? nullptr : text.c_str(),
                                      ts.empty() ? nullptr : ts.c_str(),
                                      &err);
            if (id != UINT64_MAX)
            {
                imported++;
            }
            else
            {
                if (err)
                {
                    fprintf(stderr, "warning: insert failed: %s\n", err);
                    free(err);
                    err = nullptr;
                }
            }
        }

        if (args.input_file)
            fclose(in);
        printf("Imported %zu rows\n", imported);
    }
    else
    {
        fprintf(stderr, "error: unknown command: %s\n", args.cmd.c_str());
        print_usage(argv[0]);
        rc = 1;
    }

done:
    logosdb_close(db);
    return rc;
}
