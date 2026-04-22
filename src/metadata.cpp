#include "metadata.h"

#include <nlohmann/json.hpp>

#include <cerrno>
#include <cstring>
#include <fstream>
#include <fcntl.h>
#include <sys/types.h>

#ifdef _WIN32
    #include <io.h>
#else
    #include <unistd.h>
#endif

namespace logosdb {
namespace internal {

using json = nlohmann::json;

MetadataStore::~MetadataStore() { close(); }

bool MetadataStore::open(const std::string & path, std::string & err) {
    close();
    path_ = path;

#ifdef _WIN32
    int flags = O_RDWR | O_CREAT | O_APPEND | O_BINARY;
#else
    int flags = O_RDWR | O_CREAT | O_APPEND;
#endif
    fd_ = ::open(path.c_str(), flags, 0644);
    if (fd_ < 0) {
        err = std::string("open meta: ") + strerror(errno);
        return false;
    }

    std::ifstream in(path);
    if (in.good()) {
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;

            // Try to parse as JSON
            json j;
            try {
                j = json::parse(line);
            } catch (const json::exception & e) {
                // Invalid JSON line - skip but don't fail
                continue;
            }

            // Tombstone record: {"op":"del","id":N}
            if (j.contains("op") && j["op"] == "del" && j.contains("id")) {
                uint64_t id = j["id"].get<uint64_t>();
                if (id < rows_.size() && !rows_[id].deleted) {
                    rows_[id].deleted = true;
                    ++num_deleted_;
                }
                continue;
            }

            // Data row: {"text":"...","ts":"..."}
            MetaRow r;
            if (j.contains("text")) {
                r.text = j["text"].get<std::string>();
            }
            if (j.contains("ts")) {
                r.timestamp = j["ts"].get<std::string>();
            }
            rows_.push_back(std::move(r));
        }
    }
    return true;
}

void MetadataStore::close() {
    if (fd_ >= 0) {
#ifdef _WIN32
        _close(fd_);
#else
        ::close(fd_);
#endif
        fd_ = -1;
    }
    rows_.clear();
    num_deleted_ = 0;
    path_.clear();
}

uint64_t MetadataStore::append(const char * text, const char * timestamp,
                                std::string & err) {
    if (fd_ < 0) { err = "meta not open"; return UINT64_MAX; }

    json j;
    j["text"] = text ? text : "";
    j["ts"] = timestamp ? timestamp : "";
    std::string line = j.dump() + "\n";

#ifdef _WIN32
    int written = _write(fd_, line.data(), (int)line.size());
    if (written != (int)line.size()) {
#else
    ssize_t written = ::write(fd_, line.data(), line.size());
    if (written != (ssize_t)line.size()) {
#endif
        err = std::string("write meta: ") + strerror(errno);
        return UINT64_MAX;
    }

    uint64_t id = rows_.size();
    rows_.push_back({text ? text : "", timestamp ? timestamp : ""});
    return id;
}

uint64_t MetadataStore::append_batch(const char * const * texts, const char * const * timestamps,
                                      int n, std::string & err) {
    if (fd_ < 0) { err = "meta not open"; return UINT64_MAX; }
    if (n <= 0) { return rows_.size(); }

    // Build all JSON lines and write in a single batch
    std::string batch;
    batch.reserve(n * 64);  // rough estimate

    for (int i = 0; i < n; ++i) {
        json j;
        j["text"] = texts && texts[i] ? texts[i] : "";
        j["ts"] = timestamps && timestamps[i] ? timestamps[i] : "";
        batch += j.dump();
        batch += "\n";
    }

#ifdef _WIN32
    int written = _write(fd_, batch.data(), (int)batch.size());
    if (written != (int)batch.size()) {
#else
    ssize_t written = ::write(fd_, batch.data(), batch.size());
    if (written != (ssize_t)batch.size()) {
#endif
        err = std::string("write meta batch: ") + strerror(errno);
        return UINT64_MAX;
    }

    uint64_t start_id = rows_.size();
    for (int i = 0; i < n; ++i) {
        rows_.push_back({
            texts && texts[i] ? texts[i] : "",
            timestamps && timestamps[i] ? timestamps[i] : ""
        });
    }
    return start_id;
}

bool MetadataStore::mark_deleted(uint64_t id, std::string & err) {
    if (fd_ < 0) { err = "meta not open"; return false; }
    if (id >= rows_.size()) {
        err = "delete: id out of range";
        return false;
    }
    if (rows_[id].deleted) {
        err = "delete: id already deleted";
        return false;
    }

    json j;
    j["op"] = "del";
    j["id"] = id;
    std::string line = j.dump() + "\n";

#ifdef _WIN32
    int written = _write(fd_, line.data(), (int)line.size());
    if (written != (int)line.size()) {
#else
    ssize_t written = ::write(fd_, line.data(), line.size());
    if (written != (ssize_t)line.size()) {
#endif
        err = std::string("write tombstone: ") + strerror(errno);
        return false;
    }

    rows_[id].deleted = true;
    ++num_deleted_;
    return true;
}

std::vector<uint64_t> MetadataStore::deleted_ids() const {
    std::vector<uint64_t> out;
    out.reserve(num_deleted_);
    for (size_t i = 0; i < rows_.size(); ++i) {
        if (rows_[i].deleted) out.push_back((uint64_t)i);
    }
    return out;
}

bool MetadataStore::sync(std::string & err) {
    if (fd_ < 0) { err = "meta not open"; return false; }
#ifdef _WIN32
    if (_commit(fd_) != 0) {
#else
    if (::fsync(fd_) != 0) {
#endif
        err = std::string("fsync meta: ") + strerror(errno);
        return false;
    }
    return true;
}

} // namespace internal
} // namespace logosdb
