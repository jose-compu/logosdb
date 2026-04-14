#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace logosdb {
namespace internal {

struct MetaRow {
    std::string text;
    std::string timestamp;
};

// Append-only JSONL metadata store.
// Each line: {"text":"...","ts":"..."}
class MetadataStore {
public:
    MetadataStore() = default;
    ~MetadataStore();

    MetadataStore(const MetadataStore &) = delete;
    MetadataStore & operator=(const MetadataStore &) = delete;

    bool open(const std::string & path, std::string & err);
    void close();

    uint64_t append(const char * text, const char * timestamp, std::string & err);

    size_t count() const { return rows_.size(); }

    const MetaRow * row(uint64_t idx) const {
        return idx < rows_.size() ? &rows_[idx] : nullptr;
    }

    bool sync(std::string & err);

private:
    std::string           path_;
    int                   fd_ = -1;
    std::vector<MetaRow>  rows_;
};

} // namespace internal
} // namespace logosdb
