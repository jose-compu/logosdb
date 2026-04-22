#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace logosdb {
namespace internal {

struct MetaRow {
    std::string text;
    std::string timestamp;
    bool        deleted = false;
};

// Append-only JSONL metadata store.
//
// Data rows:      {"text":"...","ts":"..."}
// Tombstone rows: {"op":"del","id":N}
//
// Tombstones mark an earlier data row as logically deleted. They do not
// occupy a row index themselves.
class MetadataStore {
public:
    MetadataStore() = default;
    ~MetadataStore();

    MetadataStore(const MetadataStore &) = delete;
    MetadataStore & operator=(const MetadataStore &) = delete;

    bool open(const std::string & path, std::string & err);
    void close();

    uint64_t append(const char * text, const char * timestamp, std::string & err);

    /* Append n metadata rows efficiently. Returns the starting id, or UINT64_MAX on error. */
    uint64_t append_batch(const char * const * texts, const char * const * timestamps,
                          int n, std::string & err);

    // Append a tombstone for `id`. Returns false if the id is out of range
    // or already deleted. `err` is set on failure.
    bool mark_deleted(uint64_t id, std::string & err);

    size_t count()         const { return rows_.size(); }
    size_t deleted_count() const { return num_deleted_; }

    bool is_deleted(uint64_t id) const {
        return id < rows_.size() && rows_[id].deleted;
    }

    const MetaRow * row(uint64_t idx) const {
        return idx < rows_.size() ? &rows_[idx] : nullptr;
    }

    // Iterate over all currently-tombstoned ids. Used by the DB on open to
    // re-apply deletion marks to a freshly-rebuilt HNSW index.
    std::vector<uint64_t> deleted_ids() const;

    bool sync(std::string & err);

private:
    std::string           path_;
    int                   fd_ = -1;
    std::vector<MetaRow>  rows_;
    size_t                num_deleted_ = 0;
};

} // namespace internal
} // namespace logosdb
