#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace logosdb
{
namespace internal
{

// Write-Ahead Log for atomic Put operations.
//
// The WAL records a Put intent before any store is modified. On crash,
// incomplete entries are replayed on the next open() to ensure consistency
// across vector storage, metadata, and HNSW index.
//
// File format (append-only binary):
//   [magic "LOGW" (4 bytes)]
//   [version uint32 (4 bytes)]
//   [sequence of WALEntry records]
//
// Each WALEntry:
//   [state uint8: 0=pending, 1=committed]
//   [dim uint32 (4 bytes)]
//   [vector_bytes uint32 (4 bytes)]
//   [vector data (dim*4 bytes)]
//   [text_len uint32 (4 bytes)]
//   [text (text_len bytes)]
//   [ts_len uint32 (4 bytes)]
//   [timestamp (ts_len bytes)]
//   [expected_id uint64 (8 bytes)]

enum class WALState : uint8_t
{
    PENDING = 0,
    COMMITTED = 1,
    ABORTED = 2
};

struct WALEntry
{
    WALState state = WALState::PENDING;
    uint32_t dim = 0;
    std::vector<float> vector;
    std::string text;
    std::string timestamp;
    uint64_t expected_id = 0;  // Expected row id (for validation)
};

class WriteAheadLog
{
  public:
    WriteAheadLog() = default;
    ~WriteAheadLog();

    WriteAheadLog(const WriteAheadLog&) = delete;
    WriteAheadLog& operator=(const WriteAheadLog&) = delete;

    // Open or create WAL file at the given path.
    bool open(const std::string& path, std::string& err);
    void close();

    // Append a new pending entry. Returns the entry offset in the file
    // (needed to mark committed later), or -1 on error.
    int64_t append_pending(const float* vec,
                           int dim,
                           const char* text,
                           const char* timestamp,
                           uint64_t expected_id,
                           std::string& err);

    // Append `n` pending entries with a SINGLE fsync at the end (chunked WAL write).
    // Returns 0 on success and fills `out_offsets` with one entry offset per row
    // (size: n). On any per-row write error returns -1 with `err` set; entries that
    // were partially written before the failure stay on disk and will be replayed
    // or discarded on the next open (same recovery contract as single-row put).
    //
    // `embeddings` must contain n*dim floats. `texts` / `timestamps` may be NULL or
    // arrays of size n with per-row NULLs allowed. `start_expected_id` is the row
    // id assigned to the first entry; subsequent entries use start_expected_id+i.
    int append_pending_batch(const float* embeddings,
                             int n,
                             int dim,
                             const char* const* texts,
                             const char* const* timestamps,
                             uint64_t start_expected_id,
                             std::vector<int64_t>& out_offsets,
                             std::string& err);

    // Mark an entry as committed by its offset.
    bool mark_committed(int64_t offset, std::string& err);

    // Mark `offsets.size()` entries as committed with a SINGLE fsync at the end.
    bool mark_committed_batch(const std::vector<int64_t>& offsets, std::string& err);

    // Replay all pending entries, calling the provided function for each.
    // Entries are marked committed after successful replay.
    // Returns number of entries replayed, or -1 on error.
    int replay_pending(std::function<bool(const WALEntry&, std::string&)> replay_fn,
                       std::string& err);

    // Sync WAL to disk.
    bool sync(std::string& err);

    // Get count of pending entries (for debugging/metrics).
    size_t pending_count() const { return pending_count_; }

  private:
    bool read_entry_at(int64_t offset, WALEntry& entry, std::string& err);
    bool write_state_at(int64_t offset, WALState state, std::string& err);

    std::string path_;
    int fd_ = -1;
    size_t pending_count_ = 0;
};

}  // namespace internal
}  // namespace logosdb
