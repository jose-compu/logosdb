#include "wal.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

namespace logosdb {
namespace internal {

static constexpr uint32_t WAL_MAGIC = 0x57474F4C;  // "LOGW" in little-endian
static constexpr uint32_t WAL_VERSION = 1;

WriteAheadLog::~WriteAheadLog() { close(); }

bool WriteAheadLog::open(const std::string & path, std::string & err) {
    close();
    path_ = path;

    fd_ = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ < 0) {
        err = std::string("wal open: ") + strerror(errno);
        return false;
    }

    struct stat st;
    if (fstat(fd_, &st) != 0) {
        err = std::string("wal fstat: ") + strerror(errno);
        close();
        return false;
    }

    if (st.st_size == 0) {
        // New file: write header
        uint32_t header[2] = {WAL_MAGIC, WAL_VERSION};
        if (::write(fd_, header, sizeof(header)) != sizeof(header)) {
            err = std::string("wal write header: ") + strerror(errno);
            close();
            return false;
        }
        pending_count_ = 0;
    } else {
        // Existing file: validate header and count pending entries
        uint32_t header[2];
        if (::pread(fd_, header, sizeof(header), 0) != sizeof(header)) {
            err = std::string("wal read header: ") + strerror(errno);
            close();
            return false;
        }
        if (header[0] != WAL_MAGIC) {
            err = "wal: bad magic";
            close();
            return false;
        }
        if (header[1] != WAL_VERSION) {
            err = "wal: version mismatch";
            close();
            return false;
        }

        // Scan for pending entries
        int64_t offset = sizeof(header);
        WALEntry entry;
        while (true) {
            if (!read_entry_at(offset, entry, err)) {
                if (err.empty()) break;  // EOF
                close();
                return false;
            }
            if (entry.state == WALState::PENDING) {
                ++pending_count_;
            }
            // Calculate next entry offset
            offset += 1;  // state byte
            offset += 4;  // dim
            offset += 4 + entry.vector.size() * sizeof(float);  // vector len + data
            offset += 4 + entry.text.size();  // text len + data
            offset += 4 + entry.timestamp.size();  // ts len + data
            offset += 8;  // expected_id
        }
        err.clear();
    }

    return true;
}

void WriteAheadLog::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    path_.clear();
    pending_count_ = 0;
}

int64_t WriteAheadLog::append_pending(const float * vec, int dim,
                                      const char * text, const char * timestamp,
                                      uint64_t expected_id,
                                      std::string & err) {
    if (fd_ < 0) { err = "wal not open"; return -1; }

    // Get current file position (where we'll write this entry)
    off_t offset = ::lseek(fd_, 0, SEEK_END);
    if (offset < 0) {
        err = std::string("wal lseek: ") + strerror(errno);
        return -1;
    }

    // Write state (PENDING)
    uint8_t state = static_cast<uint8_t>(WALState::PENDING);
    if (::write(fd_, &state, 1) != 1) {
        err = std::string("wal write state: ") + strerror(errno);
        return -1;
    }

    // Write dim
    uint32_t dim_u32 = static_cast<uint32_t>(dim);
    if (::write(fd_, &dim_u32, 4) != 4) {
        err = std::string("wal write dim: ") + strerror(errno);
        return -1;
    }

    // Write vector length and data
    uint32_t vec_bytes = dim * sizeof(float);
    if (::write(fd_, &vec_bytes, 4) != 4) {
        err = std::string("wal write vec len: ") + strerror(errno);
        return -1;
    }
    if (vec_bytes > 0 && ::write(fd_, vec, vec_bytes) != vec_bytes) {
        err = std::string("wal write vec data: ") + strerror(errno);
        return -1;
    }

    // Write text
    std::string t = text ? text : "";
    uint32_t text_len = static_cast<uint32_t>(t.size());
    if (::write(fd_, &text_len, 4) != 4) {
        err = std::string("wal write text len: ") + strerror(errno);
        return -1;
    }
    if (text_len > 0 && ::write(fd_, t.data(), text_len) != text_len) {
        err = std::string("wal write text: ") + strerror(errno);
        return -1;
    }

    // Write timestamp
    std::string ts = timestamp ? timestamp : "";
    uint32_t ts_len = static_cast<uint32_t>(ts.size());
    if (::write(fd_, &ts_len, 4) != 4) {
        err = std::string("wal write ts len: ") + strerror(errno);
        return -1;
    }
    if (ts_len > 0 && ::write(fd_, ts.data(), ts_len) != ts_len) {
        err = std::string("wal write ts: ") + strerror(errno);
        return -1;
    }

    // Write expected_id
    if (::write(fd_, &expected_id, 8) != 8) {
        err = std::string("wal write expected_id: ") + strerror(errno);
        return -1;
    }

    // Sync to ensure WAL entry is durable before we modify stores
    if (!sync(err)) {
        return -1;
    }

    ++pending_count_;
    return offset;
}

bool WriteAheadLog::mark_committed(int64_t offset, std::string & err) {
    if (fd_ < 0) { err = "wal not open"; return false; }

    if (!write_state_at(offset, WALState::COMMITTED, err)) {
        return false;
    }

    if (pending_count_ > 0) --pending_count_;
    return sync(err);
}

bool WriteAheadLog::write_state_at(int64_t offset, WALState state, std::string & err) {
    uint8_t state_byte = static_cast<uint8_t>(state);
    if (::pwrite(fd_, &state_byte, 1, offset) != 1) {
        err = std::string("wal pwrite state: ") + strerror(errno);
        return false;
    }
    return true;
}

bool WriteAheadLog::read_entry_at(int64_t offset, WALEntry & entry, std::string & err) {
    err.clear();

    // Read state
    uint8_t state_byte;
    if (::pread(fd_, &state_byte, 1, offset) != 1) {
        return false;  // EOF or error
    }
    entry.state = static_cast<WALState>(state_byte);
    offset += 1;

    // Read dim
    uint32_t dim;
    if (::pread(fd_, &dim, 4, offset) != 4) {
        err = "wal: truncated entry (dim)";
        return false;
    }
    entry.dim = dim;
    offset += 4;

    // Read vector
    uint32_t vec_bytes;
    if (::pread(fd_, &vec_bytes, 4, offset) != 4) {
        err = "wal: truncated entry (vec len)";
        return false;
    }
    offset += 4;
    if (vec_bytes > 0) {
        entry.vector.resize(vec_bytes / sizeof(float));
        if (::pread(fd_, entry.vector.data(), vec_bytes, offset) != vec_bytes) {
            err = "wal: truncated entry (vec data)";
            return false;
        }
    } else {
        entry.vector.clear();
    }
    offset += vec_bytes;

    // Read text
    uint32_t text_len;
    if (::pread(fd_, &text_len, 4, offset) != 4) {
        err = "wal: truncated entry (text len)";
        return false;
    }
    offset += 4;
    entry.text.resize(text_len);
    if (text_len > 0) {
        if (::pread(fd_, &entry.text[0], text_len, offset) != text_len) {
            err = "wal: truncated entry (text data)";
            return false;
        }
    }
    offset += text_len;

    // Read timestamp
    uint32_t ts_len;
    if (::pread(fd_, &ts_len, 4, offset) != 4) {
        err = "wal: truncated entry (ts len)";
        return false;
    }
    offset += 4;
    entry.timestamp.resize(ts_len);
    if (ts_len > 0) {
        if (::pread(fd_, &entry.timestamp[0], ts_len, offset) != ts_len) {
            err = "wal: truncated entry (ts data)";
            return false;
        }
    }
    offset += ts_len;

    // Read expected_id
    if (::pread(fd_, &entry.expected_id, 8, offset) != 8) {
        err = "wal: truncated entry (expected_id)";
        return false;
    }

    return true;
}

int WriteAheadLog::replay_pending(
    std::function<bool(const WALEntry &, std::string &)> replay_fn,
    std::string & err) {
    if (fd_ < 0) { err = "wal not open"; return -1; }

    int64_t offset = 8;  // Skip header (magic + version)
    int replayed = 0;
    WALEntry entry;

    while (true) {
        // Peek at next entry state
        uint8_t state_byte;
        ssize_t r = ::pread(fd_, &state_byte, 1, offset);
        if (r == 0) break;  // EOF
        if (r < 0) {
            err = std::string("wal replay pread: ") + strerror(errno);
            return -1;
        }

        // Read full entry
        if (!read_entry_at(offset, entry, err)) {
            if (err.empty()) break;
            return -1;
        }

        // Only replay pending entries
        if (entry.state == WALState::PENDING) {
            if (!replay_fn(entry, err)) {
                return -1;
            }
            // Mark as committed after successful replay
            if (!write_state_at(offset, WALState::COMMITTED, err)) {
                return -1;
            }
            if (pending_count_ > 0) --pending_count_;
            ++replayed;
        }

        // Advance to next entry
        offset += 1;  // state
        offset += 4;  // dim
        offset += 4 + entry.vector.size() * sizeof(float);  // vector
        offset += 4 + entry.text.size();  // text
        offset += 4 + entry.timestamp.size();  // timestamp
        offset += 8;  // expected_id
    }

    return replayed;
}

bool WriteAheadLog::sync(std::string & err) {
    if (fd_ < 0) { err = "wal not open"; return false; }
    if (::fsync(fd_) != 0) {
        err = std::string("wal fsync: ") + strerror(errno);
        return false;
    }
    return true;
}

} // namespace internal
} // namespace logosdb
