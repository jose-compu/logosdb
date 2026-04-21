#include "storage.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace logosdb {
namespace internal {

static size_t row_stride(int dim) { return (size_t)dim * sizeof(float); }

static bool checked_file_size(uint64_t n_rows, int dim, size_t & out) {
    size_t stride = row_stride(dim);
    if (stride > 0 && n_rows > (SIZE_MAX - sizeof(StorageHeader)) / stride) {
        return false; // would overflow
    }
    out = sizeof(StorageHeader) + n_rows * stride;
    return true;
}

VectorStorage::~VectorStorage() { close(); }

bool VectorStorage::open(const std::string & path, int dim, std::string & err) {
    close();
    path_ = path;

    fd_ = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ < 0) {
        err = std::string("open: ") + strerror(errno);
        return false;
    }

    struct stat st;
    if (fstat(fd_, &st) != 0) {
        err = std::string("fstat: ") + strerror(errno);
        close();
        return false;
    }
    file_size_ = (size_t)st.st_size;

    if (file_size_ == 0) {
        header_ = {};
        header_.dim = (uint32_t)dim;
        header_.n_rows = 0;
        if (::ftruncate(fd_, sizeof(StorageHeader)) != 0) {
            err = std::string("ftruncate: ") + strerror(errno);
            close();
            return false;
        }
        if (::pwrite(fd_, &header_, sizeof(header_), 0) != sizeof(header_)) {
            err = "failed to write header";
            close();
            return false;
        }
        file_size_ = sizeof(StorageHeader);
    } else {
        if (file_size_ < sizeof(StorageHeader)) {
            err = "file too small for header";
            close();
            return false;
        }
        if (::pread(fd_, &header_, sizeof(header_), 0) != sizeof(header_)) {
            err = "failed to read header";
            close();
            return false;
        }
        if (header_.magic != 0x4C4F474F) {
            err = "bad magic";
            close();
            return false;
        }
        if (header_.dim != (uint32_t)dim && dim > 0 && header_.dim > 0) {
            err = "dimension mismatch (file=" + std::to_string(header_.dim) +
                  " requested=" + std::to_string(dim) + ")";
            close();
            return false;
        }
        if (dim > 0 && header_.dim == 0) {
            header_.dim = (uint32_t)dim;
        }
        // Validate n_rows against actual file size to detect corruption.
        size_t expected_size = 0;
        if (!checked_file_size(header_.n_rows, (int)header_.dim, expected_size)
            || expected_size > file_size_) {
            size_t stride = row_stride((int)header_.dim);
            uint64_t max_rows = stride > 0
                ? (file_size_ - sizeof(StorageHeader)) / stride : 0;
            header_.n_rows = max_rows;
        }
    }

    if (!remap(err)) {
        close();
        return false;
    }
    return true;
}

void VectorStorage::close() {
    unmap();
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    header_ = {};
    file_size_ = 0;
}

uint64_t VectorStorage::append(const float * vec, int dim, std::string & err) {
    if (fd_ < 0) { err = "not open"; return UINT64_MAX; }
    if ((uint32_t)dim != header_.dim) {
        err = "dim mismatch on append";
        return UINT64_MAX;
    }

    size_t new_size = 0;
    if (!checked_file_size(header_.n_rows + 1, dim, new_size)) {
        err = "storage size overflow";
        return UINT64_MAX;
    }
    size_t stride = row_stride(dim);
    size_t offset = new_size - stride;

    if (::ftruncate(fd_, (off_t)new_size) != 0) {
        err = std::string("ftruncate: ") + strerror(errno);
        return UINT64_MAX;
    }
    if (::pwrite(fd_, vec, stride, (off_t)offset) != (ssize_t)stride) {
        err = "pwrite vec failed";
        return UINT64_MAX;
    }

    uint64_t id = header_.n_rows;
    header_.n_rows++;
    file_size_ = new_size;

    if (::pwrite(fd_, &header_, sizeof(header_), 0) != sizeof(header_)) {
        err = "pwrite header failed";
        return UINT64_MAX;
    }

    if (!remap(err)) return UINT64_MAX;
    return id;
}

uint64_t VectorStorage::append_batch(const float * data, int n, int dim, std::string & err) {
    if (fd_ < 0) { err = "not open"; return UINT64_MAX; }
    if (n <= 0) { return header_.n_rows; }
    if ((uint32_t)dim != header_.dim) {
        err = "dim mismatch on batch append";
        return UINT64_MAX;
    }

    size_t new_size = 0;
    if (!checked_file_size(header_.n_rows + n, dim, new_size)) {
        err = "storage size overflow";
        return UINT64_MAX;
    }

    // Single ftruncate to final size
    if (::ftruncate(fd_, (off_t)new_size) != 0) {
        err = std::string("ftruncate: ") + strerror(errno);
        return UINT64_MAX;
    }

    // Single pwrite for all vectors
    size_t stride = row_stride(dim);
    size_t total_bytes = (size_t)n * stride;
    size_t offset = sizeof(StorageHeader) + header_.n_rows * stride;

    if (::pwrite(fd_, data, total_bytes, (off_t)offset) != (ssize_t)total_bytes) {
        err = "pwrite batch vec failed";
        return UINT64_MAX;
    }

    uint64_t start_id = header_.n_rows;
    header_.n_rows += n;
    file_size_ = new_size;

    // Update header once
    if (::pwrite(fd_, &header_, sizeof(header_), 0) != sizeof(header_)) {
        err = "pwrite header failed";
        return UINT64_MAX;
    }

    // Single remap at the end
    if (!remap(err)) return UINT64_MAX;
    return start_id;
}

const float * VectorStorage::row(uint64_t idx) const {
    if (!map_base_ || idx >= header_.n_rows) return nullptr;
    size_t stride = row_stride((int)header_.dim);
    if (stride > 0 && idx > (SIZE_MAX - sizeof(StorageHeader)) / stride)
        return nullptr;
    size_t offset = sizeof(StorageHeader) + idx * stride;
    if (offset + stride > map_size_) return nullptr;
    return reinterpret_cast<const float *>(map_base_ + offset);
}

const float * VectorStorage::data() const {
    if (!map_base_ || header_.n_rows == 0) return nullptr;
    return reinterpret_cast<const float *>(map_base_ + sizeof(StorageHeader));
}

bool VectorStorage::sync(std::string & err) {
    if (fd_ < 0) { err = "not open"; return false; }
    if (::fsync(fd_) != 0) {
        err = std::string("fsync: ") + strerror(errno);
        return false;
    }
    return true;
}

bool VectorStorage::remap(std::string & err) {
    unmap();
    struct stat st;
    if (fstat(fd_, &st) != 0) {
        err = std::string("fstat: ") + strerror(errno);
        return false;
    }
    map_size_ = (size_t)st.st_size;
    if (map_size_ == 0) return true;
    map_base_ = (uint8_t *)mmap(nullptr, map_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (map_base_ == MAP_FAILED) {
        map_base_ = nullptr;
        err = std::string("mmap: ") + strerror(errno);
        return false;
    }
    return true;
}

void VectorStorage::unmap() {
    if (map_base_ && map_size_ > 0) {
        munmap(map_base_, map_size_);
    }
    map_base_ = nullptr;
    map_size_ = 0;
}

} // namespace internal
} // namespace logosdb
