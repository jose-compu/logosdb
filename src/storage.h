#pragma once

#include "platform.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace logosdb {
namespace internal {

// Fixed-stride binary vector storage with mmap support.
// File layout: [header (32 bytes)] [row_0 (dim*4 bytes)] [row_1] ...
//
// Header:
//   uint32_t magic   = 0x4C4F474F ("LOGO")
//   uint32_t version = 1
//   uint32_t dim
//   uint32_t reserved
//   uint64_t n_rows
//   uint64_t reserved2

struct StorageHeader {
    uint32_t magic    = 0x4C4F474F;
    uint32_t version  = 1;
    uint32_t dim      = 0;
    uint32_t reserved = 0;
    uint64_t n_rows   = 0;
    uint64_t reserved2 = 0;
};

static_assert(sizeof(StorageHeader) == 32, "header must be 32 bytes");

class VectorStorage {
public:
    VectorStorage() = default;
    ~VectorStorage();

    VectorStorage(const VectorStorage &) = delete;
    VectorStorage & operator=(const VectorStorage &) = delete;

    bool open(const std::string & path, int dim, std::string & err);
    void close();

    uint64_t append(const float * vec, int dim, std::string & err);

    /* Append n vectors efficiently. Returns the starting id, or UINT64_MAX on error.
     * 'data' must contain n * dim floats. */
    uint64_t append_batch(const float * data, int n, int dim, std::string & err);

    size_t      n_rows() const { return header_.n_rows; }
    int         dim()    const { return (int)header_.dim; }

    // Pointer to the i-th row (valid while file is open/mapped).
    const float * row(uint64_t idx) const;

    // Pointer to the start of all vector data (for bulk tensor load).
    const float * data() const;

    bool sync(std::string & err);

private:
    bool remap(std::string & err);
    void unmap();

    std::string    path_;
    int            fd_        = -1;
    StorageHeader  header_    = {};
    uint8_t *      map_base_  = nullptr;
    size_t         map_size_  = 0;
    size_t         file_size_ = 0;
    platform::MappedFile platform_map_{};  // For Windows memory mapping
};

} // namespace internal
} // namespace logosdb
