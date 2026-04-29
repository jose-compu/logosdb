#pragma once

#include "platform.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace logosdb {
namespace internal {

// Data type for vector storage precision
enum StorageDtype {
    DTYPE_FLOAT32 = 0,  // 4 bytes per dimension (default)
    DTYPE_FLOAT16 = 1,  // 2 bytes per dimension
    DTYPE_INT8    = 2,  // 1 byte per dimension
};

// Fixed-stride binary vector storage with mmap support.
// File layout: [header (32 bytes)] [row_0] [row_1] ...
//
// Header (v2):
//   uint32_t magic   = 0x4C4F474F ("LOGO")
//   uint32_t version = 2
//   uint32_t dim
//   uint32_t dtype   = 0=float32, 1=float16, 2=int8
//   uint64_t n_rows
//   float    scale   (for int8 quantization, 1.0 for others)

struct StorageHeader {
    uint32_t magic    = 0x4C4F474F;
    uint32_t version  = 2;
    uint32_t dim      = 0;
    uint32_t dtype    = 0;        // StorageDtype value
    uint64_t n_rows   = 0;
    float    scale    = 1.0f;     // For int8 dequantization
    float    reserved = 0.0f;     // Padding to 32 bytes
};

static_assert(sizeof(StorageHeader) == 32, "header must be 32 bytes");

// Get bytes per element for a given dtype
size_t dtype_size(StorageDtype dtype);

// Convert float32 to float16 (IEEE 754 half-precision)
uint16_t float32_to_float16(float f);

// Convert float16 to float32
float float16_to_float32(uint16_t h);

// Quantize float32 array to int8 with given scale
void quantize_float32_to_int8(const float * src, int8_t * dst, int dim, float scale);

// Dequantize int8 array to float32 with given scale
void dequantize_int8_to_float32(const int8_t * src, float * dst, int dim, float scale);

// Compute optimal int8 scale for a vector (max absolute value / 127.0f)
float compute_int8_scale(const float * vec, int dim);

class VectorStorage {
public:
    VectorStorage() = default;
    ~VectorStorage();

    VectorStorage(const VectorStorage &) = delete;
    VectorStorage & operator=(const VectorStorage &) = delete;

    // Open with explicit dtype (for creating new databases)
    bool open(const std::string & path, int dim, StorageDtype dtype, std::string & err);

    // Open with default float32 dtype (backward compatibility)
    bool open(const std::string & path, int dim, std::string & err) {
        return open(path, dim, DTYPE_FLOAT32, err);
    }

    void close();

    uint64_t append(const float * vec, int dim, std::string & err);

    /* Append n vectors efficiently. Returns the starting id, or UINT64_MAX on error.
     * 'data' must contain n * dim floats. */
    uint64_t append_batch(const float * data, int n, int dim, std::string & err);

    size_t      n_rows() const { return header_.n_rows; }
    int         dim()    const { return (int)header_.dim; }
    StorageDtype dtype() const { return static_cast<StorageDtype>(header_.dtype); }

    // Pointer to the i-th row (valid while file is open/mapped).
    // Note: For reduced precision, this returns a pointer to the raw storage.
    // Use row_to_float32() to get dequantized data.
    const void * row_raw(uint64_t idx) const;

    // Get a row and dequantize to float32 (caller must provide buffer of size dim)
    void row_to_float32(uint64_t idx, float * out) const;

    // Pointer to the start of all vector data (for bulk tensor load).
    // Note: This is raw storage, not dequantized.
    const void * data_raw() const;

    // Dequantize all vectors to float32 buffer
    void data_to_float32(float * out) const;

    bool sync(std::string & err);

private:
    bool remap(std::string & err);
    void unmap();
    bool reserve_mapping(size_t min_size, std::string & err);
    bool extend_mapping_if_needed(std::string & err);
    size_t row_stride_bytes() const;

    std::string    path_;
    int            fd_        = -1;
    StorageHeader  header_    = {};
    uint8_t *      map_base_  = nullptr;
    size_t         map_size_  = 0;        // Currently mapped size
    size_t         file_size_ = 0;        // Current file size
    size_t         reserved_size_ = 0;    // Reserved address space size
    static constexpr size_t DEFAULT_RESERVE_SIZE = 1ULL << 30;  // 1 GB reservation
    platform::MappedFile platform_map_{};  // For Windows memory mapping
};

} // namespace internal
} // namespace logosdb
