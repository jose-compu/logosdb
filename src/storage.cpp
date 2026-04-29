#include "storage.h"
#include "platform.h"

#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
    #include <io.h>
#else
    #include <unistd.h>
#endif

namespace logosdb {
namespace internal {

/* ── Dtype utilities ─────────────────────────────────────────────────── */

size_t dtype_size(StorageDtype dtype) {
    switch (dtype) {
        case DTYPE_FLOAT16: return 2;
        case DTYPE_INT8:    return 1;
        case DTYPE_FLOAT32: default: return 4;
    }
}

// IEEE 754 float32 to float16 conversion
uint16_t float32_to_float16(float f) {
    // Bit-level reinterpretation
    union { float f; uint32_t u; } v = { f };
    uint32_t u = v.u;

    uint32_t sign = (u >> 31) & 0x1;
    uint32_t exp = (u >> 23) & 0xFF;
    uint32_t mant = u & 0x7FFFFF;

    // Handle special cases
    if (exp == 0xFF) {  // Inf or NaN
        if (mant != 0) return (sign << 15) | 0x7C00 | (mant >> 13);  // NaN
        return (sign << 15) | 0x7C00;  // Inf
    }

    // Convert exponent
    int32_t new_exp = (int32_t)exp - 127 + 15;

    if (new_exp >= 31) {
        // Overflow to infinity
        return (sign << 15) | 0x7C00;
    } else if (new_exp <= 0) {
        // Underflow to zero or denormal
        if (new_exp < -10) {
            return sign << 15;  // Zero
        }
        // Denormal
        mant = (mant | 0x800000) >> (1 - new_exp);
        return (sign << 15) | (mant >> 13);
    }

    // Normal case
    return (sign << 15) | (new_exp << 10) | (mant >> 13);
}

float float16_to_float32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            // Zero
            return sign ? -0.0f : 0.0f;
        }
        // Denormal
        float f = mant / 1024.0f;
        return sign ? -f * 0.00006103515625f : f * 0.00006103515625f;
    } else if (exp == 31) {
        if (mant != 0) {
            // NaN
            union { uint32_t u; float f; } v = { (sign << 31) | 0x7FC00000 | (mant << 13) };
            return v.f;
        }
        // Inf
        union { uint32_t u; float f; } v = { (sign << 31) | 0x7F800000 };
        return v.f;
    }

    // Normal
    uint32_t u = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    union { uint32_t u; float f; } v = { u };
    return v.f;
}

void quantize_float32_to_int8(const float * src, int8_t * dst, int dim, float scale) {
    if (scale == 0.0f) {
        for (int i = 0; i < dim; ++i) dst[i] = 0;
        return;
    }
    float inv_scale = 127.0f / scale;
    for (int i = 0; i < dim; ++i) {
        int32_t val = (int32_t)std::round(src[i] * inv_scale);
        // Clamp to [-127, 127] (reserve -128 for future)
        if (val > 127) val = 127;
        if (val < -127) val = -127;
        dst[i] = (int8_t)val;
    }
}

void dequantize_int8_to_float32(const int8_t * src, float * dst, int dim, float scale) {
    float scale_127 = scale / 127.0f;
    for (int i = 0; i < dim; ++i) {
        dst[i] = (float)src[i] * scale_127;
    }
}

float compute_int8_scale(const float * vec, int dim) {
    float max_abs = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float abs_val = std::abs(vec[i]);
        if (abs_val > max_abs) max_abs = abs_val;
    }
    return max_abs;
}

/* ── VectorStorage implementation ───────────────────────────────────── */

static size_t row_stride(int dim, StorageDtype dtype) {
    return (size_t)dim * dtype_size(dtype);
}

static bool checked_file_size(uint64_t n_rows, int dim, StorageDtype dtype, size_t & out) {
    size_t stride = row_stride(dim, dtype);
    if (stride > 0 && n_rows > (SIZE_MAX - sizeof(StorageHeader)) / stride) {
        return false; // would overflow
    }
    out = sizeof(StorageHeader) + n_rows * stride;
    return true;
}

VectorStorage::~VectorStorage() { close(); }

bool VectorStorage::open(const std::string & path, int dim, StorageDtype dtype, std::string & err) {
    close();
    path_ = path;

#ifdef _WIN32
    int flags = O_RDWR | O_CREAT | O_BINARY;
#else
    int flags = O_RDWR | O_CREAT;
#endif
    fd_ = ::open(path.c_str(), flags, 0644);
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
        // Create new file with specified dtype
        header_ = {};
        header_.version = 2;
        header_.dim = (uint32_t)dim;
        header_.dtype = (uint32_t)dtype;
        header_.n_rows = 0;
        header_.scale = 1.0f;
        if (!platform::file_truncate(fd_, sizeof(StorageHeader))) {
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

        // Handle version compatibility
        if (header_.version == 0 || header_.version == 1) {
            // v1 files are float32 only - upgrade to v2 in memory
            header_.version = 2;
            header_.dtype = DTYPE_FLOAT32;
            header_.scale = 1.0f;
            header_.reserved = 0.0f;
        } else if (header_.version != 2) {
            err = "unsupported file version: " + std::to_string(header_.version);
            close();
            return false;
        }

        // Check dtype compatibility
        StorageDtype file_dtype = static_cast<StorageDtype>(header_.dtype);
        if (file_dtype != dtype && file_dtype != DTYPE_FLOAT32) {
            // Allow opening existing files even if requested dtype differs
            // We'll use the file's dtype
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
        if (!checked_file_size(header_.n_rows, (int)header_.dim, file_dtype, expected_size)
            || expected_size > file_size_) {
            size_t stride = row_stride((int)header_.dim, file_dtype);
            uint64_t max_rows = stride > 0
                ? (file_size_ - sizeof(StorageHeader)) / stride : 0;
            header_.n_rows = max_rows;
        }
    }

    if (!reserve_mapping(file_size_, err)) {
        close();
        return false;
    }
    return true;
}

void VectorStorage::close() {
    unmap();
    if (fd_ >= 0) {
#ifdef _WIN32
        _close(fd_);
#else
        ::close(fd_);
#endif
        fd_ = -1;
    }
    header_ = {};
    file_size_ = 0;
    reserved_size_ = 0;
}

uint64_t VectorStorage::append(const float * vec, int dim, std::string & err) {
    if (fd_ < 0) { err = "not open"; return UINT64_MAX; }
    if ((uint32_t)dim != header_.dim) {
        err = "dim mismatch on append";
        return UINT64_MAX;
    }

    StorageDtype dtype = static_cast<StorageDtype>(header_.dtype);
    size_t new_size = 0;
    if (!checked_file_size(header_.n_rows + 1, dim, dtype, new_size)) {
        err = "storage size overflow";
        return UINT64_MAX;
    }
    size_t stride = row_stride(dim, dtype);
    size_t offset = new_size - stride;

    if (!platform::file_truncate(fd_, new_size)) {
        err = std::string("ftruncate: ") + strerror(errno);
        return UINT64_MAX;
    }

    // Convert and write based on dtype
    if (dtype == DTYPE_FLOAT32) {
        if (::pwrite(fd_, vec, stride, (off_t)offset) != (ssize_t)stride) {
            err = "pwrite vec failed";
            return UINT64_MAX;
        }
    } else if (dtype == DTYPE_FLOAT16) {
        std::vector<uint16_t> half(dim);
        for (int i = 0; i < dim; ++i) {
            half[i] = float32_to_float16(vec[i]);
        }
        if (::pwrite(fd_, half.data(), stride, (off_t)offset) != (ssize_t)stride) {
            err = "pwrite float16 vec failed";
            return UINT64_MAX;
        }
    } else if (dtype == DTYPE_INT8) {
        // Update global scale if needed
        float vec_scale = compute_int8_scale(vec, dim);
        if (vec_scale > header_.scale) {
            header_.scale = vec_scale;
        }
        std::vector<int8_t> quantized(dim);
        quantize_float32_to_int8(vec, quantized.data(), dim, header_.scale);
        if (::pwrite(fd_, quantized.data(), stride, (off_t)offset) != (ssize_t)stride) {
            err = "pwrite int8 vec failed";
            return UINT64_MAX;
        }
    }

    uint64_t id = header_.n_rows;
    header_.n_rows++;
    file_size_ = new_size;

    if (::pwrite(fd_, &header_, sizeof(header_), 0) != sizeof(header_)) {
        err = "pwrite header failed";
        return UINT64_MAX;
    }

    // Extend mapping if file grew beyond current mapping
    if (!extend_mapping_if_needed(err)) return UINT64_MAX;
    return id;
}

uint64_t VectorStorage::append_batch(const float * data, int n, int dim, std::string & err) {
    if (fd_ < 0) { err = "not open"; return UINT64_MAX; }
    if (n <= 0) { return header_.n_rows; }
    if ((uint32_t)dim != header_.dim) {
        err = "dim mismatch on batch append";
        return UINT64_MAX;
    }

    StorageDtype dtype = static_cast<StorageDtype>(header_.dtype);
    size_t new_size = 0;
    if (!checked_file_size(header_.n_rows + n, dim, dtype, new_size)) {
        err = "storage size overflow";
        return UINT64_MAX;
    }

    // Single ftruncate to final size
    if (!platform::file_truncate(fd_, new_size)) {
        err = std::string("ftruncate: ") + strerror(errno);
        return UINT64_MAX;
    }

    size_t stride = row_stride(dim, dtype);
    size_t total_bytes = (size_t)n * stride;
    size_t offset = sizeof(StorageHeader) + header_.n_rows * stride;

    // Convert and write based on dtype
    if (dtype == DTYPE_FLOAT32) {
        if (::pwrite(fd_, data, total_bytes, (off_t)offset) != (ssize_t)total_bytes) {
            err = "pwrite batch vec failed";
            return UINT64_MAX;
        }
    } else if (dtype == DTYPE_FLOAT16) {
        std::vector<uint16_t> half((size_t)n * dim);
        for (int i = 0; i < n * dim; ++i) {
            half[i] = float32_to_float16(data[i]);
        }
        if (::pwrite(fd_, half.data(), total_bytes, (off_t)offset) != (ssize_t)total_bytes) {
            err = "pwrite batch float16 vec failed";
            return UINT64_MAX;
        }
    } else if (dtype == DTYPE_INT8) {
        // Compute max scale across batch
        float max_scale = header_.scale;
        for (int i = 0; i < n; ++i) {
            float vec_scale = compute_int8_scale(data + i * dim, dim);
            if (vec_scale > max_scale) max_scale = vec_scale;
        }
        if (max_scale > header_.scale) {
            header_.scale = max_scale;
        }

        std::vector<int8_t> quantized((size_t)n * dim);
        for (int i = 0; i < n; ++i) {
            quantize_float32_to_int8(data + i * dim, quantized.data() + i * dim, dim, header_.scale);
        }
        if (::pwrite(fd_, quantized.data(), total_bytes, (off_t)offset) != (ssize_t)total_bytes) {
            err = "pwrite batch int8 vec failed";
            return UINT64_MAX;
        }
    }

    uint64_t start_id = header_.n_rows;
    header_.n_rows += n;
    file_size_ = new_size;

    // Update header once
    if (::pwrite(fd_, &header_, sizeof(header_), 0) != sizeof(header_)) {
        err = "pwrite header failed";
        return UINT64_MAX;
    }

    // Extend mapping if needed (single remap at end, or just extend if reserved)
    if (!extend_mapping_if_needed(err)) return UINT64_MAX;
    return start_id;
}

size_t VectorStorage::row_stride_bytes() const {
    return row_stride((int)header_.dim, static_cast<StorageDtype>(header_.dtype));
}

const void * VectorStorage::row_raw(uint64_t idx) const {
    if (!map_base_ || idx >= header_.n_rows) return nullptr;
    size_t stride = row_stride_bytes();
    if (stride > 0 && idx > (SIZE_MAX - sizeof(StorageHeader)) / stride)
        return nullptr;
    size_t offset = sizeof(StorageHeader) + idx * stride;
    if (offset + stride > map_size_) return nullptr;
    return map_base_ + offset;
}

void VectorStorage::row_to_float32(uint64_t idx, float * out) const {
    const void * raw = row_raw(idx);
    if (!raw) return;

    int dim = (int)header_.dim;
    StorageDtype dtype = static_cast<StorageDtype>(header_.dtype);

    if (dtype == DTYPE_FLOAT32) {
        std::memcpy(out, raw, dim * sizeof(float));
    } else if (dtype == DTYPE_FLOAT16) {
        const uint16_t * half = static_cast<const uint16_t *>(raw);
        for (int i = 0; i < dim; ++i) {
            out[i] = float16_to_float32(half[i]);
        }
    } else if (dtype == DTYPE_INT8) {
        const int8_t * q = static_cast<const int8_t *>(raw);
        dequantize_int8_to_float32(q, out, dim, header_.scale);
    }
}

const void * VectorStorage::data_raw() const {
    if (!map_base_ || header_.n_rows == 0) return nullptr;
    return map_base_ + sizeof(StorageHeader);
}

void VectorStorage::data_to_float32(float * out) const {
    if (!map_base_ || header_.n_rows == 0) return;

    int dim = (int)header_.dim;
    StorageDtype dtype = static_cast<StorageDtype>(header_.dtype);
    size_t n = header_.n_rows;

    if (dtype == DTYPE_FLOAT32) {
        std::memcpy(out, data_raw(), n * dim * sizeof(float));
    } else if (dtype == DTYPE_FLOAT16) {
        const uint16_t * half = static_cast<const uint16_t *>(data_raw());
        for (size_t i = 0; i < n * dim; ++i) {
            out[i] = float16_to_float32(half[i]);
        }
    } else if (dtype == DTYPE_INT8) {
        const int8_t * q = static_cast<const int8_t *>(data_raw());
        for (size_t i = 0; i < n; ++i) {
            dequantize_int8_to_float32(q + i * dim, out + i * dim, dim, header_.scale);
        }
    }
}

bool VectorStorage::sync(std::string & err) {
    if (fd_ < 0) { err = "not open"; return false; }
    if (platform::file_sync(fd_) != 0) {
        err = std::string("fsync: ") + strerror(errno);
        return false;
    }
    return true;
}

bool VectorStorage::reserve_mapping(size_t min_size, std::string & err) {
    unmap();

    // Reserve at least DEFAULT_RESERVE_SIZE or min_size, rounded up
    reserved_size_ = std::max(DEFAULT_RESERVE_SIZE, min_size);
    reserved_size_ = (reserved_size_ + 4095) & ~4095ULL;  // Round up to page size

#ifdef _WIN32
    // Windows: use platform mapping
    platform::mmap_close(platform_map_);
    if (!platform::mmap_reserve(path_, reserved_size_, platform_map_, err)) {
        return false;
    }
    map_base_ = platform_map_.data;
    map_size_ = platform::mmap_commit(platform_map_, file_size_);
#elif defined(__linux__)
    // Linux: use MAP_NORESERVE to reserve address space without committing memory
    map_base_ = (uint8_t *)mmap(nullptr, reserved_size_, PROT_READ,
                                  MAP_SHARED | MAP_NORESERVE, fd_, 0);
    if (map_base_ == MAP_FAILED) {
        map_base_ = nullptr;
        err = std::string("mmap reserve: ") + strerror(errno);
        return false;
    }
    // Advise that we don't need the pages beyond current file size yet
    if (file_size_ < reserved_size_) {
        madvise(map_base_ + file_size_, reserved_size_ - file_size_, MADV_DONTNEED);
    }
    map_size_ = file_size_;
#else
    // macOS and others: fall back to regular mmap of current file size
    // We don't reserve extra space on macOS due to lack of MAP_NORESERVE
    map_base_ = (uint8_t *)mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (map_base_ == MAP_FAILED) {
        map_base_ = nullptr;
        err = std::string("mmap: ") + strerror(errno);
        return false;
    }
    reserved_size_ = file_size_;
    map_size_ = file_size_;
#endif
    return true;
}

bool VectorStorage::extend_mapping_if_needed(std::string & err) {
    // If file hasn't grown beyond mapped size, nothing to do
    if (file_size_ <= map_size_) return true;

    // If we have enough reserved space, just extend the active mapping
    if (file_size_ <= reserved_size_) {
#ifdef _WIN32
        // Windows: commit more pages
        map_size_ = platform::mmap_commit(platform_map_, file_size_);
#elif defined(__linux__)
        // Linux: with MAP_SHARED, the mapping automatically covers new file data
        // Just update our tracked size
        map_size_ = file_size_;
#else
        // macOS: need to remap since we didn't reserve extra space
        if (!remap(err)) return false;
#endif
        return true;
    }

    // Need to grow reservation - do a full remap
    return reserve_mapping(file_size_, err);
}

bool VectorStorage::remap(std::string & err) {
    // Try to use reserve_mapping for better performance
    return reserve_mapping(file_size_, err);
}

void VectorStorage::unmap() {
#ifdef _WIN32
    platform::mmap_close(platform_map_);
    map_base_ = nullptr;
    map_size_ = 0;
    reserved_size_ = 0;
#else
    if (map_base_ && reserved_size_ > 0) {
        munmap(map_base_, reserved_size_);
    }
    map_base_ = nullptr;
    map_size_ = 0;
    reserved_size_ = 0;
#endif
}

} // namespace internal
} // namespace logosdb
