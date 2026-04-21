#pragma once

// Platform abstraction layer for cross-platform compatibility
// Handles differences between POSIX (Linux/macOS) and Windows

#ifdef _WIN32
    #define LOGOSDB_WINDOWS
    #define NOMINMAX
    #include <windows.h>
    #include <io.h>
    #include <direct.h>
#else
    #define LOGOSDB_POSIX
    #include <unistd.h>
    #include <sys/mman.h>
    #include <string.h>  // For strdup
#endif

#include <cstddef>
#include <cstdint>
#include <string>

namespace logosdb {
namespace internal {
namespace platform {

// Memory-mapped file abstraction
struct MappedFile {
#ifdef LOGOSDB_WINDOWS
    HANDLE file_handle = INVALID_HANDLE_VALUE;
    HANDLE map_handle = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif
    uint8_t* data = nullptr;
    size_t size = 0;
};

// File operations
bool file_exists(const std::string& path);
bool file_truncate(int fd, size_t size);
int file_sync(int fd);

// Memory mapping
bool mmap_open(const std::string& path, size_t& out_size, MappedFile& out_map, std::string& err);
bool mmap_resize(MappedFile& map, size_t new_size, std::string& err);
void mmap_close(MappedFile& map);

// String utilities
char* string_duplicate(const char* str);

// Platform-specific wrapper macros
#ifdef _WIN32
    inline bool file_truncate(int fd, size_t size) {
        return _chsize_s(fd, size) == 0;
    }
    inline int file_sync(int fd) {
        return _commit(fd);
    }
    inline char* string_duplicate(const char* str) {
        return _strdup(str);
    }
#else
    inline bool file_truncate(int fd, size_t size) {
        return ftruncate(fd, size) == 0;
    }
    inline int file_sync(int fd) {
        return fsync(fd);
    }
    inline char* string_duplicate(const char* str) {
        return strdup(str);
    }
#endif

} // namespace platform
} // namespace internal
} // namespace logosdb
