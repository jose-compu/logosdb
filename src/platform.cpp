#include "platform.h"

#include <cerrno>
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
namespace platform {

bool file_exists(const std::string& path) {
#ifdef _WIN32
    struct _stat st;
    return _stat(path.c_str(), &st) == 0;
#else
    struct stat st;
    return stat(path.c_str(), &st) == 0;
#endif
}

#ifdef _WIN32
// Windows memory mapping implementation

bool mmap_open(const std::string& path, size_t& out_size, MappedFile& out_map, std::string& err) {
    out_map = {}; // Clear

    HANDLE file_handle = CreateFileA(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ,
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );

    if (file_handle == INVALID_HANDLE_VALUE) {
        err = "CreateFileA failed: " + std::to_string(GetLastError());
        return false;
    }

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file_handle, &file_size)) {
        err = "GetFileSizeEx failed: " + std::to_string(GetLastError());
        CloseHandle(file_handle);
        return false;
    }

    if (file_size.QuadPart == 0) {
        // Empty file - no mapping needed
        out_map.file_handle = file_handle;
        out_map.data = nullptr;
        out_map.size = 0;
        out_size = 0;
        return true;
    }

    HANDLE map_handle = CreateFileMapping(
        file_handle,
        nullptr,
        PAGE_READONLY,
        0,
        0,
        nullptr
    );

    if (map_handle == INVALID_HANDLE_VALUE) {
        err = "CreateFileMapping failed: " + std::to_string(GetLastError());
        CloseHandle(file_handle);
        return false;
    }

    void* data = MapViewOfFile(
        map_handle,
        FILE_MAP_READ,
        0,
        0,
        0
    );

    if (!data) {
        err = "MapViewOfFile failed: " + std::to_string(GetLastError());
        CloseHandle(map_handle);
        CloseHandle(file_handle);
        return false;
    }

    out_map.file_handle = file_handle;
    out_map.map_handle = map_handle;
    out_map.data = static_cast<uint8_t*>(data);
    out_map.size = static_cast<size_t>(file_size.QuadPart);
    out_size = out_map.size;
    return true;
}

void mmap_close(MappedFile& map) {
    if (map.data) {
        UnmapViewOfFile(map.data);
        map.data = nullptr;
    }
    if (map.map_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(map.map_handle);
        map.map_handle = INVALID_HANDLE_VALUE;
    }
    if (map.file_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(map.file_handle);
        map.file_handle = INVALID_HANDLE_VALUE;
    }
    map.size = 0;
}

bool mmap_resize(MappedFile& map, size_t new_size, std::string& err) {
    // On Windows, we need to close and reopen the mapping
    // This is used when the file grows
    mmap_close(map);

    // For simplicity on Windows, we don't remap here - the caller
    // should use mmap_open again after closing the file
    err = "mmap_resize not implemented for Windows - use close/reopen pattern";
    return false;
}

bool mmap_reserve(const std::string& path, size_t reserve_size, MappedFile& out_map, std::string& err) {
    out_map = {}; // Clear

    // First, get the current file size
    HANDLE file_handle = CreateFileA(
        path.c_str(),
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE,  // Allow others to write (file growth)
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        nullptr
    );

    if (file_handle == INVALID_HANDLE_VALUE) {
        err = "CreateFileA failed: " + std::to_string(GetLastError());
        return false;
    }

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(file_handle, &file_size)) {
        err = "GetFileSizeEx failed: " + std::to_string(GetLastError());
        CloseHandle(file_handle);
        return false;
    }

    // Create a file mapping that reserves virtual address space
    // We use a large maximum size for reservation, but only commit what's needed
    HANDLE map_handle = CreateFileMapping(
        file_handle,
        nullptr,
        PAGE_READONLY,
        0,
        static_cast<DWORD>(reserve_size),  // Maximum size for reservation
        nullptr
    );

    if (map_handle == INVALID_HANDLE_VALUE) {
        err = "CreateFileMapping failed: " + std::to_string(GetLastError());
        CloseHandle(file_handle);
        return false;
    }

    // Map only the current file size initially
    void* data = MapViewOfFile(
        map_handle,
        FILE_MAP_READ,
        0,
        0,
        static_cast<size_t>(file_size.QuadPart)  // Initial view size
    );

    if (!data) {
        err = "MapViewOfFile failed: " + std::to_string(GetLastError());
        CloseHandle(map_handle);
        CloseHandle(file_handle);
        return false;
    }

    out_map.file_handle = file_handle;
    out_map.map_handle = map_handle;
    out_map.data = static_cast<uint8_t*>(data);
    out_map.size = static_cast<size_t>(file_size.QuadPart);
    return true;
}

size_t mmap_commit(MappedFile& map, size_t file_size) {
#ifdef _WIN32
    // On Windows with file mapping, the view automatically extends
    // when the file grows (as long as we're within the mapping's max size)
    // We just need to unmap and remap to the new size
    if (map.data) {
        UnmapViewOfFile(map.data);
    }

    void* data = MapViewOfFile(
        map.map_handle,
        FILE_MAP_READ,
        0,
        0,
        file_size  // New view size
    );

    if (!data) {
        return map.size;  // Failed, keep old size
    }

    map.data = static_cast<uint8_t*>(data);
    map.size = file_size;
    return map.size;
#else
    (void)map;
    return file_size;
#endif
}

#else
// POSIX memory mapping implementation (Linux/macOS)

bool mmap_open(const std::string& path, size_t& out_size, MappedFile& out_map, std::string& err) {
    out_map = {}; // Clear

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        err = std::string("open: ") + strerror(errno);
        return false;
    }

    struct stat st;
    if (fstat(fd, &st) != 0) {
        err = std::string("fstat: ") + strerror(errno);
        ::close(fd);
        return false;
    }

    size_t size = static_cast<size_t>(st.st_size);

    if (size == 0) {
        // Empty file - no mapping needed
        out_map.fd = fd;
        out_map.data = nullptr;
        out_map.size = 0;
        out_size = 0;
        return true;
    }

    void* data = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) {
        err = std::string("mmap: ") + strerror(errno);
        ::close(fd);
        return false;
    }

    out_map.fd = fd;
    out_map.data = static_cast<uint8_t*>(data);
    out_map.size = size;
    out_size = size;
    return true;
}

void mmap_close(MappedFile& map) {
    if (map.data && map.size > 0) {
        munmap(map.data, map.size);
    }
    if (map.fd >= 0) {
        ::close(map.fd);
    }
    map.data = nullptr;
    map.fd = -1;
    map.size = 0;
}

bool mmap_resize(MappedFile& map, size_t new_size, std::string& err) {
    // On POSIX, unmap and remap
    if (map.data && map.size > 0) {
        munmap(map.data, map.size);
    }

    void* data = mmap(nullptr, new_size, PROT_READ, MAP_SHARED, map.fd, 0);
    if (data == MAP_FAILED) {
        err = std::string("mmap resize: ") + strerror(errno);
        return false;
    }

    map.data = static_cast<uint8_t*>(data);
    map.size = new_size;
    return true;
}

// Stub implementations for POSIX - reservation is handled directly in storage.cpp
bool mmap_reserve(const std::string& path, size_t reserve_size, MappedFile& out_map, std::string& err) {
    (void)path;
    (void)reserve_size;
    (void)out_map;
    err = "mmap_reserve not implemented for POSIX - use storage.cpp implementation";
    return false;
}

size_t mmap_commit(MappedFile& map, size_t file_size) {
    (void)map;
    return file_size;
}

#endif

} // namespace platform
} // namespace internal
} // namespace logosdb
