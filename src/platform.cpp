#include "platform.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cerrno>
#include <cstring>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace logosdb
{
namespace internal
{
namespace platform
{

bool file_exists(const std::string& path)
{
#ifdef _WIN32
    struct _stat st;
    return _stat(path.c_str(), &st) == 0;
#else
    struct stat st;
    return stat(path.c_str(), &st) == 0;
#endif
}

#ifdef _WIN32
// Windows memory mapping implementation.
//
// Design notes:
//  - CreateFileMapping returns NULL on failure (not INVALID_HANDLE_VALUE).
//  - PAGE_READONLY mappings cannot exceed the current file size, so the Linux
//    MAP_NORESERVE over-reservation trick is not available.  Instead we simply
//    map the current file size and re-create the mapping object whenever the
//    file grows (mmap_commit).  The file_handle is kept open across commits so
//    that GetFileSizeEx always returns the live size written by the POSIX fd.

// Helper: create a mapping + view for the current size of an already-open file.
// Returns true on success; map_handle / data / size are filled in out_map.
// Caller must pre-set out_map.file_handle before calling.
static bool win_map_current(MappedFile& out_map, std::string& err)
{
    LARGE_INTEGER fs;
    if (!GetFileSizeEx(out_map.file_handle, &fs))
    {
        err = "GetFileSizeEx failed: " + std::to_string(GetLastError());
        return false;
    }

    if (fs.QuadPart == 0)
    {
        out_map.map_handle = INVALID_HANDLE_VALUE;
        out_map.data = nullptr;
        out_map.size = 0;
        return true;
    }

    HANDLE mh = CreateFileMapping(out_map.file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mh == NULL)
    {
        err = "CreateFileMapping failed: " + std::to_string(GetLastError());
        return false;
    }

    void* data = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (!data)
    {
        err = "MapViewOfFile failed: " + std::to_string(GetLastError());
        CloseHandle(mh);
        return false;
    }

    out_map.map_handle = mh;
    out_map.data = static_cast<uint8_t*>(data);
    out_map.size = static_cast<size_t>(fs.QuadPart);
    return true;
}

bool mmap_open(const std::string& path, size_t& out_size, MappedFile& out_map, std::string& err)
{
    out_map = {};

    HANDLE fh = CreateFileA(path.c_str(),
                            GENERIC_READ,
                            FILE_SHARE_READ,
                            nullptr,
                            OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL,
                            nullptr);
    if (fh == INVALID_HANDLE_VALUE)
    {
        err = "CreateFileA failed: " + std::to_string(GetLastError());
        return false;
    }

    out_map.file_handle = fh;
    if (!win_map_current(out_map, err))
    {
        CloseHandle(fh);
        out_map = {};
        return false;
    }

    out_size = out_map.size;
    return true;
}

void mmap_close(MappedFile& map)
{
    if (map.data)
    {
        UnmapViewOfFile(map.data);
        map.data = nullptr;
    }
    if (map.map_handle != INVALID_HANDLE_VALUE)
    {
        CloseHandle(map.map_handle);
        map.map_handle = INVALID_HANDLE_VALUE;
    }
    if (map.file_handle != INVALID_HANDLE_VALUE)
    {
        CloseHandle(map.file_handle);
        map.file_handle = INVALID_HANDLE_VALUE;
    }
    map.size = 0;
}

bool mmap_resize(MappedFile& map, size_t new_size, std::string& err)
{
    // On Windows, close and reopen the whole mapping.
    // (mmap_commit is the preferred path for growth; this is a fallback.)
    (void)new_size;
    mmap_close(map);
    err = "mmap_resize: use mmap_commit for in-place growth on Windows";
    return false;
}

// mmap_reserve: open the file keeping it share-writable so that the POSIX fd
// owned by VectorStorage can append data while the mapping stays open.
// True VA over-reservation is not supported for read-only mappings on Windows;
// we simply map the current file contents and re-map on each commit.
bool mmap_reserve(const std::string& path,
                  size_t reserve_size,
                  MappedFile& out_map,
                  std::string& err)
{
    out_map = {};
    (void)reserve_size;  // cannot over-reserve with PAGE_READONLY on Windows

    HANDLE fh = CreateFileA(path.c_str(),
                            GENERIC_READ,
                            FILE_SHARE_READ | FILE_SHARE_WRITE,
                            nullptr,
                            OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL,
                            nullptr);
    if (fh == INVALID_HANDLE_VALUE)
    {
        err = "CreateFileA failed: " + std::to_string(GetLastError());
        return false;
    }

    out_map.file_handle = fh;
    if (!win_map_current(out_map, err))
    {
        CloseHandle(fh);
        out_map = {};
        return false;
    }
    return true;
}

// mmap_commit: re-create the mapping object so it covers the new (larger) file
// size.  The file_handle stays open; only map_handle and the view are recycled.
size_t mmap_commit(MappedFile& map, size_t file_size)
{
    if (map.file_handle == INVALID_HANDLE_VALUE)
        return 0;

    // Release the old view and mapping object.
    if (map.data)
    {
        UnmapViewOfFile(map.data);
        map.data = nullptr;
    }
    if (map.map_handle != INVALID_HANDLE_VALUE)
    {
        CloseHandle(map.map_handle);
        map.map_handle = INVALID_HANDLE_VALUE;
    }

    if (file_size == 0)
    {
        map.size = 0;
        return 0;
    }

    // Create a fresh mapping covering the current (grown) file.
    HANDLE mh = CreateFileMapping(map.file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mh == NULL)
        return map.size;

    void* data = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, file_size);
    if (!data)
    {
        CloseHandle(mh);
        return map.size;
    }

    map.map_handle = mh;
    map.data = static_cast<uint8_t*>(data);
    map.size = file_size;
    return file_size;
}

#else
// POSIX memory mapping implementation (Linux/macOS)

bool mmap_open(const std::string& path, size_t& out_size, MappedFile& out_map, std::string& err)
{
    out_map = {};  // Clear

    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
    {
        err = std::string("open: ") + strerror(errno);
        return false;
    }

    struct stat st;
    if (fstat(fd, &st) != 0)
    {
        err = std::string("fstat: ") + strerror(errno);
        ::close(fd);
        return false;
    }

    size_t size = static_cast<size_t>(st.st_size);

    if (size == 0)
    {
        // Empty file - no mapping needed
        out_map.fd = fd;
        out_map.data = nullptr;
        out_map.size = 0;
        out_size = 0;
        return true;
    }

    void* data = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED)
    {
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

void mmap_close(MappedFile& map)
{
    if (map.data && map.size > 0)
    {
        munmap(map.data, map.size);
    }
    if (map.fd >= 0)
    {
        ::close(map.fd);
    }
    map.data = nullptr;
    map.fd = -1;
    map.size = 0;
}

bool mmap_resize(MappedFile& map, size_t new_size, std::string& err)
{
    // On POSIX, unmap and remap
    if (map.data && map.size > 0)
    {
        munmap(map.data, map.size);
    }

    void* data = mmap(nullptr, new_size, PROT_READ, MAP_SHARED, map.fd, 0);
    if (data == MAP_FAILED)
    {
        err = std::string("mmap resize: ") + strerror(errno);
        return false;
    }

    map.data = static_cast<uint8_t*>(data);
    map.size = new_size;
    return true;
}

// Stub implementations for POSIX - reservation is handled directly in storage.cpp
bool mmap_reserve(const std::string& path,
                  size_t reserve_size,
                  MappedFile& out_map,
                  std::string& err)
{
    (void)path;
    (void)reserve_size;
    (void)out_map;
    err = "mmap_reserve not implemented for POSIX - use storage.cpp implementation";
    return false;
}

size_t mmap_commit(MappedFile& map, size_t file_size)
{
    (void)map;
    return file_size;
}

#endif

}  // namespace platform
}  // namespace internal
}  // namespace logosdb
