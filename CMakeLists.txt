cmake_minimum_required(VERSION 3.15)
project(logosdb VERSION 0.9.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MSVC-specific compile flags
if(MSVC)
    add_compile_options(
        /utf-8        # Source and execution charset: UTF-8
        /EHsc         # C++ exception handling
        /W3           # Warning level 3
        /wd4068       # Suppress unknown pragma warnings (for clang pragmas in vendored headers)
    )
    # Suppress deprecation warnings for POSIX-named functions (_open, _write, etc.)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_DEPRECATE)
endif()

option(LOGOSDB_BUILD_TOOLS  "Build CLI and benchmark tools" ON)
option(LOGOSDB_BUILD_TESTS  "Build unit tests"              ON)
option(LOGOSDB_BUILD_PYTHON "Build Python bindings (pybind11)" OFF)

# hnswlib and nlohmann/json are header-only
add_library(hnswlib INTERFACE)
target_include_directories(hnswlib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# Core library
add_library(logosdb
    src/logosdb.cpp
    src/storage.cpp
    src/metadata.cpp
    src/hnsw_index.cpp
    src/wal.cpp
    src/platform.cpp
)
target_include_directories(logosdb PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(logosdb PRIVATE hnswlib nlohmann_json)

# Position-independent code is required when the static library is linked
# into a shared object (the Python extension is a .so).
set_target_properties(logosdb PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(APPLE)
    # No extra link flags needed for mmap on macOS
elseif(UNIX)
    target_link_libraries(logosdb PRIVATE pthread)
endif()

if(LOGOSDB_BUILD_TOOLS)
    add_executable(logosdb-cli tools/logosdb-cli.cpp)
    target_link_libraries(logosdb-cli PRIVATE logosdb)
    # MSVC's format-string checker cannot parse "%" PRIu64 macro concatenation
    # and emits false-positive C4474/C4477.  Suppress only for this target.
    if(MSVC)
        target_compile_options(logosdb-cli PRIVATE /wd4474 /wd4477)
    endif()

    add_executable(logosdb-bench tools/logosdb-bench.cpp)
    target_link_libraries(logosdb-bench PRIVATE logosdb)
endif()

if(LOGOSDB_BUILD_TESTS)
    find_package(Threads REQUIRED)
    # Single-header doctest is vendored in tests/
    # Stress suite ("stress") is excluded by default; run with:
    #   ./logosdb-test --test-suite=stress
    add_executable(logosdb-test tests/test_doctest.cpp tests/test_stress.cpp)
    target_include_directories(logosdb-test PRIVATE tests)
    target_link_libraries(logosdb-test PRIVATE logosdb Threads::Threads)

    enable_testing()
    add_test(NAME logosdb_doctest COMMAND logosdb-test --test-suite-exclude=stress)
endif()

option(LOGOSDB_BUILD_FUZZ "Build libFuzzer harness for JSON metadata parsing (Clang, Linux)" OFF)
if(LOGOSDB_BUILD_FUZZ)
    if(APPLE)
        message(
            WARNING
                "LOGOSDB_BUILD_FUZZ skipped: Apple Clang does not ship libFuzzer here. Use Linux + Clang (see CI)."
        )
    elseif(WIN32)
        message(
            WARNING
                "LOGOSDB_BUILD_FUZZ skipped: libFuzzer is not available on Windows. Use Linux + Clang (see CI)."
        )
    else()
        add_executable(logosdb-fuzz-jsonl tools/fuzz_jsonl_metadata.cpp)
        target_link_libraries(logosdb-fuzz-jsonl PRIVATE nlohmann_json)
        target_compile_options(logosdb-fuzz-jsonl PRIVATE -fsanitize=fuzzer-no-link,address -O1 -g)
        target_link_options(logosdb-fuzz-jsonl PRIVATE -fsanitize=fuzzer,address)
    endif()
endif()

if(LOGOSDB_BUILD_PYTHON)
    # Discover Python and pybind11. When built by scikit-build-core,
    # pybind11 is provided via the build isolation environment.
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(_core MODULE python/src/bindings.cpp)
    target_link_libraries(_core PRIVATE logosdb)
    target_compile_definitions(_core PRIVATE
        LOGOSDB_PY_VERSION="${PROJECT_VERSION}"
    )

    # Install the extension module inside the `logosdb` package directory
    # so it is importable as `logosdb._core`.
    install(TARGETS _core LIBRARY DESTINATION logosdb)
endif()
