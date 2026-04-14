cmake_minimum_required(VERSION 3.14)
project(logosdb VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# hnswlib is header-only
add_library(hnswlib INTERFACE)
target_include_directories(hnswlib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# Core library
add_library(logosdb
    src/logosdb.cpp
    src/storage.cpp
    src/metadata.cpp
    src/hnsw_index.cpp
)
target_include_directories(logosdb PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(logosdb PRIVATE hnswlib)

if(APPLE)
    # No extra link flags needed for mmap on macOS
elseif(UNIX)
    target_link_libraries(logosdb PRIVATE pthread)
endif()

# CLI tool
add_executable(logosdb-cli tools/logosdb-cli.cpp)
target_link_libraries(logosdb-cli PRIVATE logosdb)

# Benchmark tool
add_executable(logosdb-bench tools/logosdb-bench.cpp)
target_link_libraries(logosdb-bench PRIVATE logosdb)

# Tests
add_executable(logosdb-test tests/test_basic.cpp)
target_link_libraries(logosdb-test PRIVATE logosdb)

enable_testing()
add_test(NAME logosdb_basic COMMAND logosdb-test)
