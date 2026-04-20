cmake_minimum_required(VERSION 3.15)
project(logosdb VERSION 0.2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

    add_executable(logosdb-bench tools/logosdb-bench.cpp)
    target_link_libraries(logosdb-bench PRIVATE logosdb)
endif()

if(LOGOSDB_BUILD_TESTS)
    add_executable(logosdb-test tests/test_basic.cpp)
    target_link_libraries(logosdb-test PRIVATE logosdb)

    enable_testing()
    add_test(NAME logosdb_basic COMMAND logosdb-test)
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
