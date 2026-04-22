#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace logosdb {
namespace internal {

/* Distance metrics - match the public API constants */
enum DistanceMetric {
    DIST_IP = 0,      /* Inner product */
    DIST_COSINE = 1,  /* Cosine (IP on normalized vectors) */
    DIST_L2 = 2       /* Euclidean distance */
};

struct HnswParams {
    int    dim             = 0;
    size_t max_elements    = 1000000;
    int    ef_construction = 200;
    int    M               = 16;
    int    ef_search       = 50;
    int    distance        = DIST_IP;  /* DIST_IP, DIST_COSINE, or DIST_L2 */
};

// Thin wrapper around hnswlib for inner-product search on L2-normalized vectors.
class HnswIndex {
public:
    HnswIndex() = default;
    ~HnswIndex();

    HnswIndex(const HnswIndex &) = delete;
    HnswIndex & operator=(const HnswIndex &) = delete;

    // Create a new index or load an existing one from disk.
    bool open(const std::string & path, const HnswParams & params, std::string & err);
    void close();

    // Add a vector with the given internal label (row index).
    bool add(uint64_t label, const float * vec, std::string & err);

    // Mark `label` as deleted. Excluded from subsequent search results.
    // Returns false if the label is not in the index.
    bool mark_deleted(uint64_t label, std::string & err);

    // True if `label` exists in the index and is currently marked deleted.
    // Returns false if the label is missing.
    bool is_deleted(uint64_t label) const;

    // True if `label` is currently present (known) in the index, regardless
    // of its deletion state.
    bool has_label(uint64_t label) const;

    // Search for top-k nearest. Returns (label, distance) pairs sorted by distance desc
    // (inner product — higher is more similar).
    std::vector<std::pair<uint64_t, float>>
    search(const float * query, int top_k, std::string & err) const;

    size_t count() const;

    bool save(std::string & err) const;

private:
    struct Impl;
    Impl * impl_ = nullptr;
    std::string path_;
};

} // namespace internal
} // namespace logosdb
