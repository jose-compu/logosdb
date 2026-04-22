#include "hnsw_index.h"

#include <hnswlib/hnswlib/hnswlib.h>
#include <fstream>
#include <cstring>

namespace logosdb {
namespace internal {

/* Index file header for distance metric persistence */
struct IndexHeader {
    char     magic[8];        /* "LOGOSDB\0" */
    uint32_t version;         /* 1 */
    int32_t  distance;      /* DistanceMetric value */
    int32_t  dim;           /* vector dimension */
    uint8_t  reserved[12];  /* padding to 32 bytes */

    IndexHeader() {
        std::memset(this, 0, sizeof(*this));
        std::memcpy(magic, "LOGOSDB\0", 8);
        version = 1;
        distance = DIST_IP;
        dim = 0;
    }
};

struct HnswIndex::Impl {
    hnswlib::SpaceInterface<float> * space = nullptr;
    hnswlib::HierarchicalNSW<float> * alg = nullptr;
    HnswParams params;
    DistanceMetric distance_metric = DIST_IP;

    ~Impl() {
        delete alg;
        delete space;
    }
};

HnswIndex::~HnswIndex() { close(); }

static hnswlib::SpaceInterface<float> * create_space(DistanceMetric dist, int dim) {
    switch (dist) {
        case DIST_L2:
            return new hnswlib::L2Space(dim);
        case DIST_IP:
        case DIST_COSINE:
        default:
            return new hnswlib::InnerProductSpace(dim);
    }
}

/* Get the metadata path from the index path */
static std::string get_meta_path(const std::string & index_path) {
    return index_path + ".meta";
}

bool HnswIndex::open(const std::string & path, const HnswParams & params,
                      std::string & err) {
    close();
    path_ = path;

    impl_ = new Impl();
    impl_->params = params;
    impl_->distance_metric = static_cast<DistanceMetric>(params.distance);

    // Check for metadata file with stored distance metric
    std::string meta_path = get_meta_path(path);
    std::ifstream meta_in(meta_path, std::ios::binary);
    if (meta_in.good()) {
        IndexHeader header;
        meta_in.read(reinterpret_cast<char*>(&header), sizeof(header));
        meta_in.close();

        if (std::memcmp(header.magic, "LOGOSDB\0", 8) == 0 && header.version == 1) {
            // Check distance metric compatibility
            DistanceMetric stored_dist = static_cast<DistanceMetric>(header.distance);
            if (stored_dist != impl_->distance_metric) {
                err = "distance metric mismatch: stored=" + std::to_string(header.distance) +
                      ", requested=" + std::to_string(params.distance);
                close();
                return false;
            }
            // Check dimension
            if (header.dim != params.dim) {
                err = "dimension mismatch: stored=" + std::to_string(header.dim) +
                      ", requested=" + std::to_string(params.dim);
                close();
                return false;
            }
        }
    }

    // Try loading existing index.
    std::ifstream in(path, std::ios::binary);
    if (in.good() && in.peek() != std::ifstream::traits_type::eof()) {
        in.close();
        try {
            impl_->space = create_space(impl_->distance_metric, params.dim);
            impl_->alg = new hnswlib::HierarchicalNSW<float>(
                impl_->space, path, false, params.max_elements, true);
            impl_->alg->setEf(params.ef_search);
            return true;
        } catch (const std::exception & e) {
            err = std::string("hnsw load: ") + e.what();
            close();
            return false;
        }
    }
    in.close();

    // Create new index
    try {
        impl_->space = create_space(impl_->distance_metric, params.dim);
        impl_->alg = new hnswlib::HierarchicalNSW<float>(
            impl_->space, params.max_elements, params.M, params.ef_construction);
        impl_->alg->setEf(params.ef_search);
    } catch (const std::exception & e) {
        err = std::string("hnsw init: ") + e.what();
        close();
        return false;
    }
    return true;
}

void HnswIndex::close() {
    delete impl_;
    impl_ = nullptr;
    path_.clear();
}

/* Normalize a vector to unit length (for cosine similarity) */
static void l2_normalize(float * vec, int dim) {
    float norm = 0.0f;
    for (int i = 0; i < dim; ++i) {
        norm += vec[i] * vec[i];
    }
    norm = std::sqrt(norm);
    if (norm > 0.0f) {
        for (int i = 0; i < dim; ++i) {
            vec[i] /= norm;
        }
    }
}

bool HnswIndex::add(uint64_t label, const float * vec, std::string & err) {
    if (!impl_ || !impl_->alg) { err = "index not open"; return false; }
    try {
        if (impl_->distance_metric == DIST_COSINE) {
            // Make a copy and normalize for cosine similarity
            std::vector<float> normalized(vec, vec + impl_->params.dim);
            l2_normalize(normalized.data(), impl_->params.dim);
            impl_->alg->addPoint(normalized.data(), (hnswlib::labeltype)label);
        } else {
            impl_->alg->addPoint(vec, (hnswlib::labeltype)label);
        }
    } catch (const std::exception & e) {
        err = std::string("hnsw add: ") + e.what();
        return false;
    }
    return true;
}

bool HnswIndex::mark_deleted(uint64_t label, std::string & err) {
    if (!impl_ || !impl_->alg) { err = "index not open"; return false; }
    try {
        impl_->alg->markDelete((hnswlib::labeltype)label);
    } catch (const std::exception & e) {
        err = std::string("hnsw markDelete: ") + e.what();
        return false;
    }
    return true;
}

bool HnswIndex::is_deleted(uint64_t label) const {
    if (!impl_ || !impl_->alg) return false;
    auto & lookup = impl_->alg->label_lookup_;
    std::unique_lock<std::mutex> lk(impl_->alg->label_lookup_lock);
    auto it = lookup.find((hnswlib::labeltype)label);
    if (it == lookup.end()) return false;
    auto internal_id = it->second;
    lk.unlock();
    return impl_->alg->isMarkedDeleted(internal_id);
}

bool HnswIndex::has_label(uint64_t label) const {
    if (!impl_ || !impl_->alg) return false;
    auto & lookup = impl_->alg->label_lookup_;
    std::unique_lock<std::mutex> lk(impl_->alg->label_lookup_lock);
    return lookup.find((hnswlib::labeltype)label) != lookup.end();
}

std::vector<std::pair<uint64_t, float>>
HnswIndex::search(const float * query, int top_k, std::string & err) const {
    std::vector<std::pair<uint64_t, float>> results;
    if (!impl_ || !impl_->alg) { err = "index not open"; return results; }
    if (impl_->alg->cur_element_count == 0) return results;
    if (top_k <= 0) return results;

    size_t k = std::min((size_t)top_k, (size_t)impl_->alg->cur_element_count);

    // Prepare query (normalize for cosine)
    std::vector<float> normalized_query;
    const float * search_vec = query;
    if (impl_->distance_metric == DIST_COSINE) {
        normalized_query.assign(query, query + impl_->params.dim);
        l2_normalize(normalized_query.data(), impl_->params.dim);
        search_vec = normalized_query.data();
    }

    try {
        auto pq = impl_->alg->searchKnn(search_vec, k);
        results.reserve(pq.size());
        while (!pq.empty()) {
            auto & top = pq.top();
            float score;
            if (impl_->distance_metric == DIST_L2) {
                // L2: distance is already correct, invert so higher = more similar
                score = 1.0f / (1.0f + top.first);
            } else {
                // IP/Cosine: hnswlib returns (1 - ip) as distance; convert back
                score = 1.0f - top.first;
            }
            results.push_back({(uint64_t)top.second, score});
            pq.pop();
        }
        // Reverse so highest score is first.
        std::reverse(results.begin(), results.end());
    } catch (const std::exception & e) {
        err = std::string("hnsw search: ") + e.what();
    }
    return results;
}

size_t HnswIndex::count() const {
    if (!impl_ || !impl_->alg) return 0;
    return impl_->alg->cur_element_count;
}

bool HnswIndex::save(std::string & err) const {
    if (!impl_ || !impl_->alg) { err = "index not open"; return false; }
    try {
        impl_->alg->saveIndex(path_);

        // Write metadata file with distance metric
        std::string meta_path = get_meta_path(path_);
        std::ofstream meta_out(meta_path, std::ios::binary | std::ios::trunc);
        if (!meta_out.good()) {
            err = "cannot write index metadata";
            return false;
        }

        IndexHeader header;
        header.distance = impl_->distance_metric;
        header.dim = impl_->params.dim;

        meta_out.write(reinterpret_cast<const char*>(&header), sizeof(header));
        meta_out.close();
    } catch (const std::exception & e) {
        err = std::string("hnsw save: ") + e.what();
        return false;
    }
    return true;
}

} // namespace internal
} // namespace logosdb
