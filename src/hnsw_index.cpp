#include "hnsw_index.h"

#include <hnswlib/hnswlib/hnswlib.h>
#include <fstream>

namespace logosdb {
namespace internal {

struct HnswIndex::Impl {
    hnswlib::InnerProductSpace space;
    hnswlib::HierarchicalNSW<float> * alg = nullptr;
    HnswParams params;

    explicit Impl(int dim) : space(dim) {}
    ~Impl() { delete alg; }
};

HnswIndex::~HnswIndex() { close(); }

bool HnswIndex::open(const std::string & path, const HnswParams & params,
                      std::string & err) {
    close();
    path_ = path;

    impl_ = new Impl(params.dim);
    impl_->params = params;

    // Try loading existing index.
    std::ifstream probe(path, std::ios::binary);
    if (probe.good() && probe.peek() != std::ifstream::traits_type::eof()) {
        probe.close();
        try {
            impl_->alg = new hnswlib::HierarchicalNSW<float>(
                &impl_->space, path, false, params.max_elements, true);
            impl_->alg->setEf(params.ef_search);
            return true;
        } catch (const std::exception & e) {
            err = std::string("hnsw load: ") + e.what();
            close();
            return false;
        }
    }
    probe.close();

    try {
        impl_->alg = new hnswlib::HierarchicalNSW<float>(
            &impl_->space, params.max_elements, params.M, params.ef_construction);
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

bool HnswIndex::add(uint64_t label, const float * vec, std::string & err) {
    if (!impl_ || !impl_->alg) { err = "index not open"; return false; }
    try {
        impl_->alg->addPoint(vec, (hnswlib::labeltype)label);
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
    try {
        auto pq = impl_->alg->searchKnn(query, k);
        results.reserve(pq.size());
        while (!pq.empty()) {
            auto & top = pq.top();
            // hnswlib inner-product returns (1 - ip) as distance; convert back.
            float score = 1.0f - top.first;
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
    } catch (const std::exception & e) {
        err = std::string("hnsw save: ") + e.what();
        return false;
    }
    return true;
}

} // namespace internal
} // namespace logosdb
