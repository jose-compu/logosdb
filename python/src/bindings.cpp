#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <logosdb/logosdb.h>

#include <memory>
#include <string>
#include <vector>

namespace py = pybind11;

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

static std::vector<float> to_vector(const FloatArray & arr, const char * name) {
    auto buf = arr.request();
    if (buf.ndim != 1) {
        throw py::value_error(std::string(name) + " must be a 1-D float32 array");
    }
    const float * data = static_cast<const float *>(buf.ptr);
    return std::vector<float>(data, data + buf.shape[0]);
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "LogosDB — fast semantic vector database (HNSW + mmap)";

#ifdef LOGOSDB_PY_VERSION
    m.attr("__version__") = LOGOSDB_PY_VERSION;
#else
    m.attr("__version__") = LOGOSDB_VERSION_STRING;
#endif
    m.attr("LOGOSDB_VERSION") = LOGOSDB_VERSION_STRING;

    // Distance metric constants
    m.attr("DIST_IP")     = LOGOSDB_DIST_IP;
    m.attr("DIST_COSINE") = LOGOSDB_DIST_COSINE;
    m.attr("DIST_L2")     = LOGOSDB_DIST_L2;

    // ── SearchHit ──────────────────────────────────────────────────
    py::class_<logosdb::SearchHit>(m, "SearchHit",
        "A single search result: id, score, and optional text/timestamp metadata.")
        .def_readonly("id",        &logosdb::SearchHit::id)
        .def_readonly("score",     &logosdb::SearchHit::score)
        .def_readonly("text",      &logosdb::SearchHit::text)
        .def_readonly("timestamp", &logosdb::SearchHit::timestamp)
        .def("__repr__", [](const logosdb::SearchHit & h) {
            return "SearchHit(id=" + std::to_string(h.id) +
                   ", score=" + std::to_string(h.score) +
                   ", text=" + h.text +
                   ", timestamp=" + h.timestamp + ")";
        })
        .def("__iter__", [](const logosdb::SearchHit & h) {
            return py::iter(py::make_tuple(h.id, h.score, h.text, h.timestamp));
        });

    // ── DB ─────────────────────────────────────────────────────────
    py::class_<logosdb::DB>(m, "DB",
        "A LogosDB database handle. One process per database path.")
        .def(py::init([](const std::string & path,
                         int dim,
                         size_t max_elements,
                         int ef_construction,
                         int M,
                         int ef_search,
                         int distance) {
            logosdb::Options o;
            o.dim              = dim;
            o.max_elements     = max_elements;
            o.ef_construction  = ef_construction;
            o.M                = M;
            o.ef_search        = ef_search;
            o.distance         = distance;
            return std::make_unique<logosdb::DB>(path, o);
        }),
        py::arg("path"),
        py::arg("dim"),
        py::arg("max_elements")    = 1000000,
        py::arg("ef_construction") = 200,
        py::arg("M")               = 16,
        py::arg("ef_search")       = 50,
        py::arg("distance")        = LOGOSDB_DIST_IP,
        R"doc(Open or create a database at `path`.

Args:
    path: directory to store vectors/metadata/index files.
    dim: vector dimensionality. Fixed at creation; enforced on reopen.
    max_elements: HNSW capacity (default 1,000,000).
    ef_construction: HNSW build-time search width (default 200).
    M: HNSW graph out-degree (default 16).
    ef_search: HNSW query-time search width (default 50).
    distance: distance metric (DIST_IP, DIST_COSINE, or DIST_L2; default DIST_IP).
)doc")

        .def("put",
            [](logosdb::DB & self,
               FloatArray embedding,
               const std::string & text,
               const std::string & timestamp) {
                return self.put(to_vector(embedding, "embedding"), text, timestamp);
            },
            py::arg("embedding"),
            py::arg("text") = "",
            py::arg("timestamp") = "",
            "Insert a vector. Returns the new row id.")

        .def("delete",
            [](logosdb::DB & self, uint64_t id) { self.del(id); },
            py::arg("id"),
            "Tombstone the row with the given id. Excluded from future searches.")

        .def("update",
            [](logosdb::DB & self,
               uint64_t id,
               FloatArray embedding,
               const std::string & text,
               const std::string & timestamp) {
                return self.update(id, to_vector(embedding, "embedding"), text, timestamp);
            },
            py::arg("id"),
            py::arg("embedding"),
            py::arg("text") = "",
            py::arg("timestamp") = "",
            "Replace the row with the given id. Returns the NEW id (delete + put).")

        .def("search",
            [](logosdb::DB & self, FloatArray query, int top_k) {
                return self.search(to_vector(query, "query"), top_k);
            },
            py::arg("query"),
            py::arg("top_k") = 5,
            "Top-k approximate nearest-neighbor search. Returns list[SearchHit].")

        .def("search_ts_range",
            [](logosdb::DB & self,
               FloatArray query,
               int top_k,
               const std::string & ts_from,
               const std::string & ts_to,
               int candidate_k) {
                return self.search_ts_range(
                    to_vector(query, "query"), top_k, ts_from, ts_to, candidate_k);
            },
            py::arg("query"),
            py::arg("top_k") = 5,
            py::arg("ts_from") = "",
            py::arg("ts_to") = "",
            py::arg("candidate_k") = 0,
            R"doc(Search with timestamp range filter.

Args:
    query: query vector (1-D float32 numpy array).
    top_k: number of results to return.
    ts_from: start timestamp (ISO 8601, inclusive), empty for no lower bound.
    ts_to: end timestamp (ISO 8601, inclusive), empty for no upper bound.
    candidate_k: internal fetch multiplier (default 10x top_k for good recall).

Returns:
    list[SearchHit]: filtered results sorted by score.
)doc")

        .def("count",
            &logosdb::DB::count,
            "Total row count including tombstoned rows.")

        .def("count_live",
            &logosdb::DB::count_live,
            "Live row count (total minus tombstoned).")

        .def_property_readonly("dim",
            &logosdb::DB::dim,
            "Vector dimensionality.")

        .def("raw_vectors",
            [](py::object self) {
                auto & db = self.cast<logosdb::DB &>();
                size_t n = 0;
                int    d = 0;
                const float * p = db.raw_vectors(n, d);
                if (!p || n == 0) {
                    // Empty zero-copy placeholder (0 rows × d cols).
                    return py::array_t<float>(
                        std::vector<py::ssize_t>{0, (py::ssize_t)d});
                }
                // Zero-copy view into the mmap-backed storage. The array keeps
                // the DB Python object alive via `self` as its base.
                std::vector<py::ssize_t> shape   = {(py::ssize_t)n, (py::ssize_t)d};
                std::vector<py::ssize_t> strides = {
                    (py::ssize_t)(d * sizeof(float)),
                    (py::ssize_t)sizeof(float)
                };
                py::array_t<float> arr(shape, strides, p, self);
                // Mark as read-only.
                py::detail::array_proxy(arr.ptr())->flags &=
                    ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
                return arr;
            },
            R"doc(Return a read-only (n_rows, dim) numpy array view over the vector store.

The returned array is zero-copy and shares memory with the mmap-backed
storage; do NOT modify it. The array holds a reference to the DB, so
it keeps the database open for the lifetime of the array.
)doc")

        .def("__len__", &logosdb::DB::count_live)

        .def("__repr__", [](const logosdb::DB & db) {
            return "DB(dim=" + std::to_string(db.dim()) +
                   ", count=" + std::to_string(db.count()) +
                   ", count_live=" + std::to_string(db.count_live()) + ")";
        });
}
