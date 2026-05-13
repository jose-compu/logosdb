#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <logosdb/logosdb.h>

#include <cstring>
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

    m.def("compact",
          [](const std::string& src, const std::string& dst) { logosdb::compact(src, dst); },
          py::arg("src"),
          py::arg("dst"),
          "Copy live rows from src database directory into dst (drops tombstones; dst must be "
          "empty).");

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

        .def("put_batch",
            [](logosdb::DB & self,
               py::array_t<float, py::array::c_style | py::array::forcecast> embeddings,
               py::object texts_obj,
               py::object timestamps_obj) {
                auto buf = embeddings.request();
                if (buf.ndim != 2) {
                    throw py::value_error("embeddings must be a 2-D float32 array (n, dim)");
                }
                int n = (int)buf.shape[0];
                int dim = (int)buf.shape[1];
                if (dim != self.dim()) {
                    throw py::value_error("embeddings dim does not match DB.dim");
                }
                std::vector<float> flat(static_cast<size_t>(n) * static_cast<size_t>(dim));
                std::memcpy(flat.data(), buf.ptr, flat.size() * sizeof(float));

                std::vector<std::string> texts;
                if (!texts_obj.is_none()) {
                    texts = py::cast<std::vector<std::string>>(texts_obj);
                    if ((int)texts.size() != n)
                        throw py::value_error("len(texts) must equal embeddings.shape[0]");
                }
                std::vector<std::string> timestamps;
                if (!timestamps_obj.is_none()) {
                    timestamps = py::cast<std::vector<std::string>>(timestamps_obj);
                    if ((int)timestamps.size() != n)
                        throw py::value_error("len(timestamps) must equal embeddings.shape[0]");
                }
                std::vector<uint64_t> ids;
                {
                    py::gil_scoped_release release;
                    ids = self.put_batch(flat, n, texts, timestamps);
                }
                return ids;
            },
            py::arg("embeddings"),
            py::arg("texts") = py::none(),
            py::arg("timestamps") = py::none(),
            R"doc(Insert n vectors in one chunked, WAL-aware call.

Args:
    embeddings: 2-D float32 array of shape (n, dim).
    texts: optional list[str] of length n (per-row metadata; "" allowed).
    timestamps: optional list[str] of length n (ISO 8601 or empty).

Returns:
    list[int]: the n new row ids (contiguous).

Durability matches per-row `put`: rows are WAL-protected before being
written to the vector/metadata stores; the WAL is fsynced once per chunk
(default chunk size 1024; override with `LOGOSDB_BATCH_CHUNK_SIZE`).
)doc")

        .def("export_ndjson",
            [](logosdb::DB & self,
               const std::string & out_path,
               uint64_t start_id,
               uint64_t end_id_exclusive) {
                py::gil_scoped_release release;
                self.export_ndjson(out_path, start_id, end_id_exclusive);
            },
            py::arg("out_path"),
            py::arg("start_id") = 0,
            py::arg("end_id_exclusive") = 0,
            "Stream live rows in [start_id, end_id_exclusive) to `out_path` as NDJSON. "
            "Bounded memory (O(dim) per row). Pass end_id_exclusive=0 for 'until count()'.")

        .def("import_ndjson",
            [](logosdb::DB & self,
               const std::string & in_path,
               int chunk_size,
               const std::string & checkpoint_path,
               bool resume) {
                py::gil_scoped_release release;
                self.import_ndjson(in_path, chunk_size, checkpoint_path, resume);
            },
            py::arg("in_path"),
            py::arg("chunk_size") = 1024,
            py::arg("checkpoint_path") = "",
            py::arg("resume") = false,
            "Stream NDJSON into the DB with chunked, WAL-aware put_batch. "
            "Optional `checkpoint_path` lets a later call `--resume` past the byte_offset.")

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

        .def("sync",
            &logosdb::DB::sync,
            "Flush WAL, persist HNSW index, and fsync vector + metadata stores.")

        .def(
            "stats",
            [](const logosdb::DB& self) {
                logosdb_stats_t s = self.get_stats();
                py::dict d;
                d["rows_total"] = s.rows_total;
                d["rows_live"] = s.rows_live;
                d["tombstones"] = s.tombstones;
                d["index_elements"] = s.index_elements;
                d["wal_pending"] = s.wal_pending;
                d["distance_metric"] = s.distance_metric;
                d["storage_dtype"] = s.storage_dtype;
                d["put_success"] = s.put_success;
                d["put_failed"] = s.put_failed;
                d["put_batch_success"] = s.put_batch_success;
                d["put_batch_failed"] = s.put_batch_failed;
                d["search_success"] = s.search_success;
                d["search_failed"] = s.search_failed;
                d["search_ts_success"] = s.search_ts_success;
                d["search_ts_failed"] = s.search_ts_failed;
                d["delete_success"] = s.delete_success;
                d["delete_failed"] = s.delete_failed;
                d["update_success"] = s.update_success;
                d["update_failed"] = s.update_failed;
                d["sync_calls"] = s.sync_calls;
                return d;
            },
            "Operation counters and store health fields.")

        .def_property_readonly("dim",
            &logosdb::DB::dim,
            "Vector dimensionality.")

        .def("raw_vectors",
            [](py::object self) {
                auto & db = self.cast<logosdb::DB &>();
                size_t n = 0;
                int    d = 0;
                // raw_vectors now returns a vector<float> (handles quantized storage)
                std::vector<float> data = db.raw_vectors(n, d);
                if (data.empty() || n == 0) {
                    // Empty array (0 rows × d cols).
                    return py::array_t<float>(
                        std::vector<py::ssize_t>{0, (py::ssize_t)d});
                }
                // Return a numpy array. This copies the data (required because
                // data is a local vector that will be destroyed).
                std::vector<py::ssize_t> shape   = {(py::ssize_t)n, (py::ssize_t)d};
                py::array_t<float> arr(shape, data.data());
                // Mark as read-only.
                py::detail::array_proxy(arr.ptr())->flags &=
                    ~py::detail::npy_api::NPY_ARRAY_WRITEABLE_;
                return arr;
            },
            R"doc(Return a read-only (n_rows, dim) numpy array copy of the vector store.

The returned array is a COPY of the vector data. For reduced-precision
storage (float16/int8), vectors are dequantized to float32 before return.
This is safe to read but should NOT be modified.
)doc")

        .def("__len__", &logosdb::DB::count_live)

        .def("__repr__", [](const logosdb::DB & db) {
            return "DB(dim=" + std::to_string(db.dim()) +
                   ", count=" + std::to_string(db.count()) +
                   ", count_live=" + std::to_string(db.count_live()) + ")";
        });
}
