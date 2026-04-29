/**
 * Node.js N-API bindings for LogosDB
 *
 * Provides JavaScript bindings for the LogosDB C++ API.
 */

#include <napi.h>
#include <logosdb/logosdb.h>
#include <cstring>
#include <string>
#include <vector>

using namespace Napi;

// Helper to convert JS array to float vector
static std::vector<float> JsArrayToFloatVector(const Array& arr) {
    std::vector<float> vec;
    vec.reserve(arr.Length());
    for (size_t i = 0; i < arr.Length(); i++) {
        vec.push_back(arr.Get(i).As<Number>().FloatValue());
    }
    return vec;
}

// Helper to convert SearchHit to JS object
static Object SearchHitToJsObject(Env env, const logosdb_search_result_t* result, int index) {
    Object obj = Object::New(env);

    uint64_t id = logosdb_result_id(result, index);
    float score = logosdb_result_score(result, index);
    const char* text = logosdb_result_text(result, index);
    const char* timestamp = logosdb_result_timestamp(result, index);

    obj.Set("id", Number::New(env, static_cast<double>(id)));
    obj.Set("score", Number::New(env, score));
    if (text) {
        obj.Set("text", String::New(env, text));
    } else {
        obj.Set("text", env.Null());
    }
    if (timestamp) {
        obj.Set("timestamp", String::New(env, timestamp));
    } else {
        obj.Set("timestamp", env.Null());
    }

    return obj;
}

// DB wrapper class
class DBWrapper : public ObjectWrap<DBWrapper> {
public:
    static Object Init(Env env, Object exports) {
        Function func = DefineClass(env, "DB", {
            InstanceMethod("put", &DBWrapper::Put),
            InstanceMethod("search", &DBWrapper::Search),
            InstanceMethod("searchTsRange", &DBWrapper::SearchTsRange),
            InstanceMethod("update", &DBWrapper::Update),
            InstanceMethod("delete", &DBWrapper::Delete),
            InstanceMethod("count", &DBWrapper::Count),
            InstanceMethod("countLive", &DBWrapper::CountLive),
            InstanceMethod("dim", &DBWrapper::Dim),
            InstanceMethod("close", &DBWrapper::Close),
        });

        exports.Set("DB", func);
        return exports;
    }

    DBWrapper(const CallbackInfo& info) : ObjectWrap<DBWrapper>(info) {
        Env env = info.Env();

        if (info.Length() < 1 || !info[0].IsString()) {
            throw TypeError::New(env, "Path (string) expected as first argument");
        }

        std::string path = info[0].As<String>().Utf8Value();
        int dim = 128;  // default
        size_t max_elements = 1000000;
        int ef_construction = 200;
        int M = 16;
        int ef_search = 50;
        int distance = 0;  // LOGOSDB_DIST_IP

        // Parse options object if provided
        if (info.Length() > 1 && info[1].IsObject()) {
            Object options = info[1].As<Object>();

            if (options.Has("dim")) {
                dim = options.Get("dim").As<Number>().Int32Value();
            }
            if (options.Has("maxElements")) {
                max_elements = static_cast<size_t>(
                    options.Get("maxElements").As<Number>().Int64Value()
                );
            }
            if (options.Has("efConstruction")) {
                ef_construction = options.Get("efConstruction").As<Number>().Int32Value();
            }
            if (options.Has("M")) {
                M = options.Get("M").As<Number>().Int32Value();
            }
            if (options.Has("efSearch")) {
                ef_search = options.Get("efSearch").As<Number>().Int32Value();
            }
            if (options.Has("distance")) {
                distance = options.Get("distance").As<Number>().Int32Value();
            }
        }

        // Create options
        logosdb_options_t* opts = logosdb_options_create();
        logosdb_options_set_dim(opts, dim);
        logosdb_options_set_max_elements(opts, max_elements);
        logosdb_options_set_ef_construction(opts, ef_construction);
        logosdb_options_set_M(opts, M);
        logosdb_options_set_ef_search(opts, ef_search);
        logosdb_options_set_distance(opts, distance);

        char* err = nullptr;
        db_ = logosdb_open(path.c_str(), opts, &err);
        logosdb_options_destroy(opts);

        if (!db_) {
            std::string error_msg = err ? err : "Unknown error opening database";
            if (err) free(err);
            throw Error::New(env, error_msg);
        }

        dim_ = dim;
    }

    ~DBWrapper() {
        if (db_) {
            logosdb_close(db_);
            db_ = nullptr;
        }
    }

private:
    logosdb_t* db_ = nullptr;
    int dim_ = 0;

    void Put(const CallbackInfo& info) {
        Env env = info.Env();

        if (info.Length() < 1 || !info[0].IsArray()) {
            throw TypeError::New(env, "Embedding (array) expected as first argument");
        }

        Array embeddingArr = info[0].As<Array>();
        std::vector<float> embedding = JsArrayToFloatVector(embeddingArr);

        if ((int)embedding.size() != dim_) {
            throw Error::New(env, "Embedding dimension mismatch");
        }

        const char* text = nullptr;
        const char* timestamp = nullptr;

        if (info.Length() > 1 && info[1].IsString()) {
            text_str_ = info[1].As<String>().Utf8Value();
            text = text_str_.c_str();
        }

        if (info.Length() > 2 && info[2].IsString()) {
            ts_str_ = info[2].As<String>().Utf8Value();
            timestamp = ts_str_.c_str();
        }

        char* err = nullptr;
        uint64_t id = logosdb_put(db_, embedding.data(), dim_, text, timestamp, &err);

        if (id == UINT64_MAX) {
            std::string error_msg = err ? err : "Failed to insert";
            if (err) free(err);
            throw Error::New(env, error_msg);
        }

        if (err) free(err);

        info.GetReturnValue().Set(Number::New(env, static_cast<double>(id)));
    }

    // Storage for string references during put calls
    std::string text_str_;
    std::string ts_str_;

    void Search(const CallbackInfo& info) {
        Env env = info.Env();

        if (info.Length() < 1 || !info[0].IsArray()) {
            throw TypeError::New(env, "Query embedding (array) expected as first argument");
        }

        Array queryArr = info[0].As<Array>();
        std::vector<float> query = JsArrayToFloatVector(queryArr);

        if ((int)query.size() != dim_) {
            throw Error::New(env, "Query dimension mismatch");
        }

        int top_k = 10;
        if (info.Length() > 1 && info[1].IsNumber()) {
            top_k = info[1].As<Number>().Int32Value();
        }

        char* err = nullptr;
        logosdb_search_result_t* result = logosdb_search(db_, query.data(), dim_, top_k, &err);

        if (!result) {
            std::string error_msg = err ? err : "Search failed";
            if (err) free(err);
            throw Error::New(env, error_msg);
        }

        if (err) free(err);

        int count = logosdb_result_count(result);
        Array results = Array::New(env, count);

        for (int i = 0; i < count; i++) {
            results.Set(i, SearchHitToJsObject(env, result, i));
        }

        logosdb_result_free(result);

        info.GetReturnValue().Set(results);
    }

    void SearchTsRange(const CallbackInfo& info) {
        Env env = info.Env();

        if (info.Length() < 1 || !info[0].IsArray()) {
            throw TypeError::New(env, "Query embedding (array) expected");
        }

        Array queryArr = info[0].As<Array>();
        std::vector<float> query = JsArrayToFloatVector(queryArr);

        if ((int)query.size() != dim_) {
            throw Error::New(env, "Query dimension mismatch");
        }

        int top_k = 10;
        std::string ts_from;
        std::string ts_to;
        int candidate_k = 0;

        if (info.Length() > 1 && info[1].IsObject()) {
            Object options = info[1].As<Object>();

            if (options.Has("topK")) {
                top_k = options.Get("topK").As<Number>().Int32Value();
            }
            if (options.Has("tsFrom")) {
                ts_from = options.Get("tsFrom").As<String>().Utf8Value();
            }
            if (options.Has("tsTo")) {
                ts_to = options.Get("tsTo").As<String>().Utf8Value();
            }
            if (options.Has("candidateK")) {
                candidate_k = options.Get("candidateK").As<Number>().Int32Value();
            }
        }

        if (candidate_k < top_k) {
            candidate_k = top_k * 10;  // Default 10x
        }

        char* err = nullptr;
        logosdb_search_result_t* result = logosdb_search_ts_range(
            db_, query.data(), dim_, top_k,
            ts_from.empty() ? nullptr : ts_from.c_str(),
            ts_to.empty() ? nullptr : ts_to.c_str(),
            candidate_k,
            &err
        );

        if (!result) {
            std::string error_msg = err ? err : "Search failed";
            if (err) free(err);
            throw Error::New(env, error_msg);
        }

        if (err) free(err);

        int count = logosdb_result_count(result);
        Array results = Array::New(env, count);

        for (int i = 0; i < count; i++) {
            results.Set(i, SearchHitToJsObject(env, result, i));
        }

        logosdb_result_free(result);

        info.GetReturnValue().Set(results);
    }

    void Update(const CallbackInfo& info) {
        Env env = info.Env();

        if (info.Length() < 2 || !info[0].IsNumber() || !info[1].IsArray()) {
            throw TypeError::New(env, "Expected (id, embedding, [text], [timestamp])");
        }

        uint64_t id = static_cast<uint64_t>(info[0].As<Number>().Int64Value());

        Array embeddingArr = info[1].As<Array>();
        std::vector<float> embedding = JsArrayToFloatVector(embeddingArr);

        if ((int)embedding.size() != dim_) {
            throw Error::New(env, "Embedding dimension mismatch");
        }

        const char* text = nullptr;
        const char* timestamp = nullptr;

        if (info.Length() > 2 && info[2].IsString()) {
            text_str_ = info[2].As<String>().Utf8Value();
            text = text_str_.c_str();
        }

        if (info.Length() > 3 && info[3].IsString()) {
            ts_str_ = info[3].As<String>().Utf8Value();
            timestamp = ts_str_.c_str();
        }

        char* err = nullptr;
        uint64_t new_id = logosdb_update(db_, id, embedding.data(), dim_, text, timestamp, &err);

        if (new_id == UINT64_MAX) {
            std::string error_msg = err ? err : "Update failed";
            if (err) free(err);
            throw Error::New(env, error_msg);
        }

        if (err) free(err);

        info.GetReturnValue().Set(Number::New(env, static_cast<double>(new_id)));
    }

    void Delete(const CallbackInfo& info) {
        Env env = info.Env();

        if (info.Length() < 1 || !info[0].IsNumber()) {
            throw TypeError::New(env, "ID (number) expected");
        }

        uint64_t id = static_cast<uint64_t>(info[0].As<Number>().Int64Value());

        char* err = nullptr;
        int rc = logosdb_delete(db_, id, &err);

        if (rc != 0) {
            std::string error_msg = err ? err : "Delete failed";
            if (err) free(err);
            throw Error::New(env, error_msg);
        }

        if (err) free(err);
    }

    void Count(const CallbackInfo& info) {
        Env env = info.Env();
        size_t count = logosdb_count(db_);
        info.GetReturnValue().Set(Number::New(env, static_cast<double>(count)));
    }

    void CountLive(const CallbackInfo& info) {
        Env env = info.Env();
        size_t count = logosdb_count_live(db_);
        info.GetReturnValue().Set(Number::New(env, static_cast<double>(count)));
    }

    void Dim(const CallbackInfo& info) {
        Env env = info.Env();
        info.GetReturnValue().Set(Number::New(env, dim_));
    }

    void Close(const CallbackInfo& info) {
        if (db_) {
            logosdb_close(db_);
            db_ = nullptr;
        }
    }
};

// Constants
static void DefineConstants(Env env, Object exports) {
    exports.Set("DIST_IP", Number::New(env, LOGOSDB_DIST_IP));
    exports.Set("DIST_COSINE", Number::New(env, LOGOSDB_DIST_COSINE));
    exports.Set("DIST_L2", Number::New(env, LOGOSDB_DIST_L2));
    exports.Set("VERSION", String::New(env, LOGOSDB_VERSION_STRING));
}

// Module initialization
static Object Init(Env env, Object exports) {
    DBWrapper::Init(env, exports);
    DefineConstants(env, exports);
    return exports;
}

NODE_API_MODULE(logosdb, Init)
