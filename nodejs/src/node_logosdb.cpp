/**
 * Node.js N-API bindings for LogosDB
 *
 * Provides JavaScript bindings for the LogosDB C++ API.
 */

#include <logosdb/logosdb.h>

#include <napi.h>

#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

// Helper to convert JS array to float vector
static std::vector<float> JsArrayToFloatVector(const Napi::Array& arr)
{
    std::vector<float> vec;
    vec.reserve(arr.Length());
    for (size_t i = 0; i < arr.Length(); i++)
    {
        vec.push_back(arr.Get(i).As<Napi::Number>().FloatValue());
    }
    return vec;
}

static std::vector<float> JsValueToFloatVector(Napi::Env env, const Napi::Value& val)
{
    if (val.IsArray())
        return JsArrayToFloatVector(val.As<Napi::Array>());
    if (val.IsTypedArray() && val.As<Napi::TypedArray>().TypedArrayType() == napi_float32_array)
    {
        Napi::Float32Array fa = val.As<Napi::Float32Array>();
        const size_t len = fa.ElementLength();
        std::vector<float> vec(len);
        std::memcpy(vec.data(), fa.Data(), len * sizeof(float));
        return vec;
    }
    throw Napi::TypeError::New(env, "Expected Float32Array or number[]");
}

static bool EmbeddingValueIsValid(const Napi::Value& val)
{
    return val.IsArray() ||
           (val.IsTypedArray() && val.As<Napi::TypedArray>().TypedArrayType() == napi_float32_array);
}

// Helper to convert SearchHit to JS object
static Napi::Object
SearchHitToJsObject(Napi::Env env, const logosdb_search_result_t* result, int index)
{
    Napi::Object obj = Napi::Object::New(env);

    uint64_t id = logosdb_result_id(result, index);
    float score = logosdb_result_score(result, index);
    const char* text = logosdb_result_text(result, index);
    const char* timestamp = logosdb_result_timestamp(result, index);

    obj.Set("id", Napi::Number::New(env, static_cast<double>(id)));
    obj.Set("score", Napi::Number::New(env, score));
    if (text)
    {
        obj.Set("text", Napi::String::New(env, text));
    }
    else
    {
        obj.Set("text", env.Null());
    }
    if (timestamp)
    {
        obj.Set("timestamp", Napi::String::New(env, timestamp));
    }
    else
    {
        obj.Set("timestamp", env.Null());
    }

    return obj;
}

static Napi::Array SearchResultToJsArray(Napi::Env env, logosdb_search_result_t* result)
{
    const int count = logosdb_result_count(result);
    Napi::Array results = Napi::Array::New(env, count);
    for (int i = 0; i < count; i++)
        results.Set(i, SearchHitToJsObject(env, result, i));
    return results;
}

class SearchAsyncWorker : public Napi::AsyncWorker
{
  public:
    SearchAsyncWorker(Napi::Function& callback,
                      Napi::Promise::Deferred deferred,
                      logosdb_t* db,
                      std::vector<float> query,
                      int dim,
                      int top_k)
        : Napi::AsyncWorker(callback),
          deferred_(deferred),
          db_(db),
          query_(std::move(query)),
          dim_(dim),
          top_k_(top_k)
    {
    }

    void Execute() override
    {
        char* err = nullptr;
        result_ = logosdb_search(db_, query_.data(), dim_, top_k_, &err);
        if (!result_)
        {
            error_msg_ = err ? err : "Search failed";
            if (err)
                free(err);
        }
        else if (err)
        {
            free(err);
        }
    }

    void OnOK() override
    {
        Napi::HandleScope scope(Env());
        if (!result_)
        {
            deferred_.Reject(Napi::Error::New(Env(), error_msg_).Value());
            return;
        }
        Napi::Array hits = SearchResultToJsArray(Env(), result_);
        logosdb_result_free(result_);
        result_ = nullptr;
        deferred_.Resolve(hits);
    }

    void OnError(const Napi::Error& e) override { deferred_.Reject(e.Value()); }

  private:
    Napi::Promise::Deferred deferred_;
    logosdb_t* db_;
    std::vector<float> query_;
    int dim_;
    int top_k_;
    logosdb_search_result_t* result_ = nullptr;
    std::string error_msg_;
};

class PutBatchAsyncWorker : public Napi::AsyncWorker
{
  public:
    PutBatchAsyncWorker(Napi::Function& callback,
                        Napi::Promise::Deferred deferred,
                        logosdb_t* db,
                        int dim,
                        std::vector<float> embeddings,
                        int n,
                        std::vector<const char*> text_ptrs,
                        std::vector<const char*> ts_ptrs,
                        bool has_texts,
                        bool has_timestamps,
                        std::vector<std::string> text_storage,
                        std::vector<std::string> ts_storage)
        : Napi::AsyncWorker(callback),
          deferred_(deferred),
          db_(db),
          dim_(dim),
          embeddings_(std::move(embeddings)),
          n_(n),
          text_ptrs_(std::move(text_ptrs)),
          ts_ptrs_(std::move(ts_ptrs)),
          has_texts_(has_texts),
          has_timestamps_(has_timestamps),
          text_storage_(std::move(text_storage)),
          ts_storage_(std::move(ts_storage))
    {
    }

    void Execute() override
    {
        out_ids_.resize(static_cast<size_t>(n_));
        char* err = nullptr;
        const int rc = logosdb_put_batch(db_,
                                         embeddings_.data(),
                                         n_,
                                         dim_,
                                         has_texts_ ? text_ptrs_.data() : nullptr,
                                         has_timestamps_ ? ts_ptrs_.data() : nullptr,
                                         out_ids_.data(),
                                         &err);
        if (rc != 0)
        {
            error_msg_ = err ? err : "put_batch failed";
            if (err)
                free(err);
        }
        else if (err)
        {
            free(err);
        }
    }

    void OnOK() override
    {
        Napi::HandleScope scope(Env());
        if (!error_msg_.empty())
        {
            deferred_.Reject(Napi::Error::New(Env(), error_msg_).Value());
            return;
        }
        Napi::Array result = Napi::Array::New(Env(), static_cast<uint32_t>(n_));
        for (int i = 0; i < n_; ++i)
            result.Set(static_cast<uint32_t>(i),
                       Napi::Number::New(Env(), static_cast<double>(out_ids_[static_cast<size_t>(i)])));
        deferred_.Resolve(result);
    }

    void OnError(const Napi::Error& e) override { deferred_.Reject(e.Value()); }

  private:
    Napi::Promise::Deferred deferred_;
    logosdb_t* db_;
    int dim_;
    int n_;
    std::vector<float> embeddings_;
    std::vector<const char*> text_ptrs_;
    std::vector<const char*> ts_ptrs_;
    bool has_texts_;
    bool has_timestamps_;
    std::vector<std::string> text_storage_;
    std::vector<std::string> ts_storage_;
    std::vector<uint64_t> out_ids_;
    std::string error_msg_;
};

class SearchBatchAsyncWorker : public Napi::AsyncWorker
{
  public:
    SearchBatchAsyncWorker(Napi::Function& callback,
                           Napi::Promise::Deferred deferred,
                           logosdb_t* db,
                           int dim,
                           int n,
                           int k,
                           std::vector<float> flat,
                           logosdb_id_filter_t filter,
                           std::vector<uint64_t> allow_store,
                           std::vector<uint64_t> deny_store,
                           std::string ts_from,
                           std::string ts_to)
        : Napi::AsyncWorker(callback),
          deferred_(deferred),
          db_(db),
          dim_(dim),
          n_(n),
          k_(k),
          flat_(std::move(flat)),
          filter_(filter),
          allow_store_(std::move(allow_store)),
          deny_store_(std::move(deny_store)),
          ts_from_(std::move(ts_from)),
          ts_to_(std::move(ts_to))
    {
        if (!allow_store_.empty())
        {
            filter_.allow = allow_store_.data();
            filter_.n_allow = allow_store_.size();
        }
        if (!deny_store_.empty())
        {
            filter_.deny = deny_store_.data();
            filter_.n_deny = deny_store_.size();
        }
        filter_.ts_from = ts_from_.empty() ? nullptr : ts_from_.c_str();
        filter_.ts_to = ts_to_.empty() ? nullptr : ts_to_.c_str();
    }

    void Execute() override
    {
        out_ids_.resize(static_cast<size_t>(n_) * static_cast<size_t>(k_));
        out_scores_.resize(static_cast<size_t>(n_) * static_cast<size_t>(k_));
        char* err = nullptr;
        if (logosdb_search_batch(db_,
                                 flat_.data(),
                                 n_,
                                 dim_,
                                 k_,
                                 &filter_,
                                 out_ids_.data(),
                                 out_scores_.data(),
                                 &err) != 0)
        {
            error_msg_ = err ? err : "search_batch failed";
            if (err)
                free(err);
        }
        else if (err)
        {
            free(err);
        }
    }

    void OnOK() override
    {
        Napi::HandleScope scope(Env());
        if (!error_msg_.empty())
        {
            deferred_.Reject(Napi::Error::New(Env(), error_msg_).Value());
            return;
        }
        Napi::Array idRows = Napi::Array::New(Env(), n_);
        Napi::Array scoreRows = Napi::Array::New(Env(), n_);
        for (int qi = 0; qi < n_; ++qi)
        {
            Napi::Array idRow = Napi::Array::New(Env(), k_);
            Napi::Array scoreRow = Napi::Array::New(Env(), k_);
            for (int kj = 0; kj < k_; ++kj)
            {
                const size_t idx =
                    static_cast<size_t>(qi) * static_cast<size_t>(k_) + static_cast<size_t>(kj);
                const uint64_t id = out_ids_[idx];
                const float sc = out_scores_[idx];
                if (id == UINT64_MAX)
                    idRow.Set(kj, Env().Null());
                else
                    idRow.Set(kj, Napi::Number::New(Env(), static_cast<double>(id)));
                scoreRow.Set(kj, Napi::Number::New(Env(), sc));
            }
            idRows.Set(qi, idRow);
            scoreRows.Set(qi, scoreRow);
        }
        Napi::Object out = Napi::Object::New(Env());
        out.Set("ids", idRows);
        out.Set("scores", scoreRows);
        deferred_.Resolve(out);
    }

    void OnError(const Napi::Error& e) override { deferred_.Reject(e.Value()); }

  private:
    Napi::Promise::Deferred deferred_;
    logosdb_t* db_;
    int dim_;
    int n_;
    int k_;
    std::vector<float> flat_;
    logosdb_id_filter_t filter_;
    std::vector<uint64_t> allow_store_;
    std::vector<uint64_t> deny_store_;
    std::string ts_from_;
    std::string ts_to_;
    std::vector<uint64_t> out_ids_;
    std::vector<float> out_scores_;
    std::string error_msg_;
};

static void FillIdFilterFromJs(Napi::Object opts,
                               logosdb_id_filter_t* f,
                               std::vector<uint64_t>& allow_store,
                               std::vector<uint64_t>& deny_store,
                               std::string& ts_from,
                               std::string& ts_to)
{
    logosdb_id_filter_init(f);
    if (opts.Has("allow"))
    {
        Napi::Value av = opts.Get("allow");
        if (av.IsArray())
        {
            Napi::Array arr = av.As<Napi::Array>();
            allow_store.reserve(arr.Length());
            for (uint32_t i = 0; i < arr.Length(); ++i)
                allow_store.push_back(
                    static_cast<uint64_t>(arr.Get(i).As<Napi::Number>().Int64Value()));
            f->allow = allow_store.data();
            f->n_allow = allow_store.size();
        }
    }
    if (opts.Has("deny"))
    {
        Napi::Value dv = opts.Get("deny");
        if (dv.IsArray())
        {
            Napi::Array arr = dv.As<Napi::Array>();
            deny_store.reserve(arr.Length());
            for (uint32_t i = 0; i < arr.Length(); ++i)
                deny_store.push_back(
                    static_cast<uint64_t>(arr.Get(i).As<Napi::Number>().Int64Value()));
            f->deny = deny_store.data();
            f->n_deny = deny_store.size();
        }
    }
    if (opts.Has("minScore") && !opts.Get("minScore").IsNull() &&
        !opts.Get("minScore").IsUndefined())
        f->min_score = opts.Get("minScore").As<Napi::Number>().FloatValue();
    if (opts.Has("tsFrom"))
        ts_from = opts.Get("tsFrom").As<Napi::String>().Utf8Value();
    if (opts.Has("tsTo"))
        ts_to = opts.Get("tsTo").As<Napi::String>().Utf8Value();
    f->ts_from = ts_from.empty() ? nullptr : ts_from.c_str();
    f->ts_to = ts_to.empty() ? nullptr : ts_to.c_str();
    if (opts.Has("skipEmptyTs"))
        f->skip_empty_ts = opts.Get("skipEmptyTs").As<Napi::Boolean>().Value() ? 1 : 0;
}

// DB wrapper class
class DBWrapper : public Napi::ObjectWrap<DBWrapper>
{
  public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports)
    {
        Napi::Function func =
            DefineClass(env,
                        "DB",
                        {
                            InstanceMethod("put", &DBWrapper::Put),
                            InstanceMethod("putBatch", &DBWrapper::PutBatch),
                            InstanceMethod("get", &DBWrapper::Get),
                            InstanceMethod("search", &DBWrapper::Search),
                            InstanceMethod("searchFiltered", &DBWrapper::SearchFiltered),
                            InstanceMethod("searchBatch", &DBWrapper::SearchBatch),
                            InstanceMethod("getMany", &DBWrapper::GetMany),
                            InstanceMethod("searchTsRange", &DBWrapper::SearchTsRange),
                            InstanceMethod("searchAsync", &DBWrapper::SearchAsync),
                            InstanceMethod("putBatchAsync", &DBWrapper::PutBatchAsync),
                            InstanceMethod("searchBatchAsync", &DBWrapper::SearchBatchAsync),
                            InstanceMethod("indexCapacity", &DBWrapper::IndexCapacity),
                            InstanceMethod("update", &DBWrapper::Update),
                            InstanceMethod("delete", &DBWrapper::Delete),
                            InstanceMethod("count", &DBWrapper::Count),
                            InstanceMethod("countLive", &DBWrapper::CountLive),
                            InstanceMethod("liveIds", &DBWrapper::LiveIds),
                            InstanceMethod("rawVectors", &DBWrapper::RawVectors),
                            InstanceMethod("exportNdjson", &DBWrapper::ExportNdjson),
                            InstanceMethod("dim", &DBWrapper::Dim),
                            InstanceMethod("sync", &DBWrapper::Sync),
                            InstanceMethod("stats", &DBWrapper::Stats),
                            InstanceMethod("close", &DBWrapper::Close),
                        });

        exports.Set("DB", func);
        return exports;
    }

    DBWrapper(const Napi::CallbackInfo& info)
        : Napi::ObjectWrap<DBWrapper>(info)
    {
        Napi::Env env = info.Env();

        if (info.Length() < 1 || !info[0].IsString())
        {
            throw Napi::TypeError::New(env, "Path (string) expected as first argument");
        }

        std::string path = info[0].As<Napi::String>().Utf8Value();
        int dim = 128;  // default
        size_t max_elements = 1000000;
        size_t max_capacity = 0;
        int ef_construction = 200;
        int M = 16;
        int ef_search = 50;
        int distance = 0;  // LOGOSDB_DIST_IP
        int dtype = LOGOSDB_DTYPE_FLOAT32;

        // Parse options object if provided
        if (info.Length() > 1 && info[1].IsObject())
        {
            Napi::Object options = info[1].As<Napi::Object>();

            if (options.Has("dim"))
            {
                dim = options.Get("dim").As<Napi::Number>().Int32Value();
            }
            if (options.Has("maxElements"))
            {
                max_elements =
                    static_cast<size_t>(options.Get("maxElements").As<Napi::Number>().Int64Value());
            }
            if (options.Has("maxCapacity"))
            {
                max_capacity =
                    static_cast<size_t>(options.Get("maxCapacity").As<Napi::Number>().Int64Value());
            }
            if (options.Has("efConstruction"))
            {
                ef_construction = options.Get("efConstruction").As<Napi::Number>().Int32Value();
            }
            if (options.Has("M"))
            {
                M = options.Get("M").As<Napi::Number>().Int32Value();
            }
            if (options.Has("efSearch"))
            {
                ef_search = options.Get("efSearch").As<Napi::Number>().Int32Value();
            }
            if (options.Has("distance"))
            {
                distance = options.Get("distance").As<Napi::Number>().Int32Value();
            }
            if (options.Has("dtype"))
            {
                dtype = options.Get("dtype").As<Napi::Number>().Int32Value();
            }
        }

        // Create options
        logosdb_options_t* opts = logosdb_options_create();
        logosdb_options_set_dim(opts, dim);
        logosdb_options_set_max_elements(opts, max_elements);
        logosdb_options_set_max_capacity(opts, max_capacity);
        logosdb_options_set_ef_construction(opts, ef_construction);
        logosdb_options_set_M(opts, M);
        logosdb_options_set_ef_search(opts, ef_search);
        logosdb_options_set_distance(opts, distance);
        if (logosdb_options_set_dtype(opts, dtype) != 0)
        {
            logosdb_options_destroy(opts);
            throw Napi::TypeError::New(
                env, "Invalid dtype (use DTYPE_FLOAT32, DTYPE_FLOAT16, or DTYPE_INT8)");
        }

        char* err = nullptr;
        db_ = logosdb_open(path.c_str(), opts, &err);
        logosdb_options_destroy(opts);

        if (!db_)
        {
            std::string error_msg = err ? err : "Unknown error opening database";
            if (err)
                free(err);
            throw Napi::Error::New(env, error_msg);
        }

        dim_ = dim;
    }

    ~DBWrapper()
    {
        if (db_)
        {
            logosdb_close(db_);
            db_ = nullptr;
        }
    }

  private:
    logosdb_t* db_ = nullptr;
    int dim_ = 0;

    Napi::Value Put(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();

        if (info.Length() < 1 || !EmbeddingValueIsValid(info[0]))
        {
            throw Napi::TypeError::New(env, "Embedding (Float32Array or number[]) expected");
        }

        std::vector<float> embedding = JsValueToFloatVector(env, info[0]);

        if ((int)embedding.size() != dim_)
        {
            throw Napi::Error::New(env, "Embedding dimension mismatch");
        }

        const char* text = nullptr;
        const char* timestamp = nullptr;

        if (info.Length() > 1 && info[1].IsString())
        {
            text_str_ = info[1].As<Napi::String>().Utf8Value();
            text = text_str_.c_str();
        }

        if (info.Length() > 2 && info[2].IsString())
        {
            ts_str_ = info[2].As<Napi::String>().Utf8Value();
            timestamp = ts_str_.c_str();
        }

        char* err = nullptr;
        uint64_t id = logosdb_put(db_, embedding.data(), dim_, text, timestamp, &err);

        if (id == UINT64_MAX)
        {
            std::string error_msg = err ? err : "Failed to insert";
            if (err)
                free(err);
            throw Napi::Error::New(env, error_msg);
        }

        if (err)
            free(err);

        return Napi::Number::New(env, static_cast<double>(id));
    }

    // Storage for string references during put calls
    std::string text_str_;
    std::string ts_str_;

    // putBatch(embeddings: number[], n: number, texts?: (string|null)[], timestamps?:
    // (string|null)[]) Inserts n vectors in a single WAL fsync — O(N) faster than N individual
    // put() calls. embeddings must be a flat array of n*dim numbers (row-major). Returns a JS array
    // of n assigned row IDs.
    Napi::Value PutBatch(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();

        if (info.Length() < 2 || !EmbeddingValueIsValid(info[0]) || !info[1].IsNumber())
        {
            throw Napi::TypeError::New(env,
                                       "putBatch(embeddings: Float32Array|number[], n: number, ...)");
        }

        int n = info[1].As<Napi::Number>().Int32Value();
        if (n <= 0)
        {
            return Napi::Array::New(env, 0);
        }

        std::vector<float> all = JsValueToFloatVector(env, info[0]);
        if (static_cast<int>(all.size()) < n * dim_)
        {
            throw Napi::RangeError::New(env, "embeddings length must be >= n * dim");
        }
        std::vector<float> embeddings(all.begin(), all.begin() + static_cast<size_t>(n) * dim_);

        std::vector<std::string> text_storage;
        std::vector<const char*> text_ptrs;
        bool has_texts = info.Length() > 2 && info[2].IsArray();
        if (has_texts)
        {
            Napi::Array textsArr = info[2].As<Napi::Array>();
            text_storage.resize(n);
            text_ptrs.resize(n, nullptr);
            for (int i = 0; i < n; i++)
            {
                Napi::Value v = textsArr.Get(static_cast<uint32_t>(i));
                if (v.IsString())
                {
                    text_storage[i] = v.As<Napi::String>().Utf8Value();
                    text_ptrs[i] = text_storage[i].c_str();
                }
            }
        }

        std::vector<std::string> ts_storage;
        std::vector<const char*> ts_ptrs;
        bool has_timestamps = info.Length() > 3 && info[3].IsArray();
        if (has_timestamps)
        {
            Napi::Array tsArr = info[3].As<Napi::Array>();
            ts_storage.resize(n);
            ts_ptrs.resize(n, nullptr);
            for (int i = 0; i < n; i++)
            {
                Napi::Value v = tsArr.Get(static_cast<uint32_t>(i));
                if (v.IsString())
                {
                    ts_storage[i] = v.As<Napi::String>().Utf8Value();
                    ts_ptrs[i] = ts_storage[i].c_str();
                }
            }
        }

        std::vector<uint64_t> out_ids(n);
        char* err = nullptr;
        int rc = logosdb_put_batch(db_,
                                   embeddings.data(),
                                   n,
                                   dim_,
                                   has_texts ? text_ptrs.data() : nullptr,
                                   has_timestamps ? ts_ptrs.data() : nullptr,
                                   out_ids.data(),
                                   &err);

        if (rc != 0)
        {
            std::string error_msg = err ? err : "put_batch failed";
            if (err)
                free(err);
            throw Napi::Error::New(env, error_msg);
        }
        if (err)
            free(err);

        Napi::Array result = Napi::Array::New(env, static_cast<size_t>(n));
        for (int i = 0; i < n; i++)
        {
            result.Set(static_cast<uint32_t>(i),
                       Napi::Number::New(env, static_cast<double>(out_ids[i])));
        }
        return result;
    }

    Napi::Value Get(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();

        if (info.Length() < 1 || !info[0].IsNumber())
        {
            throw Napi::TypeError::New(env, "ID (number) expected");
        }

        uint64_t id = static_cast<uint64_t>(info[0].As<Napi::Number>().Int64Value());

        char* err = nullptr;
        logosdb_row_t* row = logosdb_get(db_, id, &err);
        if (!row)
        {
            std::string error_msg = err ? err : "Get failed";
            if (err)
                free(err);
            throw Napi::Error::New(env, error_msg);
        }
        if (err)
            free(err);

        Napi::Object obj = Napi::Object::New(env);
        obj.Set("id", Napi::Number::New(env, static_cast<double>(logosdb_row_id(row))));
        int d = logosdb_row_dim(row);
        const float* emb = logosdb_row_embedding(row);
        Napi::Float32Array embArr = Napi::Float32Array::New(env, static_cast<size_t>(d));
        if (emb && d > 0)
            std::memcpy(embArr.Data(), emb, static_cast<size_t>(d) * sizeof(float));
        obj.Set("embedding", embArr);
        const char* text = logosdb_row_text(row);
        if (text)
        {
            obj.Set("text", Napi::String::New(env, text));
        }
        else
        {
            obj.Set("text", env.Null());
        }
        const char* ts = logosdb_row_timestamp(row);
        if (ts)
        {
            obj.Set("timestamp", Napi::String::New(env, ts));
        }
        else
        {
            obj.Set("timestamp", env.Null());
        }
        logosdb_row_free(row);
        return obj;
    }

    Napi::Value Search(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();

        if (info.Length() < 1 || !EmbeddingValueIsValid(info[0]))
        {
            throw Napi::TypeError::New(env, "Query (Float32Array or number[]) expected");
        }

        std::vector<float> query = JsValueToFloatVector(env, info[0]);

        if ((int)query.size() != dim_)
        {
            throw Napi::Error::New(env, "Query dimension mismatch");
        }

        int top_k = 10;
        if (info.Length() > 1 && info[1].IsNumber())
        {
            top_k = info[1].As<Napi::Number>().Int32Value();
        }

        char* err = nullptr;
        logosdb_search_result_t* result = logosdb_search(db_, query.data(), dim_, top_k, &err);

        if (!result)
        {
            std::string error_msg = err ? err : "Search failed";
            if (err)
                free(err);
            throw Napi::Error::New(env, error_msg);
        }

        if (err)
            free(err);

        int count = logosdb_result_count(result);
        Napi::Array results = Napi::Array::New(env, count);

        for (int i = 0; i < count; i++)
        {
            results.Set(i, SearchHitToJsObject(env, result, i));
        }

        logosdb_result_free(result);

        return results;
    }

    Napi::Value SearchFiltered(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();
        if (info.Length() < 2 || !EmbeddingValueIsValid(info[0]) || !info[1].IsObject())
            throw Napi::TypeError::New(env, "Expected (query, options)");
        std::vector<float> query = JsValueToFloatVector(env, info[0]);
        if ((int)query.size() != dim_)
            throw Napi::Error::New(env, "Query dimension mismatch");
        Napi::Object opts = info[1].As<Napi::Object>();
        int top_k = opts.Has("k") ? opts.Get("k").As<Napi::Number>().Int32Value() : 10;
        logosdb_id_filter_t f;
        std::vector<uint64_t> allow_store, deny_store;
        std::string ts_from, ts_to;
        FillIdFilterFromJs(opts, &f, allow_store, deny_store, ts_from, ts_to);
        char* err = nullptr;
        logosdb_search_result_t* result =
            logosdb_search_filtered(db_, query.data(), dim_, top_k, &f, &err);
        if (!result)
        {
            std::string msg = err ? err : "Search failed";
            if (err)
                free(err);
            throw Napi::Error::New(env, msg);
        }
        int count = logosdb_result_count(result);
        Napi::Array results = Napi::Array::New(env, count);
        for (int i = 0; i < count; i++)
            results.Set(i, SearchHitToJsObject(env, result, i));
        logosdb_result_free(result);
        return results;
    }

    Napi::Value SearchBatch(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();
        if (info.Length() < 2 || !info[0].IsArray())
            throw Napi::TypeError::New(env, "Expected (queries, k, [filter])");
        Napi::Array queriesArr = info[0].As<Napi::Array>();
        const int n = static_cast<int>(queriesArr.Length());
        int k = info[1].As<Napi::Number>().Int32Value();
        std::vector<float> flat(static_cast<size_t>(n) * static_cast<size_t>(dim_));
        for (int i = 0; i < n; ++i)
        {
            auto row =
                JsArrayToFloatVector(queriesArr.Get(static_cast<uint32_t>(i)).As<Napi::Array>());
            if ((int)row.size() != dim_)
                throw Napi::Error::New(env, "Query dimension mismatch");
            std::memcpy(flat.data() + static_cast<size_t>(i) * static_cast<size_t>(dim_),
                        row.data(),
                        static_cast<size_t>(dim_) * sizeof(float));
        }
        logosdb_id_filter_t f;
        std::vector<uint64_t> allow_store, deny_store;
        std::string ts_from, ts_to;
        if (info.Length() > 2 && info[2].IsObject())
            FillIdFilterFromJs(
                info[2].As<Napi::Object>(), &f, allow_store, deny_store, ts_from, ts_to);
        else
            logosdb_id_filter_init(&f);
        std::vector<uint64_t> out_ids(static_cast<size_t>(n) * static_cast<size_t>(k));
        std::vector<float> out_scores(static_cast<size_t>(n) * static_cast<size_t>(k));
        char* err = nullptr;
        if (logosdb_search_batch(
                db_, flat.data(), n, dim_, k, &f, out_ids.data(), out_scores.data(), &err) != 0)
        {
            std::string msg = err ? err : "search_batch failed";
            if (err)
                free(err);
            throw Napi::Error::New(env, msg);
        }
        Napi::Array idRows = Napi::Array::New(env, n);
        Napi::Array scoreRows = Napi::Array::New(env, n);
        for (int qi = 0; qi < n; ++qi)
        {
            Napi::Array idRow = Napi::Array::New(env, k);
            Napi::Array scoreRow = Napi::Array::New(env, k);
            for (int kj = 0; kj < k; ++kj)
            {
                size_t idx =
                    static_cast<size_t>(qi) * static_cast<size_t>(k) + static_cast<size_t>(kj);
                uint64_t id = out_ids[idx];
                float sc = out_scores[idx];
                if (id == UINT64_MAX)
                    idRow.Set(kj, env.Null());
                else
                    idRow.Set(kj, Napi::Number::New(env, static_cast<double>(id)));
                scoreRow.Set(kj, Napi::Number::New(env, sc));
            }
            idRows.Set(qi, idRow);
            scoreRows.Set(qi, scoreRow);
        }
        Napi::Object out = Napi::Object::New(env);
        out.Set("ids", idRows);
        out.Set("scores", scoreRows);
        return out;
    }

    Napi::Value GetMany(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();
        if (info.Length() < 1 || !info[0].IsArray())
            throw Napi::TypeError::New(env, "Expected ids array");
        Napi::Array arr = info[0].As<Napi::Array>();
        std::vector<uint64_t> ids;
        ids.reserve(arr.Length());
        for (uint32_t i = 0; i < arr.Length(); ++i)
            ids.push_back(static_cast<uint64_t>(arr.Get(i).As<Napi::Number>().Int64Value()));
        std::vector<float> out(ids.size() * static_cast<size_t>(dim_));
        char* err = nullptr;
        if (logosdb_get_many(db_, ids.data(), ids.size(), out.data(), out.size(), &err) != 0)
        {
            std::string msg = err ? err : "get_many failed";
            if (err)
                free(err);
            throw Napi::Error::New(env, msg);
        }
        Napi::Float32Array flat =
            Napi::Float32Array::New(env, ids.size() * static_cast<size_t>(dim_));
        std::memcpy(flat.Data(), out.data(), out.size() * sizeof(float));
        return flat;
    }

    Napi::Value SearchTsRange(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();

        if (info.Length() < 1 || !EmbeddingValueIsValid(info[0]))
        {
            throw Napi::TypeError::New(env, "Query (Float32Array or number[]) expected");
        }

        std::vector<float> query = JsValueToFloatVector(env, info[0]);

        if ((int)query.size() != dim_)
        {
            throw Napi::Error::New(env, "Query dimension mismatch");
        }

        int top_k = 10;
        std::string ts_from;
        std::string ts_to;
        int candidate_k = 0;
        int skip_empty_ts = 0;

        if (info.Length() > 1 && info[1].IsObject())
        {
            Napi::Object options = info[1].As<Napi::Object>();

            if (options.Has("topK"))
            {
                top_k = options.Get("topK").As<Napi::Number>().Int32Value();
            }
            if (options.Has("tsFrom"))
            {
                ts_from = options.Get("tsFrom").As<Napi::String>().Utf8Value();
            }
            if (options.Has("tsTo"))
            {
                ts_to = options.Get("tsTo").As<Napi::String>().Utf8Value();
            }
            if (options.Has("candidateK"))
            {
                candidate_k = options.Get("candidateK").As<Napi::Number>().Int32Value();
            }
            if (options.Has("skipEmptyTs"))
            {
                skip_empty_ts = options.Get("skipEmptyTs").As<Napi::Boolean>().Value() ? 1 : 0;
            }
        }

        if (candidate_k < top_k)
        {
            candidate_k = top_k * 10;  // Default 10x
        }

        char* err = nullptr;
        logosdb_search_result_t* result =
            logosdb_search_ts_range_ex(db_,
                                    query.data(),
                                    dim_,
                                    top_k,
                                    ts_from.empty() ? nullptr : ts_from.c_str(),
                                    ts_to.empty() ? nullptr : ts_to.c_str(),
                                    candidate_k,
                                    skip_empty_ts,
                                    &err);

        if (!result)
        {
            std::string error_msg = err ? err : "Search failed";
            if (err)
                free(err);
            throw Napi::Error::New(env, error_msg);
        }

        if (err)
            free(err);

        int count = logosdb_result_count(result);
        Napi::Array results = Napi::Array::New(env, count);

        for (int i = 0; i < count; i++)
        {
            results.Set(i, SearchHitToJsObject(env, result, i));
        }

        logosdb_result_free(result);

        return results;
    }

    Napi::Value Update(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();

        if (info.Length() < 2 || !info[0].IsNumber() || !EmbeddingValueIsValid(info[1]))
        {
            throw Napi::TypeError::New(env, "Expected (id, embedding, [text], [timestamp])");
        }

        uint64_t id = static_cast<uint64_t>(info[0].As<Napi::Number>().Int64Value());

        std::vector<float> embedding = JsValueToFloatVector(env, info[1]);

        if ((int)embedding.size() != dim_)
        {
            throw Napi::Error::New(env, "Embedding dimension mismatch");
        }

        const char* text = nullptr;
        const char* timestamp = nullptr;

        if (info.Length() > 2 && info[2].IsString())
        {
            text_str_ = info[2].As<Napi::String>().Utf8Value();
            text = text_str_.c_str();
        }

        if (info.Length() > 3 && info[3].IsString())
        {
            ts_str_ = info[3].As<Napi::String>().Utf8Value();
            timestamp = ts_str_.c_str();
        }

        char* err = nullptr;
        uint64_t new_id = logosdb_update(db_, id, embedding.data(), dim_, text, timestamp, &err);

        if (new_id == UINT64_MAX)
        {
            std::string error_msg = err ? err : "Update failed";
            if (err)
                free(err);
            throw Napi::Error::New(env, error_msg);
        }

        if (err)
            free(err);

        return Napi::Number::New(env, static_cast<double>(new_id));
    }

    Napi::Value Delete(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();

        if (info.Length() < 1 || !info[0].IsNumber())
        {
            throw Napi::TypeError::New(env, "ID (number) expected");
        }

        uint64_t id = static_cast<uint64_t>(info[0].As<Napi::Number>().Int64Value());

        char* err = nullptr;
        int rc = logosdb_delete(db_, id, &err);

        if (rc != 0)
        {
            std::string error_msg = err ? err : "Delete failed";
            if (err)
                free(err);
            throw Napi::Error::New(env, error_msg);
        }

        if (err)
            free(err);
        return env.Undefined();
    }

    Napi::Value IndexCapacity(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();
        return Napi::Number::New(env, static_cast<double>(logosdb_index_capacity(db_)));
    }

    Napi::Value SearchAsync(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();
        if (!db_)
            throw Napi::Error::New(env, "Database is closed");
        if (info.Length() < 1 || !EmbeddingValueIsValid(info[0]))
            throw Napi::TypeError::New(env, "searchAsync(query, topK | options)");
        std::vector<float> query = JsValueToFloatVector(env, info[0]);
        if ((int)query.size() != dim_)
            throw Napi::Error::New(env, "Query dimension mismatch");
        int top_k = 10;
        if (info.Length() > 1)
        {
            if (info[1].IsNumber())
                top_k = info[1].As<Napi::Number>().Int32Value();
            else if (info[1].IsObject() && info[1].As<Napi::Object>().Has("topK"))
                top_k = info[1].As<Napi::Object>().Get("topK").As<Napi::Number>().Int32Value();
        }
        Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
        Napi::Function cb = Napi::Function::New(env, [](const Napi::CallbackInfo&) {});
        auto* worker =
            new SearchAsyncWorker(cb, deferred, db_, std::move(query), dim_, top_k);
        worker->Queue();
        return deferred.Promise();
    }

    Napi::Value PutBatchAsync(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();
        if (!db_)
            throw Napi::Error::New(env, "Database is closed");
        if (info.Length() < 2 || !EmbeddingValueIsValid(info[0]) || !info[1].IsNumber())
            throw Napi::TypeError::New(env, "putBatchAsync(embeddings, n, texts?, timestamps?)");
        const int n = info[1].As<Napi::Number>().Int32Value();
        if (n <= 0)
        {
            Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
            deferred.Resolve(Napi::Array::New(env, 0));
            return deferred.Promise();
        }
        std::vector<float> all = JsValueToFloatVector(env, info[0]);
        if ((int)all.size() < n * dim_)
            throw Napi::RangeError::New(env, "embeddings length must be >= n * dim");
        std::vector<float> embeddings(all.begin(), all.begin() + static_cast<size_t>(n) * dim_);

        std::vector<std::string> text_storage;
        std::vector<const char*> text_ptrs;
        bool has_texts = info.Length() > 2 && info[2].IsArray();
        if (has_texts)
        {
            Napi::Array textsArr = info[2].As<Napi::Array>();
            text_storage.resize(n);
            text_ptrs.resize(n, nullptr);
            for (int i = 0; i < n; i++)
            {
                Napi::Value v = textsArr.Get(static_cast<uint32_t>(i));
                if (v.IsString())
                {
                    text_storage[i] = v.As<Napi::String>().Utf8Value();
                    text_ptrs[i] = text_storage[i].c_str();
                }
            }
        }

        std::vector<std::string> ts_storage;
        std::vector<const char*> ts_ptrs;
        bool has_timestamps = info.Length() > 3 && info[3].IsArray();
        if (has_timestamps)
        {
            Napi::Array tsArr = info[3].As<Napi::Array>();
            ts_storage.resize(n);
            ts_ptrs.resize(n, nullptr);
            for (int i = 0; i < n; i++)
            {
                Napi::Value v = tsArr.Get(static_cast<uint32_t>(i));
                if (v.IsString())
                {
                    ts_storage[i] = v.As<Napi::String>().Utf8Value();
                    ts_ptrs[i] = ts_storage[i].c_str();
                }
            }
        }

        Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
        Napi::Function cb = Napi::Function::New(env, [](const Napi::CallbackInfo&) {});
        auto* worker = new PutBatchAsyncWorker(cb,
                                               deferred,
                                               db_,
                                               dim_,
                                               std::move(embeddings),
                                               n,
                                               std::move(text_ptrs),
                                               std::move(ts_ptrs),
                                               has_texts,
                                               has_timestamps,
                                               std::move(text_storage),
                                               std::move(ts_storage));
        worker->Queue();
        return deferred.Promise();
    }

    Napi::Value SearchBatchAsync(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();
        if (!db_)
            throw Napi::Error::New(env, "Database is closed");
        if (info.Length() < 2 || !info[0].IsArray())
            throw Napi::TypeError::New(env, "searchBatchAsync(queries, k, filter?)");
        Napi::Array queriesArr = info[0].As<Napi::Array>();
        const int n = static_cast<int>(queriesArr.Length());
        const int k = info[1].As<Napi::Number>().Int32Value();
        std::vector<float> flat(static_cast<size_t>(n) * static_cast<size_t>(dim_));
        for (int i = 0; i < n; ++i)
        {
            Napi::Value qv = queriesArr.Get(static_cast<uint32_t>(i));
            std::vector<float> row = JsValueToFloatVector(env, qv);
            if ((int)row.size() != dim_)
                throw Napi::Error::New(env, "Query dimension mismatch");
            std::memcpy(flat.data() + static_cast<size_t>(i) * static_cast<size_t>(dim_),
                        row.data(),
                        static_cast<size_t>(dim_) * sizeof(float));
        }
        logosdb_id_filter_t f;
        std::vector<uint64_t> allow_store, deny_store;
        std::string ts_from, ts_to;
        if (info.Length() > 2 && info[2].IsObject())
            FillIdFilterFromJs(info[2].As<Napi::Object>(), &f, allow_store, deny_store, ts_from, ts_to);
        else
            logosdb_id_filter_init(&f);
        Napi::Promise::Deferred deferred = Napi::Promise::Deferred::New(env);
        Napi::Function cb = Napi::Function::New(env, [](const Napi::CallbackInfo&) {});
        auto* worker = new SearchBatchAsyncWorker(cb,
                                                  deferred,
                                                  db_,
                                                  dim_,
                                                  n,
                                                  k,
                                                  std::move(flat),
                                                  f,
                                                  std::move(allow_store),
                                                  std::move(deny_store),
                                                  std::move(ts_from),
                                                  std::move(ts_to));
        worker->Queue();
        return deferred.Promise();
    }

    Napi::Value Count(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();
        size_t count = logosdb_count(db_);
        return Napi::Number::New(env, static_cast<double>(count));
    }

    Napi::Value CountLive(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();
        size_t count = logosdb_count_live(db_);
        return Napi::Number::New(env, static_cast<double>(count));
    }

    Napi::Value LiveIds(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();
        if (!db_)
        {
            throw Napi::Error::New(env, "Database is closed");
        }

        size_t n_out = 0;
        if (logosdb_live_ids(db_, nullptr, 0, &n_out) != 0 || n_out == 0)
            return Napi::Array::New(env, 0);

        std::vector<uint64_t> ids(n_out);
        if (logosdb_live_ids(db_, ids.data(), ids.size(), &n_out) != 0)
            throw Napi::Error::New(env, "liveIds failed");

        Napi::Array arr = Napi::Array::New(env, n_out);
        for (size_t i = 0; i < n_out; ++i)
            arr.Set(i, Napi::Number::New(env, static_cast<double>(ids[i])));
        return arr;
    }

    Napi::Value RawVectors(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();
        if (!db_)
        {
            throw Napi::Error::New(env, "Database is closed");
        }

        size_t n_rows = 0;
        int dim = 0;
        if (logosdb_raw_vectors_decode(db_, nullptr, 0, &n_rows, &dim) != 0 || n_rows == 0 ||
            dim <= 0)
            return Napi::Float32Array::New(env, 0);

        const size_t nfloats = n_rows * static_cast<size_t>(dim);
        std::vector<float> buf(nfloats);
        size_t nr = 0;
        int d = 0;
        if (logosdb_raw_vectors_decode(db_, buf.data(), buf.size(), &nr, &d) != 0)
            throw Napi::Error::New(env, "rawVectors failed");

        Napi::Float32Array arr = Napi::Float32Array::New(env, nfloats);
        std::memcpy(arr.Data(), buf.data(), nfloats * sizeof(float));
        return arr;
    }

    Napi::Value ExportNdjson(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();
        if (!db_)
        {
            throw Napi::Error::New(env, "Database is closed");
        }
        if (info.Length() < 1 || !info[0].IsString())
        {
            throw Napi::TypeError::New(env, "Path (string) expected");
        }

        std::string out_path = info[0].As<Napi::String>().Utf8Value();
        uint64_t start_id = 0;
        uint64_t end_id_exclusive = 0;
        if (info.Length() > 1 && info[1].IsObject())
        {
            Napi::Object opts = info[1].As<Napi::Object>();
            if (opts.Has("startId"))
                start_id =
                    static_cast<uint64_t>(opts.Get("startId").As<Napi::Number>().Int64Value());
            if (opts.Has("endIdExclusive"))
                end_id_exclusive = static_cast<uint64_t>(
                    opts.Get("endIdExclusive").As<Napi::Number>().Int64Value());
        }

        char* err = nullptr;
        int rc = logosdb_export_ndjson(db_, out_path.c_str(), start_id, end_id_exclusive, &err);
        if (rc != 0)
        {
            std::string error_msg = err ? err : "exportNdjson failed";
            if (err)
                free(err);
            throw Napi::Error::New(env, error_msg);
        }
        if (err)
            free(err);
        return env.Undefined();
    }

    Napi::Value Dim(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();
        return Napi::Number::New(env, dim_);
    }

    Napi::Value Sync(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();
        if (!db_)
        {
            throw Napi::Error::New(env, "Database is closed");
        }
        char* err = nullptr;
        int rc = logosdb_sync(db_, &err);
        if (rc != 0)
        {
            std::string error_msg = err ? err : "sync failed";
            if (err)
                free(err);
            throw Napi::Error::New(env, error_msg);
        }
        if (err)
            free(err);
        return env.Undefined();
    }

    static Napi::Object StatsToJsObject(Napi::Env env, const logosdb_stats_t& s)
    {
        Napi::Object obj = Napi::Object::New(env);
        obj.Set("rows_total", Napi::Number::New(env, static_cast<double>(s.rows_total)));
        obj.Set("rows_live", Napi::Number::New(env, static_cast<double>(s.rows_live)));
        obj.Set("tombstones", Napi::Number::New(env, static_cast<double>(s.tombstones)));
        obj.Set("index_elements", Napi::Number::New(env, static_cast<double>(s.index_elements)));
        obj.Set("wal_pending", Napi::Number::New(env, static_cast<double>(s.wal_pending)));
        obj.Set("distance_metric", Napi::Number::New(env, s.distance_metric));
        obj.Set("storage_dtype", Napi::Number::New(env, s.storage_dtype));
        obj.Set("put_success", Napi::Number::New(env, static_cast<double>(s.put_success)));
        obj.Set("put_failed", Napi::Number::New(env, static_cast<double>(s.put_failed)));
        obj.Set("put_batch_success",
                Napi::Number::New(env, static_cast<double>(s.put_batch_success)));
        obj.Set("put_batch_failed",
                Napi::Number::New(env, static_cast<double>(s.put_batch_failed)));
        obj.Set("search_success", Napi::Number::New(env, static_cast<double>(s.search_success)));
        obj.Set("search_failed", Napi::Number::New(env, static_cast<double>(s.search_failed)));
        obj.Set("search_ts_success",
                Napi::Number::New(env, static_cast<double>(s.search_ts_success)));
        obj.Set("search_ts_failed",
                Napi::Number::New(env, static_cast<double>(s.search_ts_failed)));
        obj.Set("delete_success", Napi::Number::New(env, static_cast<double>(s.delete_success)));
        obj.Set("delete_failed", Napi::Number::New(env, static_cast<double>(s.delete_failed)));
        obj.Set("update_success", Napi::Number::New(env, static_cast<double>(s.update_success)));
        obj.Set("update_failed", Napi::Number::New(env, static_cast<double>(s.update_failed)));
        obj.Set("sync_calls", Napi::Number::New(env, static_cast<double>(s.sync_calls)));
        return obj;
    }

    Napi::Value Stats(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();
        if (!db_)
        {
            throw Napi::Error::New(env, "Database is closed");
        }
        logosdb_stats_t s{};
        logosdb_stats(db_, &s);
        return StatsToJsObject(env, s);
    }

    Napi::Value Close(const Napi::CallbackInfo& info)
    {
        Napi::Env env = info.Env();
        if (db_)
        {
            logosdb_close(db_);
            db_ = nullptr;
        }
        return env.Undefined();
    }
};

// Constants
static void DefineConstants(Napi::Env env, Napi::Object exports)
{
    exports.Set("DIST_IP", Napi::Number::New(env, LOGOSDB_DIST_IP));
    exports.Set("DIST_COSINE", Napi::Number::New(env, LOGOSDB_DIST_COSINE));
    exports.Set("DIST_L2", Napi::Number::New(env, LOGOSDB_DIST_L2));
    exports.Set("DTYPE_FLOAT32", Napi::Number::New(env, LOGOSDB_DTYPE_FLOAT32));
    exports.Set("DTYPE_FLOAT16", Napi::Number::New(env, LOGOSDB_DTYPE_FLOAT16));
    exports.Set("DTYPE_INT8", Napi::Number::New(env, LOGOSDB_DTYPE_INT8));
    exports.Set("VERSION", Napi::String::New(env, LOGOSDB_VERSION_STRING));
}

static Napi::Value Compact(const Napi::CallbackInfo& info)
{
    Napi::Env env = info.Env();
    if (info.Length() < 2 || !info[0].IsString() || !info[1].IsString())
    {
        throw Napi::TypeError::New(env, "Expected compact(src: string, dst: string)");
    }
    std::string src = info[0].As<Napi::String>().Utf8Value();
    std::string dst = info[1].As<Napi::String>().Utf8Value();
    char* err = nullptr;
    int rc = logosdb_compact(src.c_str(), dst.c_str(), &err);
    if (rc != 0)
    {
        std::string error_msg = err ? err : "compact failed";
        if (err)
            free(err);
        throw Napi::Error::New(env, error_msg);
    }
    if (err)
        free(err);
    return env.Undefined();
}

// Module initialization
static Napi::Object Init(Napi::Env env, Napi::Object exports)
{
    DBWrapper::Init(env, exports);
    DefineConstants(env, exports);
    exports.Set("compact", Napi::Function::New(env, Compact, "compact"));
    return exports;
}

NODE_API_MODULE(logosdb, Init)
