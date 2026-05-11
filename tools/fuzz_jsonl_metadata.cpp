/**
 * libFuzzer harness: arbitrary bytes -> nlohmann::json::parse (same family as metadata JSONL rows).
 * Build: cmake -DLOGOSDB_BUILD_FUZZ=ON -DCMAKE_CXX_COMPILER=clang++ ... && cmake --build .
 * Run: ./logosdb-fuzz-jsonl -max_total_time=30
 */

#include <cstddef>
#include <cstdint>
#include <string>

#include <nlohmann/json.hpp>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    if (size > 256 * 1024)
        return 0;

    std::string s(reinterpret_cast<const char*>(data), size);
    try
    {
        auto j = nlohmann::json::parse(s);
        (void)j.is_object();
        (void)j.is_array();
    }
    catch (const nlohmann::json::exception&)
    {
    }
    return 0;
}
