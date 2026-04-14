#include "metadata.h"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>

namespace logosdb {
namespace internal {

// Minimal JSON helpers — no external dependency.

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;
        }
    }
    return out;
}

static std::string json_unescape(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            switch (s[i + 1]) {
                case '"':  out += '"';  ++i; break;
                case '\\': out += '\\'; ++i; break;
                case 'n':  out += '\n'; ++i; break;
                case 'r':  out += '\r'; ++i; break;
                case 't':  out += '\t'; ++i; break;
                default:   out += s[i];
            }
        } else {
            out += s[i];
        }
    }
    return out;
}

static std::string extract_field(const std::string & line, const std::string & key) {
    std::string needle = "\"" + key + "\":\"";
    auto pos = line.find(needle);
    if (pos == std::string::npos) return "";
    pos += needle.size();
    auto end = line.find('"', pos);
    while (end != std::string::npos && end > 0 && line[end - 1] == '\\') {
        end = line.find('"', end + 1);
    }
    if (end == std::string::npos) return "";
    return json_unescape(line.substr(pos, end - pos));
}

MetadataStore::~MetadataStore() { close(); }

bool MetadataStore::open(const std::string & path, std::string & err) {
    close();
    path_ = path;

    fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_APPEND, 0644);
    if (fd_ < 0) {
        err = std::string("open meta: ") + strerror(errno);
        return false;
    }

    std::ifstream in(path);
    if (in.good()) {
        std::string line;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            MetaRow r;
            r.text      = extract_field(line, "text");
            r.timestamp = extract_field(line, "ts");
            rows_.push_back(std::move(r));
        }
    }
    return true;
}

void MetadataStore::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    rows_.clear();
    path_.clear();
}

uint64_t MetadataStore::append(const char * text, const char * timestamp,
                                std::string & err) {
    if (fd_ < 0) { err = "meta not open"; return UINT64_MAX; }

    std::string t  = text      ? json_escape(text)      : "";
    std::string ts = timestamp ? json_escape(timestamp)  : "";
    std::string line = "{\"text\":\"" + t + "\",\"ts\":\"" + ts + "\"}\n";

    ssize_t written = ::write(fd_, line.data(), line.size());
    if (written != (ssize_t)line.size()) {
        err = std::string("write meta: ") + strerror(errno);
        return UINT64_MAX;
    }

    uint64_t id = rows_.size();
    rows_.push_back({text ? text : "", timestamp ? timestamp : ""});
    return id;
}

bool MetadataStore::sync(std::string & err) {
    if (fd_ < 0) { err = "meta not open"; return false; }
    if (::fsync(fd_) != 0) {
        err = std::string("fsync meta: ") + strerror(errno);
        return false;
    }
    return true;
}

} // namespace internal
} // namespace logosdb
