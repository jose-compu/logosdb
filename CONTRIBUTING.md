# Contributing to LogosDB

Thank you for your interest in contributing to LogosDB. This document outlines the workflow, style expectations, and how to get started.

## Quick Start

### Prerequisites

- CMake 3.16+ (3.20+ recommended)
- C++17-capable compiler (GCC 9+, Clang 12+, Apple Clang 13+, MSVC 2019+)
- Python 3.9+ (for Python bindings and tests)
- Git with submodule support
- Node.js 18+ (optional — only for building **logosdb-mcp-server** / Claude Code `.claude/mcp.json` in this repo)

### Building

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/jose-compu/logosdb.git
cd logosdb

# Configure and build
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --parallel

# Or with Ninja (faster)
cmake -DCMAKE_BUILD_TYPE=Release -G Ninja ..
ninja
```

### Claude Code (optional)

This repo’s [`.claude/mcp.json`](.claude/mcp.json) points at **`./mcp/dist/index.js`**. From the repo root:

```bash
npm install
```

That builds the MCP server (`mcp` workspace `prepare` → `tsc`). Use `npm run mcp:build` after MCP TypeScript changes.

The root workspace also lists **`nodejs`** so `logosdb-mcp-server` resolves the **`logosdb`** native package from this repo. After changing C++ under `src/` or `include/`, refresh the vendored copy and rebuild the addon:

```bash
npm run vendor-core -w logosdb
npm run build -w logosdb
```

### Publishing npm packages (`logosdb`, `logosdb-mcp-server`, `n8n-nodes-logosdb`)

Versions are **`nodejs/package.json`** (`logosdb`) and **`mcp/package.json`** (`logosdb-mcp-server`). Keep them aligned for coordinated releases; summarize user-facing changes in repo root **`CHANGELOG`**.

From the **repository root**:

```bash
npm install
```

The root **`package.json`** defines **`workspaces`** for **`mcp`**, **`nodejs`**, and **`n8n`**, so a single install wires **`logosdb-mcp-server`** and **`n8n-nodes-logosdb`** to the local **`logosdb`** package. Use **`npm run npm:verify`** from the repo root (not from **`n8n/`** alone).

```bash
npm run npm:verify
```

Optional tarball inspection (writes to **`/tmp`**):

```bash
npm run npm:pack
```

Dry-run publish (lists packed files; does not upload):

```bash
npm publish -w logosdb --dry-run
npm publish -w logosdb-mcp-server --dry-run
```

**Order:** publish **`logosdb` first**, then **`logosdb-mcp-server`** (and **`n8n-nodes-logosdb`** last), so the registry satisfies each `logosdb@^X.Y.Z` dependency for non-workspace installs.

```bash
cd nodejs && npm publish
cd ../mcp && npm publish
cd ../n8n && npm publish
```

Do **not** run **`npm publish`** from the repo root private workspace package.

Optional: generate N-API binaries for GitHub releases from **`nodejs/`** with **`npm run native:prebuild`** / **`npm run native:prebuild-upload`** (requires a working **`prebuild`** / node-gyp toolchain).

Build targets:
- `logosdb` — static library (`liblogosdb.a`)
- `logosdb-cli` — command-line tool
- `logosdb-bench` — benchmark utility
- `logosdb-test` — unit tests

### Running Tests

```bash
cd build

# Run C++ tests
ctest --output-on-failure

# Run Python tests
pip install ".[test]"
pytest tests/python/
```

### Running the Benchmark

```bash
cd build
./logosdb-bench --dim 2048 --counts 1000,10000,100000
```

## Code Style

We follow these conventions:

- **C/C++**: Google C++ Style Guide (roughly)
  - 4-space indentation
  - `snake_case` for functions/variables
  - `PascalCase` for classes
  - `SCREAMING_SNAKE_CASE` for macros/constants
  - Braces on same line
- **Python**: PEP 8 with 4-space indentation
- **Filenames**: `snake_case.cpp`, `snake_case.h`

**Formatting tools available:**
- Use **`clang-format` 18.x** (see **Before you push** below); run `clang-format -i src/yourfile.cpp` on individual files if you prefer
- Run `clang-tidy src/yourfile.cpp` to check for common issues
- CI enforces formatting with **`clang-format-18 --dry-run -Werror`** (see **`.github/workflows/ci.yml`**)

**Before you push:** CI runs **`clang-format-18`** (Ubuntu package). Apple’s default **`clang-format`** from Xcode is often a **newer** LLVM (e.g. 19–22) and formats the same files **differently**, which makes local “clean” runs still fail in GitHub Actions. Use **18.x** locally so output matches CI.

**macOS (Homebrew):**

```bash
brew install llvm@18
export PATH="/opt/homebrew/opt/llvm@18/bin:$PATH"   # Intel: /usr/local/opt/llvm@18/bin
clang-format --version   # should report 18.x

find src include tests tools \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) | \
  grep -v 'doctest.h' | \
  xargs clang-format --dry-run -Werror
```

Apply fixes in one pass (same `PATH`):

```bash
find src include tests tools \( -name '*.cpp' -o -name '*.h' -o -name '*.hpp' \) | \
  grep -v 'doctest.h' | \
  xargs clang-format -i
```

**Linux:** install **`clang-format-18`** from your distro (CI uses `apt install clang-format-18`) and use the same `find … | xargs clang-format-18 …` commands as in **`.github/workflows/ci.yml`**.

Also run **`npm run lint`** / **`npm run format:check`** in **`mcp/`** and **`n8n/`** when you change those packages, and **`npm run npm:verify`** before a release that touches the Node addon or MCP.

### Style Guidelines

- Keep functions focused and under 50 lines when possible
- Prefer RAII and smart pointers; avoid raw `new/delete`
- Use `const` and `constexpr` aggressively
- Document public API in header files with brief comments
- Internal code: comments explain "why", not "what"

## Branch and PR Workflow

1. **Fork** the repository (or create a branch if you have access)
2. **Branch naming**: `feature/description`, `fix/description`, `docs/description`
3. **Commit messages**: Follow conventional commits style
   - `feat: add batch ingest API`
   - `fix: handle zero-dim vectors gracefully`
   - `docs: update sizing guide with int8 numbers`
4. **Pull Request**: Fill out the PR template checklist
5. **CI**: Ensure all checks pass (build + tests on Linux/macOS)

### PR Checklist

Before submitting:

- [ ] C/C++ formatting passes (`clang-format-18 --dry-run -Werror` on Linux, or **`clang-format` 18.x** from Homebrew `llvm@18` on macOS — see **CONTRIBUTING**)
- [ ] Tests added for new functionality
- [ ] Existing tests pass (`ctest`, `pytest`)
- [ ] Benchmarks run if performance-related
- [ ] CHANGELOG updated for user-facing changes
- [ ] Documentation updated (README, headers, guides)
- [ ] Code compiles without warnings on GCC and Clang

## Development Setup Tips

### CMake Presets (optional)

Create `CMakeUserPresets.json` for local development:

```json
{
  "version": 2,
  "configurePresets": [
    {
      "name": "dev",
      "generator": "Ninja",
      "binaryDir": "build-dev",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "CMAKE_EXPORT_COMPILE_COMMANDS": "ON"
      }
    }
  ]
}
```

### IDE Integration

- **VS Code**: Install CMake Tools and C/C++ extensions
- **CLion**: Open the project root; CMake is auto-detected
- **vim/neovim**: Use `compile_commands.json` from CMake for LSP

### Python Development

```bash
# Editable install for Python work
pip install -e ".[test,examples]"

# Run specific test
pytest tests/python/test_smoke.py -v
```

## Testing Guidelines

### C++ Tests

- Add tests in `tests/` directory
- Use existing test patterns in `tests/test_doctest.cpp` (doctest); long-running cases live in `tests/test_stress.cpp` under the `stress` suite (opt-in via `--test-suite=stress`)
- Tests run via CTest; add with `add_test()` in CMake

### Python Tests

- Use `pytest`
- Tests in `tests/python/`
- Mock external services; keep tests fast and deterministic

### Test Coverage

Aim for:
- New features: unit tests covering edge cases
- Bug fixes: regression test that would have caught the bug
- API changes: tests for C and C++ wrappers

## Documentation

- **Public headers**: Document in Doxygen style (brief, params, returns)
- **README.md**: Update for user-facing changes
- **docs/**: Long-form guides (sizing, RAG patterns, etc.)
- **Changelog**: Add entry under appropriate version in CHANGELOG

## Getting Help

- Open an issue for bugs or feature requests
- Use the issue templates (bug report, feature request)
- For security issues, see [SECURITY.md](SECURITY.md)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
