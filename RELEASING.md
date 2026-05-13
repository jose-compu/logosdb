# Releasing `logosdb` to PyPI

This repository publishes binary wheels (built with
[`cibuildwheel`](https://cibuildwheel.pypa.io/)) and a source distribution
to [PyPI](https://pypi.org/project/logosdb/) via GitHub Actions. Authentication
uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC), so no API tokens are stored anywhere.

## One-time setup

You need to do this **before** the first tag push succeeds. It takes ~2 min.

### 1. Reserve the name on PyPI

https://pypi.org/manage/account/publishing/ → *"Add a new pending publisher"*:

| Field              | Value         |
|--------------------|---------------|
| PyPI Project Name  | `logosdb`     |
| Owner              | `jose-compu`  |
| Repository name    | `logosdb`     |
| Workflow name      | `publish.yml` |
| Environment name   | `pypi`        |

Environment name may be left blank (matches *Any*); using `pypi` lets you
add GitHub-side protection rules that scope only to release uploads.

### 2. (Optional) Create the `pypi` GitHub Environment

`Settings → Environments → New environment → pypi`. Enabling
*"Required reviewers"* is highly recommended so a human has to approve every
upload. GitHub will auto-create the environment on first use even without
this, but without a protection rule the upload happens unattended.

That's it — the workflow already references the environment by name.

## Cutting a release

```bash
# 1. Bump the version in all three files. They MUST match.
#    - pyproject.toml            → [project].version
#    - include/logosdb/logosdb.h → LOGOSDB_VERSION_* and LOGOSDB_VERSION_STRING
#    - CMakeLists.txt            → project(logosdb VERSION ...)

# 2. Update the CHANGELOG entry with the release date and notes.

git add pyproject.toml include/logosdb/logosdb.h CMakeLists.txt CHANGELOG
git commit -m "Release v0.2.0"
git push origin main

# 3. Tag and push.
git tag -a v0.2.0 -m "logosdb 0.2.0"
git push origin v0.2.0
```

The tag push triggers `.github/workflows/publish.yml`:

1. **`build_wheels`** runs on 4 runners (Linux x86_64, Linux arm64, macOS x86_64,
   macOS arm64), invoking `cibuildwheel` to produce one wheel per supported
   CPython version (3.9–3.13). Each wheel is tested against the pytest suite
   inside the isolated build environment.
2. **`build_sdist`** builds the source distribution.
3. **`publish_pypi`** waits for reviewer approval (if configured) and then
   uploads the wheels + sdist to PyPI.

If a build or test fails, nothing is uploaded — the job simply stops before
the publish step. Because PyPI never allows re-uploading the same version,
always bump the version number when retrying (`0.2.0` → `0.2.1`) rather than
deleting and re-pushing a tag.

## Verifying a release after it lands

```bash
python -m venv /tmp/v && . /tmp/v/bin/activate
pip install --upgrade logosdb
python -c "
import logosdb, numpy as np
db = logosdb.DB('/tmp/smoke', dim=8)
db.put(np.ones(8, dtype='f4'))
print('ok:', db.count(), logosdb.__version__)
"
```

## Manual dry-run (no tag)

To exercise the full wheel + sdist matrix without publishing anything:

```
Actions → Publish → Run workflow → target: dry_run
```

This builds every wheel and the sdist, uploads them as workflow artifacts
(downloadable from the run summary), and stops. No PyPI upload.

If you want to publish from the `main` branch without tagging (e.g. a
hot-fix where tagging hasn't happened yet), choose `target: pypi` instead.
It still goes through the `pypi` environment's protection rules.

## Supported platforms

- Linux x86_64  (manylinux2014-compatible)
- Linux aarch64 (manylinux2014-compatible)
- macOS 11+ x86_64
- macOS 11+ arm64
- CPython 3.9, 3.10, 3.11, 3.12, 3.13

Windows is not yet supported; see
[issue #10](https://github.com/jose-compu/logosdb/issues/10).

## What an sdist user gets

PyPI users on unsupported platforms (or with `--no-binary logosdb`) will fall
back to the sdist, which requires:

- CMake ≥ 3.15
- A C++17 compiler
- Python development headers

The sdist includes `third_party/hnswlib/` so no submodules are needed.

## Manual publish cheat sheet

Use this when GitHub Actions is unavailable, you are cutting an out-of-band release, or you simply want to push artifacts from your workstation. **Always** finish a release commit first (versions bumped, `CHANGELOG` updated, tag pushed) before running any of the commands below.

Replace `${VERSION}` with the release you are publishing (e.g. `0.9.0`). The repository root is referred to as `${REPO}`:

```bash
REPO=/Users/joseignacio/Documents/GitHub/ai-misc/semantic-sandwich/logosdb
VERSION=0.9.0
```

### Order

Publish in this order so each `^X.Y.Z` dependency resolves on the registry by the time the next package is fetched:

1. **`logosdb`** (native Node addon — `nodejs/`)
2. **`logosdb-mcp-server`** (MCP — `mcp/`)
3. **`n8n-nodes-logosdb`** (n8n — `n8n/`, optional)
4. **`logosdb` on PyPI** (Python wheels + sdist)

### npm: native addon (`logosdb`)

Publish from **`nodejs/`** (not the monorepo root — the root `package.json` is a private workspace that would trip `EPRIVATE`). `prepublishOnly` refreshes `deps/core/` from the canonical sources.

```bash
cd "${REPO}/nodejs"
npm publish
```

Sanity check:

```bash
npm view logosdb version    # should print ${VERSION}
```

### npm: MCP server (`logosdb-mcp-server`)

```bash
cd "${REPO}/mcp"
npm publish
```

Sanity check:

```bash
npm view logosdb-mcp-server version
```

### npm: n8n node (`n8n-nodes-logosdb`, optional)

Only publish when the n8n surface changed; otherwise skip. Requires the new `logosdb` to be live on the registry so `^${VERSION}` resolves.

```bash
cd "${REPO}/n8n"
npm publish
```

### PyPI: wheels + sdist

The GitHub Actions release workflow is preferred (see [Cutting a release](#cutting-a-release)). For a manual upload, build locally and use `twine`:

```bash
cd "${REPO}"
rm -rf dist build
python -m build
python -m twine upload "dist/logosdb-${VERSION}"*
```

`twine` reads credentials from `~/.pypirc` or the `TWINE_USERNAME` / `TWINE_PASSWORD` env vars; for Trusted Publishing prefer the GitHub Actions path instead.

### Post-publish smoke checks

```bash
npm view logosdb version
npm view logosdb-mcp-server version
npm view n8n-nodes-logosdb version

python -m venv /tmp/v && . /tmp/v/bin/activate
pip install --upgrade "logosdb==${VERSION}"
python -c "import logosdb; print(logosdb.__version__)"
```
