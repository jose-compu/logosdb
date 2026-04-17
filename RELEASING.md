# Releasing `logosdb` to PyPI

This repository publishes binary wheels (built with
[`cibuildwheel`](https://cibuildwheel.pypa.io/)) and a source distribution
to [PyPI](https://pypi.org/project/logosdb/) via GitHub Actions. Authentication
uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
(OIDC), so no API tokens are stored anywhere.

## One-time setup

You need to do this **before** the first tag push succeeds. It takes ~5 min.

### 1. Reserve the name on TestPyPI and PyPI

- https://test.pypi.org/manage/account/publishing/
- https://pypi.org/manage/account/publishing/

On each site click *"Add a new pending publisher"* and fill in:

| Field                     | Value                                  |
|---------------------------|----------------------------------------|
| PyPI Project Name         | `logosdb`                              |
| Owner                     | `jose-compu`                           |
| Repository name           | `logosdb`                              |
| Workflow name             | `publish.yml`                          |
| Environment name (PyPI)   | `pypi`                                 |
| Environment name (TestPyPI) | `testpypi`                           |

### 2. Create the two GitHub environments

Go to `Settings → Environments → New environment` in the GitHub repo and create
both `testpypi` and `pypi`. For `pypi` it is highly recommended to enable
*"Required reviewers"* so a human has to approve the upload.

That's it — the workflow already references these environments by name.

## Cutting a release

```bash
# 1. Bump the version in both files. They MUST match.
#    - pyproject.toml        → [project].version
#    - include/logosdb/logosdb.h → LOGOSDB_VERSION_* and LOGOSDB_VERSION_STRING

# 2. Update the CHANGELOG entry with the release date and notes.

git add pyproject.toml include/logosdb/logosdb.h CHANGELOG
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
3. **`publish_testpypi`** uploads everything to TestPyPI.
4. **`publish_pypi`** (waits for reviewer approval if configured) uploads the
   same artifacts to PyPI.

## Verifying a TestPyPI release before PyPI

The `publish_pypi` job only runs after `publish_testpypi` finishes. If you've
enabled *Required reviewers* on the `pypi` environment, GitHub will pause and
request approval; use that window to spot-check TestPyPI:

```bash
python -m venv /tmp/v && . /tmp/v/bin/activate
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            logosdb==0.2.0
python -c "
import logosdb, numpy as np
db = logosdb.DB('/tmp/smoke', dim=8)
db.put(np.ones(8, dtype='f4'))
print('ok:', db.count(), logosdb.__version__)
"
```

If something looks wrong: decline the review in GitHub, delete the bad tag
(`git tag -d v0.2.0 && git push --delete origin v0.2.0`), fix, and start over
with a new version (`0.2.1`). PyPI *never* lets you re-upload the same version.

## Manual dry-run (no tag)

To exercise the full wheel matrix without publishing:

```
Actions → Publish → Run workflow → target: testpypi_only
```

This builds every wheel, uploads to TestPyPI, and stops. No PyPI upload.

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
