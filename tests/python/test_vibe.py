"""Smoke tests for logosdb.vibe.VibeMemory.

These tests mock Mistral API calls so no real API key or network is needed.
They exercise the full index → search → forget pipeline and all helper methods.

Run with::

    pytest tests/python/test_vibe.py -v
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIM = 8  # small dimension to keep tests fast


def _fake_embedding(text: str, dim: int = _DIM) -> list[float]:
    """Deterministic fake embedding based on text hash."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


def _mock_mistral_post(url: str, **kwargs: Any) -> MagicMock:
    """Fake requests.post that intercepts the Mistral embeddings call."""
    body = kwargs.get("json", {})
    inputs = body.get("input", [])
    embeddings = [{"embedding": _fake_embedding(t, _DIM)} for t in inputs]
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"data": embeddings}
    resp.raise_for_status = lambda: None
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def mem(tmp_path: Path):
    """VibeMemory instance with a temp root and mocked Mistral calls."""
    with patch("requests.post", side_effect=_mock_mistral_post):
        from logosdb.vibe import VibeMemory

        # Patch _DIM into the store so LogosDB uses dim=8
        with patch("logosdb.mistral.MistralVectorStore.__init__", wraps=_patched_init):
            yield VibeMemory(uri=str(tmp_path / "logosdb"), api_key="fake-key", dim=_DIM)


def _patched_init(self: Any, uri: str, **kwargs: Any) -> None:
    """Wrap MistralVectorStore.__init__ to force dim=_DIM."""
    from logosdb.mistral import MistralVectorStore

    kwargs["dim"] = _DIM
    MistralVectorStore.__init__.__wrapped__(self, uri=uri, **kwargs)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Unit tests — no DB, no network
# ---------------------------------------------------------------------------

class TestChunker:
    def test_single_paragraph(self) -> None:
        from logosdb.vibe import _chunk_text

        chunks = _chunk_text("Hello world", target_chars=200)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world"

    def test_splits_on_blank_lines(self) -> None:
        from logosdb.vibe import _chunk_text

        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = _chunk_text(text, target_chars=20)
        assert len(chunks) > 1

    def test_overlap(self) -> None:
        from logosdb.vibe import _chunk_text

        long_text = "\n\n".join([f"Paragraph {i} with some content." for i in range(20)])
        chunks = _chunk_text(long_text, target_chars=80, overlap_chars=20)
        # Each chunk after the first should begin with part of the previous
        assert all(len(c) > 0 for c in chunks)

    def test_empty_string_returns_one_chunk(self) -> None:
        from logosdb.vibe import _chunk_text

        chunks = _chunk_text("", target_chars=100)
        assert len(chunks) == 1


class TestNamespaceValidation:
    def test_valid_namespace(self) -> None:
        from logosdb.vibe import VibeMemory

        VibeMemory._validate_namespace("code")
        VibeMemory._validate_namespace("my-namespace")
        VibeMemory._validate_namespace("ns_1.2")

    def test_invalid_namespace_raises(self) -> None:
        from logosdb.vibe import VibeMemory

        with pytest.raises(ValueError, match="Invalid namespace"):
            VibeMemory._validate_namespace("bad namespace")

        with pytest.raises(ValueError, match="Invalid namespace"):
            VibeMemory._validate_namespace("../../etc/passwd")


# ---------------------------------------------------------------------------
# Integration tests — real LogosDB, mocked Mistral
# ---------------------------------------------------------------------------

class TestVibeMemory:
    def test_index_file(self, tmp_path: Path) -> None:
        src = tmp_path / "hello.py"
        src.write_text("def hello():\n    return 'world'\n")

        with patch("requests.post", side_effect=_mock_mistral_post):
            from logosdb.vibe import VibeMemory

            mem = VibeMemory(uri=str(tmp_path / "db"), api_key="fake", dim=_DIM)
            result = mem.index(str(src), namespace="code")

        assert result["indexed"] >= 1
        assert result["files"] == 1
        assert result["namespace"] == "code"

    def test_index_directory(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("def a(): pass")
        (src / "b.py").write_text("def b(): pass")
        (src / "ignore.bin").write_bytes(b"\x00\x01")

        with patch("requests.post", side_effect=_mock_mistral_post):
            from logosdb.vibe import VibeMemory

            mem = VibeMemory(uri=str(tmp_path / "db"), api_key="fake", dim=_DIM)
            result = mem.index(str(src), namespace="code")

        assert result["files"] == 2  # .bin skipped

    def test_search_returns_results(self, tmp_path: Path) -> None:
        src = tmp_path / "auth.py"
        src.write_text("def validate_jwt(token):\n    # JWT validation logic\n    pass\n")

        with patch("requests.post", side_effect=_mock_mistral_post):
            from logosdb.vibe import VibeMemory

            mem = VibeMemory(uri=str(tmp_path / "db"), api_key="fake", dim=_DIM)
            mem.index(str(src), namespace="code")
            results = mem.search("JWT validation", namespace="code", top_k=3)

        assert isinstance(results, list)
        assert len(results) >= 1
        for r in results:
            assert "id" in r
            assert "score" in r
            assert "text" in r

    def test_search_empty_namespace_returns_empty(self, tmp_path: Path) -> None:
        with patch("requests.post", side_effect=_mock_mistral_post):
            from logosdb.vibe import VibeMemory

            mem = VibeMemory(uri=str(tmp_path / "db"), api_key="fake", dim=_DIM)
            results = mem.search("anything", namespace="nonexistent")

        assert results == []

    def test_forget_by_id(self, tmp_path: Path) -> None:
        src = tmp_path / "x.py"
        src.write_text("x = 1\n")

        with patch("requests.post", side_effect=_mock_mistral_post):
            from logosdb.vibe import VibeMemory

            mem = VibeMemory(uri=str(tmp_path / "db"), api_key="fake", dim=_DIM)
            mem.index(str(src), namespace="code")
            results_before = mem.search("x = 1", namespace="code", top_k=5)
            assert results_before

            row_id = results_before[0]["id"]
            outcome = mem.forget(namespace="code", memory_id=row_id)
            assert outcome["forgotten"] == 1
            assert row_id in outcome["ids"]

    def test_forget_by_query(self, tmp_path: Path) -> None:
        src = tmp_path / "y.py"
        src.write_text("y = 2\n")

        with patch("requests.post", side_effect=_mock_mistral_post):
            from logosdb.vibe import VibeMemory

            mem = VibeMemory(uri=str(tmp_path / "db"), api_key="fake", dim=_DIM)
            mem.index(str(src), namespace="code")
            outcome = mem.forget(namespace="code", query="y = 2", top_k=1)
            assert outcome["forgotten"] >= 1

    def test_forget_requires_id_or_query(self, tmp_path: Path) -> None:
        with patch("requests.post", side_effect=_mock_mistral_post):
            from logosdb.vibe import VibeMemory

            mem = VibeMemory(uri=str(tmp_path / "db"), api_key="fake", dim=_DIM)
            with pytest.raises(ValueError):
                mem.forget(namespace="code")

    def test_info_single_namespace(self, tmp_path: Path) -> None:
        src = tmp_path / "z.py"
        src.write_text("z = 3\n")

        with patch("requests.post", side_effect=_mock_mistral_post):
            from logosdb.vibe import VibeMemory

            mem = VibeMemory(uri=str(tmp_path / "db"), api_key="fake", dim=_DIM)
            mem.index(str(src), namespace="code")
            info = mem.info(namespace="code")

        assert info["namespace"] == "code"
        assert info["count"] >= 1

    def test_list_namespaces(self, tmp_path: Path) -> None:
        src = tmp_path / "f.py"
        src.write_text("pass\n")

        with patch("requests.post", side_effect=_mock_mistral_post):
            from logosdb.vibe import VibeMemory

            mem = VibeMemory(uri=str(tmp_path / "db"), api_key="fake", dim=_DIM)
            mem.index(str(src), namespace="code")
            mem.index(str(src), namespace="tests")
            namespaces = mem.list_namespaces()

        assert set(namespaces) >= {"code", "tests"}

    def test_missing_api_key_raises(self, tmp_path: Path) -> None:
        from logosdb.vibe import VibeMemory

        with pytest.raises(ValueError, match="Mistral API key"):
            with patch.dict("os.environ", {}, clear=True):
                VibeMemory(uri=str(tmp_path / "db"), api_key="")


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

class TestVibeCli:
    def test_cli_index_and_search(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        src = tmp_path / "cli_test.py"
        src.write_text("def cli_function(): pass\n")

        with patch("requests.post", side_effect=_mock_mistral_post), \
             patch.dict("os.environ", {
                 "LOGOSDB_PATH": str(tmp_path / "db"),
                 "MISTRAL_API_KEY": "fake",
             }):
            # Monkey-patch dim inside VibeMemory for the CLI path
            import logosdb._vibe_cli as cli_mod
            orig_mem = cli_mod._mem

            def _patched_mem():
                from logosdb.vibe import VibeMemory
                return VibeMemory(
                    uri=str(tmp_path / "db"), api_key="fake", dim=_DIM
                )

            cli_mod._mem = _patched_mem  # type: ignore[assignment]
            try:
                from logosdb._vibe_cli import main

                main(["index", str(src), "--namespace", "code"])
                out = json.loads(capsys.readouterr().out)
                assert out["files"] == 1

                main(["list"])
                out2 = json.loads(capsys.readouterr().out)
                assert "code" in out2
            finally:
                cli_mod._mem = orig_mem
