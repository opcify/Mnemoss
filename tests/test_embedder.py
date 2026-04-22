"""Embedder tests: FakeEmbedder deterministic, LocalEmbedder integration-marked,
OpenAIEmbedder / GeminiEmbedder mocked via their respective SDKs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mnemoss.encoder.embedder import (
    DEFAULT_GEMINI_DIM,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_LOCAL_DIM,
    DEFAULT_OPENAI_DIM,
    FakeEmbedder,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    make_embedder,
)


def test_fake_embedder_deterministic() -> None:
    e1 = FakeEmbedder(dim=32)
    e2 = FakeEmbedder(dim=32)
    v1 = e1.embed(["hello", "world"])
    v2 = e2.embed(["hello", "world"])
    assert v1.shape == (2, 32)
    assert v1.dtype == np.float32
    assert np.allclose(v1, v2)


def test_fake_embedder_normalized() -> None:
    e = FakeEmbedder(dim=16)
    v = e.embed(["x"])
    assert np.isclose(np.linalg.norm(v[0]), 1.0, atol=1e-5)


def test_fake_embedder_handles_empty() -> None:
    e = FakeEmbedder(dim=8)
    v = e.embed([])
    assert v.shape == (0, 8)
    assert v.dtype == np.float32


def test_make_embedder_local_default() -> None:
    e = make_embedder("local")
    assert isinstance(e, LocalEmbedder)
    assert e.dim == DEFAULT_LOCAL_DIM


def test_make_embedder_local_explicit() -> None:
    e = make_embedder("local:all-MiniLM-L6-v2")
    assert isinstance(e, LocalEmbedder)


def test_make_embedder_openai_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        make_embedder("openai")


def test_make_embedder_gemini_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        make_embedder("gemini")


def test_make_embedder_gemini_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    e = make_embedder("gemini")
    assert isinstance(e, GeminiEmbedder)
    assert e.dim == DEFAULT_GEMINI_DIM
    assert e.embedder_id == f"gemini:{DEFAULT_GEMINI_MODEL}:{DEFAULT_GEMINI_DIM}"


def test_make_embedder_gemini_explicit_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    e = make_embedder("gemini:text-embedding-004")
    assert isinstance(e, GeminiEmbedder)
    # Unknown model → dim resolved at first embed.
    assert e.dim == -1


def test_make_embedder_passthrough_instance() -> None:
    e = FakeEmbedder()
    assert make_embedder(e) is e


def test_make_embedder_unknown_spec_raises() -> None:
    with pytest.raises(ValueError):
        make_embedder("chroma")


def test_openai_embedder_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    embedder = OpenAIEmbedder()
    assert embedder.dim == DEFAULT_OPENAI_DIM

    # Mock the OpenAI client so the test never hits the network.
    mock_client = MagicMock()
    mock_data = [MagicMock(embedding=[0.1] * DEFAULT_OPENAI_DIM) for _ in range(2)]
    mock_client.embeddings.create.return_value = MagicMock(data=mock_data)

    with patch.object(embedder, "_client", mock_client):
        vectors = embedder.embed(["hello", "world"])

    assert vectors.shape == (2, DEFAULT_OPENAI_DIM)
    assert vectors.dtype == np.float32
    mock_client.embeddings.create.assert_called_once()


def test_gemini_embedder_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
    embedder = GeminiEmbedder()
    assert embedder.dim == DEFAULT_GEMINI_DIM

    mock_client = MagicMock()
    mock_embeddings = [MagicMock(values=[0.1] * DEFAULT_GEMINI_DIM) for _ in range(2)]
    mock_client.models.embed_content.return_value = MagicMock(embeddings=mock_embeddings)

    with patch.object(embedder, "_client", mock_client):
        vectors = embedder.embed(["hello", "world"])

    assert vectors.shape == (2, DEFAULT_GEMINI_DIM)
    assert vectors.dtype == np.float32
    call_kwargs = mock_client.models.embed_content.call_args.kwargs
    assert call_kwargs["model"] == DEFAULT_GEMINI_MODEL
    assert call_kwargs["contents"] == ["hello", "world"]
    # Native-dim calls skip the config (no MRL truncation).
    assert "config" not in call_kwargs


def test_gemini_embedder_mrl_renormalizes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Truncated-MRL outputs come back un-normalized; embedder must renormalize."""
    monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
    embedder = GeminiEmbedder(dim=768)
    assert embedder.dim == 768

    # Simulate the API returning raw (un-normalized) 768-d vectors.
    raw = [[3.0] + [0.0] * 767, [0.0, 4.0] + [0.0] * 766]
    mock_client = MagicMock()
    mock_embeddings = [MagicMock(values=v) for v in raw]
    mock_client.models.embed_content.return_value = MagicMock(embeddings=mock_embeddings)

    with patch.object(embedder, "_client", mock_client):
        vectors = embedder.embed(["a", "b"])

    assert vectors.shape == (2, 768)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    # Truncated-dim call must pass an EmbedContentConfig via `config`.
    assert "config" in mock_client.models.embed_content.call_args.kwargs


def test_gemini_embedder_handles_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gk-test")
    embedder = GeminiEmbedder()
    v = embedder.embed([])
    assert v.shape == (0, DEFAULT_GEMINI_DIM)
    assert v.dtype == np.float32


@pytest.mark.integration
def test_local_embedder_runs_on_multilingual_text() -> None:
    """Smoke-test the real sentence-transformers model (downloads on first run)."""
    e = LocalEmbedder()
    v = e.embed(["hello", "你好", "こんにちは"])
    assert v.shape == (3, DEFAULT_LOCAL_DIM)
    assert v.dtype == np.float32
