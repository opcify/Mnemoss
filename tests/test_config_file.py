"""Tests for the TOML config-file loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from mnemoss.core.config_file import (
    MnemossFileConfig,
    load_config_file,
)
from mnemoss.encoder import (
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
)
from mnemoss.llm import (
    AnthropicClient,
    GeminiClient,
    MockLLMClient,
    OpenAIClient,
)


def _write(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def test_load_explicit_path_builds_local_embedder(tmp_path: Path) -> None:
    cfg_path = _write(
        tmp_path / "mnemoss.toml",
        """
        [embedder]
        provider = "local"
        """,
    )
    cfg = load_config_file(cfg_path)
    assert cfg is not None
    assert cfg.source_path == cfg_path
    assert cfg.embedder.provider == "local"
    assert cfg.llm.provider is None
    emb = cfg.build_embedder()
    assert isinstance(emb, LocalEmbedder)
    assert cfg.build_llm() is None


def test_missing_explicit_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config_file(tmp_path / "nope.toml")


def test_discovery_returns_none_when_nothing_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MNEMOSS_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))  # move ~/.mnemoss out of the way
    assert load_config_file() is None


def test_discovery_picks_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_path = _write(
        tmp_path / "custom.toml",
        """
        [embedder]
        provider = "local"
        model = "all-MiniLM-L6-v2"
        """,
    )
    monkeypatch.setenv("MNEMOSS_CONFIG", str(cfg_path))
    cfg = load_config_file()
    assert cfg is not None
    assert cfg.embedder.model == "all-MiniLM-L6-v2"


def test_gemini_embedder_from_config(tmp_path: Path) -> None:
    cfg_path = _write(
        tmp_path / "mnemoss.toml",
        """
        [embedder]
        provider = "gemini"
        model = "gemini-embedding-001"
        api_key = "fake-key"
        dim = 1536
        """,
    )
    cfg = load_config_file(cfg_path)
    assert cfg is not None
    emb = cfg.build_embedder()
    assert isinstance(emb, GeminiEmbedder)
    assert emb.dim == 1536
    assert emb.embedder_id == "gemini:gemini-embedding-001:1536"


def test_openai_embedder_from_config(tmp_path: Path) -> None:
    cfg_path = _write(
        tmp_path / "mnemoss.toml",
        """
        [embedder]
        provider = "openai"
        api_key = "sk-fake"
        """,
    )
    cfg = load_config_file(cfg_path)
    assert cfg is not None
    emb = cfg.build_embedder()
    assert isinstance(emb, OpenAIEmbedder)


def test_llm_providers_from_config(tmp_path: Path) -> None:
    for provider, expected_cls in [
        ("openai", OpenAIClient),
        ("anthropic", AnthropicClient),
        ("gemini", GeminiClient),
        ("mock", MockLLMClient),
    ]:
        cfg_path = _write(
            tmp_path / f"{provider}.toml",
            f"""
            [embedder]
            provider = "local"

            [llm]
            provider = "{provider}"
            api_key = "fake"
            """,
        )
        cfg = load_config_file(cfg_path)
        assert cfg is not None
        llm = cfg.build_llm()
        assert isinstance(llm, expected_cls), provider


def test_unknown_section_raises(tmp_path: Path) -> None:
    cfg_path = _write(
        tmp_path / "mnemoss.toml",
        """
        [storage]
        root = "/tmp"
        """,
    )
    with pytest.raises(ValueError, match="Unknown section"):
        load_config_file(cfg_path)


def test_unknown_embedder_key_raises(tmp_path: Path) -> None:
    cfg_path = _write(
        tmp_path / "mnemoss.toml",
        """
        [embedder]
        provider = "local"
        garbage = "x"
        """,
    )
    with pytest.raises(ValueError, match="embedder"):
        load_config_file(cfg_path)


def test_unknown_embedder_provider_raises(tmp_path: Path) -> None:
    cfg_path = _write(
        tmp_path / "mnemoss.toml",
        """
        [embedder]
        provider = "chroma"
        """,
    )
    cfg = load_config_file(cfg_path)
    assert cfg is not None
    with pytest.raises(ValueError, match="Unknown embedder provider"):
        cfg.build_embedder()


def test_unknown_llm_provider_raises(tmp_path: Path) -> None:
    cfg_path = _write(
        tmp_path / "mnemoss.toml",
        """
        [embedder]
        provider = "local"

        [llm]
        provider = "bard"
        """,
    )
    cfg = load_config_file(cfg_path)
    assert cfg is not None
    with pytest.raises(ValueError, match="Unknown llm provider"):
        cfg.build_llm()


def test_api_key_env_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Omitting api_key in the file falls back to the provider env var."""
    monkeypatch.setenv("GEMINI_API_KEY", "env-key")
    cfg_path = _write(
        tmp_path / "mnemoss.toml",
        """
        [embedder]
        provider = "gemini"
        """,
    )
    cfg = load_config_file(cfg_path)
    assert cfg is not None
    # Should not raise — env var supplies the key.
    emb = cfg.build_embedder()
    assert isinstance(emb, GeminiEmbedder)


def test_mnemoss_from_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mnemoss import Mnemoss

    cfg_path = _write(
        tmp_path / "mnemoss.toml",
        """
        [embedder]
        provider = "local"
        """,
    )
    monkeypatch.setenv("MNEMOSS_CONFIG", str(cfg_path))
    mem = Mnemoss.from_config_file(workspace="w")
    assert isinstance(mem, Mnemoss)


def test_mnemoss_from_config_file_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from mnemoss import Mnemoss

    monkeypatch.delenv("MNEMOSS_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(FileNotFoundError):
        Mnemoss.from_config_file(workspace="w")


def test_dataclass_shape() -> None:
    """Sanity check that MnemossFileConfig is the expected dataclass shape."""
    assert hasattr(MnemossFileConfig, "__dataclass_fields__")
