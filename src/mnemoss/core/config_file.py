"""TOML config-file loader for cloud embedder + LLM settings.

A single ``mnemoss.toml`` picks the embedder and LLM provider, their
model id, and an API key. API keys in the file are optional — if
absent, the provider client falls back to its env var
(``OPENAI_API_KEY`` / ``GEMINI_API_KEY`` / ``ANTHROPIC_API_KEY``).

Example::

    [embedder]
    provider = "gemini"               # "local" | "openai" | "gemini"
    model    = "gemini-embedding-001" # optional
    api_key  = "..."                  # optional
    dim      = 3072                   # optional

    [llm]
    provider = "gemini"               # "openai" | "anthropic" | "gemini" | "mock"
    model    = "gemini-2.5-flash"     # optional
    api_key  = "..."                  # optional

Discovery order when ``path`` is omitted:

1. ``$MNEMOSS_CONFIG``
2. ``./mnemoss.toml`` (current working directory)
3. ``~/.mnemoss/config.toml``

``load_config_file()`` returns ``None`` if no file is found (callers
can then fall back to their own defaults).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - exercised on 3.10 only
    import tomli as tomllib  # type: ignore[no-redef,import-not-found]

from mnemoss.encoder import (
    Embedder,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
)
from mnemoss.encoder.embedder import DEFAULT_LOCAL_MODEL
from mnemoss.llm import (
    AnthropicClient,
    GeminiClient,
    LLMClient,
    MockLLMClient,
    OpenAIClient,
)

_KNOWN_EMBEDDER_KEYS = {"provider", "model", "api_key", "dim"}
_KNOWN_LLM_KEYS = {"provider", "model", "api_key"}
_KNOWN_SECTIONS = {"embedder", "llm"}


@dataclass
class EmbedderFileConfig:
    provider: str = "local"
    model: str | None = None
    api_key: str | None = None
    dim: int | None = None


@dataclass
class LLMFileConfig:
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None


@dataclass
class MnemossFileConfig:
    """Parsed ``mnemoss.toml`` contents."""

    embedder: EmbedderFileConfig
    llm: LLMFileConfig
    source_path: Path | None = None

    def build_embedder(self) -> Embedder:
        """Instantiate the configured embedder."""

        p = self.embedder.provider.lower()
        if p == "local":
            return LocalEmbedder(model_name=self.embedder.model or DEFAULT_LOCAL_MODEL)
        if p == "openai":
            kwargs: dict[str, Any] = {}
            if self.embedder.model:
                kwargs["model"] = self.embedder.model
            if self.embedder.api_key:
                kwargs["api_key"] = self.embedder.api_key
            if self.embedder.dim is not None:
                kwargs["dim"] = self.embedder.dim
            return OpenAIEmbedder(**kwargs)
        if p == "gemini":
            kwargs = {}
            if self.embedder.model:
                kwargs["model"] = self.embedder.model
            if self.embedder.api_key:
                kwargs["api_key"] = self.embedder.api_key
            if self.embedder.dim is not None:
                kwargs["dim"] = self.embedder.dim
            return GeminiEmbedder(**kwargs)
        raise ValueError(
            f"Unknown embedder provider {self.embedder.provider!r} "
            f"(expected 'local' | 'openai' | 'gemini')"
        )

    def build_llm(self) -> LLMClient | None:
        """Instantiate the configured LLM client, or ``None`` if unset."""

        if self.llm.provider is None:
            return None
        p = self.llm.provider.lower()
        if p == "mock":
            return MockLLMClient()
        kwargs: dict[str, Any] = {}
        if self.llm.model:
            kwargs["model"] = self.llm.model
        if self.llm.api_key:
            kwargs["api_key"] = self.llm.api_key
        if p == "openai":
            return OpenAIClient(**kwargs)
        if p == "anthropic":
            return AnthropicClient(**kwargs)
        if p == "gemini":
            return GeminiClient(**kwargs)
        raise ValueError(
            f"Unknown llm provider {self.llm.provider!r} "
            f"(expected 'openai' | 'anthropic' | 'gemini' | 'mock')"
        )


def _default_search_paths() -> list[Path]:
    paths: list[Path] = []
    env = os.environ.get("MNEMOSS_CONFIG")
    if env:
        paths.append(Path(env).expanduser())
    paths.append(Path.cwd() / "mnemoss.toml")
    paths.append(Path.home() / ".mnemoss" / "config.toml")
    return paths


def _parse(data: dict[str, Any], source: Path | None) -> MnemossFileConfig:
    unknown_sections = set(data.keys()) - _KNOWN_SECTIONS
    if unknown_sections:
        raise ValueError(f"Unknown section(s) in mnemoss config: {sorted(unknown_sections)}")

    emb_raw = data.get("embedder", {}) or {}
    unknown = set(emb_raw.keys()) - _KNOWN_EMBEDDER_KEYS
    if unknown:
        raise ValueError(f"Unknown key(s) in [embedder]: {sorted(unknown)}")
    embedder_cfg = EmbedderFileConfig(
        provider=str(emb_raw.get("provider", "local")),
        model=emb_raw.get("model"),
        api_key=emb_raw.get("api_key"),
        dim=int(emb_raw["dim"]) if "dim" in emb_raw else None,
    )

    llm_raw = data.get("llm", {}) or {}
    unknown = set(llm_raw.keys()) - _KNOWN_LLM_KEYS
    if unknown:
        raise ValueError(f"Unknown key(s) in [llm]: {sorted(unknown)}")
    llm_cfg = LLMFileConfig(
        provider=llm_raw.get("provider"),
        model=llm_raw.get("model"),
        api_key=llm_raw.get("api_key"),
    )

    return MnemossFileConfig(embedder=embedder_cfg, llm=llm_cfg, source_path=source)


def load_config_file(path: str | Path | None = None) -> MnemossFileConfig | None:
    """Load a Mnemoss TOML config.

    If ``path`` is given it must exist. If omitted, search the default
    locations (see module docstring); return ``None`` when nothing is
    found.
    """

    if path is not None:
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Mnemoss config file not found: {p}")
        with p.open("rb") as f:
            data = tomllib.load(f)
        return _parse(data, source=p)

    for candidate in _default_search_paths():
        if candidate.exists():
            with candidate.open("rb") as f:
                data = tomllib.load(f)
            return _parse(data, source=candidate)
    return None
