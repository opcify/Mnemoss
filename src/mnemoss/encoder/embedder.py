"""Embedding backends.

Three shipped implementations plus a test helper:

- ``LocalEmbedder`` — default zero-config; sentence-transformers
  ``paraphrase-multilingual-MiniLM-L12-v2`` (384d, 50+ languages).
  Lazy-imports the heavy dependency so test-time imports stay cheap.
- ``OpenAIEmbedder`` — opt-in via the ``openai`` extra; ``text-embedding-3-small``
  at native 1536d. Reads ``OPENAI_API_KEY`` unless ``api_key`` is provided.
- ``GeminiEmbedder`` — opt-in via the ``gemini`` extra; ``gemini-embedding-001``
  at native 3072d (MRL supports 768/1536/3072 via ``dim=``). Reads
  ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``) unless ``api_key`` is provided.
- ``FakeEmbedder`` — deterministic hash-based vectors for tests.

All embedders return ``float32`` arrays with ``shape == (n, dim)``.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Protocol

import numpy as np

DEFAULT_LOCAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_LOCAL_DIM = 384
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_DIM = 1536
DEFAULT_GEMINI_MODEL = "gemini-embedding-001"
DEFAULT_GEMINI_DIM = 3072


class Embedder(Protocol):
    """Embedder interface.

    Implementations must return ``float32`` ndarrays. ``embedder_id`` is
    pinned into the workspace schema so mixed-embedder reads raise.
    """

    dim: int
    embedder_id: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class LocalEmbedder:
    """sentence-transformers embedder. Lazy model load on first ``embed``."""

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL) -> None:
        self._model_name = model_name
        # Kept untyped to avoid an import-time dependency on
        # sentence_transformers (which is heavy). Resolved to a real
        # SentenceTransformer instance inside ``_ensure_model``.
        self._model: Any = None
        # Dim pinned for the shipped default. Other models would need a
        # one-off ``encode(["x"]).shape[1]`` probe.
        if model_name == DEFAULT_LOCAL_MODEL:
            self.dim = DEFAULT_LOCAL_DIM
        else:
            self.dim = -1  # resolved at first embed
        self.embedder_id = f"local:{model_name}"

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self._model_name)
        self._model = model
        if self.dim < 0:
            self.dim = int(model.get_sentence_embedding_dimension() or 0)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_model()
        assert self._model is not None
        vectors = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)


class OpenAIEmbedder:
    """OpenAI Embeddings API. Opt-in via the ``[openai]`` extra."""

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        api_key: str | None = None,
        dim: int | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise RuntimeError("OpenAIEmbedder requires OPENAI_API_KEY (env var or api_key=)")
        self.dim = (
            dim
            if dim is not None
            else (DEFAULT_OPENAI_DIM if model == DEFAULT_OPENAI_MODEL else -1)
        )
        self.embedder_id = f"openai:{model}:{self.dim}"
        self._client = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        from openai import OpenAI

        self._client = OpenAI(api_key=self._api_key)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim if self.dim > 0 else 0), dtype=np.float32)
        self._ensure_client()
        assert self._client is not None
        response = self._client.embeddings.create(input=texts, model=self._model)
        vectors = [item.embedding for item in response.data]
        arr = np.asarray(vectors, dtype=np.float32)
        if self.dim < 0:
            self.dim = arr.shape[1]
            self.embedder_id = f"openai:{self._model}:{self.dim}"
        return arr


class GeminiEmbedder:
    """Google Gemini Embeddings API. Opt-in via the ``[gemini]`` extra.

    Uses the ``google-genai`` SDK. Default model ``gemini-embedding-001``
    emits 3072-d vectors natively; Matryoshka (MRL) truncation to 768 or
    1536 is supported via ``dim=`` (the API returns a shorter vector and
    we L2-renormalize as Google recommends). Reads ``GEMINI_API_KEY`` or
    ``GOOGLE_API_KEY`` from the environment unless ``api_key`` is passed.
    """

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_MODEL,
        api_key: str | None = None,
        dim: int | None = None,
    ) -> None:
        self._model = model
        self._api_key = (
            api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        if not self._api_key:
            raise RuntimeError(
                "GeminiEmbedder requires GEMINI_API_KEY or GOOGLE_API_KEY (env var or api_key=)"
            )
        self.dim = (
            dim
            if dim is not None
            else (DEFAULT_GEMINI_DIM if model == DEFAULT_GEMINI_MODEL else -1)
        )
        self.embedder_id = f"gemini:{model}:{self.dim}"
        self._client: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from google import genai
        except ImportError as e:
            raise RuntimeError(
                "GeminiEmbedder needs the 'google-genai' package — "
                "install with `pip install mnemoss[gemini]`"
            ) from e
        self._client = genai.Client(api_key=self._api_key)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim if self.dim > 0 else 0), dtype=np.float32)
        self._ensure_client()
        assert self._client is not None

        config: Any = None
        if self.dim > 0 and self.dim != DEFAULT_GEMINI_DIM:
            from google.genai import types as genai_types

            config = genai_types.EmbedContentConfig(output_dimensionality=self.dim)

        kwargs: dict[str, Any] = {"model": self._model, "contents": texts}
        if config is not None:
            kwargs["config"] = config
        response = self._client.models.embed_content(**kwargs)
        vectors = [item.values for item in response.embeddings]
        arr = np.asarray(vectors, dtype=np.float32)

        # MRL-truncated outputs are not normalized by the API; renormalize.
        if self.dim > 0 and self.dim != DEFAULT_GEMINI_DIM:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms

        if self.dim < 0:
            self.dim = arr.shape[1]
            self.embedder_id = f"gemini:{self._model}:{self.dim}"
        return arr


class FakeEmbedder:
    """Deterministic hash-based embedder for tests.

    Produces normalized vectors so dot-product ≈ cosine. Similar-prefix
    texts do not necessarily produce similar vectors — that's a feature
    (avoids false semantic signal leaking into unit tests that care only
    about plumbing).
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim
        self.embedder_id = f"fake:{dim}"

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            seed = int.from_bytes(
                hashlib.sha256(text.encode("utf-8")).digest()[:8], "big", signed=False
            )
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.dim).astype(np.float32)
            norm = float(np.linalg.norm(v))
            out[i] = v / norm if norm > 0 else v
        return out


def make_embedder(spec: str | Embedder) -> Embedder:
    """Factory used by ``Mnemoss(...)``.

    ``spec`` can be:
      * ``"local"``                    — ``LocalEmbedder()``
      * ``"openai"``                   — ``OpenAIEmbedder()``
      * ``"openai:<model-id>"``        — explicit OpenAI model
      * ``"local:<model-name>"``       — explicit local model
      * ``"gemini"``                   — ``GeminiEmbedder()``
      * ``"gemini:<model-id>"``        — explicit Gemini model
      * an ``Embedder`` instance       — passed through

    Anything else raises ``ValueError``.
    """

    if not isinstance(spec, str):
        return spec
    if spec == "local":
        return LocalEmbedder()
    if spec == "openai":
        return OpenAIEmbedder()
    if spec == "gemini":
        return GeminiEmbedder()
    if spec.startswith("openai:"):
        return OpenAIEmbedder(model=spec[len("openai:") :])
    if spec.startswith("local:"):
        return LocalEmbedder(model_name=spec[len("local:") :])
    if spec.startswith("gemini:"):
        return GeminiEmbedder(model=spec[len("gemini:") :])
    raise ValueError(f"Unknown embedder spec: {spec!r}")
