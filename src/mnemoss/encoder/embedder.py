"""Embedding backends.

Two shipped implementations plus a test helper:

- ``LocalEmbedder`` — default zero-config; sentence-transformers
  ``paraphrase-multilingual-MiniLM-L12-v2`` (384d, 50+ languages).
  Lazy-imports the heavy dependency so test-time imports stay cheap.
- ``OpenAIEmbedder`` — opt-in via the ``openai`` extra; ``text-embedding-3-small``
  at native 1536d. Reads ``OPENAI_API_KEY`` unless ``api_key`` is provided.
- ``FakeEmbedder`` — deterministic hash-based vectors for tests.

All embedders return ``float32`` arrays with ``shape == (n, dim)``.
"""

from __future__ import annotations

import hashlib
import os
from typing import Protocol

import numpy as np

DEFAULT_LOCAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_LOCAL_DIM = 384
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_DIM = 1536


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
        self._model = None
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

        self._model = SentenceTransformer(self._model_name)
        if self.dim < 0:
            self.dim = int(self._model.get_sentence_embedding_dimension() or 0)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        self._ensure_model()
        assert self._model is not None
        vectors = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
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
            raise RuntimeError(
                "OpenAIEmbedder requires OPENAI_API_KEY (env var or api_key=)"
            )
        self.dim = dim if dim is not None else (
            DEFAULT_OPENAI_DIM if model == DEFAULT_OPENAI_MODEL else -1
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
      * an ``Embedder`` instance       — passed through

    Anything else raises ``ValueError``.
    """

    if not isinstance(spec, str):
        return spec
    if spec == "local":
        return LocalEmbedder()
    if spec == "openai":
        return OpenAIEmbedder()
    if spec.startswith("openai:"):
        return OpenAIEmbedder(model=spec[len("openai:") :])
    if spec.startswith("local:"):
        return LocalEmbedder(model_name=spec[len("local:") :])
    raise ValueError(f"Unknown embedder spec: {spec!r}")
