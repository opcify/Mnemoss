"""Encoder layer — embedders and the simplified event encoder."""

from mnemoss.encoder.embedder import (
    Embedder,
    FakeEmbedder,
    GeminiEmbedder,
    LocalEmbedder,
    OpenAIEmbedder,
    make_embedder,
)

__all__ = [
    "Embedder",
    "FakeEmbedder",
    "GeminiEmbedder",
    "LocalEmbedder",
    "OpenAIEmbedder",
    "make_embedder",
]
