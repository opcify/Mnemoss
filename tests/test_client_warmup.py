"""Regression test for the embedder warmup fix.

Origin: 2026-04-27 — `Mnemoss._ensure_open` called `embed([""])` to warm
the embedder's lazy model load. OpenAI's text-embedding-3-* endpoints
reject empty strings with a 400 BadRequest, so the dreaming-validation
harness blew up on first observe. `LocalEmbedder` happily accepts empty
strings, hiding the bug locally.

The fix swaps the warmup payload to `["warmup"]`. This test pins that
behavior with an embedder that rejects empty strings the same way OpenAI
does, so a future regression surfaces immediately instead of only at
first cloud-embedder use.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mnemoss import FormulaParams, Mnemoss, StorageParams


class StrictNonEmptyEmbedder:
    """Embedder that mimics OpenAI's empty-string rejection (HTTP 400)."""

    dim = 4
    embedder_id = "strict-nonempty:4"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        for t in texts:
            if not t:
                raise ValueError(
                    "BadRequestError(400): input must contain non-empty strings"
                )
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            out[i, sum(ord(c) for c in t) % self.dim] = 1.0
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        out = np.where(norm > 0, out / norm, out)
        return out


@pytest.mark.asyncio
async def test_warmup_payload_is_non_empty_string(tmp_path: Path) -> None:
    embedder = StrictNonEmptyEmbedder()
    mem = Mnemoss(
        workspace="warmup",
        embedding_model=embedder,
        formula=FormulaParams(noise_scale=0.0),
        storage=StorageParams(root=tmp_path),
    )
    try:
        # _ensure_open() runs warmup. If it ever regresses to embed([""])
        # the StrictNonEmptyEmbedder raises and observe() blows up.
        await mem.observe(role="user", content="hello")
    finally:
        await mem.close()

    # Warmup ran at least once and the payload was non-empty.
    assert embedder.calls, "embedder.embed was never called"
    warmup_call = embedder.calls[0]
    assert warmup_call == ["warmup"], (
        f"expected warmup payload ['warmup'], got {warmup_call!r}"
    )
