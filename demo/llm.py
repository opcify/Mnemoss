"""LLM adapters for the simulation.

Two implementations:

- :class:`StubLLM` — returns canned responses in order. Used for CI
  smoke tests and any scene run that doesn't need real generation.
  Zero network calls, fully deterministic.
- :class:`GeminiLLM` — real generation via Gemini 2.5 Flash free tier.
  Lazy import so missing deps / missing API key only fire when the
  caller asks for it. Used ONCE at recording time to generate the
  trace committed to the repo; replay is deterministic from the
  committed trace.

The :class:`LLM` protocol is intentionally minimal — one async
method. The scenario code doesn't care whether the LLM is real or
stubbed.
"""

from __future__ import annotations

import os
from typing import Any, Protocol


class LLM(Protocol):
    """Minimal async chat-like LLM.

    Attributes
    ----------
    llm_id:
        Short label written into the trace (``"stub"`` /
        ``"gemini-2.5-flash"`` / etc.) so replay UIs can show which
        model actually produced the responses.
    """

    llm_id: str

    async def generate(self, user_message: str, memory_context: list[str]) -> str:
        """Produce an assistant response given the user message and
        a list of pre-formatted memory-context snippets."""


class StubLLM:
    """Canned responses, consumed in order.

    Parameters
    ----------
    responses:
        One string per call. If the caller burns through the list,
        subsequent calls return a placeholder so the trace stays
        well-formed rather than raising mid-scene.
    """

    llm_id = "stub"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._idx = 0

    async def generate(self, user_message: str, memory_context: list[str]) -> str:
        if self._idx >= len(self._responses):
            return "(stub ran out of scripted responses)"
        response = self._responses[self._idx]
        self._idx += 1
        return response


class GeminiLLM:
    """Real responses via Gemini 2.5 Flash (free tier).

    Reads ``GEMINI_API_KEY`` or ``GOOGLE_API_KEY`` at construction
    time. Raises ``RuntimeError`` if neither is set — callers should
    fall back to :class:`StubLLM` for offline work.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        *,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self.llm_id = f"gemini:{model}"
        self._api_key = (
            api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        )
        if not self._api_key:
            raise RuntimeError(
                "GeminiLLM requires GEMINI_API_KEY or GOOGLE_API_KEY (env var or api_key=)."
            )
        self._client: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            from google import genai
        except ImportError as e:  # pragma: no cover — extras check
            raise RuntimeError(
                "GeminiLLM needs the 'google-genai' package — install "
                "with `pip install mnemoss[gemini]`."
            ) from e
        self._client = genai.Client(api_key=self._api_key)

    async def generate(self, user_message: str, memory_context: list[str]) -> str:
        """Build a simple system + user prompt using the recalled memories."""

        self._ensure_client()
        assert self._client is not None

        memory_block = ""
        if memory_context:
            bullets = "\n".join(f"- {m}" for m in memory_context)
            memory_block = (
                "You are responding to a user you have chatted with before. "
                "Relevant memories recalled from prior turns:\n"
                f"{bullets}\n\n"
                "Use them naturally if they're relevant. Do not parrot them "
                "back verbatim; weave them into your reply.\n\n"
            )

        prompt = (
            f"{memory_block}User: {user_message}\n\nRespond in 1-2 sentences, warm but concise."
        )

        # Gemini SDK is sync; wrap in asyncio.to_thread so the
        # simulate loop stays async-friendly.
        import asyncio

        def _call() -> str:
            resp = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
            )
            return (resp.text or "").strip() or "(empty response)"

        return await asyncio.to_thread(_call)
