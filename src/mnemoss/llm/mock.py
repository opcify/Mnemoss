"""In-memory LLM stand-in for tests.

``MockLLMClient(responses=[...])`` consumes canned responses in order,
one per ``complete_text`` / ``complete_json`` call. ``MockLLMClient(
callback=fn)`` delegates to ``fn(prompt)`` for dynamic responses. Every
call is recorded on ``client.calls`` as ``(method, prompt)`` so tests
can assert on prompt shape.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class MockLLMClient:
    def __init__(
        self,
        *,
        responses: list[Any] | None = None,
        callback: Callable[[str], Any] | None = None,
        model: str = "mock",
    ) -> None:
        if responses is None and callback is None:
            responses = []
        self._responses: list[Any] = list(responses) if responses else []
        self._callback = callback
        self._model = model
        self.calls: list[tuple[str, str]] = []

    @property
    def model(self) -> str:
        return self._model

    def _next(self, method: str, prompt: str) -> Any:
        self.calls.append((method, prompt))
        if self._callback is not None:
            return self._callback(prompt)
        if not self._responses:
            raise RuntimeError(
                "MockLLMClient exhausted its canned responses "
                f"(call #{len(self.calls)}, method={method!r})"
            )
        return self._responses.pop(0)

    async def complete_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        r = self._next("text", prompt)
        if not isinstance(r, str):
            raise TypeError(
                f"MockLLMClient expected str for complete_text, got {type(r).__name__}"
            )
        return r

    async def complete_json(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        r = self._next("json", prompt)
        if not isinstance(r, dict):
            raise TypeError(
                f"MockLLMClient expected dict for complete_json, got {type(r).__name__}"
            )
        return r
