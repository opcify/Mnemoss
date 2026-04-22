"""Provider-agnostic LLM client Protocol + OpenAI/Anthropic implementations.

The two methods on the Protocol are the minimum surface Dreaming needs:

- ``complete_text``: free-form continuation. Currently unused by the
  core pipeline but kept for future phases / integrators.
- ``complete_json``: structured output. P3 Consolidate uses this.

Both are ``async``. Provider SDKs are lazy-imported inside the first
call so the Protocol itself (and ``MockLLMClient``) have no runtime
dependency on the SDKs.
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimum surface a dreaming LLM provider must implement."""

    @property
    def model(self) -> str: ...

    async def complete_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str: ...

    async def complete_json(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]: ...


class OpenAIClient:
    """Uses OpenAI's chat completions + ``response_format=json_object`` for
    structured output. Requires the ``openai`` optional extra."""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._client: Any = None

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise RuntimeError(
                    "OpenAIClient needs the 'openai' package — "
                    "install with `pip install mnemoss[openai]`"
                ) from e
            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def complete_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    async def complete_json(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
            temperature=temperature,
        )
        raw = response.choices[0].message.content or "{}"
        parsed: dict[str, Any] = json.loads(raw)
        return parsed


class GeminiClient:
    """Uses Google's Gemini API via the ``google-genai`` SDK.

    JSON mode is driven by ``response_mime_type="application/json"``,
    but we still defensively parse the text in case the model slips a
    preamble in. Requires the ``gemini`` optional extra (installs
    ``google-genai``).
    """

    def __init__(
        self,
        *,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._client: Any = None

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from google import genai
            except ImportError as e:
                raise RuntimeError(
                    "GeminiClient needs the 'google-genai' package — "
                    "install with `pip install mnemoss[gemini]`"
                ) from e
            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = genai.Client(**kwargs)
        return self._client

    def _config(
        self,
        *,
        max_tokens: int,
        temperature: float,
        json_mode: bool,
    ) -> Any:
        from google.genai import types as genai_types

        kwargs: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            kwargs["response_mime_type"] = "application/json"
        return genai_types.GenerateContentConfig(**kwargs)

    async def complete_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        client = self._get_client()
        response = await client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._config(
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=False,
            ),
        )
        return (response.text or "").strip()

    async def complete_json(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        client = self._get_client()
        response = await client.aio.models.generate_content(
            model=self._model,
            contents=prompt,
            config=self._config(
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=True,
            ),
        )
        text = (response.text or "").strip()
        if not text:
            return {}
        # response_mime_type usually yields clean JSON, but fall back to the
        # defensive extractor in case the model emits a preamble or fences.
        try:
            parsed: dict[str, Any] = json.loads(text)
            return parsed
        except json.JSONDecodeError:
            return _extract_first_json_object(text)


class AnthropicClient:
    """Uses Anthropic's Messages API. Anthropic doesn't ship a native
    JSON mode, so ``complete_json`` appends an instruction and parses
    the first JSON object it can recover from the response."""

    def __init__(
        self,
        *,
        model: str = "claude-haiku-4-5-20251001",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._client: Any = None

    @property
    def model(self) -> str:
        return self._model

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as e:
                raise RuntimeError(
                    "AnthropicClient needs the 'anthropic' package — "
                    "install with `pip install mnemoss[anthropic]`"
                ) from e
            kwargs: dict[str, Any] = {}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = AsyncAnthropic(**kwargs)
        return self._client

    async def complete_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        client = self._get_client()
        response = await client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return _text_from_anthropic_response(response)

    async def complete_json(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        instructed = (
            prompt + "\n\nRespond with a single JSON object only. No prose, no markdown fences."
        )
        client = self._get_client()
        response = await client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": instructed}],
        )
        text = _text_from_anthropic_response(response)
        return _extract_first_json_object(text)


def _text_from_anthropic_response(response: Any) -> str:
    # Anthropic returns content as a list of blocks; concatenate text parts.
    chunks = []
    for block in getattr(response, "content", []):
        text = getattr(block, "text", None)
        if text is not None:
            chunks.append(text)
    return "".join(chunks)


def _extract_first_json_object(text: str) -> dict[str, Any]:
    """Recover the first JSON object from ``text``.

    Strips markdown fences if present, then finds the outermost
    ``{`` ... ``}`` run and parses it. Raises ``ValueError`` if no
    parseable object is found.
    """

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].startswith("```"):
            lines.pop()
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in response: {text!r}")
    # Walk until we balance braces to handle nested JSON safely.
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(stripped)):
        c = stripped[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                parsed: dict[str, Any] = json.loads(stripped[start : i + 1])
                return parsed
    raise ValueError(f"Unbalanced braces in JSON response: {text!r}")
