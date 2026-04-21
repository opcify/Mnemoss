# ruff: noqa: B027
# Optional hooks intentionally have empty bodies without @abstractmethod —
# they mirror Hermes's actual MemoryProvider contract where subclasses
# override only the hooks they care about. B027 flags this as a smell
# for "real" ABCs, but the pattern is load-bearing here.
"""Fallback ``MemoryProvider`` base class for development outside Hermes.

When this package is installed as a Hermes plugin at
``plugins/memory/mnemoss/``, the real base class comes from
``agent.memory_provider``. When running unit tests standalone (no
hermes-agent on the path), this minimal stub provides the same ABC
surface so the provider class can still import, subclass, and be tested.

Keep this stub in lockstep with the actual Hermes contract; if Hermes
adds a required abstract method, mirror it here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MemoryProvider(ABC):
    """Stand-in for ``agent.memory_provider.MemoryProvider``.

    Only the abstract bits Hermes requires are declared. Everything the
    concrete provider overrides is left as plain methods here so the
    subclass's ``super()`` calls (if any) land on a benign default.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def initialize(self, session_id: str, **kwargs: Any) -> None: ...

    @abstractmethod
    def get_tool_schemas(self) -> list[dict[str, Any]]: ...

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None: ...

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None: ...

    def handle_tool_call(
        self, tool_name: str, args: dict[str, Any], **kwargs: Any
    ) -> str:
        raise NotImplementedError

    def shutdown(self) -> None: ...

    def on_turn_start(
        self, turn_number: int, message: str, **kwargs: Any
    ) -> None: ...

    def on_session_end(self, messages: list[dict[str, Any]]) -> None: ...

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        return ""

    def on_delegation(
        self,
        task: str,
        result: str,
        *,
        child_session_id: str = "",
        **kwargs: Any,
    ) -> None: ...

    def on_memory_write(self, action: str, target: str, content: str) -> None: ...

    def get_config_schema(self) -> list[dict[str, Any]]:
        return []

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None: ...
