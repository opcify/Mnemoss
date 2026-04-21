"""LLM client abstractions used by Dreaming phases that require text
generation (Extract, Refine, Generalize).

Mnemoss never lets an LLM make *system decisions* — the formula
governs retrieval ranking, tier migration, and disposal — but LLMs are
used for content generation inside the Cold Path. The Protocol here
lets callers swap providers without changing the dreaming code.
"""

from mnemoss.llm.client import (
    AnthropicClient,
    LLMClient,
    OpenAIClient,
)
from mnemoss.llm.mock import MockLLMClient

__all__ = [
    "AnthropicClient",
    "LLMClient",
    "MockLLMClient",
    "OpenAIClient",
]
