"""Core types and configuration.

This subpackage holds the pure-data types that every other package
consumes: ``Memory``, ``Event``, ``RawMessage`` (core/types.py) and
the ``*Params`` dataclasses + ``SCHEMA_VERSION`` constant
(core/config.py). Nothing in here imports from the rest of Mnemoss
— it's the dependency floor.
"""

from mnemoss.core.config import (
    SCHEMA_VERSION,
    EncoderParams,
    FormulaParams,
    MnemossConfig,
    SegmentationParams,
    StorageParams,
)
from mnemoss.core.types import (
    Event,
    IndexTier,
    Memory,
    MemoryType,
    RawMessage,
    Relation,
    Tombstone,
)

__all__ = [
    "SCHEMA_VERSION",
    "EncoderParams",
    "Event",
    "FormulaParams",
    "IndexTier",
    "Memory",
    "MemoryType",
    "MnemossConfig",
    "RawMessage",
    "Relation",
    "SegmentationParams",
    "StorageParams",
    "Tombstone",
]
