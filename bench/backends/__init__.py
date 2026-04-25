"""Backend adapters for the launch-comparison benchmark.

Each adapter implements the ``MemoryBackend`` protocol in ``base.py``.
Backends sit behind a common async interface so the benchmark harness
in ``bench/launch_comparison.py`` can call ``observe`` and ``recall``
uniformly across Mnemoss, Mem0, and naive-RAG (Chroma).
"""

from bench.backends.base import MemoryBackend, RecallHit

__all__ = ["MemoryBackend", "RecallHit"]
