"""Benchmark harnesses for Mnemoss.

These are standalone runnable scripts, not pytest tests — run a
benchmark with ``python -m bench.bench_recall``. Benchmarks live
outside ``tests/`` so they're never part of the default test suite
(CI stays fast) but are easy to execute when tuning or investigating
regressions.

Each bench script prints a human-readable table + emits the same
numbers as JSON on the last line so an external tracker can parse
them.
"""
