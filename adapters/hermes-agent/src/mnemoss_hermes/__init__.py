"""Mnemoss memory plugin for Hermes Agent.

See the package-level README for installation. The entry points are the
Hermes plugin ``register(ctx)`` hook and the :class:`MnemossMemoryProvider`
class, either of which can be wired into a Hermes deployment.
"""

from mnemoss_hermes.provider import MnemossMemoryProvider, register

__all__ = ["MnemossMemoryProvider", "register"]
