"""Background dream scheduling.

Mnemoss's dream triggers normally fire only when the caller invokes
``mem.dream(...)``. The scheduler automates the two trigger types that
depend on wall-clock time rather than in-context signals:

- **nightly** — fires once per calendar day at a configured time
- **idle**    — fires after N seconds of no ``observe()`` activity

The three remaining triggers (``session_end``, ``surprise``,
``cognitive_load``) stay caller-driven because they reflect semantic
events the scheduler can't detect.
"""

from mnemoss.scheduler.scheduler import DreamScheduler, SchedulerConfig

__all__ = ["DreamScheduler", "SchedulerConfig"]
