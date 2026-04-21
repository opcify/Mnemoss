"""Working-memory buffer tests."""

from __future__ import annotations

from mnemoss.working import WorkingMemory


def test_fifo_evicts_oldest() -> None:
    wm = WorkingMemory(capacity=3)
    for mid in ["a", "b", "c", "d"]:
        wm.append("alice", mid)
    # "a" was evicted; order is b, c, d
    assert wm.active_set("alice") == ["b", "c", "d"]


def test_append_moves_existing_to_most_recent() -> None:
    wm = WorkingMemory(capacity=3)
    for mid in ["a", "b", "c"]:
        wm.append("alice", mid)
    wm.append("alice", "a")  # re-add
    assert wm.active_set("alice") == ["b", "c", "a"]


def test_per_agent_isolation() -> None:
    wm = WorkingMemory(capacity=3)
    wm.append("alice", "m1")
    wm.append("bob", "m2")
    # Ambient buffer is independent
    wm.append(None, "m_shared")
    # Alice sees her own + ambient; Bob sees his own + ambient.
    assert "m1" in wm.active_set("alice")
    assert "m_shared" in wm.active_set("alice")
    assert "m2" not in wm.active_set("alice")
    assert "m2" in wm.active_set("bob")


def test_ambient_caller_sees_only_ambient() -> None:
    wm = WorkingMemory()
    wm.append("alice", "private")
    wm.append(None, "ambient")
    assert wm.active_set(None) == ["ambient"]


def test_clear_scoped() -> None:
    wm = WorkingMemory()
    wm.append("alice", "m1")
    wm.append(None, "m2")
    wm.clear("alice")
    assert wm.active_set("alice") == ["m2"]  # only ambient left
    assert wm.active_set(None) == ["m2"]
