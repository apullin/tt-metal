# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-only cover for the ping-pong resync that keeps an eager collective off a skewed semaphore.

The getters hand out ping-pong semaphores by alternating a HOST-side parity index. A traced replay
re-runs the recorded collectives on device without going through them, so the host parity stops
tracking the device's last-used semaphore and the next eager collective waits on one its neighbour
has already signalled — a deadlock with no error message. Both halves of the invariant are asserted
here because zeroing either side alone still leaves the two disagreeing, and a pool left out stays
skewed and hangs the next time an eager op reaches for it.

No device: the manager is built without its constructor and the ttnn call is captured, so this runs
in CI, which is where a missed pool has to be caught — on hardware it surfaces as a hang.
"""

import pytest

from ...parallel.manager import CCLManager

# (pool attribute, its parity attribute) — every pool the getters alternate.
POOLS = [
    ("rs_ping_pong_semaphores", "rs_ping_pong_idx"),
    ("rs_ping_pong_semaphores_fused", "rs_ping_pong_idx_fused"),
    ("ag_ping_pong_semaphores", "ag_ping_pong_idx"),
    ("exp_ring_ping_pong_semaphores", "exp_ring_ping_pong_idx"),
    ("np_ping_pong_semaphores", "np_ping_pong_idx"),
    ("sr_ping_pong_semaphores", "sr_ping_pong_idx"),
    ("barrier_semaphores", "barrier_idx"),
]


class FakeSem:
    """Stands in for a device global semaphore; records what it was reset to."""

    def __init__(self, tag):
        self.tag = tag
        self.value = 7  # non-zero: a reset that skips this pool leaves the 7 visible


@pytest.fixture
def manager(monkeypatch):
    """A CCLManager with every pool populated and every parity SKEWED.

    Built via __new__ so no mesh device is needed. The parities start at 1 — the state a replay
    leaves behind — so an implementation that only zeroes device values fails the parity assertions
    instead of passing by accident.
    """
    m = CCLManager.__new__(CCLManager)
    for pool_attr, idx_attr in POOLS:
        setattr(m, pool_attr, {axis: [FakeSem(f"{pool_attr}[{axis}][{i}]") for i in range(2)] for axis in (0, 1)})
        setattr(m, idx_attr, [1, 1])

    reset_calls = []
    monkeypatch.setattr(
        "models.tt_dit.parallel.manager.ttnn.reset_global_semaphore_value",
        lambda sem, val: (reset_calls.append((sem, val)), setattr(sem, "value", val))[0],
    )
    m._reset_calls = reset_calls
    return m


def test_every_pool_is_zeroed(manager):
    """A pool the reset skips stays skewed and deadlocks the next eager op that reaches for it."""
    manager.reset_global_semaphores()

    for pool_attr, _ in POOLS:
        for axis in (0, 1):
            for sem in getattr(manager, pool_attr)[axis]:
                assert sem.value == 0, f"{sem.tag} was left at {sem.value}: pool {pool_attr} not reset"


def test_every_host_parity_is_reset(manager):
    """The device half alone is not enough — the skew that hangs lives in the HOST index."""
    manager.reset_global_semaphores()

    for _, idx_attr in POOLS:
        assert getattr(manager, idx_attr) == [0, 0], f"{idx_attr} still skewed: host and device disagree"


def test_resets_both_axes_of_every_pool(manager):
    """Guards the axis loop: resetting axis 0 only leaves axis 1 to hang on the next collective."""
    manager.reset_global_semaphores()

    expected = sum(len(getattr(manager, pool)[axis]) for pool, _ in POOLS for axis in (0, 1))
    assert len(manager._reset_calls) == expected, "not every semaphore in every axis was reset"
    assert all(val == 0 for _, val in manager._reset_calls), "a semaphore was reset to a non-zero value"
