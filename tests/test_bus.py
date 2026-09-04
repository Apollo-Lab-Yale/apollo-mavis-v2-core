"""CommandBus / LatestSlot semantics (design doc 01-core §15, §18)."""

from __future__ import annotations

import threading
import time

from apollo_mavis_v2_core.bus import Command, CommandBus, CommandResult, LatestSlot

N_PRODUCERS = 8
CMDS_PER_PRODUCER = 25


def test_mpsc_every_future_resolves_exactly_once() -> None:
    bus = CommandBus(maxsize=N_PRODUCERS * CMDS_PER_PRODUCER)
    submitted: dict[str, object] = {}
    submitted_lock = threading.Lock()
    handled: list[str] = []
    start = threading.Barrier(N_PRODUCERS + 1)
    done = threading.Event()

    def produce() -> None:
        start.wait()
        for i in range(CMDS_PER_PRODUCER):
            cmd = Command(op="noop", args={"i": i})
            fut = bus.submit(cmd)
            with submitted_lock:
                submitted[cmd.corr_id] = fut

    def handler(cmd: Command) -> CommandResult:
        handled.append(cmd.corr_id)
        return CommandResult(corr_id=cmd.corr_id, ok=True, detail="ok")

    def drainer() -> None:
        start.wait()
        total = 0
        deadline = time.monotonic() + 10.0
        while total < N_PRODUCERS * CMDS_PER_PRODUCER and time.monotonic() < deadline:
            total += bus.drain(handler)
        done.set()

    threads = [threading.Thread(target=produce) for _ in range(N_PRODUCERS)]
    threads.append(threading.Thread(target=drainer))
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)
    assert done.is_set()

    assert len(submitted) == N_PRODUCERS * CMDS_PER_PRODUCER
    # exactly once: every corr_id handled a single time, every future resolved
    assert sorted(handled) == sorted(submitted)
    for corr_id, fut in submitted.items():
        assert fut.done()
        result = fut.result(timeout=0)
        assert result.ok
        assert result.corr_id == corr_id


def test_bus_full_immediate_nack() -> None:
    bus = CommandBus(maxsize=2)
    kept = [bus.submit(Command(op="a")), bus.submit(Command(op="b"))]
    overflow_cmd = Command(op="c")
    overflow = bus.submit(overflow_cmd)
    assert overflow.done()  # resolved immediately, producer never blocked
    result = overflow.result(timeout=0)
    assert result == CommandResult(corr_id=overflow_cmd.corr_id, ok=False, detail="bus full")
    assert not kept[0].done() and not kept[1].done()

    drained = bus.drain(lambda cmd: CommandResult(corr_id=cmd.corr_id, ok=True))
    assert drained == 2
    assert all(f.result(timeout=0).ok for f in kept)


def test_handler_exception_resolves_not_ok() -> None:
    bus = CommandBus()
    cmd = Command(op="boom")
    fut = bus.submit(cmd)
    exc = RuntimeError("kaboom")

    def handler(_: Command) -> CommandResult:
        raise exc

    assert bus.drain(handler) == 1
    result = fut.result(timeout=0)
    assert not result.ok
    assert result.corr_id == cmd.corr_id
    assert result.detail == repr(exc)


def test_command_defaults() -> None:
    a, b = Command(op="x"), Command(op="x")
    assert a.corr_id != b.corr_id and len(a.corr_id) == 32
    assert a.source == "ws" and a.args == {}


def test_latest_slot_overwrite() -> None:
    slot: LatestSlot[int] = LatestSlot()
    assert slot.get() is None
    slot.put(1)
    slot.put(2)
    got = slot.get()
    assert got is not None
    value, put_mono = got
    assert value == 2
    assert put_mono <= time.monotonic()
    # get() does not consume
    assert slot.get() is not None


def test_latest_slot_wait_fresh_timeout() -> None:
    slot: LatestSlot[int] = LatestSlot()
    t0 = time.monotonic()
    assert slot.wait_fresh(timeout=0.05) is None
    assert time.monotonic() - t0 < 1.0
    # a value put BEFORE the call never satisfies wait_fresh
    slot.put(7)
    assert slot.wait_fresh(timeout=0.05) is None


def test_latest_slot_wait_fresh_wakes_on_put() -> None:
    slot: LatestSlot[str] = LatestSlot()
    slot.put("stale")

    def put_later() -> None:
        time.sleep(0.05)
        slot.put("fresh")

    t = threading.Thread(target=put_later)
    t.start()
    got = slot.wait_fresh(timeout=5.0)
    t.join()
    assert got is not None
    assert got[0] == "fresh"
