"""Command bus & typed latest-value queues (design doc 01-core §15).

The Dora-migration seam (spine §2): discrete ops flow through one MPSC bus
with correlation IDs; high-rate data through depth-1 :class:`LatestSlot`s.
Core owns both primitives; the runtime instantiates and wires them
(04-runtime §4). Both are thread-*safe* but thread-*owning* nothing — core
never starts threads (§16).
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Command:
    """One discrete operation submitted to the control loop."""

    op: str  # ActionName values + internal ops ("end_session", "load_profile", ...)
    args: dict[str, Any] = field(default_factory=dict)
    corr_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    source: Literal["ws", "rest", "internal", "dora"] = "ws"  # "dora": external bus (14-dora)


@dataclass(frozen=True)
class CommandResult:
    """Outcome of one :class:`Command`; ``detail`` surfaces as ``AckMsg.detail``."""

    corr_id: str
    ok: bool
    detail: str = ""


class CommandBus:
    """Thread-safe MPSC command queue.

    Producers: any thread / asyncio loop (asyncio callers await the returned
    future via ``asyncio.wrap_future``). Consumer: ONE thread — the runtime
    ControlLoop's — drains at each tick boundary. Every submitted future
    resolves exactly once.
    """

    def __init__(self, maxsize: int = 256) -> None:
        self._maxsize = int(maxsize)
        self._lock = threading.Lock()
        self._queue: deque[tuple[Command, Future[CommandResult]]] = deque()

    def submit(self, cmd: Command) -> Future[CommandResult]:
        """Enqueue ``cmd``; returns a future keyed by ``cmd.corr_id``.

        A full queue resolves the future immediately with ``ok=False,
        detail="bus full"`` — a WS handler is never blocked.
        """
        future: Future[CommandResult] = Future()
        with self._lock:
            if len(self._queue) < self._maxsize:
                self._queue.append((cmd, future))
                return future
        future.set_result(CommandResult(corr_id=cmd.corr_id, ok=False, detail="bus full"))
        return future

    def drain(self, handler: Callable[[Command], CommandResult]) -> int:
        """Pop all queued commands, run ``handler`` synchronously on each, and
        resolve each future with its result.

        A handler exception resolves as ``ok=False, detail=repr(exc)``; long
        ops should resolve immediately ``ok=True, detail="accepted"`` and
        report progress via telemetry. Returns the number of commands drained.
        """
        with self._lock:
            batch = list(self._queue)
            self._queue.clear()
        for cmd, future in batch:
            # Skip commands whose future was cancelled while queued (e.g. via
            # asyncio.wrap_future): cancel() returning True promises the call
            # never runs. After this the future is RUNNING and uncancellable.
            if not future.set_running_or_notify_cancel():
                continue
            try:
                result = handler(cmd)
            except Exception as exc:
                result = CommandResult(corr_id=cmd.corr_id, ok=False, detail=repr(exc))
            future.set_result(result)
        return len(batch)


class LatestSlot(Generic[T]):
    """Depth-1 latest-value slot (lock + ``threading.Event``); with
    :class:`CommandBus` the ONLY inter-thread structure in the stack.

    One writer per slot; readers never mutate payloads (§15 conventions).
    Staleness decisions use the value's own ``t_mono``, not ``put_mono``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._event = threading.Event()  # set by the NEXT put; swapped each put
        self._value: T | None = None
        self._put_mono = 0.0
        self._has_value = False

    def put(self, value: T) -> None:
        """Publish ``value``, overwriting any unread predecessor."""
        with self._lock:
            self._value = value
            self._put_mono = time.monotonic()
            self._has_value = True
            event, self._event = self._event, threading.Event()
        event.set()

    def get(self) -> tuple[T, float] | None:
        """Newest ``(value, put_mono)``; ``None`` if nothing was ever put."""
        with self._lock:
            if not self._has_value:
                return None
            return (self._value, self._put_mono)  # type: ignore[return-value]

    def wait_fresh(self, timeout: float) -> tuple[T, float] | None:
        """Block for the next :meth:`put` AFTER this call starts.

        Returns the newest ``(value, put_mono)`` once a fresh put lands within
        ``timeout`` seconds; values put before the call never satisfy it.
        Returns ``None`` on timeout. Safe for multiple concurrent waiters.
        """
        with self._lock:
            event = self._event
        if not event.wait(timeout):
            return None
        with self._lock:
            return (self._value, self._put_mono)  # type: ignore[return-value]
