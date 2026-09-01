"""Teleop held-key provider contract (design doc 01-core §5.2).

The runtime's WebSocket bridge implements this; the control loop consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class HeldState:
    """Snapshot of the controller's currently held movement keys."""

    held: frozenset[str]  # KeyboardEvent.code values, movement keys only
    seq: int  # KeysMsg.seq (monotonic; stale seq dropped upstream)
    rx_mono: float  # SERVER receive time — the watchdog feed

    def __post_init__(self) -> None:
        object.__setattr__(self, "held", frozenset(self.held))


@runtime_checkable
class TeleopInput(Protocol):
    """Latest-value view of the held-key channel."""

    def latest(self) -> HeldState | None:
        """Newest held-state; ``None`` = no controller connected."""
        ...
