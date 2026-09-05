"""Workcell implementation facade (design doc 01-core §5.1).

Bundles the arms and cameras of one physical or simulated cell behind a
single bring-up/tear-down surface so the runtime treats hardware and sim
identically (spine §3.3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from ..state import ArmState
from .arm import ArmInterface
from .camera import CameraInterface


class WorkcellInterface(ABC):
    """A set of arms + cameras with collective lifecycle."""

    arms: dict[str, ArmInterface]  # keyed by ArmConfig.id, config order
    cameras: dict[str, CameraInterface]

    @property
    @abstractmethod
    def kind(self) -> Literal["hardware", "sim"]: ...

    @abstractmethod
    def start(self) -> None:
        """Parallel per-arm bring-up.

        Component failures are collected into
        :class:`WorkcellBringupError` (``.statuses`` maps component id to its
        :class:`BringupError` or ``None``) — one arm failing is not fatal to
        its siblings. Sim implementations also start their monotonic-paced
        ``mj_step`` thread here so the runtime sees a live cell either way.
        """

    @abstractmethod
    def stop(self) -> None:
        """Tear down in reverse bring-up order; idempotent."""

    @abstractmethod
    def states(self) -> dict[str, ArmState]:
        """Latest snapshot per arm (same keys as :attr:`arms`)."""

    def drain_events(self) -> list[Any]:
        """Pop the driver events queued since the last call (phase-09b; additive).

        Hardware workcells return the per-arm driver events (fault, recovered,
        reseed, Studio-conflict, rail, gripper, stale) the runtime control loop
        consumes once per tick to drive FAULT -> RECOVERING -> RUNNING
        (04-runtime §15); the events are hardware-package types, hence
        ``Any``. Non-abstract: sim and fake workcells inherit this ``[]``.
        """
        return []
