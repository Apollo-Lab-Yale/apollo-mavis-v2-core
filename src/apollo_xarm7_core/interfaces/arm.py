"""Arm implementation facade (design doc 01-core §5.1).

Implemented by ``apollo_xarm7_hardware`` (xArm SDK) and ``apollo_xarm7_sim``
(MuJoCo); consumed by the runtime control loop, which treats both identically
(spine §3.3).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..state import ArmState, GripperCommand


class ArmInterface(ABC):
    """One arm (7 joints, optional linear rail as an 8th slot)."""

    @abstractmethod
    def connect(self) -> None:
        """Blocking bring-up; raises the typed §16 exceptions on failure."""

    @abstractmethod
    def disconnect(self) -> None:
        """Tear down the connection; idempotent (safe when never connected)."""

    @abstractmethod
    def get_state(self) -> ArmState:
        """Lock-free latest snapshot; never blocks (staleness via ``stale``)."""

    @abstractmethod
    def command_joints(self, q: np.ndarray) -> None:
        """Set the latest-wins joint target for the 100 Hz servo streamer.

        Non-blocking: only stores the target. ``len(q) == dof``; rad, with
        ``q[7]`` = rail position (m) when present — the rail slot feeds the
        sparse rail path. NaN or wrong shape raises :class:`CommandError`
        (the runtime ControlLoop — the single gate chokepoint — sanitizes
        first).
        """

    @abstractmethod
    def command_gripper(self, cmd: GripperCommand) -> None:
        """Queue a gripper command; <= 10 Hz advised."""

    @abstractmethod
    def command_rail(self, pos_m: float) -> None:
        """Absolute rail target, clamped to ``[0, se3.RAIL_TRAVEL_M]``.

        Raises :class:`RailUnavailableError` when ``not has_rail``.
        """

    @abstractmethod
    def clear_errors(self) -> None:
        """User-initiated recovery (LATCHED -> RECOVERING)."""

    @abstractmethod
    def stop(self) -> None:
        """Software stop; idempotent (safe to call twice)."""

    @property
    @abstractmethod
    def dof(self) -> int:
        """7 or 8 (rail arms); fixed at connect."""

    @property
    @abstractmethod
    def has_rail(self) -> bool:
        """Whether the 8th (rail) slot exists; fixed at connect."""

    @property
    @abstractmethod
    def gripper_force_capable(self) -> bool:
        """True only for force-capable grippers (xarm_g2)."""
