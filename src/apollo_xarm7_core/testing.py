"""Canonical deterministic fakes (design doc 01-core §18).

THE fake set for the hardware/sim/runtime test suites — pure python + numpy,
no threads, no wall clock: :meth:`FakeArm.step` advances an internal
monotonic time, so tests are fully deterministic.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .errors import BringupError, CommandError, RailUnavailableError, WorkcellBringupError
from .interfaces.arm import ArmInterface
from .interfaces.camera import CameraInterface
from .interfaces.workcell import WorkcellInterface
from .se3 import RAIL_TRAVEL_M
from .state import ArmState, CameraFrame, GripperCommand, GripperState
from .types import Pose


class FakeArm(ArmInterface):
    """Deterministic in-memory :class:`ArmInterface`.

    ``command_joints`` stores a latest-wins target; :meth:`step` slews ``q``
    toward it at ``max_joint_speed_rad_s`` (rail slot at
    ``max_rail_speed_m_s``). ``ee_pose`` is ``Pose.identity()`` unless a
    simple FK stub ``fk(q) -> Pose`` is provided.
    """

    def __init__(
        self,
        arm_id: str = "arm0",
        *,
        has_rail: bool = False,
        gripper_force_capable: bool = False,
        q0: np.ndarray | None = None,
        max_joint_speed_rad_s: float = 1.0,
        max_rail_speed_m_s: float = 0.2,
        fk=None,
    ) -> None:
        self.arm_id = arm_id
        self._has_rail = bool(has_rail)
        self._gripper_force_capable = bool(gripper_force_capable)
        self._dof = 8 if has_rail else 7
        q = np.zeros(self._dof) if q0 is None else np.asarray(q0, dtype=np.float64).copy()
        if q.shape != (self._dof,):
            raise ValueError(f"q0 must have shape ({self._dof},), got {q.shape}")
        self._q = q
        self._dq = np.zeros(self._dof)
        self._target = q.copy()
        self._max_joint_speed = float(max_joint_speed_rad_s)
        self._max_rail_speed = float(max_rail_speed_m_s)
        self._fk = fk
        self._t_mono = 0.0
        self._open_frac = 1.0
        self._error_code = 0
        self._warn_code = 0
        self.connected = False
        self.gripper_commands: list[GripperCommand] = []

    # -- lifecycle -------------------------------------------------------
    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def stop(self) -> None:
        """Halt: pin the target to the current configuration; safe twice."""
        self._target = self._q.copy()

    # -- commands (latest-wins, non-blocking) -----------------------------
    def command_joints(self, q: np.ndarray) -> None:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (self._dof,):
            raise CommandError(f"expected shape ({self._dof},), got {q.shape}")
        if not np.all(np.isfinite(q)):
            raise CommandError(f"joint target must be finite, got {q}")
        target = q.copy()
        if self._has_rail:
            target[7] = min(max(target[7], 0.0), RAIL_TRAVEL_M)
        self._target = target

    def command_gripper(self, cmd: GripperCommand) -> None:
        self.gripper_commands.append(cmd)
        self._open_frac = cmd.open_frac

    def command_rail(self, pos_m: float) -> None:
        if not self._has_rail:
            raise RailUnavailableError(f"arm {self.arm_id!r} has no rail")
        pos_m = float(pos_m)
        if not np.isfinite(pos_m):
            raise CommandError(f"rail target must be finite, got {pos_m}")
        self._target[7] = min(max(pos_m, 0.0), RAIL_TRAVEL_M)

    # -- error injection ---------------------------------------------------
    def inject_error(self, code: int, warn: int = 0) -> None:
        """Latch an xArm-style error/warn code until :meth:`clear_errors`."""
        self._error_code = int(code)
        self._warn_code = int(warn)

    def clear_errors(self) -> None:
        self._error_code = 0
        self._warn_code = 0

    # -- deterministic time ------------------------------------------------
    def step(self, dt: float) -> None:
        """Advance internal time by ``dt`` s, slewing ``q`` toward the target."""
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        max_delta = np.full(self._dof, self._max_joint_speed * dt)
        if self._has_rail:
            max_delta[7] = self._max_rail_speed * dt
        prev = self._q
        self._q = prev + np.clip(self._target - prev, -max_delta, max_delta)
        self._dq = (self._q - prev) / dt
        if self._has_rail:
            self._dq[7] = 0.0  # rail velocity unobservable (§4)
        self._t_mono += dt

    def get_state(self) -> ArmState:
        return ArmState(
            arm_id=self.arm_id,
            q=self._q.copy(),
            dq=self._dq.copy(),
            ee_pose=self._fk(self._q) if self._fk is not None else Pose.identity(),
            gripper=GripperState(open_frac=self._open_frac),
            rail_pos_m=float(self._q[7]) if self._has_rail else None,
            error_code=self._error_code,
            warn_code=self._warn_code,
            mode=1,
            state=0,
            stale=False,
            t_mono=self._t_mono,
            wallclock_ns=int(self._t_mono * 1e9),
        )

    # -- properties ----------------------------------------------------------
    @property
    def dof(self) -> int:
        return self._dof

    @property
    def has_rail(self) -> bool:
        return self._has_rail

    @property
    def gripper_force_capable(self) -> bool:
        return self._gripper_force_capable


class FakeCamera(CameraInterface):
    """Deterministic :class:`CameraInterface`: every :meth:`latest` call after
    :meth:`start` yields a fresh gradient frame with an incrementing ``seq``
    (internal time advances by ``1/fps`` per frame; no capture thread)."""

    def __init__(
        self,
        camera_id: str = "cam0",
        resolution: tuple[int, int] = (64, 48),
        fps: float = 30.0,
    ) -> None:
        self._camera_id = camera_id
        self._resolution = (int(resolution[0]), int(resolution[1]))
        self._fps = float(fps)
        self._seq = 0
        self._t_mono = 0.0
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.started = False

    def latest(self) -> CameraFrame | None:
        if not self.started:
            return None
        w, h = self._resolution
        seq = self._seq
        rgb = np.empty((h, w, 3), dtype=np.uint8)
        rgb[..., 0] = ((np.arange(w) + seq) % 256)[None, :]
        rgb[..., 1] = ((np.arange(h) + seq) % 256)[:, None]
        rgb[..., 2] = seq % 256
        frame = CameraFrame(
            camera_id=self._camera_id,
            rgb=rgb,
            t_mono=self._t_mono,
            wallclock_ns=int(self._t_mono * 1e9),
            seq=seq,
        )
        self._seq += 1
        self._t_mono += 1.0 / self._fps
        return frame

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    @property
    def fps(self) -> float:
        return self._fps


class FakeWorkcell(WorkcellInterface):
    """Deterministic :class:`WorkcellInterface` bundling fakes.

    ``start``/``stop`` are idempotent and keep call-count bookkeeping;
    :meth:`step` advances every arm's internal clock by ``dt``.
    """

    def __init__(
        self,
        arms: dict[str, FakeArm] | None = None,
        cameras: dict[str, FakeCamera] | None = None,
        kind: Literal["hardware", "sim"] = "sim",
    ) -> None:
        self.arms: dict[str, ArmInterface] = dict(
            arms if arms is not None else {"arm0": FakeArm("arm0")}
        )
        self.cameras: dict[str, CameraInterface] = dict(
            cameras if cameras is not None else {"cam0": FakeCamera("cam0")}
        )
        self._kind: Literal["hardware", "sim"] = kind
        self.started = False
        self.start_calls = 0
        self.stop_calls = 0

    @property
    def kind(self) -> Literal["hardware", "sim"]:
        return self._kind

    def start(self) -> None:
        self.start_calls += 1
        if self.started:
            return
        statuses: dict[str, BringupError | None] = {}
        for arm_id, arm in self.arms.items():
            try:
                arm.connect()
                statuses[arm_id] = None
            except BringupError as exc:  # sibling arms still brought up
                statuses[arm_id] = exc
        for camera_id, camera in self.cameras.items():
            try:
                camera.start()
                statuses[camera_id] = None
            except BringupError as exc:
                statuses[camera_id] = exc
        self.started = True
        if any(v is not None for v in statuses.values()):
            raise WorkcellBringupError(statuses)

    def stop(self) -> None:
        self.stop_calls += 1
        if not self.started:
            return
        for camera in reversed(list(self.cameras.values())):
            camera.stop()
        for arm in reversed(list(self.arms.values())):
            arm.stop()
            arm.disconnect()
        self.started = False

    def states(self) -> dict[str, ArmState]:
        return {arm_id: arm.get_state() for arm_id, arm in self.arms.items()}

    def step(self, dt: float) -> None:
        """Advance every arm's deterministic clock by ``dt`` s."""
        for arm in self.arms.values():
            if isinstance(arm, FakeArm):
                arm.step(dt)
