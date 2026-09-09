"""Hot-path state snapshots (design doc 01-core §4).

Frozen numpy-bearing dataclasses published at up to 100 Hz. Never serialized
directly — telemetry converts to the pydantic models in ``protocol.telemetry``
at 25 Hz. Published arrays are immutable by convention (§16); with
``APOLLO_CORE_DEBUG=1`` immutability is enforced via ``setflags``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .errors import CommandError
from .types import Pose

_DEBUG = os.environ.get("APOLLO_CORE_DEBUG") == "1"


def _freeze(arr: np.ndarray) -> np.ndarray:
    if _DEBUG:
        arr = arr.copy()
        arr.setflags(write=False)
    return arr


@dataclass(frozen=True, eq=False)
class GripperState:
    open_frac: float  # [0,1]; classic gripper: pulses/850 over the 0.085 m span
    moving: bool | None = None  # None = unknowable (gripper fw < 3.4.3)
    grasped: bool | None = None  # grasp status bit / convergence heuristic
    current: float | None = None  # A; only fw >= 2.7.100 monitor stream

    def __post_init__(self) -> None:
        f = float(self.open_frac)
        if not np.isfinite(f) or not 0.0 <= f <= 1.0:
            raise ValueError(f"open_frac must be in [0, 1], got {self.open_frac}")
        object.__setattr__(self, "open_frac", f)


@dataclass(frozen=True, eq=False)
class GripperCommand:
    open_frac: float  # target [0,1]; 1 = fully open
    force: float | None = None  # [0,1]; honored ONLY by force-capable grippers (G2)
    speed: float | None = None  # [0,1]; classic gripper silently ignores

    def __post_init__(self) -> None:
        f = float(self.open_frac)
        if not np.isfinite(f) or not 0.0 <= f <= 1.0:
            raise CommandError(f"GripperCommand.open_frac must be in [0, 1], got {f}")
        object.__setattr__(self, "open_frac", f)
        for name in ("force", "speed"):
            val = getattr(self, name)
            if val is None:
                continue
            val = float(val)
            if not np.isfinite(val) or not 0.0 <= val <= 1.0:
                raise CommandError(f"GripperCommand.{name} must be in [0, 1], got {val}")
            object.__setattr__(self, name, val)


@dataclass(frozen=True, eq=False)
class ArmState:
    arm_id: str
    q: np.ndarray  # (7,)|(8,) rad; q[7] = rail position (m) when present
    dq: np.ndarray  # same shape; rail slot 0.0 (rail velocity unobservable)
    ee_pose: Pose  # TCP in the arm_base frame
    gripper: GripperState
    rail_pos_m: float | None  # None = no rail; == q[7] otherwise
    error_code: int  # xArm codes (0 ok; 22 self-coll, 31 collision, 35 boundary,
    warn_code: int  #   111 rail comms). Sim fills 0.
    mode: int  # controller mode (1 = servo)
    state: int  # controller state (0 = ready)
    stale: bool  # report age > 0.15 s (~15 frames) -> safety gate holds
    t_mono: float  # source report timestamp, monotonic s
    wallclock_ns: int

    def __post_init__(self) -> None:
        q = np.asarray(self.q, dtype=np.float64)
        dq = np.asarray(self.dq, dtype=np.float64)
        if q.shape not in ((7,), (8,)):
            raise ValueError(f"q must have shape (7,) or (8,), got {q.shape}")
        if dq.shape != q.shape:
            raise ValueError(f"dq shape {dq.shape} must match q shape {q.shape}")
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(dq))):
            raise ValueError("q/dq must be finite")
        if q.shape == (8,):
            if self.rail_pos_m is None:
                raise ValueError("8-DoF q requires rail_pos_m (== q[7])")
            if abs(float(self.rail_pos_m) - float(q[7])) > 1e-9:
                raise ValueError(
                    f"rail_pos_m ({self.rail_pos_m}) must equal q[7] ({q[7]})"
                )
            object.__setattr__(self, "rail_pos_m", float(self.rail_pos_m))
        elif self.rail_pos_m is not None:
            raise ValueError("7-DoF q requires rail_pos_m=None")
        object.__setattr__(self, "q", _freeze(q))
        object.__setattr__(self, "dq", _freeze(dq))

    @property
    def dof(self) -> int:
        return int(self.q.shape[0])

    @property
    def has_rail(self) -> bool:
        return self.rail_pos_m is not None


@dataclass(frozen=True, eq=False)
class CameraFrame:
    camera_id: str
    rgb: np.ndarray  # (H, W, 3) uint8, RGB order
    t_mono: float
    wallclock_ns: int
    seq: int = 0
    # additive (phase-12; 14-dora §4.2 "Depth"): an aligned depth image, (H, W) uint16 in
    # units of ``depth_scale_m`` (mm at the default), same H/W as ``rgb``; None = the
    # producer has no depth (v4l2 / sim colour-only). Recorder and VideoHub ignore it.
    depth: np.ndarray | None = None
    depth_scale_m: float = 0.001

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must have shape (H, W, 3), got {rgb.shape}")
        if rgb.dtype != np.uint8:
            raise ValueError(f"rgb must be uint8, got {rgb.dtype}")
        object.__setattr__(self, "rgb", _freeze(rgb))
        if self.depth is not None:
            depth = np.asarray(self.depth)
            if depth.shape != rgb.shape[:2]:
                raise ValueError(
                    f"depth must have shape {rgb.shape[:2]} (the rgb H, W), got {depth.shape}"
                )
            if depth.dtype != np.uint16:
                raise ValueError(f"depth must be uint16, got {depth.dtype}")
            object.__setattr__(self, "depth", _freeze(depth))
        scale = float(self.depth_scale_m)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"depth_scale_m must be > 0, got {self.depth_scale_m}")
        object.__setattr__(self, "depth_scale_m", scale)
