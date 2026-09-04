"""Geometry primitives and frame references (design doc 01-core §3).

Units are meters and radians throughout. Quaternions are ``(w, x, y, z)``
with the canonical sign convention ``w >= 0``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, NamedTuple

import numpy as np

from .errors import FrameRefError

Vec3 = np.ndarray  # (3,) float64
Quat = np.ndarray  # (4,) float64, (w, x, y, z), canonical w >= 0

_DEBUG = os.environ.get("APOLLO_CORE_DEBUG") == "1"


def _as_f64(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite, got {arr}")
    if _DEBUG:
        arr = arr.copy()
        arr.setflags(write=False)
    return arr


@dataclass(frozen=True, eq=False)
class Pose:
    """SE3 pose: ``position`` in meters, ``orientation`` as wxyz unit quaternion.

    As a ``Transform`` it maps child-frame coordinates into the parent frame.
    """

    position: Vec3
    orientation: Quat

    def __post_init__(self) -> None:
        # Local import: se3 depends on types for constants only, not on Pose.
        from . import se3

        object.__setattr__(self, "position", _as_f64(self.position, (3,), "position"))
        q = _as_f64(self.orientation, (4,), "orientation")
        norm = float(np.linalg.norm(q))
        if norm < 1e-9:
            raise ValueError("orientation quaternion has ~zero norm")
        q = se3.quat_normalize(q)  # fresh array; safe to freeze
        if _DEBUG:
            q.setflags(write=False)
        object.__setattr__(self, "orientation", q)

    @staticmethod
    def identity() -> Pose:
        return Pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))

    def compose(self, other: Pose) -> Pose:
        from . import se3

        return se3.pose_mul(self, other)

    def inverse(self) -> Pose:
        from . import se3

        return se3.pose_inv(self)

    def to_matrix(self) -> np.ndarray:
        from . import se3

        m = np.eye(4)
        m[:3, :3] = se3.quat_to_mat(self.orientation)
        m[:3, 3] = self.position
        return m

    @classmethod
    def from_matrix(cls, m: np.ndarray) -> Pose:
        from . import se3

        m = np.asarray(m, dtype=np.float64)
        if m.shape != (4, 4):
            raise ValueError(f"expected (4, 4) homogeneous matrix, got {m.shape}")
        return cls(m[:3, 3].copy(), se3.mat_to_quat(m[:3, :3]))


Transform = Pose  # SE3 alias: maps child-frame coords to parent


@dataclass(frozen=True, eq=False)
class Twist:
    """Spatial velocity plus the auxiliary rail / gripper rates."""

    v: Vec3
    w: Vec3
    rail_v: float = 0.0  # m/s
    grip_v: float = 0.0  # open-fraction/s

    def __post_init__(self) -> None:
        object.__setattr__(self, "v", _as_f64(self.v, (3,), "v"))
        object.__setattr__(self, "w", _as_f64(self.w, (3,), "w"))
        for name in ("rail_v", "grip_v"):
            val = float(getattr(self, name))
            if not np.isfinite(val):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, val)

    @staticmethod
    def zero() -> Twist:
        return Twist(np.zeros(3), np.zeros(3))


FrameRef = str  # "world" | "arm_base:<arm_id>" | "camera:<camera_id>" | "ee:<arm_id>"

_FRAME_KINDS: tuple[str, ...] = ("world", "arm_base", "camera", "ee")


class ParsedFrame(NamedTuple):
    kind: Literal["world", "arm_base", "camera", "ee"]
    ident: str | None


def parse_frame(ref: FrameRef) -> ParsedFrame:
    """Parse and validate a FrameRef string; raises :class:`FrameRefError`."""
    if not isinstance(ref, str) or not ref:
        raise FrameRefError(f"frame ref must be a non-empty string, got {ref!r}")
    if ref == "world":
        return ParsedFrame("world", None)
    kind, sep, ident = ref.partition(":")
    if kind == "world":
        raise FrameRefError("'world' takes no identifier")
    if kind not in _FRAME_KINDS:
        raise FrameRefError(f"unknown frame kind {kind!r} in {ref!r}")
    if not sep or not ident:
        raise FrameRefError(f"frame kind {kind!r} requires an identifier: {ref!r}")
    if ":" in ident:
        raise FrameRefError(f"frame identifier may not contain ':': {ref!r}")
    return ParsedFrame(kind, ident)  # type: ignore[arg-type]


def frame_ref(kind: str, ident: str | None = None) -> FrameRef:
    """Build a canonical FrameRef; raises :class:`FrameRefError` on misuse."""
    if kind == "world":
        if ident is not None:
            raise FrameRefError("'world' takes no identifier")
        return "world"
    if kind not in _FRAME_KINDS:
        raise FrameRefError(f"unknown frame kind {kind!r}")
    if not ident:
        raise FrameRefError(f"frame kind {kind!r} requires an identifier")
    ref = f"{kind}:{ident}"
    parse_frame(ref)
    return ref


__all__ = [
    "Vec3",
    "Quat",
    "Pose",
    "Transform",
    "Twist",
    "FrameRef",
    "ParsedFrame",
    "parse_frame",
    "frame_ref",
]
