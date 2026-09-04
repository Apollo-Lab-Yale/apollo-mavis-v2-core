"""Session and REST body models (design doc 01-core §12).

Bodies for the ``/api`` surface (04-runtime §13.1); runtime defines no wire
model of its own.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

from apollo_mavis_v2_core.types import FrameRef, parse_frame

Mode = Literal["teleop", "collect", "dagger", "inference"]

START_FROM_RE = r"^(keep_current|profile:[A-Za-z0-9_\-]+)$"
_START_FROM_PATTERN = re.compile(START_FROM_RE)


class SessionSpec(BaseModel):
    """POST /api/session body."""

    mode: Mode
    kind: Literal["hardware", "sim"]  # honored iff config available (binding; else 409)
    arms: list[str]  # participating arm ids
    frames: dict[str, FrameRef]  # per-arm RECORDING frame; never affects control math
    sim_scene: str | None = None  # required when kind == "sim"
    digital_twin_scene: str | None = None  # required when kind == "hardware"
    start_from: str = "keep_current"  # START_FROM_RE
    task: str | None = None  # dataset task string (collect/dagger: required)
    policy: str | None = None  # checkpoint id (dagger/inference); None = latest /
    #   promoted deploy ckpt (409 if none promoted — 12-dagger §9)

    @field_validator("start_from")
    @classmethod
    def _start_from_matches(cls, v: str) -> str:
        if not _START_FROM_PATTERN.fullmatch(v):
            raise ValueError(f"start_from must match {START_FROM_RE}, got {v!r}")
        return v

    @model_validator(mode="after")
    def _cross_field(self) -> SessionSpec:
        extra = set(self.frames) - set(self.arms)
        if extra:
            raise ValueError(f"frames keys must be a subset of arms: {sorted(extra)}")
        for arm_id, ref in self.frames.items():
            parsed = parse_frame(ref)  # raises FrameRefError (a ValueError)
            if parsed.kind == "ee":
                raise ValueError(f"frames[{arm_id!r}] may not be an ee: frame: {ref!r}")
        if self.mode in ("collect", "dagger") and not self.task:
            raise ValueError(f"mode {self.mode!r} requires a task string")
        return self


class SessionInfo(BaseModel):
    """POST/GET /api/session response."""

    session_id: str
    epoch: str
    mode: Mode
    arms: list[str]
    streams: list[str]  # video ids: camera ids + "sim" and/or "twin"
    state: str  # SessionState value


class ArmStatusInfo(BaseModel):
    """Landing-page arm card."""

    arm_id: str
    ip: str | None
    connected: bool
    has_rail: bool
    gripper: Literal["xarm", "xarm_g2", "none"]
    gripper_force_capable: bool
    error_code: int
    joint_limits: list[tuple[float, float]]  # 7 rad pairs; + [0.0, 0.65] m for rail arms


class CameraInfo(BaseModel):
    """Landing-page camera card."""

    camera_id: str
    kind: Literal["v4l2", "realsense", "sim"]
    label: str
    resolution: tuple[int, int]
    fps: int
    live: bool  # pre-session preview available (~15 fps)


class WorkcellStatus(BaseModel):
    """GET /api/workcell response."""

    kind: Literal["hardware", "sim"]
    available_kinds: list[Literal["hardware", "sim"]]
    arms: list[ArmStatusInfo]
    cameras: list[CameraInfo]
    policies_available: bool = False  # enables DAgger/Inference launch (05-ui §8.1)


class SceneInfo(BaseModel):
    """GET /api/scenes?kind=sim|twin row."""

    scene_id: str
    label: str
    num_arms: int
    rail_flags: list[bool]
    cameras: list[str]
    kind: Literal["sim", "twin"]


class ProfileInfo(BaseModel):
    """GET /api/profiles row (full posture via GET /api/profiles/{id})."""

    profile_id: str
    name: str
    arms: list[str]
    notes: str
    created_at: str
    is_initial_condition: bool


class PolicyInfo(BaseModel):
    """GET /api/policies row (04-runtime §13.1)."""

    policy_id: str  # "{run_id}/v{n:06d}" | "{run_id}/deploy/v{k:03d}"
    path: str
    action_space: Literal["delta_ee", "abs_ee", "joint"]
    action_frame: str  # FrameRef: "arm_base:<id>" | "world" | "camera:<id>"
    policy_version: int
    promoted: bool = False  # deploy checkpoints only (12-dagger §9)


__all__ = [
    "Mode",
    "START_FROM_RE",
    "SessionSpec",
    "SessionInfo",
    "ArmStatusInfo",
    "CameraInfo",
    "WorkcellStatus",
    "SceneInfo",
    "ProfileInfo",
    "PolicyInfo",
]
