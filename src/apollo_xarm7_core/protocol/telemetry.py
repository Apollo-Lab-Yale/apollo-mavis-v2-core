"""Telemetry WebSocket messages (design doc 01-core §11).

Server -> all ``/ws/telemetry`` clients at 25 Hz (20-30 band). Shapes match
05-ui §2; *additive* fields extend the UI shape (the UI ignores unknowns).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from apollo_xarm7_core.dagger.types import ControlMode, TrainerStatus
from apollo_xarm7_core.schemas.safety import CollisionReport


class PoseMsg(BaseModel):
    """Wire pose: position in m, orientation as wxyz unit quaternion."""

    position: tuple[float, float, float]  # m
    orientation: tuple[float, float, float, float]  # wxyz


class ArmTelemetry(BaseModel):
    """Per-arm telemetry block."""

    arm_id: str
    connected: bool
    q: list[float]  # rad, len 7
    rail_pos_m: float | None  # None = no rail; 0-0.65
    ee_pose: PoseMsg
    gripper_open_frac: float
    error_code: int  # 0 = ok (xArm code otherwise)
    warn_code: int = 0  # additive
    stale: bool = False  # additive
    goto: Literal["planning", "executing", "failed"] | None = None
    # joint-panel lifecycle; "failed" transient


class ClearanceItem(BaseModel):
    """One monitored geometry pair at the measured config."""

    pair: tuple[str, str]
    dist_m: float


class EpisodeStatus(BaseModel):
    """Episode recorder status (collect/DAgger)."""

    state: Literal["idle", "recording", "saving"]
    index: int | None
    frames: int
    duration_s: float


class DaggerStatus(BaseModel):
    """DAgger session block; full shape per 12-dagger §11 (canonical)."""

    control_mode: ControlMode
    engaged_arm: str | None
    frozen_arms: list[str] = []
    policy_version: str | None  # "{run_id}/v{n:06d}"; updates only at swaps
    staged_version: str | None = None
    episodes_labeled: int = 0
    takeover_rate_ep: float = 0.0  # human-frame fraction, current episode
    takeover_rate_run: float = 0.0  # rolling mean, last 10 episodes
    new_label_frames: int = 0
    trainer: TrainerStatus | None = None


class InferenceStatus(BaseModel):
    """Inference block; same gate machinery, takeover = SAFETY ESCAPE."""

    control_mode: ControlMode
    engaged_arm: str | None = None  # additive vs 05-ui
    policy_version: str | None


class SessionTelemetry(BaseModel):
    """Additive session-lifecycle block (04-runtime §13.3)."""

    state: str  # SessionState value
    start_from_progress: float | None = None  # 0-1 during START_FROM
    plan_status: str | None = None
    trainer_alive: bool | None = None


class TelemetryMsg(BaseModel):
    """One 25 Hz telemetry frame."""

    t: Literal["telemetry"] = "telemetry"
    seq: int
    ts: float  # server monotonic, s
    epoch: str
    active_arm: str | None  # server-authoritative (Tab cycles it)
    controller_connected: bool
    arms: list[ArmTelemetry]
    collision: CollisionReport  # §6 model
    clearances: list[ClearanceItem]  # top-5 monitored pairs, measured config
    episode: EpisodeStatus | None  # None in teleop/inference
    dagger: DaggerStatus | None  # None outside DAgger
    inference: InferenceStatus | None  # None outside inference
    session: SessionTelemetry | None = None  # additive


__all__ = [
    "PoseMsg",
    "ArmTelemetry",
    "ClearanceItem",
    "EpisodeStatus",
    "DaggerStatus",
    "InferenceStatus",
    "SessionTelemetry",
    "TelemetryMsg",
]
