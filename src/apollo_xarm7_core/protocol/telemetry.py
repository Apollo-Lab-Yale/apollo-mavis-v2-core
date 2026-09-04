"""Telemetry WebSocket messages (design doc 01-core §11).

Server -> all ``/ws/telemetry`` clients at 25 Hz (20-30 band). Shapes match
05-ui §2; *additive* fields extend the UI shape (the UI ignores unknowns).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from apollo_xarm7_core.dagger.types import ControlMode, TrainerStatus
from apollo_xarm7_core.schemas.safety import CollisionReport


class PoseMsg(BaseModel):
    """Wire pose: position in m, orientation as wxyz unit quaternion."""

    position: tuple[float, float, float]  # m
    orientation: tuple[float, float, float, float]  # wxyz


# ``protocol.tracker`` builds on PoseMsg and TrackerTelemetry (below) embeds its
# status model, so the import must follow PoseMsg to keep the cycle importable;
# the package ``__init__`` loads this module before ``tracker``.
from apollo_xarm7_core.protocol.tracker import TrackerCalibrationStatus  # noqa: E402


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


class TrackerSettingsMsg(BaseModel):
    """Live tracker teleop settings echoed in telemetry (13-tracker §3.5, §4).

    The ``filter_*`` fields are the *effective* One Euro pose-filter settings
    (config defaults, overridable live via ``tracker_settings``); they default
    here so pre-filter producers still validate.
    """

    yaw_deg: float  # lighthouse-world -> MJCF-world yaw alignment
    pos_scale: float  # tracker displacement -> EE displacement gain
    follow_rotation: bool  # orientation deltas applied when True
    filter_enabled: bool = True  # One Euro pose filter active (additive)
    filter_min_cutoff_hz: float = 1.0  # One Euro min cutoff, Hz (additive)
    filter_beta: float = 0.05  # One Euro speed coefficient (additive)


class ControllerTelemetry(BaseModel):
    """Raw Vive-controller input state (13-tracker §1.1), additive.

    Mirrors the libsurvive button/axis events for the tracked object. Axis
    conventions: ``trigger`` 0..1; ``trackpad_x``/``trackpad_y`` -1..1 with
    +y = top. ``trackpad_touch`` is finger contact, ``trackpad_click`` the
    physical press. ``menu`` + ``system`` is the pairing combo (never mapped).
    """

    trigger: float = 0.0  # analog pull, 0..1
    trigger_pressed: bool = False  # trigger click (button 0)
    trackpad_touch: bool = False  # finger on the pad (TOUCH_DOWN/UP)
    trackpad_click: bool = False  # pad pressed (button 1)
    trackpad_x: float = 0.0  # -1..1
    trackpad_y: float = 0.0  # -1..1, +y = top
    grip: bool = False  # button 7
    menu: bool = False  # button 6
    system: bool = False  # button 3


class TrackerTelemetry(BaseModel):
    """Vive-tracker block (13-tracker §3.5), additive.

    Device fields are populated even without a session; session fields
    (``engaged_arm``, ``anchor_tcp``, ``target_tcp``) are ``None`` otherwise.
    ``controller`` echoes the raw controller inputs (``None`` when the backend
    reports no controller) and ``device_held`` the key codes the runtime
    injects from them (13-tracker §1.1); a stale sample yields an empty list.
    ``device_action`` is the last device-sourced discrete action (e.g.
    ``"switch_arm"``), cleared by the runtime ~1 s after it fired.
    ``pose_filtered`` is the aligned pose after the One Euro filter (§4), i.e.
    what the anchor/delta math actually consumes; ``None`` when no sample.
    ``calibration`` mirrors ``GET /api/tracker/calibration`` (protocol.tracker)
    so the Devices-page wizard follows progress without polling.
    """

    backend: Literal["libsurvive", "fake", "none"]
    status: Literal["no_backend", "starting", "searching", "tracking", "stale", "error"]
    detail: str = ""  # human-readable reason for error/no_backend
    object_name: str = ""  # libsurvive object used (e.g. "WM0")
    seq: int = 0  # last sample sequence number
    rate_hz: float = 0.0  # measured sample rate
    age_s: float | None = None  # now - rx_mono of the last sample
    pose_raw: PoseMsg | None = None  # lighthouse world
    pose_world: PoseMsg | None = None  # after yaw alignment
    pose_filtered: PoseMsg | None = None  # world, after alignment + pose filter (§4)
    clutch: bool = False  # tracker_clutch currently held
    engaged_arm: str | None = None  # arm being driven while clutched
    anchor_tcp: PoseMsg | None = None  # EE pose at engagement (world)
    target_tcp: PoseMsg | None = None  # current tracker-derived EE target (world)
    settings: TrackerSettingsMsg
    controller: ControllerTelemetry | None = None  # raw controller inputs (§1.1)
    device_held: list[str] = Field(default_factory=list)  # codes injected from controller
    device_action: str | None = None  # last device-sourced discrete action (~1 s latch)
    charging: bool | None = None  # controller on external (USB) power; None = not reported
    calibration: TrackerCalibrationStatus | None = None  # additive (phase-10 wizard)


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
    tracker: TrackerTelemetry | None = None  # additive (13-tracker §3.5)


__all__ = [
    "PoseMsg",
    "ArmTelemetry",
    "ClearanceItem",
    "EpisodeStatus",
    "DaggerStatus",
    "InferenceStatus",
    "SessionTelemetry",
    "TrackerSettingsMsg",
    "ControllerTelemetry",
    "TrackerTelemetry",
    "TelemetryMsg",
]
