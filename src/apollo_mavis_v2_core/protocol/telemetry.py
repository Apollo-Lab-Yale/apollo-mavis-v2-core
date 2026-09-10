"""Telemetry WebSocket messages (design doc 01-core §11).

Server -> all ``/ws/telemetry`` clients at 25 Hz (20-30 band). Shapes match
05-ui §2; *additive* fields extend the UI shape (the UI ignores unknowns).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from apollo_mavis_v2_core.dagger.types import ControlMode, TrainerStatus
from apollo_mavis_v2_core.protocol.external import ExternalStatus, TrainerStatusAnnounce
from apollo_mavis_v2_core.protocol.hardware_monitor import HardwareMonitorTelemetry
from apollo_mavis_v2_core.protocol.microphone import MicStatus
from apollo_mavis_v2_core.schemas.safety import CollisionReport


class PoseMsg(BaseModel):
    """Wire pose: position in m, orientation as wxyz unit quaternion."""

    position: tuple[float, float, float]  # m
    orientation: tuple[float, float, float, float]  # wxyz


# ``protocol.tracker`` builds on PoseMsg and TrackerTelemetry (below) embeds its
# status model, so the import must follow PoseMsg to keep the cycle importable;
# the package ``__init__`` loads this module before ``tracker``.
from apollo_mavis_v2_core.protocol.tracker import TrackerCalibrationStatus  # noqa: E402


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
    fault_detail: str = ""  # additive (phase-09b): controller fault this arm is stopped
    #   for, e.g. "controller error 24: Speed Exceeds Limit"; "" = none
    recovering: bool = False  # additive (phase-09b): recovery ran (session RECOVERING),
    #   streaming resumes once the operator re-grips the clutch


class ClearanceItem(BaseModel):
    """One monitored geometry pair at the measured config."""

    pair: tuple[str, str]
    dist_m: float


class EpisodeStatus(BaseModel):
    """Episode recorder status (collect/DAgger).

    ``returning`` (additive, 2026-09-07; 04-runtime §10.5): after a save or a
    discard the arms are being driven back to the session's return profile (the
    ``start_from`` profile, else the workcell's initial condition) on the twin
    planner; ``episode_new`` is refused meanwhile. The dataset fields are
    additive too: ``repo_id`` / ``total_episodes`` / ``total_frames`` mirror the
    dataset manifest (saved episodes), ``detail`` is a short human-readable note
    ("returning to profile 'ready'", "return cancelled: movement key",
    "recorder degraded ..."). Deletion is immediate (one directory, 10-frames
    §11.7), so there is no pending-deletion list.
    """

    state: Literal["idle", "recording", "saving", "returning"]
    index: int | None
    frames: int
    duration_s: float
    repo_id: str | None = None
    total_episodes: int = 0
    total_frames: int = 0
    detail: str = ""
    frames_skipped: int = 0  # additive (2026-09-07): idle frames the action filter dropped
    #   in the open episode (04-runtime §10.5)


class DatasetExportTelemetry(BaseModel):
    """``TelemetryMsg.datasets.export`` (additive, 2026-09-07; 04-runtime §10.6):
    the running / last LeRobot v3 export job. ``phase`` walks scanning -> videos
    -> data -> meta -> validating -> done | failed; ``done`` / ``total`` count the
    current phase's units (episodes for videos / data, 1 for meta / validating);
    ``detail`` names the file being written or the failure."""

    repo_id: str
    format: str
    phase: Literal["scanning", "videos", "data", "meta", "validating", "done", "failed"]
    done: int = 0
    total: int = 0
    detail: str = ""


class DatasetsTelemetry(BaseModel):
    """``TelemetryMsg.datasets`` (additive, 2026-09-07): session-less like the
    microphone block; ``export`` is None until an export has run in this process."""

    export: DatasetExportTelemetry | None = None


class OnlineDaggerStatus(BaseModel):
    """``DaggerStatus.online_dagger`` (additive, phase-14; 15-online-dagger §3/§5): the
    runtime's rollout-level shell state as the UI sees it. ``phase`` is
    ``waiting_trainer`` (until the trainer reports ``ready`` for THIS session, when
    ``wait_for_trainer_ready``) -> ``rollout`` -> ``training`` (the trainer reports
    ``training``; ``episode_new`` refused while ``pause_while_training``) -> ``rollout``
    ...; ``error`` mirrors a trainer error until a non-error status arrives. The shell
    counts kept rollouts and the session's actor split — never iterations, which are
    the trainer's business. ``detail`` is the operator-facing reason ``episode_new`` is
    refused (or the trainer's detail); ``trainer`` is the last ``TrainerStatusAnnounce``
    verbatim and ``trainer_alive`` / ``trainer_age_s`` its freshness (<=
    ``dora.policy.spec_stale_s``); ``policy_version_acting`` follows the announced
    spec / action version (a change shows as "swapped" in the panel).
    """

    session_name: str
    phase: Literal["waiting_trainer", "rollout", "training", "error"]
    rollouts_saved: int  # kept rollouts of the session (a resume continues the count)
    detail: str = ""  # why episode_new is refused, trainer detail
    trainer_alive: bool = False  # a fresh trainer_status (<= spec_stale_s)
    trainer_age_s: float | None = None
    trainer: TrainerStatusAnnounce | None = None  # verbatim last status
    policy_version_acting: int | None = None
    expert_frames_session: int = 0  # kept expert frames this session (actor == 1)
    novice_frames_session: int = 0  # kept novice frames this session (actor == 0)
    session_dir: str = ""  # ~/data/online_dagger/<session_name>


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
    policy_stale: bool = False  # additive (phase-12; promised by 04-runtime §15): the
    #   policy output is past its staleness window -> policy arms hold (12-dagger §6.3)
    online_dagger: OnlineDaggerStatus | None = None  # additive (phase-14; 15-online-dagger
    #   §5): non-null iff the session's SessionSpec.online_dagger is set; appended last


class InferenceStatus(BaseModel):
    """Inference block; same gate machinery, takeover = SAFETY ESCAPE."""

    control_mode: ControlMode
    engaged_arm: str | None = None  # additive vs 05-ui
    policy_version: str | None
    policy_stale: bool = False  # additive (phase-12): see DaggerStatus.policy_stale


class ArmBringupTelemetry(BaseModel):
    """One hardware bring-up step of one arm (phase-09c; 04-runtime §5).

    Fed from the workcell's ``status_cb`` while ``SessionTelemetry.state`` is
    ``bringup``; the Cockpit lists the rows until the session is running.
    ``step`` names the stage (``network`` / ``connect`` / ``rail`` / ``gripper``
    / ``report`` / ``frozen`` ...), ``detail`` the human-readable outcome, e.g.
    "Perception Arm frozen at last sample".
    """

    arm_id: str
    step: str  # bring-up stage name
    status: Literal["pending", "ok", "warning", "error"]
    detail: str = ""  # human-readable outcome / reason


class SessionTelemetry(BaseModel):
    """Additive session-lifecycle block (04-runtime §13.3)."""

    state: str  # SessionState value
    start_from_progress: float | None = None  # 0-1 during START_FROM
    plan_status: str | None = None
    trainer_alive: bool | None = None
    bringup: list[ArmBringupTelemetry] | None = None  # additive (phase-09c): hardware
    #   bring-up progress per arm/step; None for sim sessions and once cleared
    translate_frame: Literal["camera", "world", "base"] | None = None
    #   additive (2026-09-08): the frame the keyboard TRANSLATE keys act in for this
    #   session (runtime ``control.translate_frame``; rotations are always about the
    #   TCP axes). The runtime default is ``world`` (operator-fixed; operator decision
    #   2026-09-08, 15-pro-dagger §0 item 8); ``camera`` (the active arm's wrist
    #   camera) and ``base`` (the pre-2026-09-08 behaviour) remain selectable. The
    #   keymap labels the KEY axes, not a frame, so the overlay needs this to tell
    #   the operator what "forward" currently means. None = no session.
    fault_detail: str = ""  # additive (2026-09-08; 04-runtime §13.3): the session
    #   manager's session-level notice the per-arm rows do NOT already carry - a
    #   ``start_from`` plan the loop refused ("start_from refused: Manipulation Arm
    #   faulted (controller state 4, code C24) - use Clear errors & resume, then Go to
    #   profile") or could not plan, the outcome of a "Go to profile" / `R` return that
    #   did not arrive ("Go to profile 'shelf': the digital twin could not plan ..."),
    #   or the fault text while no arm row shows one (a fault before the first
    #   snapshot). "" = nothing to say; shown verbatim by the Cockpit's FaultBanner.


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
    filter_beta: float = 5.0  # One Euro speed coefficient, Hz per (m/s) (additive;
    #   5.0 = the runtime's effective default since 2026-09-07, 13-tracker §4)


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

    Link fields (2026-09-07, 13-tracker §3.5 "controller link"): the pose path
    and the button path are INDEPENDENT and can fail apart, which is exactly
    what happened on 2026-09-06 (poses at 135 Hz while libsurvive delivered no
    button event at all, so the clutch could never engage and the frozen
    ``controller`` state looked plausible). Therefore:

    * ``controller_age_s`` is the age of the newest controller INPUT event
      (button / touch / axis), independent of ``age_s`` (the pose age).
      ``None`` when no input event has ever been seen. A pose-fresh sample with
      a stale ``controller_age_s`` means "moving works, buttons do not".
    * ``objects`` lists the OBJECT-type devices libsurvive currently reports
      (e.g. ``["WM0"]``), so a panel can separate "not paired / dongle busy"
      (empty) from "paired, waiting for base stations".
    * ``dongle_present`` is the USB presence of the Watchman receiver
      (``28de:2101``) read from sysfs, so "unplugged" is distinguishable from
      "unpaired". ``None`` when the check is unavailable (non-Linux, no sysfs).
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
    # Controller link (additive, 2026-09-07): the button path's own liveness and
    # the pairing evidence behind it (see the class docstring).
    controller_age_s: float | None = None  # now - rx_mono of the newest INPUT event
    objects: list[str] = Field(default_factory=list)  # libsurvive OBJECT devices seen
    dongle_present: bool | None = None  # USB 28de:2101 in sysfs; None = not checked


class MicrophoneTelemetry(BaseModel):
    """Microphone block (phase-11; 04-runtime §13.3), additive.

    Every field defaults so a producer without a microphone still validates.
    One frame per telemetry tick (frame length = ``sample_rate / telemetry_hz``,
    1920 samples at 48 kHz / 25 Hz); the UI de-duplicates on ``seq``.
    ``env_min``/``env_max`` are the per-bin min/max envelope of the frame as 64
    int8 values (-127..127, time-ordered) for the scrolling oscilloscope,
    quantised RELATIVE TO THE FRAME PEAK (the loudest sample maps to +-127, so a
    quiet room keeps its shape); absolute = ``env / 127 * 10 ** (peak_dbfs /
    20)``. ``rms_dbfs``/``peak_dbfs`` are full-scale levels (0 dBFS = |1.0|),
    ``None`` when no frame has arrived. ``status`` shares ``MicStatus`` with
    ``MicrophoneInfo`` (§12).
    """

    mic_id: str = "mic_view"
    status: MicStatus = "no_backend"
    detail: str = ""  # human-readable reason for absent/error/no_backend
    seq: int = 0  # last frame sequence number
    age_s: float | None = None  # now - rx_mono of the last frame
    rate_hz: float = 0.0  # measured frame rate
    sample_rate: int = 48000  # Hz
    rms_dbfs: float | None = None  # frame RMS level
    peak_dbfs: float | None = None  # frame peak level
    clipping: bool = False  # peak >= -1 dBFS
    env_min: list[int] = []  # 64 x int8 (-127..127) rel. to frame peak, per-bin minimum
    env_max: list[int] = []  # 64 x int8 (-127..127) rel. to frame peak, per-bin maximum
    overruns: int = 0  # backend overrun / dropped-block count since start


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
    microphone: MicrophoneTelemetry | None = None  # additive (phase-11)
    external: ExternalStatus | None = None  # additive (phase-12): dora bridge + external policy
    hardware_monitor: HardwareMonitorTelemetry | None = None  # additive (phase-09a)
    datasets: DatasetsTelemetry | None = None  # additive (2026-09-07): export job progress


__all__ = [
    "PoseMsg",
    "ArmTelemetry",
    "ClearanceItem",
    "EpisodeStatus",
    "DatasetExportTelemetry",
    "DatasetsTelemetry",
    "OnlineDaggerStatus",
    "DaggerStatus",
    "InferenceStatus",
    "ArmBringupTelemetry",
    "SessionTelemetry",
    "TrackerSettingsMsg",
    "ControllerTelemetry",
    "TrackerTelemetry",
    "MicrophoneTelemetry",
    "TelemetryMsg",
]
