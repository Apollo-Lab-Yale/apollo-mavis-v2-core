"""Wire-protocol tests (design doc 01-core §10-§12, §18)."""

from __future__ import annotations

import json
from typing import get_args

import pytest
from pydantic import BaseModel, ValidationError

from apollo_mavis_v2_core.dagger.types import ControlMode, TrainerStatus
from apollo_mavis_v2_core.protocol.control import (
    AckMsg,
    ActionMsg,
    HelloMsg,
    JointTargetArgs,
    KeysMsg,
    SaveProfileArgs,
    SetInitialConditionArgs,
    TrackerSettingsArgs,
    parse_client_msg,
    validate_action_args,
)
from apollo_mavis_v2_core.protocol.hardware_monitor import (
    ArmMonitorStatus,
    ArmMonitorTelemetry,
    HardwareMonitorTelemetry,
    TwinOverlayStatus,
    TwinOverlayTelemetry,
)
from apollo_mavis_v2_core.protocol.maintenance import (
    ArmMaintenanceOp,
    ArmMaintenanceRequest,
    ArmMaintenanceResult,
    MaintenancePath,
    MaintenancePhase,
    MaintenanceProgress,
    MaintenanceStatus,
    PrePositionPlan,
    RailSweepVerdict,
)
from apollo_mavis_v2_core.protocol.microphone import MicrophoneInfo, MicStatus
from apollo_mavis_v2_core.protocol.session import (
    ArmStatusInfo,
    CameraInfo,
    PolicyInfo,
    ProfileInfo,
    SceneInfo,
    SessionInfo,
    SessionSpec,
    WorkcellStatus,
)
from apollo_mavis_v2_core.protocol.telemetry import (
    ArmBringupTelemetry,
    ArmTelemetry,
    ClearanceItem,
    ControllerTelemetry,
    DaggerStatus,
    EpisodeStatus,
    InferenceStatus,
    MicrophoneTelemetry,
    PoseMsg,
    SessionTelemetry,
    TelemetryMsg,
    TrackerSettingsMsg,
    TrackerTelemetry,
)
from apollo_mavis_v2_core.protocol.tracker import (
    CalibrationValidation,
    LighthouseStatus,
    TrackerCalibrationCommand,
    TrackerCalibrationStatus,
    YawGesturePoint,
)
from apollo_mavis_v2_core.protocol.video import is_reserved_stream
from apollo_mavis_v2_core.schemas.safety import CollisionReport

_POSE = PoseMsg(position=(0.3, 0.0, 0.4), orientation=(1.0, 0.0, 0.0, 0.0))

_ARM = ArmTelemetry(
    arm_id="arm0",
    connected=True,
    q=[0.0, -0.5, 0.0, 1.0, 0.0, 1.2, 0.0],
    rail_pos_m=0.32,
    ee_pose=_POSE,
    gripper_open_frac=0.8,
    error_code=0,
    warn_code=0,
    stale=False,
    goto="executing",
)

_TRACKER_SETTINGS = TrackerSettingsMsg(yaw_deg=90.0, pos_scale=1.5, follow_rotation=True)
# Non-default pose-filter tuning (13-tracker §4 "Pose filter").
_TRACKER_SETTINGS_TUNED = TrackerSettingsMsg(
    yaw_deg=0.0, pos_scale=1.0, follow_rotation=False,
    filter_enabled=False, filter_min_cutoff_hz=2.5, filter_beta=0.2,
)

# Controller with trigger clicked and trackpad pressed near the top edge
# (13-tracker §1.1: -> KeyC + KeyH injected).
_CONTROLLER = ControllerTelemetry(
    trigger=1.0,
    trigger_pressed=True,
    trackpad_touch=True,
    trackpad_click=True,
    trackpad_x=-0.12,
    trackpad_y=0.85,
    grip=False,
    menu=False,
    system=False,
)

# Device-only block (no session): session fields stay None.
_TRACKER_IDLE = TrackerTelemetry(
    backend="none", status="no_backend", detail="pysurvive not installed",
    settings=_TRACKER_SETTINGS,
)

# Engaged block: every optional field populated.
_TRACKER_ENGAGED = TrackerTelemetry(
    backend="libsurvive",
    status="tracking",
    object_name="WM0",
    seq=1234,
    rate_hz=248.5,
    age_s=0.004,
    pose_raw=_POSE,
    pose_world=PoseMsg(position=(0.0, 0.3, 0.4), orientation=(0.7071, 0.0, 0.0, 0.7071)),
    pose_filtered=PoseMsg(position=(0.001, 0.299, 0.4), orientation=(0.7071, 0.0, 0.0, 0.7071)),
    clutch=True,
    engaged_arm="arm0",
    anchor_tcp=_POSE,
    target_tcp=PoseMsg(position=(0.31, 0.02, 0.4), orientation=(1.0, 0.0, 0.0, 0.0)),
    settings=_TRACKER_SETTINGS_TUNED,
    controller=_CONTROLLER,
    device_held=["KeyC", "KeyH"],
    device_action="switch_arm",
)

# phase-10 calibration snapshots (13-tracker §3/§4): base-station capture in
# progress with three stations, and a completed yaw fit awaiting `apply`.
_LIGHTHOUSES = [
    LighthouseStatus(index=0, channel=1, serial="LHB-2A3B4C5D", pose=_POSE, scenes=4,
                     reference=True),
    LighthouseStatus(index=1, channel=3, serial="LHB-6E7F8091",
                     pose=PoseMsg(position=(-1.2, 0.8, 2.1),
                                  orientation=(0.5, -0.5, 0.5, 0.5)),
                     scenes=3),
    LighthouseStatus(index=2, channel=None),  # OOTX not decoded yet
]
_CALIB_BASE_STATION = TrackerCalibrationStatus(
    kind="base_station",
    phase="capturing",
    detail="scenes 4/6 — park the controller still >= 3 s at another spot",
    started_at=1_756_900_000.0,
    elapsed_s=42.5,
    scenes=4,
    lighthouses=_LIGHTHOUSES,
    stations_visible=3,
    controller_still=False,
    yaw_valid=False,
    base_station_installed_at=1_756_800_000.0,
)
_CALIB_VALIDATED = TrackerCalibrationStatus(
    kind="base_station",
    phase="done",
    detail="validation passed — install",
    scenes=7,
    lighthouses=_LIGHTHOUSES,
    stations_visible=3,
    controller_still=True,
    validation=CalibrationValidation(
        samples=2480, std_mm=(0.08, 0.11, 0.06), max_step_mm=0.4, passed=True,
    ),
    installed_path="/home/op/.config/libsurvive/config.json",
    backup_path="/home/op/.config/libsurvive/config.json.bak-20260903-141500",
)
_YAW_POINTS = [
    YawGesturePoint(label=label, pose=PoseMsg(position=pos, orientation=(1.0, 0.0, 0.0, 0.0)))
    for label, pos in (
        ("start", (0.00, 0.00, 1.00)),
        ("left", (0.25, 0.02, 1.01)),
        ("forward", (0.26, 0.27, 1.00)),
        ("right", (0.01, 0.28, 0.99)),
        ("back", (0.00, 0.01, 1.00)),
        ("up", (0.01, 0.00, 1.24)),
        ("down", (0.00, 0.01, 1.00)),
    )
]
_CALIB_YAW = TrackerCalibrationStatus(
    kind="yaw",
    phase="done",
    detail="fit ok — apply",
    started_at=1_756_900_100.0,
    elapsed_s=18.0,
    yaw_points=_YAW_POINTS,
    next_point=None,
    fitted_yaw_deg=102.1,
    fit_residual_deg=1.7,
    fit_checks=[],
    yaw_valid=False,
    yaw_calibrated_at=1_756_700_000.0,
    base_station_installed_at=1_756_800_000.0,
)
_CALIB_YAW_FAILED_FIT = TrackerCalibrationStatus(
    kind="yaw",
    phase="done",
    detail="fit checks failed — redo",
    yaw_points=_YAW_POINTS,
    fitted_yaw_deg=-77.9,
    fit_residual_deg=22.3,
    fit_checks=["leg left too short (0.04 m < 0.10 m)", "residual 22.3 deg > 15.0 deg"],
)

# phase-11 microphone blocks: no backend (all defaults) and a live RØDE frame with
# a full 64-bin int8 envelope (-127..127, time-ordered).
_ENV_MIN = [-(2 * i) for i in range(64)]  # 0 .. -126
_ENV_MAX = [2 * i for i in range(64)]  # 0 .. 126
_MIC_IDLE = MicrophoneTelemetry()
_MIC_LIVE = MicrophoneTelemetry(
    mic_id="mic_view",
    status="live",
    seq=4321,
    age_s=0.012,
    rate_hz=25.0,
    sample_rate=48000,
    rms_dbfs=-31.7,
    peak_dbfs=-0.4,
    clipping=True,
    env_min=_ENV_MIN,
    env_max=_ENV_MAX,
    overruns=2,
)
_MIC_INFO_LIVE = MicrophoneInfo(
    mic_id="mic_view",
    label="RØDE NT-USB Mini",
    kind="pulse",
    source="alsa_input.usb-R__DE_Microphones_R__DE_NT-USB_Mini_750BFEE8-00.mono-fallback",
    sample_rate=48000,
    channels=1,
    live=True,
    status="live",
)
_MIC_INFO_ABSENT = MicrophoneInfo(
    mic_id="mic_view",
    label="View arm microphone",
    kind="none",
    source=None,
    sample_rate=48000,
    live=False,
    status="absent",
    detail="no PulseAudio source matches 'NT-USB Mini'",
)

# phase-09a read-only hardware monitor + twin overlay, mirroring the 2026-09-04 live
# facts: both linear tracks unhomed (raw register 0, position meaningless), the
# Perception Arm latching controller error C19, the Manipulation Arm's overlay assuming
# rail 0.65 m. Joint 1 is the raw controller angle (keyframe = pi; no offset).
_Q_KEYFRAME = [3.14159, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_TCP_KEYFRAME = [0.207, 0.0, 0.112, 3.14159, 0.0, 0.0]  # flange pose, base frame
_MON_GRIP_RUNNING = ArmMonitorTelemetry(
    arm_id="grip",
    status="running",
    seq=812,
    age_s=0.041,
    q=_Q_KEYFRAME,
    tcp_pose=_TCP_KEYFRAME,
    rail_present=True,
    rail_homed=False,
    rail_enabled=False,
    rail_pos_m=None,  # not homed -> meaningless
    rail_raw_mm=0.0,
    gripper_open_frac=0.93,
    gripper_raw=78.4,
    error_code=0,
    warn_code=0,
    state=4,
    mode=0,
)
_MON_VIEW_ERROR = ArmMonitorTelemetry(
    arm_id="view",
    status="running",
    detail="controller error 19: End Effector Communication Error",
    seq=809,
    age_s=0.038,
    q=_Q_KEYFRAME,
    tcp_pose=_TCP_KEYFRAME,
    rail_present=True,
    rail_homed=False,
    rail_enabled=False,
    rail_raw_mm=0.0,
    gripper_open_frac=None,  # gripper "none"
    error_code=19,
    state=4,
    mode=0,
)
_OVERLAY_GRIP_LIVE = TwinOverlayTelemetry(
    stream_id="grip_wrist_align",
    camera_id="grip_wrist",
    arm_id="grip",
    status="live",
    detail="rail not homed - twin assumes 0.65 m",
    fps=11.8,
    rail_fallback_m=0.65,
    joint1_offset_rad=0.0,
    mask_fraction=0.184,
)
_OVERLAY_VIEW_WAITING = TwinOverlayTelemetry(
    stream_id="view_wrist_align", camera_id="view_wrist", arm_id="view", status="waiting",
    detail="no camera frame yet",
)
_HW_MONITOR = HardwareMonitorTelemetry(
    enabled=True,
    paused=False,
    arms=[_MON_GRIP_RUNNING, _MON_VIEW_ERROR],
    overlays=[_OVERLAY_GRIP_LIVE, _OVERLAY_VIEW_WAITING],
)
# A hardware session owns the boxes: connections released, overlays switched off.
_HW_MONITOR_PAUSED = HardwareMonitorTelemetry(
    enabled=True,
    paused=True,
    arms=[
        ArmMonitorTelemetry(arm_id="grip", status="paused", detail="hardware session active"),
        ArmMonitorTelemetry(arm_id="view", status="paused", detail="hardware session active"),
    ],
    overlays=[
        _OVERLAY_GRIP_LIVE.model_copy(update={"status": "off", "detail": "", "fps": 0.0}),
        _OVERLAY_VIEW_WAITING.model_copy(update={"status": "off", "detail": ""}),
    ],
)

# phase-09b error recovery. Per-arm session telemetry while the Manipulation Arm is stopped
# on a controller fault (C24) and after the operator's "Clear errors & resume" ran the
# recovery sequence (waiting for the clutch to be re-gripped).
_ARM_FAULTED = _ARM.model_copy(update={
    "error_code": 24, "goto": None,
    "fault_detail": "controller error 24: Speed Exceeds Limit",
})
_ARM_RECOVERING = _ARM_FAULTED.model_copy(update={"error_code": 0, "recovering": True})
# The Perception Arm after "Clear errors" from the Hardware tab, carrying the slow-poll
# read-back of the controller-side safety parameters (2026-09-04 live values: tcp_load 0 kg,
# sensitivity 1 - both wrong), then after "Apply safety settings" brought them to the
# configured 3 / 0.55 kg / (0, 0, 90) mm.
_MON_VIEW_CLEARED = _MON_VIEW_ERROR.model_copy(update={
    "detail": "", "seq": 811, "error_code": 0,
    "collision_sensitivity": 1, "tcp_load_kg": 0.0, "tcp_load_cog_mm": [0.0, 0.0, 0.0],
    "backstops_match": False,
})
_MON_VIEW_BACKSTOPS = _MON_VIEW_CLEARED.model_copy(update={
    "seq": 830, "collision_sensitivity": 3, "tcp_load_kg": 0.55,
    "tcp_load_cog_mm": [0.0, 0.0, 90.0], "backstops_match": True,
})
_MAINT_CLEAR_OK = ArmMaintenanceResult(
    arm_id="view", op="clear_errors", path="monitor", ok=True,
    detail="errors cleared (C19 -> 0)",
    sdk_codes={"clean_error": 0, "clean_warn": 0},  # never motion_enable
    before=_MON_VIEW_ERROR, after=_MON_VIEW_CLEARED,
)
_MAINT_APPLY_WARN = ArmMaintenanceResult(
    arm_id="view", op="apply_backstops", path="monitor", ok=True,
    detail="safety settings applied (sensitivity 3, payload 0.55 kg)",
    sdk_codes={  # backstops.apply_backstops order
        "set_tcp_load": 0, "set_gravity_direction": 0, "set_collision_sensitivity": 0,
        "set_self_collision_detection": 0, "set_collision_tool_model": 1,
        "set_collision_rebound": 0,
    },
    warnings=["set_collision_tool_model returned 1"],
    before=_MON_VIEW_CLEARED, after=_MON_VIEW_BACKSTOPS,
)
_MAINT_RECOVER_SESSION = ArmMaintenanceResult(
    arm_id="grip", op="recover", path="session", ok=True,
    detail="recovered - re-grip the clutch to continue",
    sdk_codes={"clean_error": 0, "clean_warn": 0, "motion_enable": 0, "set_mode": 0,
               "set_state": 0},
)
_MAINT_RECOVER_REFUSED = ArmMaintenanceResult(
    arm_id="view", op="recover", path="monitor", ok=False, detail="recover needs a session",
)
# phase-09c: home_rail is twin-gated by a full-travel sweep at the arm's CURRENT posture.
_Q_FOLDED = [3.141592653589793, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # xArm7 zero, joint 1 = pi
_SWEEP_CLEAR = RailSweepVerdict(
    scene_id="mavis_v2", inflation_m=0.025, step_m=0.005, clear=True,
    min_clearance_m=0.0178, min_clearance_at_m=0.65,
    min_clearance_pair=["grip/link2", "grip/link4"],
    q_checked=_Q_FOLDED, other_arms={"view": [*_Q_FOLDED, 0.0]},
    assumptions=["view rail unknown - used fallback 0.00 m"], sample_seq=4210,
)
_SWEEP_BLOCKED = RailSweepVerdict(
    scene_id="mavis_v2", inflation_m=0.025, step_m=0.005, clear=False,
    first_blocked_m=0.125, first_blocked_pair=["grip/gripper_finger_left", "table"],
    min_clearance_m=-0.004, min_clearance_at_m=0.13,
    min_clearance_pair=["grip/gripper_finger_left", "table"],
    q_checked=[3.14, 0.4, 0.0, 1.9, 0.0, 1.5, 0.0], other_arms={"view": [*_Q_FOLDED, 0.0]},
    sample_seq=4302,
)
_MAINT_HOME_DRY = ArmMaintenanceResult(
    arm_id="grip", op="home_rail", path="monitor", ok=True,
    detail="rail sweep clear (dry run - nothing written)", rail_sweep=_SWEEP_CLEAR,
    before=_MON_VIEW_CLEARED.model_copy(update={"arm_id": "grip", "seq": 4210}),
)
_MAINT_HOME_OK = ArmMaintenanceResult(
    arm_id="grip", op="home_rail", path="monitor", ok=True,
    detail="rail homed (on_zero 1, enabled, 0.000 m)",
    sdk_codes={  # the exact home_rail write set - never motion_enable
        "set_linear_track_back_origin": 0, "set_linear_track_enable": 0,
        "set_linear_track_speed": 0,
    },
    before=_MON_VIEW_CLEARED.model_copy(update={"arm_id": "grip", "seq": 4210}),
    after=_MON_VIEW_CLEARED.model_copy(update={
        "arm_id": "grip", "seq": 4260, "rail_pos_m": 0.0, "rail_homed": True,
        "rail_enabled": True,
    }),
    rail_sweep=_SWEEP_CLEAR,
)
_MAINT_HOME_REFUSED = ArmMaintenanceResult(
    arm_id="grip", op="home_rail", path="monitor", ok=False,
    detail="rail sweep blocked at 0.125 m: grip/gripper_finger_left <-> table",
    rail_sweep=_SWEEP_BLOCKED,
)
# phase-09d: when the CURRENT posture is not sweep-clear the runtime plans a joint path to a
# rail-safe posture (every waypoint clear for all 131 rail positions) and offers it on the
# verdict; on confirm the op runs as an asynchronous RailHomingJob (202 + job_id) whose phases
# ride ArmMonitorTelemetry.maintenance and whose final result is GET .../maintenance/last.
_PRE_POSITION_NONE = PrePositionPlan(
    needed=False, detail="current posture clears the whole rail travel",
)
_PRE_POSITION_PLAN = PrePositionPlan(
    needed=True, source="keyframe", target_q=_Q_FOLDED, waypoints=14, duration_s=38.5,
    checked_rail_positions=131, clear=True,
    detail="path to the factory-zero posture: 14 waypoints clear for all 131 rail positions",
)
_PRE_POSITION_REFUSED = PrePositionPlan(
    needed=True, source="home", target_q=_Q_FOLDED, waypoints=0, checked_rail_positions=131,
    clear=False,
    detail="no rail-safe path found - fold the arm toward the factory zero posture in Studio "
           "and retry",
)
_SWEEP_CLEAR_09D = _SWEEP_CLEAR.model_copy(update={"pre_position": _PRE_POSITION_NONE})
_SWEEP_BLOCKED_PLANNED = _SWEEP_BLOCKED.model_copy(update={"pre_position": _PRE_POSITION_PLAN})
_SWEEP_BLOCKED_UNPLANNABLE = _SWEEP_BLOCKED.model_copy(
    update={"pre_position": _PRE_POSITION_REFUSED},
)
_JOB_ID = "home_rail-grip-20260905T101500-7f3a"
_MAINT_HOME_DRY_PLANNED = ArmMaintenanceResult(
    arm_id="grip", op="home_rail", path="monitor", ok=True,
    detail="posture not clear - pre-positioning motion planned (dry run - nothing written)",
    rail_sweep=_SWEEP_BLOCKED_PLANNED,
    before=_MON_VIEW_CLEARED.model_copy(update={"arm_id": "grip", "seq": 4302}),
)
_MAINT_HOME_ACCEPTED = ArmMaintenanceResult(  # the 202 body
    arm_id="grip", op="home_rail", path="monitor", ok=True, status="accepted", job_id=_JOB_ID,
    detail="rail homing job started: pre-position (14 waypoints, ~39 s at 10 %) then home",
    rail_sweep=_SWEEP_BLOCKED_PLANNED,
    before=_MON_VIEW_CLEARED.model_copy(update={"arm_id": "grip", "seq": 4302}),
)
# The job's final result (GET .../maintenance/last): same job_id, status back to "done". The
# job drives the arm through its own driver connection, so it reports the session path
# (MaintenancePath has no "job" value - contract §1 keeps the vocabulary).
_MAINT_HOME_JOB_DONE = _MAINT_HOME_OK.model_copy(update={
    "path": "session", "job_id": _JOB_ID,
    "detail": "rail homed after pre-positioning (on_zero 1, enabled, 0.000 m); posture held",
    "rail_sweep": _SWEEP_BLOCKED_PLANNED,
})
_MAINT_HOME_JOB_FAILED = ArmMaintenanceResult(
    arm_id="grip", op="home_rail", path="session", ok=False, job_id=_JOB_ID,
    detail="positioning aborted: controller error 24 (Speed Exceeds Limit) - stopped, brakes on",
    rail_sweep=_SWEEP_BLOCKED_PLANNED,
)
_MAINT_HOME_REFUSED_09D = ArmMaintenanceResult(
    arm_id="grip", op="home_rail", path="monitor", ok=False, status="refused",
    detail="rail sweep blocked at 0.125 m and no rail-safe path found - fold the arm toward "
           "the factory zero posture in Studio and retry",
    rail_sweep=_SWEEP_BLOCKED_UNPLANNABLE,
)
_PROGRESS_QUEUED = MaintenanceProgress(op="home_rail", job_id=_JOB_ID, phase="queued")
_PROGRESS_POSITIONING = MaintenanceProgress(
    op="home_rail", job_id=_JOB_ID, phase="positioning", detail="waypoint 9 / 14",
    progress=0.55, started_at=1_757_067_300.0,
)
_PROGRESS_DONE = MaintenanceProgress(
    op="home_rail", job_id=_JOB_ID, phase="done", detail="rail homed; posture held",
    progress=1.0, started_at=1_757_067_300.0,
)
_PROGRESS_FAILED = MaintenanceProgress(
    op="home_rail", job_id=_JOB_ID, phase="failed",
    detail="positioning aborted: controller error 24", progress=0.55,
    started_at=1_757_067_300.0,
)
# During the job the monitor is paused (the job's driver owns the box) and the other arm is
# frozen at its last sample (09c D1, maintenance motion only since 09d).
_MON_GRIP_HOMING_JOB = _MON_GRIP_RUNNING.model_copy(update={
    "status": "paused", "detail": "rail homing job: positioning", "seq": 4310,
    "maintenance_busy": True, "maintenance": _PROGRESS_POSITIONING,
})
_MON_VIEW_FROZEN = _MON_VIEW_BACKSTOPS.model_copy(update={
    "status": "paused", "detail": "Perception Arm frozen at last sample (rail homing job)",
    "seq": 831,
})
_HW_MONITOR_HOMING_JOB = HardwareMonitorTelemetry(
    enabled=True, paused=True, arms=[_MON_GRIP_HOMING_JOB, _MON_VIEW_FROZEN],
)
_BRINGUP_ROWS = [
    ArmBringupTelemetry(arm_id="grip", step="network", status="ok"),
    ArmBringupTelemetry(arm_id="grip", step="connect", status="ok", detail="fw 2.5.0"),
    ArmBringupTelemetry(arm_id="grip", step="rail", status="pending"),
    ArmBringupTelemetry(arm_id="view", step="frozen", status="warning",
                        detail="Perception Arm frozen at last sample"),
]

_WIRE_MODELS: list[BaseModel] = [
    HelloMsg(epoch="ep0", session_id="s0", role="controller"),
    HelloMsg(epoch="ep0", session_id=None, role="observer"),
    KeysMsg(seq=7, ts=123.5, held=["KeyW", "ArrowRight"]),
    ActionMsg(name="switch_arm"),
    ActionMsg(name="switch_arm_prev"),
    ActionMsg(name="joint_target", args={"arm_id": "arm0", "positions": [0.0] * 8, "mode": "jog"}),
    ActionMsg(name="tracker_settings", args={"pos_scale": 1.5, "follow_rotation": False}),
    ActionMsg(name="tracker_settings", args={"filter_min_cutoff_hz": 0.5, "filter_beta": 0.0}),
    AckMsg(name="takeover_toggle", ok=False, detail="observer"),
    JointTargetArgs(arm_id="arm0", positions=[0.1] * 7, mode="goto"),
    SaveProfileArgs(name="home", notes="pre-demo"),
    SetInitialConditionArgs(profile_id="abc123"),
    SetInitialConditionArgs(),
    TrackerSettingsArgs(yaw_deg=-45.0, pos_scale=0.5, follow_rotation=False),
    TrackerSettingsArgs(filter_enabled=False, filter_min_cutoff_hz=0.05, filter_beta=5.0),
    TrackerSettingsArgs(),
    _POSE,
    _ARM,
    ClearanceItem(pair=("arm0/link5", "arm1/link3"), dist_m=0.031),
    EpisodeStatus(state="recording", index=4, frames=250, duration_s=8.3),
    DaggerStatus(
        control_mode=ControlMode.HUMAN,
        engaged_arm="arm0",
        frozen_arms=["arm1"],
        policy_version="run0/v000003",
        staged_version="run0/v000004",
        episodes_labeled=5,
        takeover_rate_ep=0.2,
        takeover_rate_run=0.15,
        new_label_frames=42,
        trainer=TrainerStatus(state="training", steps_total=100),
    ),
    InferenceStatus(control_mode=ControlMode.POLICY, engaged_arm=None, policy_version="r/v000001"),
    SessionTelemetry(state="RUNNING", start_from_progress=0.5, plan_status="planning"),
    _TRACKER_SETTINGS,
    _TRACKER_SETTINGS_TUNED,
    ControllerTelemetry(),
    _CONTROLLER,
    _TRACKER_IDLE,
    _TRACKER_ENGAGED,
    # phase-10 tracker calibration (REST + TrackerTelemetry.calibration)
    LighthouseStatus(index=0),
    *_LIGHTHOUSES,
    CalibrationValidation(),
    CalibrationValidation(samples=2480, std_mm=(61.0, 62.0, 53.0), max_step_mm=248.0,
                          threshold_std_mm=5.0, threshold_step_mm=20.0, passed=False),
    *_YAW_POINTS,
    TrackerCalibrationStatus(),
    _CALIB_BASE_STATION,
    _CALIB_VALIDATED,
    _CALIB_YAW,
    _CALIB_YAW_FAILED_FIT,
    TrackerCalibrationStatus(kind="base_station", phase="failed",
                             detail="backend is not libsurvive"),
    TrackerCalibrationStatus(kind="yaw", phase="aborted", yaw_valid=True,
                             yaw_calibrated_at=1_756_700_000.0, applied_yaw_deg=102.1),
    TrackerCalibrationCommand(kind="base_station", op="start"),
    TrackerCalibrationCommand(kind="base_station", op="validate"),
    TrackerCalibrationCommand(kind="yaw", op="capture"),
    TrackerCalibrationCommand(kind="yaw", op="capture", point="forward"),
    TrackerCalibrationCommand(kind="yaw", op="apply"),
    TrackerCalibrationCommand(kind="yaw", op="abort"),
    _TRACKER_ENGAGED.model_copy(update={"clutch": False, "calibration": _CALIB_BASE_STATION}),
    _TRACKER_IDLE.model_copy(update={"backend": "fake", "status": "tracking",
                                     "calibration": _CALIB_YAW}),
    # phase-11 microphone (TelemetryMsg.microphone + GET /api/microphones)
    _MIC_IDLE,
    _MIC_LIVE,
    MicrophoneTelemetry(status="stalled", seq=17, age_s=1.3, rate_hz=0.0,
                        rms_dbfs=-60.0, peak_dbfs=-48.2),
    MicrophoneTelemetry(status="error", detail="PortAudio: device unavailable (EBUSY)"),
    _MIC_INFO_LIVE,
    _MIC_INFO_ABSENT,
    MicrophoneInfo(mic_id="mic_view", label="View arm microphone", kind="fake", source=None,
                   sample_rate=48000, live=True, status="live"),
    # phase-09a hardware monitor + twin overlay (TelemetryMsg.hardware_monitor)
    ArmMonitorTelemetry(arm_id="grip"),
    ArmMonitorTelemetry(arm_id="view", status="connecting", detail="opening 192.168.2.219"),
    ArmMonitorTelemetry(arm_id="grip", status="stale", seq=44, age_s=1.7, q=_Q_KEYFRAME),
    ArmMonitorTelemetry(arm_id="grip", status="error", detail="connect: [Errno 111] refused"),
    _MON_GRIP_RUNNING,
    _MON_VIEW_ERROR,
    _MON_GRIP_RUNNING.model_copy(update={"rail_homed": True, "rail_enabled": True,
                                         "rail_pos_m": 0.65, "rail_raw_mm": 650.0}),
    TwinOverlayTelemetry(stream_id="grip_wrist_align", camera_id="grip_wrist", arm_id="grip"),
    TwinOverlayTelemetry(stream_id="view_wrist_align", camera_id="view_wrist", arm_id="view",
                         status="stale", detail="monitor stale", fps=12.0, mask_fraction=0.41),
    TwinOverlayTelemetry(stream_id="view_wrist_align", camera_id="view_wrist", arm_id="view",
                         status="error", detail="EGL context lost"),
    _OVERLAY_GRIP_LIVE,
    _OVERLAY_VIEW_WAITING,
    HardwareMonitorTelemetry(),
    HardwareMonitorTelemetry(enabled=False, arms=[
        ArmMonitorTelemetry(arm_id="grip", detail="apollo_mavis_v2_hardware not importable"),
    ]),
    _HW_MONITOR,
    _HW_MONITOR_PAUSED,
    # phase-09b error recovery (ArmTelemetry fault fields, monitor read-back, maintenance)
    _ARM_FAULTED,
    _ARM_RECOVERING,
    _MON_VIEW_CLEARED,
    _MON_VIEW_BACKSTOPS,
    ArmMonitorTelemetry(arm_id="grip", status="running", maintenance_busy=True),
    ArmMaintenanceRequest(op="clear_errors"),
    ArmMaintenanceRequest(op="apply_backstops"),
    ArmMaintenanceRequest(op="recover"),
    ArmMaintenanceResult(arm_id="grip", op="clear_errors", path="monitor", ok=False,
                         detail="monitor not connected"),
    _MAINT_CLEAR_OK,
    _MAINT_APPLY_WARN,
    _MAINT_RECOVER_SESSION,
    _MAINT_RECOVER_REFUSED,
    # phase-09c hardware session (home_rail + sweep verdict, speed_scale, bring-up rows)
    ArmMaintenanceRequest(op="home_rail"),
    ArmMaintenanceRequest(op="home_rail", dry_run=True),
    _SWEEP_CLEAR,
    _SWEEP_BLOCKED,
    _MAINT_HOME_DRY,
    _MAINT_HOME_OK,
    _MAINT_HOME_REFUSED,
    *_BRINGUP_ROWS,
    SessionTelemetry(state="bringup", bringup=_BRINGUP_ROWS),
    SessionTelemetry(state="running", bringup=None),
    # phase-09d rail homing with planning (PrePositionPlan, async job status, progress)
    PrePositionPlan(needed=False),
    _PRE_POSITION_NONE,
    _PRE_POSITION_PLAN,
    _PRE_POSITION_REFUSED,
    _SWEEP_CLEAR_09D,
    _SWEEP_BLOCKED_PLANNED,
    _SWEEP_BLOCKED_UNPLANNABLE,
    _MAINT_HOME_DRY_PLANNED,
    _MAINT_HOME_ACCEPTED,
    _MAINT_HOME_JOB_DONE,
    _MAINT_HOME_JOB_FAILED,
    _MAINT_HOME_REFUSED_09D,
    _PROGRESS_QUEUED,
    _PROGRESS_POSITIONING,
    _PROGRESS_DONE,
    _PROGRESS_FAILED,
    _MON_GRIP_HOMING_JOB,
    _MON_VIEW_FROZEN,
    _HW_MONITOR_HOMING_JOB,
    TelemetryMsg(
        seq=11, ts=20.0, epoch="ep0", active_arm=None, controller_connected=False,
        arms=[], collision=CollisionReport.ok(), clearances=[],
        episode=None, dagger=None, inference=None, session=None,
        hardware_monitor=_HW_MONITOR_HOMING_JOB,
    ),
    TelemetryMsg(
        seq=9, ts=18.0, epoch="ep0", active_arm="grip", controller_connected=True,
        arms=[], collision=CollisionReport.ok(), clearances=[],
        episode=None, dagger=None, inference=None,
        session=SessionTelemetry(state="bringup", bringup=_BRINGUP_ROWS),
        hardware_monitor=_HW_MONITOR_PAUSED,
    ),
    SessionSpec(
        mode="teleop", kind="hardware", arms=["grip"], frames={"grip": "arm_base:grip"},
        digital_twin_scene="mavis_v2", speed_scale=0.1,
    ),
    SessionInfo(
        session_id="s1", epoch="ep1", mode="teleop", arms=["grip"], streams=["twin"],
        state="bringup", kind="hardware", speed_scale=0.1,
    ),
    TelemetryMsg(
        seq=5, ts=16.0, epoch="ep0", active_arm="arm0", controller_connected=True,
        arms=[_ARM_FAULTED], collision=CollisionReport.ok(), clearances=[],
        episode=None, dagger=None, inference=None, session=SessionTelemetry(state="fault"),
        hardware_monitor=_HW_MONITOR_PAUSED,
    ),
    TelemetryMsg(
        seq=1,
        ts=12.0,
        epoch="ep0",
        active_arm="arm0",
        controller_connected=True,
        arms=[_ARM],
        collision=CollisionReport.ok(),
        clearances=[ClearanceItem(pair=("a", "b"), dist_m=0.05)],
        episode=EpisodeStatus(state="idle", index=None, frames=0, duration_s=0.0),
        dagger=None,
        inference=None,
        session=SessionTelemetry(state="RUNNING"),
        tracker=_TRACKER_ENGAGED,
        microphone=_MIC_LIVE,
        hardware_monitor=_HW_MONITOR,
    ),
    SessionSpec(
        mode="collect",
        kind="sim",
        arms=["arm0", "arm1"],
        frames={"arm0": "world", "arm1": "arm_base:arm1"},
        sim_scene="two_arm_table",
        task="stack the cubes",
    ),
    SessionInfo(
        session_id="s0", epoch="ep0", mode="teleop", arms=["arm0"], streams=["cam0", "sim"],
        state="RUNNING",
    ),
    ArmStatusInfo(
        arm_id="arm0",
        ip="192.168.1.203",
        connected=True,
        has_rail=True,
        gripper="xarm",
        gripper_force_capable=False,
        error_code=0,
        joint_limits=[(-3.1, 3.1)] * 7 + [(0.0, 0.65)],
    ),
    # phase-11: hardware arm rows carry the probe result; no session yet.
    ArmStatusInfo(
        arm_id="view", ip="192.168.1.186", connected=False, reachable="refused",
        has_rail=True, gripper="none", gripper_force_capable=False, error_code=0,
        joint_limits=[(-3.1, 3.1)] * 7 + [(0.0, 0.65)],
    ),
    CameraInfo(
        camera_id="cam0", kind="v4l2", label="wrist 0", resolution=(640, 480), fps=30, live=True
    ),
    CameraInfo(
        camera_id="camera1", kind="v4l2", label="camera1", resolution=(640, 480), fps=30,
        live=False,  # configured, not attached (Hardware tab draws a black tile)
    ),
    # phase-09a: digital-twin overlay rows (kind "twin"); live only while the real camera
    # is live AND the overlay is live/stale.
    CameraInfo(
        camera_id="grip_wrist_align", kind="twin", label="Manipulation · twin overlay",
        resolution=(640, 480), fps=12, live=True,
    ),
    CameraInfo(
        camera_id="view_wrist_align", kind="twin", label="Perception · twin overlay",
        resolution=(640, 480), fps=12, live=False,
    ),
    WorkcellStatus(kind="sim", available_kinds=["sim"], arms=[], cameras=[]),
    WorkcellStatus(
        kind="hardware", available_kinds=["hardware", "sim"],
        arms=[ArmStatusInfo(
            arm_id="grip", ip="192.168.1.185", connected=False, reachable="open",
            has_rail=True, gripper="xarm", gripper_force_capable=True, error_code=0,
            joint_limits=[(-3.1, 3.1)] * 7 + [(0.0, 0.65)],
        )],
        cameras=[], policies_available=False, hardware_ready=True,
    ),
    SceneInfo(
        scene_id="two_arm_table", label="Two-arm table", num_arms=2,
        rail_flags=[True, False], cameras=["cam0"], kind="sim",
    ),
    ProfileInfo(
        profile_id="abc123", name="home", arms=["arm0"], notes="",
        created_at="2026-09-01T00:00:00Z", is_initial_condition=True,
    ),
    PolicyInfo(
        policy_id="run0/v000003", path="checkpoints/run0/v000003/", action_space="delta_ee",
        action_frame="arm_base:arm0", policy_version=3, promoted=False,
    ),
]


@pytest.mark.parametrize("model", _WIRE_MODELS, ids=lambda m: type(m).__name__)
def test_wire_model_json_round_trip(model: BaseModel):
    """Every wire model survives model_dump_json -> model_validate_json."""
    assert type(model).model_validate_json(model.model_dump_json()) == model


def test_discriminated_union_parses_mixed_transcript():
    """One TypeAdapter parses the client side of a mixed control transcript."""
    transcript = [
        json.dumps({"t": "keys", "seq": 1, "ts": 0.1, "held": ["KeyW"]}),
        json.dumps({"t": "action", "name": "switch_arm", "args": {}}),
        json.dumps({"t": "keys", "seq": 2, "ts": 0.2, "held": []}),
        json.dumps({"t": "action", "name": "takeover_toggle"}),
        json.dumps(
            {
                "t": "action",
                "name": "joint_target",
                "args": {"arm_id": "arm0", "positions": [0.0] * 8, "mode": "goto"},
            }
        ),
        json.dumps({"t": "keys", "seq": 3, "ts": 0.3, "held": ["KeyC", "ArrowLeft"]}),
        json.dumps({"t": "action", "name": "switch_arm_prev"}),
        json.dumps({"t": "action", "name": "tracker_settings", "args": {"yaw_deg": 90.0}}),
    ]
    parsed = [parse_client_msg(line) for line in transcript]
    assert [type(m) for m in parsed] == [
        KeysMsg, ActionMsg, KeysMsg, ActionMsg, ActionMsg, KeysMsg, ActionMsg, ActionMsg,
    ]
    assert parsed[0].held == ["KeyW"]
    assert parsed[3].name == "takeover_toggle"
    assert parsed[5].held == ["KeyC", "ArrowLeft"]  # clutch rides KeysMsg.held
    assert parsed[6].name == "switch_arm_prev"
    assert parsed[7].args == {"yaw_deg": 90.0}

    # Bytes parse too; server messages are rejected on the client channel.
    assert isinstance(parse_client_msg(transcript[0].encode()), KeysMsg)
    with pytest.raises(ValidationError):
        parse_client_msg(json.dumps({"t": "hello", "epoch": "e", "session_id": None,
                                     "role": "controller"}))
    with pytest.raises(ValidationError):
        parse_client_msg(json.dumps({"t": "ack", "name": "switch_arm", "ok": True}))
    with pytest.raises(ValidationError):
        parse_client_msg(json.dumps({"t": "telemetry"}))


def test_controller_telemetry_defaults_are_released():
    """13-tracker §1.1: an untouched controller reads as all-zero / all-False."""
    c = ControllerTelemetry()
    assert c.trigger == 0.0 and c.trackpad_x == 0.0 and c.trackpad_y == 0.0
    assert not any([
        c.trigger_pressed, c.trackpad_touch, c.trackpad_click, c.grip, c.menu, c.system,
    ])
    assert set(ControllerTelemetry.model_fields) == {
        "trigger", "trigger_pressed", "trackpad_touch", "trackpad_click",
        "trackpad_x", "trackpad_y", "grip", "menu", "system",
    }


def test_tracker_telemetry_controller_fields_are_additive():
    """Pre-controller producers (no ``controller``/``device_held``) still parse."""
    legacy = _TRACKER_IDLE.model_dump(mode="json")
    legacy.pop("controller")
    legacy.pop("device_held")
    parsed = TrackerTelemetry.model_validate(legacy)
    assert parsed.controller is None
    assert parsed.device_held == []
    # default_factory: instances must not share the list.
    other = TrackerTelemetry.model_validate(legacy)
    parsed.device_held.append("KeyC")
    assert other.device_held == []
    # Wire form carries both new keys (UI shape), controller as a nested object.
    wire = json.loads(_TRACKER_ENGAGED.model_dump_json())
    assert wire["device_held"] == ["KeyC", "KeyH"]
    assert wire["controller"]["trigger_pressed"] is True
    assert wire["controller"]["trackpad_y"] == 0.85
    assert TrackerTelemetry.model_validate(wire) == _TRACKER_ENGAGED


def test_tracker_settings_msg_filter_fields_default_and_round_trip():
    """13-tracker §4: filter settings are additive with the runtime's defaults."""
    legacy = TrackerSettingsMsg.model_validate(
        {"yaw_deg": 0.0, "pos_scale": 1.0, "follow_rotation": True}
    )
    assert (legacy.filter_enabled, legacy.filter_min_cutoff_hz, legacy.filter_beta) == (
        True, 1.0, 0.05,
    )
    assert set(TrackerSettingsMsg.model_fields) == {
        "yaw_deg", "pos_scale", "follow_rotation",
        "filter_enabled", "filter_min_cutoff_hz", "filter_beta",
    }
    # Telemetry echoes the effective (possibly tuned) settings verbatim.
    wire = json.loads(_TRACKER_SETTINGS_TUNED.model_dump_json())
    assert wire["filter_enabled"] is False
    assert wire["filter_min_cutoff_hz"] == 2.5 and wire["filter_beta"] == 0.2
    assert TrackerSettingsMsg.model_validate(wire) == _TRACKER_SETTINGS_TUNED
    # Legacy wire form (no filter keys) still validates inside TrackerTelemetry.
    with pytest.raises(ValidationError):
        TrackerSettingsMsg.model_validate({"yaw_deg": 0.0, "pos_scale": 1.0})


def test_tracker_telemetry_filter_and_device_action_fields_are_additive():
    """Pre-filter producers (no ``pose_filtered``/``device_action``) still parse."""
    legacy = _TRACKER_ENGAGED.model_dump(mode="json")
    legacy.pop("pose_filtered")
    legacy.pop("device_action")
    for key in ("filter_enabled", "filter_min_cutoff_hz", "filter_beta"):
        legacy["settings"].pop(key)
    parsed = TrackerTelemetry.model_validate(legacy)
    assert parsed.pose_filtered is None and parsed.device_action is None
    assert parsed.settings.filter_enabled is True
    # Wire form carries both keys; pose_filtered is a nested PoseMsg.
    wire = json.loads(_TRACKER_ENGAGED.model_dump_json())
    assert wire["device_action"] == "switch_arm"
    assert wire["pose_filtered"]["position"] == [0.001, 0.299, 0.4]
    assert wire["settings"]["filter_min_cutoff_hz"] == 2.5
    assert TrackerTelemetry.model_validate(wire) == _TRACKER_ENGAGED
    # Idle block: filtered pose absent, no action fired.
    idle = json.loads(_TRACKER_IDLE.model_dump_json())
    assert idle["pose_filtered"] is None and idle["device_action"] is None
    assert idle["settings"]["filter_enabled"] is True


def test_tracker_telemetry_charging_and_calibration_fields_are_additive():
    """Pre-phase-10 producers (no ``charging``/``calibration``) still parse, and
    so do pre-2026-09-07 ones without the three controller-LINK fields."""
    assert set(TrackerTelemetry.model_fields) == {
        "backend", "status", "detail", "object_name", "seq", "rate_hz", "age_s",
        "pose_raw", "pose_world", "pose_filtered", "clutch", "engaged_arm", "anchor_tcp",
        "target_tcp", "settings", "controller", "device_held", "device_action",
        "charging", "calibration",
        # Controller link (13-tracker §3.5 item 7b): the BUTTON path's own age,
        # the pairing evidence and the receiver's USB presence.
        "controller_age_s", "objects", "dongle_present",
    }
    legacy = _TRACKER_ENGAGED.model_dump(mode="json")
    legacy.pop("charging")
    legacy.pop("calibration")
    for link_field in ("controller_age_s", "objects", "dongle_present"):
        legacy.pop(link_field)
    parsed = TrackerTelemetry.model_validate(legacy)
    assert parsed.charging is None and parsed.calibration is None
    # A producer without the link fields must read as UNKNOWN, never as "no
    # receiver" / "not paired": None / None / empty list.
    assert parsed.controller_age_s is None and parsed.dongle_present is None
    assert parsed.objects == []
    # Wire form carries both keys; calibration is the nested status snapshot.
    calibrating = _TRACKER_ENGAGED.model_copy(
        update={"charging": True, "calibration": _CALIB_BASE_STATION}
    )
    wire = json.loads(calibrating.model_dump_json())
    assert wire["charging"] is True
    assert wire["calibration"]["kind"] == "base_station"
    assert wire["calibration"]["phase"] == "capturing"
    assert wire["calibration"]["scenes"] == 4
    assert wire["calibration"]["lighthouses"][0]["reference"] is True
    assert wire["calibration"]["lighthouses"][2]["channel"] is None
    assert wire["calibration"]["yaw_valid"] is False
    assert TrackerTelemetry.model_validate(wire) == calibrating
    # Idle block: nothing reported.
    idle = json.loads(_TRACKER_IDLE.model_dump_json())
    assert idle["charging"] is None and idle["calibration"] is None
    # Nested inside a full telemetry frame too.
    frame = TelemetryMsg(
        seq=2, ts=13.0, epoch="ep0", active_arm=None, controller_connected=False,
        arms=[], collision=CollisionReport.ok(), clearances=[],
        episode=None, dagger=None, inference=None, tracker=calibrating,
    )
    parsed_frame = TelemetryMsg.model_validate_json(frame.model_dump_json())
    assert parsed_frame.tracker is not None
    assert parsed_frame.tracker.calibration == _CALIB_BASE_STATION


def test_microphone_telemetry_fields_pinned_and_defaulted():
    """phase-11: every MicrophoneTelemetry field defaults; MicStatus is shared with REST."""
    assert set(MicrophoneTelemetry.model_fields) == {
        "mic_id", "status", "detail", "seq", "age_s", "rate_hz", "sample_rate",
        "rms_dbfs", "peak_dbfs", "clipping", "env_min", "env_max", "overruns",
    }
    assert all(not f.is_required() for f in MicrophoneTelemetry.model_fields.values())
    idle = MicrophoneTelemetry()
    assert idle.mic_id == "mic_view" and idle.status == "no_backend" and idle.detail == ""
    assert (idle.seq, idle.age_s, idle.rate_hz, idle.sample_rate) == (0, None, 0.0, 48000)
    assert (idle.rms_dbfs, idle.peak_dbfs, idle.clipping) == (None, None, False)
    assert idle.env_min == [] and idle.env_max == [] and idle.overruns == 0
    # Mutable list defaults are per-instance (pydantic copies them).
    other = MicrophoneTelemetry()
    idle.env_min.append(1)
    assert other.env_min == []
    # One MicStatus vocabulary for the telemetry block and the REST row.
    assert get_args(MicStatus) == (
        "no_backend", "starting", "absent", "live", "stalled", "error",
    )
    assert MicrophoneTelemetry.model_fields["status"].annotation == MicStatus
    assert MicrophoneInfo.model_fields["status"].annotation == MicStatus
    with pytest.raises(ValidationError):
        MicrophoneTelemetry(status="tracking")  # tracker vocabulary, not MicStatus
    with pytest.raises(ValidationError):
        MicrophoneTelemetry(env_min=[0.5] * 64)  # int8 envelope, not floats
    # Wire form: 64-bin envelope survives as plain integer arrays.
    wire = json.loads(_MIC_LIVE.model_dump_json())
    assert len(wire["env_min"]) == len(wire["env_max"]) == 64
    assert min(wire["env_min"]) >= -127 and max(wire["env_max"]) <= 127
    assert wire["clipping"] is True and wire["peak_dbfs"] == -0.4
    assert MicrophoneTelemetry.model_validate(wire) == _MIC_LIVE


def test_telemetry_microphone_block_is_additive():
    """Pre-phase-11 producers (no ``microphone`` key) still parse; block rides the frame."""
    frame = TelemetryMsg(
        seq=3, ts=14.0, epoch="ep0", active_arm=None, controller_connected=False,
        arms=[], collision=CollisionReport.ok(), clearances=[],
        episode=None, dagger=None, inference=None, microphone=_MIC_LIVE,
    )
    assert list(TelemetryMsg.model_fields)[-4:] == [
        "session", "tracker", "microphone", "hardware_monitor",
    ]
    assert TelemetryMsg.model_fields["microphone"].default is None
    legacy = frame.model_dump(mode="json")
    legacy.pop("microphone")
    parsed = TelemetryMsg.model_validate(legacy)
    assert parsed.microphone is None
    wire = json.loads(frame.model_dump_json())
    assert wire["microphone"]["status"] == "live" and wire["microphone"]["seq"] == 4321
    assert wire["microphone"]["env_max"][-1] == 126
    assert TelemetryMsg.model_validate(wire) == frame
    # A frame without a device still carries the defaulted block when the runtime sends one.
    bare = frame.model_copy(update={"microphone": MicrophoneTelemetry()})
    assert json.loads(bare.model_dump_json())["microphone"]["status"] == "no_backend"


def test_microphone_info_fields_pinned():
    """phase-11: GET /api/microphones row, spelled exactly."""
    assert set(MicrophoneInfo.model_fields) == {
        "mic_id", "label", "kind", "source", "sample_rate", "channels", "live", "status",
        "detail",
    }
    required = {n for n, f in MicrophoneInfo.model_fields.items() if f.is_required()}
    assert required == {"mic_id", "label", "kind", "source", "sample_rate", "live", "status"}
    assert MicrophoneInfo.model_fields["channels"].default == 1
    assert MicrophoneInfo.model_fields["detail"].default == ""
    assert _MIC_INFO_ABSENT.channels == 1 and _MIC_INFO_ABSENT.source is None
    wire = json.loads(_MIC_INFO_LIVE.model_dump_json())
    assert wire["kind"] == "pulse" and wire["live"] is True and wire["status"] == "live"
    assert wire["source"].startswith("alsa_input.usb-R__DE_Microphones")
    assert MicrophoneInfo.model_validate(wire) == _MIC_INFO_LIVE
    with pytest.raises(ValidationError):
        MicrophoneInfo.model_validate({**wire, "kind": "alsa"})  # hw: route is banned
    with pytest.raises(ValidationError):
        MicrophoneInfo.model_validate({**wire, "status": "tracking"})
    with pytest.raises(ValidationError):
        MicrophoneInfo.model_validate({k: v for k, v in wire.items() if k != "source"})


def test_workcell_status_phase11_fields_are_additive():
    """phase-11: ``reachable`` / ``hardware_ready`` default for pre-probe producers."""
    assert set(ArmStatusInfo.model_fields) == {
        "arm_id", "ip", "connected", "reachable", "has_rail", "gripper",
        "gripper_force_capable", "error_code", "joint_limits",
    }
    assert set(WorkcellStatus.model_fields) == {
        "kind", "available_kinds", "arms", "cameras", "policies_available", "hardware_ready",
    }
    assert ArmStatusInfo.model_fields["reachable"].default == "unknown"
    assert WorkcellStatus.model_fields["hardware_ready"].default is False
    arm = ArmStatusInfo(
        arm_id="grip", ip="192.168.1.185", connected=False, reachable="unreachable",
        has_rail=True, gripper="xarm", gripper_force_capable=True, error_code=0,
        joint_limits=[(-3.1, 3.1)] * 7 + [(0.0, 0.65)],
    )
    legacy_arm = arm.model_dump(mode="json")
    legacy_arm.pop("reachable")
    assert ArmStatusInfo.model_validate(legacy_arm).reachable == "unknown"
    status = WorkcellStatus(
        kind="hardware", available_kinds=["hardware", "sim"], arms=[arm], cameras=[],
    )
    assert status.hardware_ready is False  # default: nothing probed open yet
    legacy = status.model_dump(mode="json")
    legacy.pop("hardware_ready")
    legacy["arms"][0].pop("reachable")
    parsed = WorkcellStatus.model_validate(legacy)
    assert parsed.hardware_ready is False and parsed.arms[0].reachable == "unknown"
    # ``connected`` keeps its "a session exists" meaning next to the probe result.
    ready = status.model_copy(update={
        "arms": [arm.model_copy(update={"reachable": "open"})], "hardware_ready": True,
    })
    wire = json.loads(ready.model_dump_json())
    assert wire["hardware_ready"] is True
    assert wire["arms"][0]["reachable"] == "open" and wire["arms"][0]["connected"] is False
    assert WorkcellStatus.model_validate(wire) == ready
    for bad in ("booting", "OPEN", ""):
        with pytest.raises(ValidationError):
            ArmStatusInfo.model_validate({**legacy_arm, "reachable": bad})


def test_arm_monitor_telemetry_fields_pinned_and_defaulted():
    """phase-09a: read-only monitor row, spelled exactly; only ``arm_id`` is required."""
    assert set(ArmMonitorTelemetry.model_fields) == {
        "arm_id", "status", "detail", "seq", "age_s", "q", "tcp_pose",
        "rail_present", "rail_homed", "rail_enabled", "rail_pos_m", "rail_raw_mm",
        "gripper_open_frac", "gripper_raw", "error_code", "warn_code", "state", "mode",
        # phase-09b read-back + maintenance flag
        "collision_sensitivity", "tcp_load_kg", "tcp_load_cog_mm", "backstops_match",
        "maintenance_busy",
        # phase-09d async maintenance job progress
        "maintenance",
    }
    required = {n for n, f in ArmMonitorTelemetry.model_fields.items() if f.is_required()}
    assert required == {"arm_id"}
    off = ArmMonitorTelemetry(arm_id="grip")
    assert off.status == "off" and off.detail == "" and off.seq == 0 and off.age_s is None
    assert off.q == [] and off.tcp_pose == []
    assert (off.rail_present, off.rail_homed, off.rail_enabled) == (None, None, None)
    assert (off.rail_pos_m, off.rail_raw_mm) == (None, None)
    assert (off.gripper_open_frac, off.gripper_raw) == (None, None)
    assert (off.error_code, off.warn_code, off.state, off.mode) == (0, 0, None, None)
    assert (off.collision_sensitivity, off.tcp_load_kg, off.backstops_match) == (None,) * 3
    assert off.tcp_load_cog_mm == [] and off.maintenance_busy is False
    assert off.maintenance is None
    # Mutable list defaults are per-instance (pydantic copies them).
    other = ArmMonitorTelemetry(arm_id="grip")
    off.q.append(1.0)
    assert other.q == []
    # Status vocabulary, spelled exactly; "paused" = a hardware session owns the box.
    assert get_args(ArmMonitorStatus) == (
        "off", "connecting", "running", "stale", "paused", "error",
    )
    assert ArmMonitorTelemetry.model_fields["status"].annotation == ArmMonitorStatus
    for bad in ("tracking", "live", "no_backend", "RUNNING", ""):
        with pytest.raises(ValidationError):
            ArmMonitorTelemetry(arm_id="grip", status=bad)
    with pytest.raises(ValidationError):
        ArmMonitorTelemetry()  # arm_id is required
    with pytest.raises(ValidationError):
        ArmMonitorTelemetry(arm_id="grip", q=["a"] * 7)
    with pytest.raises(ValidationError):
        ArmMonitorTelemetry(arm_id="grip", error_code=1.5)
    # Live facts (2026-09-04): unhomed track -> rail_pos_m None while rail_raw_mm reads 0;
    # the Perception Arm latches C19 with gripper "none"; q is the raw controller angle.
    wire = json.loads(_MON_GRIP_RUNNING.model_dump_json())
    assert wire["rail_present"] is True and wire["rail_homed"] is False
    assert wire["rail_pos_m"] is None and wire["rail_raw_mm"] == 0.0
    assert len(wire["q"]) == 7 and len(wire["tcp_pose"]) == 6 and wire["q"][0] == 3.14159
    assert wire["state"] == 4 and wire["mode"] == 0 and wire["gripper_open_frac"] == 0.93
    assert ArmMonitorTelemetry.model_validate(wire) == _MON_GRIP_RUNNING
    view = json.loads(_MON_VIEW_ERROR.model_dump_json())
    assert view["status"] == "running" and view["error_code"] == 19
    assert view["gripper_open_frac"] is None and view["gripper_raw"] is None
    assert view["detail"] == "controller error 19: End Effector Communication Error"
    assert ArmMonitorTelemetry.model_validate(view) == _MON_VIEW_ERROR


def test_twin_overlay_telemetry_fields_pinned_and_defaulted():
    """phase-09a: one ``<camera_id>_align`` stream; the three ids required, the rest default."""
    assert set(TwinOverlayTelemetry.model_fields) == {
        "stream_id", "camera_id", "arm_id", "status", "detail", "fps",
        "rail_fallback_m", "joint1_offset_rad", "mask_fraction",
    }
    required = {n for n, f in TwinOverlayTelemetry.model_fields.items() if f.is_required()}
    assert required == {"stream_id", "camera_id", "arm_id"}
    off = TwinOverlayTelemetry(stream_id="grip_wrist_align", camera_id="grip_wrist", arm_id="grip")
    assert off.status == "off" and off.detail == "" and off.fps == 0.0
    assert off.rail_fallback_m is None
    assert off.joint1_offset_rad == 0.0 and off.mask_fraction == 0.0
    assert get_args(TwinOverlayStatus) == ("off", "waiting", "live", "stale", "error")
    assert TwinOverlayTelemetry.model_fields["status"].annotation == TwinOverlayStatus
    for bad in ("running", "paused", "tracking", "LIVE", ""):
        with pytest.raises(ValidationError):
            TwinOverlayTelemetry.model_validate({**off.model_dump(), "status": bad})
    for key in ("stream_id", "camera_id", "arm_id"):
        with pytest.raises(ValidationError):
            TwinOverlayTelemetry.model_validate(
                {k: v for k, v in off.model_dump().items() if k != key}
            )
    # Rail fallback in use (track not homed): the block carries the assumed position and
    # the operator-facing reason. The stream id is the camera id + "_align": never a
    # camera id, never "*_wrist_cam", never a reserved session render (04-runtime §13.4).
    wire = json.loads(_OVERLAY_GRIP_LIVE.model_dump_json())
    assert wire["rail_fallback_m"] == 0.65
    assert wire["detail"] == "rail not homed - twin assumes 0.65 m"
    assert wire["status"] == "live" and wire["fps"] == 11.8 and wire["mask_fraction"] == 0.184
    assert wire["stream_id"] == f"{wire['camera_id']}_align"
    assert not wire["stream_id"].endswith("_wrist_cam")
    assert not is_reserved_stream(wire["stream_id"])
    assert TwinOverlayTelemetry.model_validate(wire) == _OVERLAY_GRIP_LIVE
    waiting = json.loads(_OVERLAY_VIEW_WAITING.model_dump_json())
    assert waiting["status"] == "waiting" and waiting["rail_fallback_m"] is None
    assert waiting["fps"] == 0.0 and waiting["mask_fraction"] == 0.0


def test_hardware_monitor_telemetry_block_pinned_and_defaulted():
    """phase-09a: the block validates empty (no hardware package) and nests both lists."""
    assert set(HardwareMonitorTelemetry.model_fields) == {"enabled", "paused", "arms", "overlays"}
    assert all(not f.is_required() for f in HardwareMonitorTelemetry.model_fields.values())
    empty = HardwareMonitorTelemetry()
    assert empty.enabled is False and empty.paused is False
    assert empty.arms == [] and empty.overlays == []
    other = HardwareMonitorTelemetry()
    empty.arms.append(_MON_GRIP_RUNNING)
    assert other.arms == []  # per-instance list defaults
    wire = json.loads(_HW_MONITOR.model_dump_json())
    assert wire["enabled"] is True and wire["paused"] is False
    assert [a["arm_id"] for a in wire["arms"]] == ["grip", "view"]
    assert [a["status"] for a in wire["arms"]] == ["running", "running"]
    assert [a["error_code"] for a in wire["arms"]] == [0, 19]
    assert [o["stream_id"] for o in wire["overlays"]] == ["grip_wrist_align", "view_wrist_align"]
    assert [o["status"] for o in wire["overlays"]] == ["live", "waiting"]
    assert HardwareMonitorTelemetry.model_validate(wire) == _HW_MONITOR
    # Paused = a hardware session owns the boxes: connections released, overlays off.
    paused = json.loads(_HW_MONITOR_PAUSED.model_dump_json())
    assert paused["enabled"] is True and paused["paused"] is True
    assert {a["status"] for a in paused["arms"]} == {"paused"}
    assert {o["status"] for o in paused["overlays"]} == {"off"}
    assert HardwareMonitorTelemetry.model_validate(paused) == _HW_MONITOR_PAUSED
    with pytest.raises(ValidationError):
        HardwareMonitorTelemetry(arms=[{"status": "running"}])  # arm_id required per row
    with pytest.raises(ValidationError):
        HardwareMonitorTelemetry(overlays=[{"arm_id": "grip"}])  # stream/camera ids required
    with pytest.raises(ValidationError):
        HardwareMonitorTelemetry(paused="later")


def test_telemetry_hardware_monitor_block_is_additive():
    """Pre-09a producers (no ``hardware_monitor`` key) still parse; the block rides the frame."""
    frame = TelemetryMsg(
        seq=4, ts=15.0, epoch="ep0", active_arm=None, controller_connected=False,
        arms=[], collision=CollisionReport.ok(), clearances=[],
        episode=None, dagger=None, inference=None, hardware_monitor=_HW_MONITOR,
    )
    assert list(TelemetryMsg.model_fields)[-4:] == [
        "session", "tracker", "microphone", "hardware_monitor",
    ]
    assert TelemetryMsg.model_fields["hardware_monitor"].default is None
    legacy = frame.model_dump(mode="json")
    legacy.pop("hardware_monitor")
    parsed = TelemetryMsg.model_validate(legacy)
    assert parsed.hardware_monitor is None
    wire = json.loads(frame.model_dump_json())
    assert wire["hardware_monitor"]["enabled"] is True
    assert wire["hardware_monitor"]["arms"][1]["error_code"] == 19
    assert wire["hardware_monitor"]["overlays"][0]["rail_fallback_m"] == 0.65
    assert TelemetryMsg.model_validate(wire) == frame
    # Session-less like tracker/microphone: no session, ``arms`` empty, monitor still live.
    assert frame.arms == [] and frame.session is None and frame.hardware_monitor is not None
    # A runtime without the hardware package still sends the defaulted block.
    bare = frame.model_copy(update={"hardware_monitor": HardwareMonitorTelemetry()})
    assert json.loads(bare.model_dump_json())["hardware_monitor"] == {
        "enabled": False, "paused": False, "arms": [], "overlays": [],
    }


def test_camera_info_kind_gains_twin():
    """phase-09a: ``kind == "twin"`` = digital-twin overlay stream (``<camera_id>_align``)."""
    assert set(CameraInfo.model_fields) == {
        "camera_id", "kind", "label", "resolution", "fps", "live",
    }
    assert get_args(CameraInfo.model_fields["kind"].annotation) == (
        "v4l2", "realsense", "sim", "twin",
    )
    row = CameraInfo(
        camera_id="grip_wrist_align", kind="twin", label="Manipulation · twin overlay",
        resolution=(640, 480), fps=12, live=True,
    )
    wire = json.loads(row.model_dump_json())
    assert wire["kind"] == "twin" and wire["resolution"] == [640, 480] and wire["fps"] == 12
    assert CameraInfo.model_validate(wire) == row
    # Existing kinds unchanged; unknown spellings rejected.
    for kind in ("v4l2", "realsense", "sim"):
        assert CameraInfo.model_validate({**wire, "kind": kind}).kind == kind
    for bad in ("overlay", "align", "TWIN", ""):
        with pytest.raises(ValidationError):
            CameraInfo.model_validate({**wire, "kind": bad})
    # Overlay rows ride /api/workcell next to the real cameras (same $defs class).
    status = WorkcellStatus(
        kind="hardware", available_kinds=["hardware", "sim"], arms=[],
        cameras=[
            CameraInfo(camera_id="grip_wrist", kind="v4l2", label="Manipulation Arm wrist",
                       resolution=(640, 480), fps=30, live=True),
            row,
        ],
    )
    parsed = WorkcellStatus.model_validate_json(status.model_dump_json())
    assert [c.kind for c in parsed.cameras] == ["v4l2", "twin"]
    assert [c.camera_id for c in parsed.cameras] == ["grip_wrist", "grip_wrist_align"]


def test_arm_monitor_telemetry_backstops_readback_is_additive():
    """phase-09b: the safety read-back + ``maintenance_busy`` default; pre-09b rows parse."""
    legacy = _MON_VIEW_BACKSTOPS.model_dump(mode="json")
    for key in ("collision_sensitivity", "tcp_load_kg", "tcp_load_cog_mm", "backstops_match",
                "maintenance_busy"):
        legacy.pop(key)
    parsed = ArmMonitorTelemetry.model_validate(legacy)
    assert parsed.collision_sensitivity is None and parsed.tcp_load_kg is None
    assert parsed.tcp_load_cog_mm == [] and parsed.backstops_match is None
    assert parsed.maintenance_busy is False
    # Live 2026-09-04 read-back before "Apply safety settings": payload 0 kg, sensitivity 1
    # on the Perception Arm -> runtime flags the mismatch; after: configured values match.
    before = json.loads(_MON_VIEW_CLEARED.model_dump_json())
    assert before["collision_sensitivity"] == 1 and before["tcp_load_kg"] == 0.0
    assert before["tcp_load_cog_mm"] == [0.0, 0.0, 0.0] and before["backstops_match"] is False
    assert before["error_code"] == 0 and before["maintenance_busy"] is False
    after = json.loads(_MON_VIEW_BACKSTOPS.model_dump_json())
    assert after["collision_sensitivity"] == 3 and after["tcp_load_kg"] == 0.55
    assert after["tcp_load_cog_mm"] == [0.0, 0.0, 90.0] and after["backstops_match"] is True
    assert ArmMonitorTelemetry.model_validate(after) == _MON_VIEW_BACKSTOPS
    busy = ArmMonitorTelemetry(arm_id="grip", status="running", maintenance_busy=True)
    assert json.loads(busy.model_dump_json())["maintenance_busy"] is True
    # The read-back is what the controller reports (an int), never a float or a word.
    with pytest.raises(ValidationError):
        ArmMonitorTelemetry(arm_id="grip", collision_sensitivity=2.5)
    with pytest.raises(ValidationError):
        ArmMonitorTelemetry(arm_id="grip", collision_sensitivity="high")
    with pytest.raises(ValidationError):
        ArmMonitorTelemetry(arm_id="grip", tcp_load_cog_mm=["x", 0, 0])
    with pytest.raises(ValidationError):
        ArmMonitorTelemetry(arm_id="grip", maintenance_busy="later")
    # Rides the block + the frame unchanged.
    block = HardwareMonitorTelemetry(enabled=True, arms=[_MON_GRIP_RUNNING, _MON_VIEW_BACKSTOPS])
    wire = json.loads(block.model_dump_json())
    assert [a["backstops_match"] for a in wire["arms"]] == [None, True]
    assert HardwareMonitorTelemetry.model_validate(wire) == block


def test_arm_telemetry_fault_fields_are_additive():
    """phase-09b: ``fault_detail`` / ``recovering`` default; pre-09b producers still parse."""
    assert set(ArmTelemetry.model_fields) == {
        "arm_id", "connected", "q", "rail_pos_m", "ee_pose", "gripper_open_frac",
        "error_code", "warn_code", "stale", "goto", "fault_detail", "recovering",
    }
    required = {n for n, f in ArmTelemetry.model_fields.items() if f.is_required()}
    assert required == {
        "arm_id", "connected", "q", "rail_pos_m", "ee_pose", "gripper_open_frac", "error_code",
    }
    assert ArmTelemetry.model_fields["fault_detail"].default == ""
    assert ArmTelemetry.model_fields["recovering"].default is False
    legacy = _ARM.model_dump(mode="json")
    legacy.pop("fault_detail")
    legacy.pop("recovering")
    parsed = ArmTelemetry.model_validate(legacy)
    assert parsed.fault_detail == "" and parsed.recovering is False
    assert parsed == _ARM
    # FAULT: the controller code AND its SDK title; RECOVERING: code cleared, flag up until
    # the operator re-grips the clutch (04-runtime §15).
    faulted = json.loads(_ARM_FAULTED.model_dump_json())
    assert faulted["error_code"] == 24 and faulted["recovering"] is False
    assert faulted["fault_detail"] == "controller error 24: Speed Exceeds Limit"
    assert faulted["goto"] is None
    assert ArmTelemetry.model_validate(faulted) == _ARM_FAULTED
    recovering = json.loads(_ARM_RECOVERING.model_dump_json())
    assert recovering["error_code"] == 0 and recovering["recovering"] is True
    assert recovering["fault_detail"] == faulted["fault_detail"]
    with pytest.raises(ValidationError):
        ArmTelemetry.model_validate({**faulted, "recovering": "soon"})
    with pytest.raises(ValidationError):
        ArmTelemetry.model_validate({**faulted, "fault_detail": 24})
    # Nested in a frame whose session block reports the FAULT state.
    frame = TelemetryMsg(
        seq=6, ts=17.0, epoch="ep0", active_arm="arm0", controller_connected=True,
        arms=[_ARM_FAULTED, _ARM], collision=CollisionReport.ok(), clearances=[],
        episode=None, dagger=None, inference=None, session=SessionTelemetry(state="fault"),
    )
    wire = json.loads(frame.model_dump_json())
    assert [a["fault_detail"] for a in wire["arms"]] == [faulted["fault_detail"], ""]
    assert wire["session"]["state"] == "fault"
    assert TelemetryMsg.model_validate(wire) == frame


def test_arm_maintenance_models_pinned():
    """phase-09b: POST /api/hardware/arms/{arm_id}/maintenance body + result, spelled exactly
    (phase-09c adds the ``home_rail`` op, ``dry_run`` and ``rail_sweep``; phase-09d ``status``
    and ``job_id`` - all additive)."""
    assert get_args(ArmMaintenanceOp) == (
        "clear_errors", "apply_backstops", "recover", "home_rail",
    )
    assert get_args(MaintenancePath) == ("monitor", "session")
    assert set(ArmMaintenanceRequest.model_fields) == {"op", "dry_run"}
    assert ArmMaintenanceRequest.model_fields["op"].is_required()
    assert ArmMaintenanceRequest.model_fields["op"].annotation == ArmMaintenanceOp
    assert ArmMaintenanceRequest.model_fields["dry_run"].default is False
    assert set(ArmMaintenanceResult.model_fields) == {
        "arm_id", "op", "path", "ok", "detail", "sdk_codes", "warnings", "before", "after",
        "rail_sweep",
        # phase-09d async job: how the op ran + the RailHomingJob id
        "status", "job_id",
    }
    required = {n for n, f in ArmMaintenanceResult.model_fields.items() if f.is_required()}
    assert required == {"arm_id", "op", "path", "ok"}
    assert ArmMaintenanceResult.model_fields["op"].annotation == ArmMaintenanceOp
    assert ArmMaintenanceResult.model_fields["path"].annotation == MaintenancePath
    bare = ArmMaintenanceResult(arm_id="grip", op="clear_errors", path="monitor", ok=True)
    assert bare.detail == "" and bare.sdk_codes == {} and bare.warnings == []
    assert bare.before is None and bare.after is None and bare.rail_sweep is None
    assert bare.status == "done" and bare.job_id is None
    # Mutable defaults are per-instance (pydantic copies them).
    other = ArmMaintenanceResult(arm_id="grip", op="clear_errors", path="monitor", ok=True)
    bare.sdk_codes["clean_error"] = 0
    bare.warnings.append("x")
    assert other.sdk_codes == {} and other.warnings == []
    # Vocabulary: SDK method names and ad-hoc spellings are not ops (home_rail IS one since
    # phase-09c - the single motion op, twin-gated).
    for bad in ("clean_error", "set_linear_track_back_origin", "home", "homing", "enable",
                "apply", "CLEAR_ERRORS", "HOME_RAIL", ""):
        with pytest.raises(ValidationError):
            ArmMaintenanceRequest(op=bad)
    with pytest.raises(ValidationError):
        ArmMaintenanceRequest(op="home_rail", dry_run="maybe")
    with pytest.raises(ValidationError):
        ArmMaintenanceResult(arm_id="grip", op="home_rail", path="monitor", ok=True,
                             rail_sweep={"clear": True})  # scene_id/inflation_m/step_m required
    for bad in ("rest", "driver", "MONITOR", ""):
        with pytest.raises(ValidationError):
            ArmMaintenanceResult(arm_id="grip", op="recover", path=bad, ok=True)
    with pytest.raises(ValidationError):
        ArmMaintenanceRequest()  # op is required
    with pytest.raises(ValidationError):
        ArmMaintenanceResult(arm_id="grip", op="recover", path="session")  # ok is required
    with pytest.raises(ValidationError):
        ArmMaintenanceResult(arm_id="grip", op="clear_errors", path="monitor", ok=True,
                             sdk_codes={"clean_error": "ok"})
    with pytest.raises(ValidationError):
        ArmMaintenanceResult(arm_id="grip", op="clear_errors", path="monitor", ok=True,
                             warnings=[1])
    with pytest.raises(ValidationError):
        ArmMaintenanceResult(arm_id="grip", op="clear_errors", path="monitor", ok=True,
                             before={"status": "running"})  # arm_id required in the sample
    # Body wire form is flat; dry_run defaults False and a legacy {"op"} body still parses.
    assert json.loads(ArmMaintenanceRequest(op="apply_backstops").model_dump_json()) == {
        "op": "apply_backstops", "dry_run": False,
    }
    assert ArmMaintenanceRequest.model_validate({"op": "recover"}).dry_run is False
    assert json.loads(ArmMaintenanceRequest(op="home_rail", dry_run=True).model_dump_json()) == {
        "op": "home_rail", "dry_run": True,
    }
    # clear_errors on the monitor path: exactly clean_error + clean_warn in call order, the
    # before/after samples show C19 -> 0, and the op NEVER enabled motion.
    clear = json.loads(_MAINT_CLEAR_OK.model_dump_json())
    assert (clear["arm_id"], clear["op"], clear["path"], clear["ok"]) == (
        "view", "clear_errors", "monitor", True,
    )
    assert list(clear["sdk_codes"]) == ["clean_error", "clean_warn"]
    assert "motion_enable" not in clear["sdk_codes"] and clear["warnings"] == []
    assert clear["before"]["error_code"] == 19 and clear["after"]["error_code"] == 0
    assert clear["before"]["arm_id"] == clear["after"]["arm_id"] == "view"
    assert ArmMaintenanceResult.model_validate(clear) == _MAINT_CLEAR_OK
    # apply_backstops: the backstops.py call order, one non-fatal warning, read-back matches.
    apply = json.loads(_MAINT_APPLY_WARN.model_dump_json())
    assert list(apply["sdk_codes"]) == [
        "set_tcp_load", "set_gravity_direction", "set_collision_sensitivity",
        "set_self_collision_detection", "set_collision_tool_model", "set_collision_rebound",
    ]
    assert apply["ok"] is True and apply["warnings"] == ["set_collision_tool_model returned 1"]
    assert apply["after"]["backstops_match"] is True and apply["after"]["tcp_load_kg"] == 0.55
    assert apply["before"]["backstops_match"] is False
    assert ArmMaintenanceResult.model_validate(apply) == _MAINT_APPLY_WARN
    # recover: session path carries no monitor samples (the monitor is paused); refused on
    # the monitor path with ok=False and the reason in detail.
    recover = json.loads(_MAINT_RECOVER_SESSION.model_dump_json())
    assert recover["path"] == "session" and recover["before"] is None
    assert recover["after"] is None and "motion_enable" in recover["sdk_codes"]
    assert ArmMaintenanceResult.model_validate(recover) == _MAINT_RECOVER_SESSION
    refused = json.loads(_MAINT_RECOVER_REFUSED.model_dump_json())
    assert refused["ok"] is False and refused["detail"] == "recover needs a session"
    assert refused["sdk_codes"] == {} and refused["path"] == "monitor"


def test_rail_sweep_verdict_pinned():
    """phase-09c: the twin sweep verdict that gates home_rail, spelled exactly."""
    assert set(RailSweepVerdict.model_fields) == {
        "scene_id", "inflation_m", "step_m", "travel_m", "clear",
        "first_blocked_m", "first_blocked_pair",
        "min_clearance_m", "min_clearance_at_m", "min_clearance_pair",
        "q_checked", "other_arms", "assumptions", "sample_seq",
        "pre_position",  # phase-09d plan (additive)
    }
    required = {n for n, f in RailSweepVerdict.model_fields.items() if f.is_required()}
    assert required == {"scene_id", "inflation_m", "step_m", "clear"}
    bare = RailSweepVerdict(scene_id="mavis_v2", inflation_m=0.025, step_m=0.005, clear=True)
    assert bare.pre_position is None
    assert bare.travel_m == 0.65  # full rail travel (CLAUDE.md: max 0.65 m)
    assert bare.first_blocked_m is None and bare.first_blocked_pair == []
    assert bare.min_clearance_m is None and bare.min_clearance_at_m is None
    assert bare.min_clearance_pair == [] and bare.q_checked == [] and bare.other_arms == {}
    assert bare.assumptions == [] and bare.sample_seq == 0
    # Mutable defaults are per-instance.
    other = RailSweepVerdict(scene_id="mavis_v2", inflation_m=0.025, step_m=0.005, clear=True)
    bare.assumptions.append("x")
    bare.other_arms["view"] = [0.0] * 8
    assert other.assumptions == [] and other.other_arms == {}
    # D4 numbers: the homing sweep uses the guardrail's 0.025 m debug inflation and 5 mm steps
    # (131 positions over 0.65 m); the verdict carries them so the UI can say so.
    assert (_SWEEP_CLEAR.inflation_m, _SWEEP_CLEAR.step_m) == (0.025, 0.005)
    assert round(_SWEEP_CLEAR.travel_m / _SWEEP_CLEAR.step_m) + 1 == 131
    # Wire form: q_checked is the 7 joints the executing monitor must re-find; other_arms
    # carries q7 + rail (8 values) for the arm that was frozen at its last sample.
    wire = json.loads(_SWEEP_CLEAR.model_dump_json())
    assert wire["clear"] is True and wire["first_blocked_m"] is None
    assert len(wire["q_checked"]) == 7 and len(wire["other_arms"]["view"]) == 8
    assert wire["assumptions"] == ["view rail unknown - used fallback 0.00 m"]
    assert wire["min_clearance_pair"] == ["grip/link2", "grip/link4"]
    assert RailSweepVerdict.model_validate(wire) == _SWEEP_CLEAR
    blocked = json.loads(_SWEEP_BLOCKED.model_dump_json())
    assert blocked["clear"] is False and blocked["first_blocked_m"] == 0.125
    assert blocked["first_blocked_pair"] == ["grip/gripper_finger_left", "table"]
    assert blocked["min_clearance_m"] < 0  # penetration at inflation
    assert RailSweepVerdict.model_validate(blocked) == _SWEEP_BLOCKED
    for bad in (
        {"scene_id": "mavis_v2", "inflation_m": 0.025, "step_m": 0.005},  # clear required
        {"scene_id": "mavis_v2", "inflation_m": "thin", "step_m": 0.005, "clear": True},
        {"scene_id": "mavis_v2", "inflation_m": 0.025, "step_m": 0.005, "clear": True,
         "first_blocked_pair": "table"},  # list, not str
        {"scene_id": "mavis_v2", "inflation_m": 0.025, "step_m": 0.005, "clear": True,
         "other_arms": {"view": "folded"}},
        {"scene_id": "mavis_v2", "inflation_m": 0.025, "step_m": 0.005, "clear": True,
         "sample_seq": 1.5},
    ):
        with pytest.raises(ValidationError):
            RailSweepVerdict.model_validate(bad)


def test_home_rail_maintenance_wire():
    """phase-09c: home_rail results - dry-run carries the verdict with zero writes, the real
    op writes exactly the three track methods (never motion_enable), a blocked sweep is
    ok=False with the verdict and nothing written."""
    dry = json.loads(_MAINT_HOME_DRY.model_dump_json())
    assert (dry["op"], dry["path"], dry["ok"]) == ("home_rail", "monitor", True)
    assert dry["sdk_codes"] == {} and dry["after"] is None  # nothing written, no after-sample
    assert dry["rail_sweep"]["clear"] is True
    assert ArmMaintenanceResult.model_validate(dry) == _MAINT_HOME_DRY
    real = json.loads(_MAINT_HOME_OK.model_dump_json())
    assert list(real["sdk_codes"]) == [
        "set_linear_track_back_origin", "set_linear_track_enable", "set_linear_track_speed",
    ]
    assert "motion_enable" not in real["sdk_codes"]
    assert real["after"]["rail_homed"] is True and real["after"]["rail_enabled"] is True
    assert real["after"]["rail_pos_m"] == 0.0 and real["before"]["arm_id"] == "grip"
    assert real["rail_sweep"] == dry["rail_sweep"]
    assert ArmMaintenanceResult.model_validate(real) == _MAINT_HOME_OK
    refused = json.loads(_MAINT_HOME_REFUSED.model_dump_json())
    assert refused["ok"] is False and refused["sdk_codes"] == {}
    assert refused["rail_sweep"]["clear"] is False
    assert refused["rail_sweep"]["first_blocked_pair"][1] == "table"
    assert refused["detail"].startswith("rail sweep blocked at 0.125 m")
    assert ArmMaintenanceResult.model_validate(refused) == _MAINT_HOME_REFUSED
    # Additive: a 09b result without rail_sweep still parses (None); the other ops keep it None.
    legacy = json.loads(_MAINT_CLEAR_OK.model_dump_json())
    legacy.pop("rail_sweep")
    assert ArmMaintenanceResult.model_validate(legacy).rail_sweep is None
    assert _MAINT_APPLY_WARN.rail_sweep is None and _MAINT_RECOVER_SESSION.rail_sweep is None


def test_pre_position_plan_pinned():
    """phase-09d: the twin-planned pre-positioning motion, spelled exactly; only ``needed`` is
    required and the defaults describe "no motion needed"."""
    assert set(PrePositionPlan.model_fields) == {
        "needed", "source", "target_q", "waypoints", "duration_s", "checked_rail_positions",
        "clear", "detail",
    }
    required = {n for n, f in PrePositionPlan.model_fields.items() if f.is_required()}
    assert required == {"needed"}
    assert get_args(PrePositionPlan.model_fields["source"].annotation) == (
        "current", "keyframe", "home", "search",
    )
    none = PrePositionPlan(needed=False)
    assert none.source == "current" and none.target_q == [] and none.waypoints == 0
    assert none.duration_s == 0.0 and none.checked_rail_positions == 0
    assert none.clear is True and none.detail == ""
    # Mutable defaults are per-instance.
    other = PrePositionPlan(needed=False)
    none.target_q.append(1.0)
    assert other.target_q == []
    # A found plan: 7 target joints, the path validated for all 131 rail positions (5 mm over
    # 0.65 m - the carriage is unknown, so the check is position-agnostic).
    wire = json.loads(_PRE_POSITION_PLAN.model_dump_json())
    assert wire["needed"] is True and wire["source"] == "keyframe" and wire["clear"] is True
    assert len(wire["target_q"]) == 7 and wire["waypoints"] == 14 and wire["duration_s"] == 38.5
    assert wire["checked_rail_positions"] == 131 == round(0.65 / 0.005) + 1
    assert PrePositionPlan.model_validate(wire) == _PRE_POSITION_PLAN
    # No plan: needed but not clear -> the op is refused; detail carries the suggestion.
    refused = json.loads(_PRE_POSITION_REFUSED.model_dump_json())
    assert refused["needed"] is True and refused["clear"] is False and refused["waypoints"] == 0
    assert "Studio" in refused["detail"]
    assert PrePositionPlan.model_validate(refused) == _PRE_POSITION_REFUSED
    for bad in ("rrt", "planned", "KEYFRAME", ""):
        with pytest.raises(ValidationError):
            PrePositionPlan(needed=True, source=bad)
    with pytest.raises(ValidationError):
        PrePositionPlan()  # needed is required
    with pytest.raises(ValidationError):
        PrePositionPlan(needed=True, target_q="folded")
    with pytest.raises(ValidationError):
        PrePositionPlan(needed=True, waypoints=3.5)
    with pytest.raises(ValidationError):
        PrePositionPlan(needed=True, checked_rail_positions="all")
    # Rides the verdict (additive): a 09c verdict without it parses with None.
    assert _SWEEP_BLOCKED_PLANNED.pre_position == _PRE_POSITION_PLAN
    assert _SWEEP_CLEAR_09D.pre_position.needed is False and _SWEEP_CLEAR.pre_position is None
    wire = json.loads(_SWEEP_BLOCKED_PLANNED.model_dump_json())
    assert wire["clear"] is False and wire["pre_position"]["waypoints"] == 14
    assert RailSweepVerdict.model_validate(wire) == _SWEEP_BLOCKED_PLANNED
    wire.pop("pre_position")
    assert RailSweepVerdict.model_validate(wire).pre_position is None
    with pytest.raises(ValidationError):
        RailSweepVerdict(scene_id="mavis_v2", inflation_m=0.025, step_m=0.005, clear=False,
                         pre_position={"source": "keyframe"})  # needed required
    with pytest.raises(ValidationError):
        RailSweepVerdict(scene_id="mavis_v2", inflation_m=0.025, step_m=0.005, clear=False,
                         pre_position="plan")


def test_maintenance_status_and_async_result_wire():
    """phase-09d: ArmMaintenanceResult.status / job_id - a 202 "accepted" carries the job id
    and the plan, the job's final result reuses the id with status "done", a plan-less op is
    "refused"; every 09b/09c result is a synchronous "done" without a job."""
    assert get_args(MaintenanceStatus) == ("done", "accepted", "refused")
    assert ArmMaintenanceResult.model_fields["status"].annotation == MaintenanceStatus
    assert ArmMaintenanceResult.model_fields["status"].default == "done"
    assert ArmMaintenanceResult.model_fields["job_id"].default is None
    for res in (_MAINT_CLEAR_OK, _MAINT_APPLY_WARN, _MAINT_RECOVER_SESSION, _MAINT_HOME_DRY,
                _MAINT_HOME_OK, _MAINT_HOME_REFUSED):
        assert (res.status, res.job_id) == ("done", None)
    legacy = json.loads(_MAINT_HOME_OK.model_dump_json())
    assert legacy["status"] == "done" and legacy["job_id"] is None
    legacy.pop("status")
    legacy.pop("job_id")
    assert ArmMaintenanceResult.model_validate(legacy) == _MAINT_HOME_OK
    # Dry run, posture not clear but plannable: 200, nothing written, the plan on the verdict.
    dry = json.loads(_MAINT_HOME_DRY_PLANNED.model_dump_json())
    assert dry["status"] == "done" and dry["job_id"] is None and dry["sdk_codes"] == {}
    assert dry["rail_sweep"]["clear"] is False
    assert dry["rail_sweep"]["pre_position"]["needed"] is True
    assert dry["rail_sweep"]["pre_position"]["waypoints"] == 14
    assert ArmMaintenanceResult.model_validate(dry) == _MAINT_HOME_DRY_PLANNED
    # 202 accepted: the job started, nothing written yet, job_id set.
    accepted = json.loads(_MAINT_HOME_ACCEPTED.model_dump_json())
    assert (accepted["status"], accepted["job_id"], accepted["ok"]) == ("accepted", _JOB_ID, True)
    assert accepted["sdk_codes"] == {} and accepted["after"] is None
    assert accepted["rail_sweep"]["pre_position"]["clear"] is True
    assert ArmMaintenanceResult.model_validate(accepted) == _MAINT_HOME_ACCEPTED
    # The job's final result (GET .../maintenance/last): "done" with the SAME job_id, the
    # exact home_rail write set, never motion_enable, the after-sample homed at 0.000 m.
    done = json.loads(_MAINT_HOME_JOB_DONE.model_dump_json())
    assert (done["status"], done["job_id"], done["ok"]) == ("done", _JOB_ID, True)
    assert list(done["sdk_codes"]) == [
        "set_linear_track_back_origin", "set_linear_track_enable", "set_linear_track_speed",
    ]
    assert "motion_enable" not in done["sdk_codes"]
    assert done["after"]["rail_homed"] is True and done["after"]["rail_pos_m"] == 0.0
    assert ArmMaintenanceResult.model_validate(done) == _MAINT_HOME_JOB_DONE
    failed = json.loads(_MAINT_HOME_JOB_FAILED.model_dump_json())
    assert (failed["status"], failed["job_id"], failed["ok"]) == ("done", _JOB_ID, False)
    assert failed["detail"].startswith("positioning aborted") and failed["sdk_codes"] == {}
    assert ArmMaintenanceResult.model_validate(failed) == _MAINT_HOME_JOB_FAILED
    # Refused: no rail-safe plan - nothing ran, no job, the suggestion in detail.
    refused = json.loads(_MAINT_HOME_REFUSED_09D.model_dump_json())
    assert (refused["status"], refused["job_id"], refused["ok"]) == ("refused", None, False)
    assert refused["sdk_codes"] == {}
    assert refused["rail_sweep"]["pre_position"]["clear"] is False
    assert "Studio" in refused["detail"]
    assert ArmMaintenanceResult.model_validate(refused) == _MAINT_HOME_REFUSED_09D
    for bad in ("pending", "running", "queued", "ok", "DONE", ""):
        with pytest.raises(ValidationError):
            ArmMaintenanceResult(arm_id="grip", op="home_rail", path="monitor", ok=True,
                                 status=bad)
    with pytest.raises(ValidationError):
        ArmMaintenanceResult(arm_id="grip", op="home_rail", path="monitor", ok=True, job_id=17)


def test_maintenance_progress_pinned_and_monitor_row_additive():
    """phase-09d: MaintenanceProgress rides ArmMonitorTelemetry.maintenance while a
    RailHomingJob runs (None otherwise); phases spelled exactly, in execution order."""
    assert get_args(MaintenancePhase) == (
        "queued", "sweeping", "planning", "connecting", "positioning",
        "homing", "verifying", "done", "failed",
    )
    assert set(MaintenanceProgress.model_fields) == {
        "op", "job_id", "phase", "detail", "progress", "started_at",
    }
    required = {n for n, f in MaintenanceProgress.model_fields.items() if f.is_required()}
    assert required == {"op", "job_id", "phase"}
    assert MaintenanceProgress.model_fields["op"].annotation == ArmMaintenanceOp
    assert MaintenanceProgress.model_fields["phase"].annotation == MaintenancePhase
    assert ArmMonitorTelemetry.model_fields["maintenance"].default is None
    queued = MaintenanceProgress(op="home_rail", job_id="j1", phase="queued")
    assert queued.detail == "" and queued.progress == 0.0 and queued.started_at is None
    for bad in ("sweep", "moving", "running", "POSITIONING", ""):
        with pytest.raises(ValidationError):
            MaintenanceProgress(op="home_rail", job_id="j1", phase=bad)
    with pytest.raises(ValidationError):
        MaintenanceProgress(op="home_rail", phase="queued")  # job_id required
    with pytest.raises(ValidationError):
        MaintenanceProgress(op="homing", job_id="j1", phase="queued")  # op vocabulary shared
    with pytest.raises(ValidationError):
        MaintenanceProgress(op="home_rail", job_id="j1", phase="homing", progress="half")
    # The row: None by default and on every pre-09d producer; the job sets it together with
    # maintenance_busy while the monitor is paused (the job's driver owns the box).
    assert ArmMonitorTelemetry(arm_id="grip").maintenance is None
    assert _MON_GRIP_RUNNING.maintenance is None and _MON_VIEW_BACKSTOPS.maintenance is None
    legacy = json.loads(_MON_GRIP_RUNNING.model_dump_json())
    assert legacy["maintenance"] is None
    legacy.pop("maintenance")
    assert ArmMonitorTelemetry.model_validate(legacy) == _MON_GRIP_RUNNING
    wire = json.loads(_MON_GRIP_HOMING_JOB.model_dump_json())
    assert wire["status"] == "paused" and wire["maintenance_busy"] is True
    assert wire["maintenance"] == {
        "op": "home_rail", "job_id": _JOB_ID, "phase": "positioning",
        "detail": "waypoint 9 / 14", "progress": 0.55, "started_at": 1_757_067_300.0,
    }
    assert ArmMonitorTelemetry.model_validate(wire) == _MON_GRIP_HOMING_JOB
    with pytest.raises(ValidationError):
        ArmMonitorTelemetry(arm_id="grip", maintenance={"phase": "queued"})  # op/job_id required
    with pytest.raises(ValidationError):
        ArmMonitorTelemetry(arm_id="grip", maintenance="positioning")
    # job_id ties the telemetry to the 202 response and to the final result.
    assert _PROGRESS_POSITIONING.job_id == _MAINT_HOME_ACCEPTED.job_id
    assert _PROGRESS_DONE.job_id == _MAINT_HOME_JOB_DONE.job_id
    assert _PROGRESS_DONE.progress == 1.0 and _PROGRESS_FAILED.phase == "failed"
    # Through the block and the 25 Hz frame: the frozen arm carries no progress.
    frame = TelemetryMsg(
        seq=11, ts=20.0, epoch="ep0", active_arm=None, controller_connected=False,
        arms=[], collision=CollisionReport.ok(), clearances=[],
        episode=None, dagger=None, inference=None, hardware_monitor=_HW_MONITOR_HOMING_JOB,
    )
    wire = json.loads(frame.model_dump_json())
    rows = wire["hardware_monitor"]["arms"]
    assert wire["hardware_monitor"]["paused"] is True
    assert [r["maintenance"] and r["maintenance"]["phase"] for r in rows] == ["positioning", None]
    assert [r["maintenance_busy"] for r in rows] == [True, False]
    assert TelemetryMsg.model_validate(wire) == frame


def test_arm_bringup_telemetry_pinned_and_session_bringup_additive():
    """phase-09c (D5): per-arm bring-up rows ride SessionTelemetry.bringup while the
    hardware session is in the bringup state; additive (None for sim / once running)."""
    assert set(ArmBringupTelemetry.model_fields) == {"arm_id", "step", "status", "detail"}
    required = {n for n, f in ArmBringupTelemetry.model_fields.items() if f.is_required()}
    assert required == {"arm_id", "step", "status"}
    assert get_args(ArmBringupTelemetry.model_fields["status"].annotation) == (
        "pending", "ok", "warning", "error",
    )
    assert ArmBringupTelemetry(arm_id="grip", step="rail", status="ok").detail == ""
    for bad in ("unhomed", "running", "OK", ""):
        with pytest.raises(ValidationError):
            ArmBringupTelemetry(arm_id="grip", step="rail", status=bad)
    with pytest.raises(ValidationError):
        ArmBringupTelemetry(arm_id="grip", status="ok")  # step required
    assert set(SessionTelemetry.model_fields) == {
        "state", "start_from_progress", "plan_status", "trainer_alive", "bringup",
    }
    assert SessionTelemetry(state="running").bringup is None
    block = SessionTelemetry(state="bringup", bringup=_BRINGUP_ROWS)
    wire = json.loads(block.model_dump_json())
    assert wire["state"] == "bringup" and len(wire["bringup"]) == 4
    assert wire["bringup"][3] == {
        "arm_id": "view", "step": "frozen", "status": "warning",
        "detail": "Perception Arm frozen at last sample",
    }
    assert SessionTelemetry.model_validate(wire) == block
    legacy = dict(wire)
    legacy.pop("bringup")
    assert SessionTelemetry.model_validate(legacy).bringup is None
    with pytest.raises(ValidationError):
        SessionTelemetry(state="bringup", bringup=[{"arm_id": "grip"}])
    with pytest.raises(ValidationError):
        SessionTelemetry(state="bringup", bringup="connecting")
    # Nested in a frame: the rows survive the 25 Hz round trip.
    frame = TelemetryMsg(
        seq=9, ts=18.0, epoch="ep0", active_arm="grip", controller_connected=True,
        arms=[], collision=CollisionReport.ok(), clearances=[],
        episode=None, dagger=None, inference=None, session=block,
    )
    wire = json.loads(frame.model_dump_json())
    assert [r["step"] for r in wire["session"]["bringup"]] == [
        "network", "connect", "rail", "frozen",
    ]
    assert TelemetryMsg.model_validate(wire) == frame


def test_session_info_kind_and_speed_scale_additive():
    """phase-09c: SessionInfo echoes the workcell kind and the speed scale; both default so a
    pre-09c producer (sim only, unscaled) still parses."""
    assert set(SessionInfo.model_fields) == {
        "session_id", "epoch", "mode", "arms", "streams", "state", "kind", "speed_scale",
    }
    legacy = {
        "session_id": "s0", "epoch": "ep0", "mode": "teleop", "arms": ["arm0"],
        "streams": ["cam0", "sim"], "state": "RUNNING",
    }
    info = SessionInfo.model_validate(legacy)
    assert (info.kind, info.speed_scale) == ("sim", 1.0)
    hw = SessionInfo.model_validate({**legacy, "kind": "hardware", "speed_scale": 0.1})
    assert (hw.kind, hw.speed_scale) == ("hardware", 0.1)
    wire = json.loads(hw.model_dump_json())
    assert wire["kind"] == "hardware" and wire["speed_scale"] == 0.1
    assert SessionInfo.model_validate(wire) == hw
    for bad in ("twin", "real", "SIM", ""):
        with pytest.raises(ValidationError):
            SessionInfo.model_validate({**legacy, "kind": bad})
    for bad in (0, 1.5, -0.1, "slow"):
        with pytest.raises(ValidationError):
            SessionInfo.model_validate({**legacy, "speed_scale": bad})


def test_tracker_calibration_model_fields_pinned():
    """phase-10 wire shapes, spelled exactly (core is the spelling authority)."""
    assert set(TrackerCalibrationStatus.model_fields) == {
        "kind", "phase", "detail", "started_at", "elapsed_s",
        "scenes", "lighthouses", "stations_visible", "controller_still", "validation",
        "installed_path", "backup_path",
        "yaw_points", "next_point", "fitted_yaw_deg", "fit_residual_deg", "fit_checks",
        "applied_yaw_deg",
        "yaw_valid", "yaw_calibrated_at", "base_station_installed_at",
    }
    assert set(TrackerCalibrationCommand.model_fields) == {"kind", "op", "point"}
    assert set(LighthouseStatus.model_fields) == {
        "index", "channel", "serial", "pose", "scenes", "reference",
    }
    assert set(CalibrationValidation.model_fields) == {
        "samples", "std_mm", "max_step_mm", "threshold_std_mm", "threshold_step_mm", "passed",
    }
    assert set(YawGesturePoint.model_fields) == {"label", "pose"}


def test_action_name_literal_rejects_unknown():
    with pytest.raises(ValidationError):
        ActionMsg(name="warp_drive")
    with pytest.raises(ValidationError):
        AckMsg(name="set_initial_profile", ok=True)  # drift ledger #1: renamed


@pytest.mark.parametrize(
    "args",
    [
        {"arm_id": "arm0", "positions": [0.0] * 8, "mode": "jog"},
        {"arm_id": "arm1", "positions": [0.1, -0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "mode": "goto"},
    ],
)
def test_joint_target_args_accept(args):
    parsed = JointTargetArgs.model_validate(args)
    assert parsed.mode in ("jog", "goto")
    assert all(isinstance(v, float) for v in parsed.positions)


@pytest.mark.parametrize(
    "args",
    [
        {"arm_id": "arm0", "positions": [0.0] * 8, "mode": "teleport"},  # bad mode literal
        {"arm_id": "arm0", "positions": [0.0] * 8},  # missing mode
        {"arm_id": "arm0", "positions": "not-a-list", "mode": "jog"},  # bad positions
        {"arm_id": "arm0", "positions": [0.0, "x"], "mode": "jog"},  # non-numeric entry
        {"positions": [0.0] * 8, "mode": "jog"},  # missing arm_id
    ],
)
def test_joint_target_args_reject(args):
    with pytest.raises(ValidationError):
        JointTargetArgs.model_validate(args)


def test_validate_action_args_per_name():
    jt = validate_action_args(
        ActionMsg(name="joint_target",
                  args={"arm_id": "arm0", "positions": [0.0] * 8, "mode": "jog"})
    )
    assert isinstance(jt, JointTargetArgs)
    sp = validate_action_args(ActionMsg(name="save_profile", args={"name": "home"}))
    assert isinstance(sp, SaveProfileArgs) and sp.notes == ""
    sic = validate_action_args(ActionMsg(name="set_initial_condition", args={}))
    assert isinstance(sic, SetInitialConditionArgs) and sic.profile_id is None
    ts = validate_action_args(ActionMsg(name="tracker_settings", args={"pos_scale": 2.0}))
    assert isinstance(ts, TrackerSettingsArgs)
    assert (ts.yaw_deg, ts.pos_scale, ts.follow_rotation) == (None, 2.0, None)
    assert (ts.filter_enabled, ts.filter_min_cutoff_hz, ts.filter_beta) == (None, None, None)
    tf = validate_action_args(
        ActionMsg(name="tracker_settings",
                  args={"filter_enabled": False, "filter_min_cutoff_hz": 0.8, "filter_beta": 0.1})
    )
    assert isinstance(tf, TrackerSettingsArgs)
    assert (tf.filter_enabled, tf.filter_min_cutoff_hz, tf.filter_beta) == (False, 0.8, 0.1)
    assert (tf.yaw_deg, tf.pos_scale, tf.follow_rotation) == (None, None, None)

    # Actions without an args model require empty args and return None.
    for name in ("switch_arm", "switch_arm_prev", "takeover_toggle", "episode_new",
                 "episode_save", "episode_discard"):
        assert validate_action_args(ActionMsg(name=name)) is None
    with pytest.raises(ValueError):
        validate_action_args(ActionMsg(name="switch_arm", args={"index": 1}))
    with pytest.raises(ValueError):
        validate_action_args(ActionMsg(name="switch_arm_prev", args={"index": 1}))
    with pytest.raises(ValidationError):
        validate_action_args(ActionMsg(name="joint_target", args={"arm_id": "arm0"}))
    with pytest.raises(ValidationError):
        validate_action_args(ActionMsg(name="tracker_settings", args={"pos_scale": 5.0}))
    with pytest.raises(ValidationError):
        validate_action_args(
            ActionMsg(name="tracker_settings", args={"filter_min_cutoff_hz": 0.0})
        )


@pytest.mark.parametrize(
    "args",
    [
        {},  # all fields omitted = unchanged
        {"yaw_deg": 90.0},
        {"yaw_deg": -180.0, "pos_scale": 0.1, "follow_rotation": False},  # lower bound
        {"pos_scale": 3.0},  # upper bound
        {"pos_scale": 1, "follow_rotation": True},  # int coerces to float
        {"filter_enabled": False},  # bypass the pose filter
        {"filter_min_cutoff_hz": 0.05, "filter_beta": 0.0},  # filter lower bounds
        {"filter_min_cutoff_hz": 50.0, "filter_beta": 5.0},  # filter upper bounds
        {"filter_min_cutoff_hz": 1, "filter_beta": 1},  # int coerces to float
        {"yaw_deg": 45.0, "filter_enabled": True, "filter_min_cutoff_hz": 1.0,
         "filter_beta": 0.05},  # mixed alignment + filter
    ],
)
def test_tracker_settings_args_accept(args):
    parsed = TrackerSettingsArgs.model_validate(args)
    assert parsed.model_dump(exclude_none=True) == {
        k: float(v) if isinstance(v, int) and not isinstance(v, bool) else v
        for k, v in args.items()
    }


@pytest.mark.parametrize(
    "args",
    [
        {"pos_scale": 5.0},  # > 3.0
        {"pos_scale": 0.05},  # < 0.1
        {"pos_scale": 0.0},
        {"pos_scale": "fast"},
        {"yaw_deg": "north"},
        {"follow_rotation": "maybe"},  # not a recognised bool spelling
        {"filter_min_cutoff_hz": 0.0},  # < 0.05 (a zero cutoff freezes the filter)
        {"filter_min_cutoff_hz": 0.01},
        {"filter_min_cutoff_hz": 100.0},  # > 50
        {"filter_min_cutoff_hz": "slow"},
        {"filter_beta": -1},  # < 0
        {"filter_beta": -0.001},
        {"filter_beta": 5.5},  # > 5
        {"filter_enabled": "maybe"},
    ],
)
def test_tracker_settings_args_reject(args):
    with pytest.raises(ValidationError):
        TrackerSettingsArgs.model_validate(args)


def _spec(**overrides) -> SessionSpec:
    base = dict(
        mode="teleop",
        kind="sim",
        arms=["arm0", "arm1"],
        frames={},
        sim_scene="scene0",
    )
    base.update(overrides)
    return SessionSpec.model_validate(base)


@pytest.mark.parametrize("value", ["keep_current", "profile:abc123", "profile:a_B-9"])
def test_start_from_accept(value):
    assert _spec(start_from=value).start_from == value


@pytest.mark.parametrize("value", ["profile:", "PROFILE:x", "keep", "", "profile:a b"])
def test_start_from_reject(value):
    with pytest.raises(ValidationError):
        _spec(start_from=value)


def test_session_spec_frames_keys_subset_of_arms():
    ok = _spec(frames={"arm0": "world", "arm1": "camera:cam0"})
    assert set(ok.frames) <= set(ok.arms)
    with pytest.raises(ValidationError):
        _spec(frames={"arm9": "world"})


def test_session_spec_frames_values_parse_and_reject_ee():
    with pytest.raises(ValidationError):
        _spec(frames={"arm0": "ee:arm0"})
    with pytest.raises(ValidationError):
        _spec(frames={"arm0": "not_a_frame"})
    with pytest.raises(ValidationError):
        _spec(frames={"arm0": "arm_base:"})


def test_session_spec_collect_dagger_require_task():
    with pytest.raises(ValidationError):
        _spec(mode="collect")
    with pytest.raises(ValidationError):
        _spec(mode="dagger", task=None)
    assert _spec(mode="collect", task="stack").task == "stack"
    assert _spec(mode="dagger", task="sort").mode == "dagger"
    assert _spec(mode="teleop").task is None  # teleop/inference: task optional
    assert _spec(mode="inference").task is None


def test_session_spec_speed_scale_bounds():
    """phase-09c (D2): speed_scale in (0, 1]; default 1.0 (unscaled); 0 and 1.5 are 422."""
    assert SessionSpec.model_fields["speed_scale"].default == 1.0
    assert _spec().speed_scale == 1.0
    for ok in (1, 1.0, 0.3, 0.1, 1e-6):
        assert _spec(speed_scale=ok).speed_scale == float(ok)
    assert isinstance(_spec(speed_scale=1).speed_scale, float)  # int coerces
    for bad in (0, 0.0, 1.5, -0.1, 2, "fast", None):
        with pytest.raises(ValidationError):
            _spec(speed_scale=bad)
    # Hardware sessions carry the same field (the tab default is 10 %); the wire form is flat.
    hw = _spec(kind="hardware", sim_scene=None, digital_twin_scene="mavis_v2",
               arms=["grip"], speed_scale=0.1)
    assert hw.speed_scale == 0.1
    assert json.loads(hw.model_dump_json())["speed_scale"] == 0.1
    # Additive: a pre-09c body without the field parses at 1.0.
    body = json.loads(_spec().model_dump_json())
    body.pop("speed_scale")
    assert SessionSpec.model_validate(body).speed_scale == 1.0
