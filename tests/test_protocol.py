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
    """Pre-phase-10 producers (no ``charging``/``calibration``) still parse."""
    assert set(TrackerTelemetry.model_fields) == {
        "backend", "status", "detail", "object_name", "seq", "rate_hz", "age_s",
        "pose_raw", "pose_world", "pose_filtered", "clutch", "engaged_arm", "anchor_tcp",
        "target_tcp", "settings", "controller", "device_held", "device_action",
        "charging", "calibration",
    }
    legacy = _TRACKER_ENGAGED.model_dump(mode="json")
    legacy.pop("charging")
    legacy.pop("calibration")
    parsed = TrackerTelemetry.model_validate(legacy)
    assert parsed.charging is None and parsed.calibration is None
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
    assert list(TelemetryMsg.model_fields)[-3:] == ["session", "tracker", "microphone"]
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
