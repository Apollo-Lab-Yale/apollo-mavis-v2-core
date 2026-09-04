"""Tracker calibration wire models (design doc 01-core §12; phase-10 §2)."""

from __future__ import annotations

import json
from typing import get_args

import pytest
from pydantic import ValidationError

from apollo_mavis_v2_core.protocol import tracker as tracker_mod
from apollo_mavis_v2_core.protocol.export_schemas import EXPORTED_MODELS
from apollo_mavis_v2_core.protocol.telemetry import PoseMsg, TrackerTelemetry
from apollo_mavis_v2_core.protocol.tracker import (
    CalibrationKind,
    CalibrationOp,
    CalibrationPhase,
    CalibrationValidation,
    LighthouseStatus,
    TrackerCalibrationCommand,
    TrackerCalibrationStatus,
    YawGesturePoint,
    YawPointLabel,
)

_POSE = PoseMsg(position=(0.1, -0.2, 1.3), orientation=(1.0, 0.0, 0.0, 0.0))


# --- literal vocabularies (phase-10 §2, spelled exactly) -----------------------


def test_literal_vocabularies_spelled_exactly():
    assert get_args(CalibrationKind) == ("none", "base_station", "yaw")
    assert get_args(CalibrationPhase) == (
        "idle", "starting", "capturing", "validating", "fitting",
        "installing", "done", "failed", "aborted",
    )
    assert get_args(CalibrationOp) == ("start", "capture", "validate", "install", "apply", "abort")
    assert get_args(YawPointLabel) == ("start", "left", "forward", "right", "back", "up", "down")


def test_module_exports_and_schema_registration():
    assert set(tracker_mod.__all__) == {
        "CalibrationKind", "CalibrationPhase", "CalibrationOp", "YawPointLabel",
        "LighthouseStatus", "CalibrationValidation", "YawGesturePoint",
        "TrackerCalibrationStatus", "TrackerCalibrationCommand",
    }
    # Package re-exports every name; REST models are exported top-level (§14).
    import apollo_mavis_v2_core.protocol as protocol

    for name in tracker_mod.__all__:
        assert getattr(protocol, name) is getattr(tracker_mod, name)
        assert name in protocol.__all__
    assert EXPORTED_MODELS["TrackerCalibrationStatus"] is TrackerCalibrationStatus
    assert EXPORTED_MODELS["TrackerCalibrationCommand"] is TrackerCalibrationCommand
    for name in ("LighthouseStatus", "CalibrationValidation", "YawGesturePoint"):
        assert name not in EXPORTED_MODELS  # ride $defs only
    # PoseMsg is shared with telemetry, not redefined.
    assert LighthouseStatus.model_fields["pose"].annotation == (PoseMsg | None)
    assert YawGesturePoint.model_fields["pose"].annotation is PoseMsg


# --- TrackerCalibrationStatus ----------------------------------------------------


def test_status_defaults_are_idle_snapshot():
    """``{}`` is the idle snapshot the runtime serves before any calibration."""
    s = TrackerCalibrationStatus.model_validate({})
    assert (s.kind, s.phase, s.detail) == ("none", "idle", "")
    assert s.started_at is None and s.elapsed_s is None
    # base-station group
    assert s.scenes == 0 and s.lighthouses == [] and s.stations_visible == 0
    assert s.controller_still is None and s.validation is None
    assert s.installed_path is None and s.backup_path is None
    # yaw group
    assert s.yaw_points == [] and s.next_point is None
    assert s.fitted_yaw_deg is None and s.fit_residual_deg is None
    assert s.fit_checks == [] and s.applied_yaw_deg is None
    # persisted state: yaw trusted until a base-station install invalidates it
    assert s.yaw_valid is True
    assert s.yaw_calibrated_at is None and s.base_station_installed_at is None
    assert s == TrackerCalibrationStatus()


def test_status_list_defaults_are_not_shared():
    a = TrackerCalibrationStatus()
    b = TrackerCalibrationStatus()
    a.lighthouses.append(LighthouseStatus(index=0))
    a.yaw_points.append(YawGesturePoint(label="start", pose=_POSE))
    a.fit_checks.append("residual too large")
    assert b.lighthouses == [] and b.yaw_points == [] and b.fit_checks == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"kind": "warp"},
        {"kind": None},
        {"phase": "running"},
        {"phase": "DONE"},
        {"next_point": "middle"},
        {"scenes": "many"},
        {"stations_visible": 2.5},
        {"controller_still": "maybe"},
        {"yaw_valid": "yes please"},
        {"lighthouses": [{"channel": 3}]},  # index required
        {"yaw_points": [{"label": "left"}]},  # pose required
        {"yaw_points": [{"label": "diagonal", "pose": _POSE.model_dump()}]},
        {"fit_checks": "leg too short"},  # must be a list
        {"validation": {"std_mm": [1.0, 2.0]}},  # 3-tuple
        {"validation": {"passed": "nope"}},
    ],
)
def test_status_rejects_bad_literals_and_shapes(overrides):
    with pytest.raises(ValidationError):
        TrackerCalibrationStatus.model_validate(overrides)


def test_status_accepts_every_kind_phase_and_point():
    for kind in get_args(CalibrationKind):
        for phase in get_args(CalibrationPhase):
            s = TrackerCalibrationStatus(kind=kind, phase=phase)
            assert (s.kind, s.phase) == (kind, phase)
    for label in get_args(YawPointLabel):
        assert TrackerCalibrationStatus(next_point=label).next_point == label
        assert YawGesturePoint(label=label, pose=_POSE).label == label


def test_status_round_trips_through_json_with_nested_models():
    s = TrackerCalibrationStatus(
        kind="base_station",
        phase="done",
        detail="validation passed — install",
        started_at=1.0,
        elapsed_s=95.5,
        scenes=6,
        lighthouses=[
            LighthouseStatus(index=0, channel=1, serial="LHB-A", pose=_POSE, scenes=6,
                             reference=True),
            LighthouseStatus(index=1, channel=7, serial="LHB-B", pose=_POSE, scenes=5),
        ],
        stations_visible=2,
        controller_still=True,
        validation=CalibrationValidation(
            samples=2500, std_mm=(0.1, 0.1, 0.08), max_step_mm=0.1, passed=True
        ),
        yaw_valid=False,
    )
    wire = json.loads(s.model_dump_json())
    assert wire["validation"]["std_mm"] == [0.1, 0.1, 0.08]  # tuple -> JSON array
    assert wire["lighthouses"][0]["pose"]["position"] == [0.1, -0.2, 1.3]
    assert wire["yaw_points"] == [] and wire["fit_checks"] == []
    assert TrackerCalibrationStatus.model_validate_json(s.model_dump_json()) == s
    # Unknown keys from a newer producer are ignored (additive protocol).
    wire["future_field"] = 1
    assert TrackerCalibrationStatus.model_validate(wire) == s


# --- CalibrationValidation ------------------------------------------------------


def test_validation_defaults_carry_acceptance_thresholds():
    """Thresholds default to the 2026-09-03 acceptance figures (std 5 mm, step 20 mm)."""
    v = CalibrationValidation()
    assert v.samples == 0 and v.std_mm == (0.0, 0.0, 0.0) and v.max_step_mm == 0.0
    assert v.threshold_std_mm == 5.0 and v.threshold_step_mm == 20.0
    assert v.passed is False


def test_validation_std_mm_is_a_fixed_3_tuple():
    v = CalibrationValidation(std_mm=[61, 62.0, 53])  # list + ints coerce
    assert v.std_mm == (61.0, 62.0, 53.0) and isinstance(v.std_mm, tuple)
    assert all(isinstance(x, float) for x in v.std_mm)
    for bad in ([1.0, 2.0], [1.0, 2.0, 3.0, 4.0], [], "1,2,3", [1.0, "x", 3.0], None):
        with pytest.raises(ValidationError):
            CalibrationValidation(std_mm=bad)
    # Round trip keeps the tuple type.
    again = CalibrationValidation.model_validate_json(v.model_dump_json())
    assert again == v and isinstance(again.std_mm, tuple)


# --- LighthouseStatus / YawGesturePoint -----------------------------------------


def test_lighthouse_status_requires_index_only():
    lh = LighthouseStatus(index=2)
    assert lh.channel is None and lh.serial is None and lh.pose is None
    assert lh.scenes == 0 and lh.reference is False
    with pytest.raises(ValidationError):
        LighthouseStatus()
    with pytest.raises(ValidationError):
        LighthouseStatus(index="two")
    with pytest.raises(ValidationError):
        LighthouseStatus(index=0, pose={"position": [0, 0, 0]})  # orientation missing


def test_yaw_gesture_point_requires_label_and_pose():
    p = YawGesturePoint(label="up", pose=_POSE)
    assert json.loads(p.model_dump_json()) == {
        "label": "up",
        "pose": {"position": [0.1, -0.2, 1.3], "orientation": [1.0, 0.0, 0.0, 0.0]},
    }
    with pytest.raises(ValidationError):
        YawGesturePoint(label="up")
    with pytest.raises(ValidationError):
        YawGesturePoint(pose=_POSE)
    with pytest.raises(ValidationError):
        YawGesturePoint(label="none", pose=_POSE)


# --- TrackerCalibrationCommand --------------------------------------------------


@pytest.mark.parametrize("kind", ["base_station", "yaw"])
@pytest.mark.parametrize("op", get_args(CalibrationOp))
def test_command_accepts_every_kind_op_pair(kind, op):
    cmd = TrackerCalibrationCommand.model_validate({"kind": kind, "op": op})
    assert (cmd.kind, cmd.op, cmd.point) == (kind, op, None)
    assert cmd.model_dump() == {"kind": kind, "op": op, "point": None}


@pytest.mark.parametrize("point", get_args(YawPointLabel))
def test_command_capture_accepts_every_point_label(point):
    cmd = TrackerCalibrationCommand(kind="yaw", op="capture", point=point)
    assert cmd.point == point
    assert TrackerCalibrationCommand.model_validate_json(cmd.model_dump_json()) == cmd


@pytest.mark.parametrize(
    "body",
    [
        {},  # kind + op required
        {"kind": "yaw"},
        {"op": "start"},
        {"kind": "none", "op": "start"},  # "none" is a status kind, not a command kind
        {"kind": "basestation", "op": "start"},
        {"kind": "yaw", "op": "restart"},
        {"kind": "yaw", "op": "Start"},
        {"kind": "yaw", "op": "capture", "point": "center"},
        {"kind": "yaw", "op": "capture", "point": 3},
        {"kind": None, "op": "start"},
    ],
)
def test_command_rejects_bad_bodies(body):
    with pytest.raises(ValidationError):
        TrackerCalibrationCommand.model_validate(body)


def test_command_is_a_flat_rest_body():
    """UI posts exactly this JSON; `point` may be omitted (None = next_point)."""
    cmd = TrackerCalibrationCommand.model_validate_json('{"kind":"base_station","op":"install"}')
    assert cmd == TrackerCalibrationCommand(kind="base_station", op="install")
    assert json.loads(cmd.model_dump_json(exclude_none=True)) == {
        "kind": "base_station", "op": "install",
    }


# --- TrackerTelemetry.calibration -----------------------------------------------


def test_tracker_telemetry_calibration_is_additive_and_defaults_none():
    base = {
        "backend": "libsurvive", "status": "tracking",
        "settings": {"yaw_deg": 102.1, "pos_scale": 1.0, "follow_rotation": True},
    }
    assert TrackerTelemetry.model_validate(base).calibration is None
    with_status = TrackerTelemetry.model_validate(
        {**base, "calibration": {"kind": "yaw", "phase": "capturing", "next_point": "left"}}
    )
    assert with_status.calibration == TrackerCalibrationStatus(
        kind="yaw", phase="capturing", next_point="left"
    )
    assert TrackerTelemetry.model_fields["calibration"].default is None
