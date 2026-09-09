"""GELLO Manipulation wire models (design doc 01-core §10-§12, §14, §20; 16-gello §7/§8;
phase-15, 2026-09-09).

Everything phase-15 added to core: the ``gello`` Mode + ``GelloSessionConfig`` and the
``SessionSpec`` cross-field rules, the ``SessionInfo`` echo, the two argless Cockpit actions,
``CommandSource.GELLO``, the ``TelemetryMsg.gello`` block, the session-less REST models of
``/api/gello*``, ``SessionAnnounce.external_arms`` and the exported schemas.
"""

from __future__ import annotations

import json
import sys
from typing import get_args

import pytest
from pydantic import ValidationError

from apollo_mavis_v2_core.protocol import KEYMAP, ActionName
from apollo_mavis_v2_core.protocol import external as ext
from apollo_mavis_v2_core.protocol import gello as gello_mod
from apollo_mavis_v2_core.protocol.control import (
    AckMsg,
    ActionMsg,
    parse_client_msg,
    validate_action_args,
)
from apollo_mavis_v2_core.protocol.export_schemas import EXPORTED_MODELS, export
from apollo_mavis_v2_core.protocol.gello import (
    GelloBackend,
    GelloCalibrateOp,
    GelloCalibrateRequest,
    GelloCalibrateResult,
    GelloDeviceStatus,
    GelloDeviceTelemetry,
    GelloInfo,
    GelloPairInfo,
    GelloPreviewRequest,
    GelloPreviewResult,
    GelloPreviewStatus,
    GelloState,
    GelloViewpointMode,
)
from apollo_mavis_v2_core.protocol.session import (
    GelloSessionConfig,
    Mode,
    SessionInfo,
    SessionSpec,
)
from apollo_mavis_v2_core.protocol.telemetry import (
    GelloTelemetry,
    GelloViewpointTelemetry,
    TelemetryMsg,
)
from apollo_mavis_v2_core.schemas.safety import CollisionEvent, CollisionReport, CommandSource

_VIEW_HOLD = [2.646, -1.598, 0.018, 1.637, 0.25, 2.007, 0.029]  # 16-gello §0 item 3
_SIGNS = [1, 1, 1, 1, 1, 1, 1]
_LEADER_Q = [3.1416, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # the mavis_v2 keyframe (fake backend default)

# The exact field order of 16-gello §8.3 — the device half first, then the session half.
_DEVICE_FIELDS = [
    "backend", "status", "detail", "port", "baud", "seq", "rate_hz", "age_s", "q_raw", "q",
    "gripper_frac", "calibrated", "joint_offsets_rad", "joint_signs",
]
_SESSION_FIELDS = [
    "state", "state_detail", "lag_rad", "max_lag_rad", "engaged_arm", "viewpoint", "paused_latched",
]


def _spec(**overrides) -> SessionSpec:
    base = dict(
        mode="gello", kind="sim", arms=["view", "grip"],
        frames={"grip": "arm_base:grip", "view": "arm_base:view"},
        sim_scene="mavis_v2_kitchen", gello={},
    )
    base.update(overrides)
    return SessionSpec.model_validate(base)


def _device(**overrides) -> dict:
    base = dict(
        backend="fake", status="connected", port="/dev/ttyUSB0", baud=1_000_000, seq=4200,
        rate_hz=99.8, age_s=0.004, q_raw=[3.1416, 1.5708, 0.0, -1.5708, 0.0, 0.0, 0.0],
        q=_LEADER_Q, gripper_frac=0.82, calibrated=True,
        joint_offsets_rad=[0.0, 1.5708, 0.0, -1.5708, 0.0, 0.0, 0.0], joint_signs=_SIGNS,
    )
    base.update(overrides)
    return base


def _frame(**overrides) -> TelemetryMsg:
    base = dict(
        seq=1, ts=1.0, epoch="ep0", active_arm="grip", controller_connected=True, arms=[],
        collision=CollisionReport.ok(), clearances=[], episode=None, dagger=None,
        inference=None,
    )
    base.update(overrides)
    return TelemetryMsg(**base)


# -- literals ------------------------------------------------------------------------------------
def test_gello_literals_are_the_16_gello_ones():
    """16-gello §8: every literal spelled once in ``protocol.gello`` and shared by the
    session, telemetry and REST models; ``gello`` is the LAST Mode (the UI's MODES pins
    the card order)."""
    assert get_args(Mode) == ("teleop", "collect", "dagger", "inference", "gello")
    assert get_args(GelloBackend) == ("dynamixel", "fake", "none")
    assert get_args(GelloDeviceStatus) == ("no_backend", "starting", "connected", "stale", "error")
    assert get_args(GelloState) == ("no_leader", "out_of_sync", "tracking", "paused", "motion")
    assert get_args(GelloViewpointMode) == ("auto", "external", "hold")
    assert get_args(GelloCalibrateOp) == ("match_arm", "gripper_open", "gripper_closed", "clear")
    assert get_args(GelloPreviewStatus) == (
        "clear", "collision", "joint_limit", "no_leader", "not_calibrated", "no_workcell",
        "scene_error",
    )
    # one spelling authority: the session block and the telemetry block share the alias
    assert GelloSessionConfig.model_fields["viewpoint"].annotation == GelloViewpointMode
    assert GelloViewpointTelemetry.model_fields["mode"].annotation == GelloViewpointMode
    assert GelloTelemetry.model_fields["backend"].annotation == GelloBackend
    assert GelloInfo.model_fields["status"].annotation == GelloDeviceStatus
    assert GelloCalibrateRequest.model_fields["op"].annotation == GelloCalibrateOp
    assert GelloPreviewResult.model_fields["status"].annotation == GelloPreviewStatus


def test_protocol_gello_is_a_dependency_free_leaf():
    """Like ``protocol.microphone``: pydantic + typing only, so ``protocol.session`` and
    ``protocol.telemetry`` can both import it without a cycle."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(gello_mod))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported == {"__future__", "typing", "pydantic"}, imported
    # and it is importable on its own, before the rest of the package
    code = "import apollo_mavis_v2_core.protocol.gello as g; print(len(g.__all__))"
    import subprocess

    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == str(len(gello_mod.__all__)) == "13"


# -- session -------------------------------------------------------------------------------------
def test_gello_session_config_defaults_strict_and_schema():
    """16-gello D1 / §8.1: ONE field, ``viewpoint`` (auto default); ``extra="forbid"`` so a
    leader setting typed here by mistake (port, joint_signs, ...) is a 422, never silently
    dropped; the JSON schema carries ``additionalProperties: false`` for the sheet."""
    assert list(GelloSessionConfig.model_fields) == ["viewpoint"]
    cfg = GelloSessionConfig()
    assert cfg.viewpoint == "auto"
    assert cfg.model_dump() == {"viewpoint": "auto"}
    for mode in ("auto", "external", "hold"):
        assert GelloSessionConfig(viewpoint=mode).viewpoint == mode
        assert GelloSessionConfig.model_validate_json(
            GelloSessionConfig(viewpoint=mode).model_dump_json()).viewpoint == mode
    for bad in ("AUTO", "none", "policy", "", None):
        with pytest.raises(ValidationError):
            GelloSessionConfig(viewpoint=bad)
    for stale_key in ("port", "joint_signs", "engage_tol_rad", "scene_id", "viewpiont"):
        with pytest.raises(ValidationError) as ei:
            GelloSessionConfig.model_validate({stale_key: 1})
        assert ei.value.errors()[0]["type"] == "extra_forbidden"
    schema = GelloSessionConfig.model_json_schema()
    assert schema["additionalProperties"] is False
    assert "required" not in schema  # everything defaults
    assert schema["properties"]["viewpoint"] == {
        "type": "string", "enum": ["auto", "external", "hold"], "default": "auto",
        "title": "Viewpoint",
    }


def test_session_spec_gello_accepts_the_d1_body():
    """16-gello D1 / §11: the sheet posts ``{mode: gello, kind, arms: [both], frames,
    sim_scene | digital_twin_scene, speed_scale, start_from: keep_current, gello:
    {viewpoint}}`` — nothing else; ``gello`` is appended LAST and defaults to None."""
    assert list(SessionSpec.model_fields)[-1] == "gello"
    assert SessionSpec.model_fields["gello"].default is None
    spec = _spec()
    assert spec.mode == "gello" and spec.gello == GelloSessionConfig()
    assert spec.gello.viewpoint == "auto"
    assert (spec.task, spec.dataset, spec.policy, spec.online_dagger) == (None, None, None, None)
    assert spec.policy_source == "checkpoint" and spec.start_from == "keep_current"
    assert spec.return_to_start is True and spec.action_filter.enabled is True  # defaults
    for mode in ("auto", "external", "hold"):
        assert _spec(gello={"viewpoint": mode}).gello.viewpoint == mode
    hw = _spec(kind="hardware", sim_scene=None, digital_twin_scene="mavis_v2_kitchen",
               speed_scale=0.5, gello={"viewpoint": "hold"})
    assert hw.kind == "hardware" and hw.gello.viewpoint == "hold" and hw.speed_scale == 0.5
    # the explicit defaults survive the JSON round trip on the same body (wire invariant)
    body = json.loads(spec.model_dump_json())
    assert body["gello"] == {"viewpoint": "auto"} and body["start_from"] == "keep_current"
    assert body["return_to_start"] is True and body["policy_source"] == "checkpoint"
    assert SessionSpec.model_validate(body) == spec
    assert SessionSpec.model_validate_json(spec.model_dump_json()) == spec
    # the nested block is validated too
    with pytest.raises(ValidationError):
        _spec(gello={"viewpoint": "policy"})
    with pytest.raises(ValidationError) as ei:
        _spec(gello={"viewpoint": "auto", "port": "/dev/ttyUSB0"})
    assert ei.value.errors()[0]["type"] == "extra_forbidden"


def test_session_spec_gello_rejects_every_foreign_field():
    """16-gello §8.1: mode gello REQUIRES the block, ``start_from == keep_current``, no task /
    dataset / policy / online_dagger, ``policy_source == checkpoint`` and the default
    return_to_start / action_filter — each with its own message, evaluated before the
    generic mode rules (the online_dagger precedent)."""
    with pytest.raises(ValidationError, match="mode 'gello' requires a gello block"):
        _spec(gello=None)
    body = json.loads(_spec().model_dump_json())
    body.pop("gello")
    with pytest.raises(ValidationError, match="requires a gello block"):
        SessionSpec.model_validate(body)  # a pre-phase-15 body cannot launch gello by accident
    with pytest.raises(ValidationError, match="requires start_from 'keep_current'"):
        _spec(start_from="profile:abc123")
    with pytest.raises(ValidationError, match="mode 'gello' takes no task"):
        _spec(task="open the fridge")
    with pytest.raises(ValidationError, match="mode 'gello' takes no dataset") as ei:
        _spec(dataset="bc_demo/fridge")
    assert "collect-mode field" not in str(ei.value)  # the specific message, not the generic
    with pytest.raises(ValidationError, match="mode 'gello' takes no dataset"):
        _spec(dataset_resume=True)
    with pytest.raises(ValidationError, match="takes no policy checkpoint"):
        _spec(policy="run0/deploy/v001")
    with pytest.raises(ValidationError, match="takes no online_dagger block") as ei:
        _spec(online_dagger={"session_name": "s1"})
    assert "online_dagger requires mode dagger" not in str(ei.value)
    with pytest.raises(ValidationError, match="requires policy_source 'checkpoint'") as ei:
        _spec(policy_source="external")
    assert "dagger|inference" not in str(ei.value)  # the gello rule wins over the generic one
    # the generic default-only rules still hold for gello
    with pytest.raises(ValidationError, match="return_to_start is a collect / dagger-mode"):
        _spec(return_to_start=False)
    with pytest.raises(ValidationError, match="action_filter is a collect / dagger-mode"):
        _spec(action_filter={"enabled": False})
    # the frames rules are unchanged (checked first)
    with pytest.raises(ValidationError, match="frames keys must be a subset of arms"):
        _spec(frames={"grip": "arm_base:grip", "other": "world"})
    with pytest.raises(ValidationError, match="may not be an ee: frame"):
        _spec(frames={"grip": "ee:grip"})


def test_session_spec_gello_block_is_gello_mode_only():
    """A non-null ``gello`` on any other mode is an error; None is inert everywhere, and a
    pre-phase-15 body (no key) parses as None — additive."""
    for mode, extra in (("teleop", {}), ("collect", {"task": "t"}),
                        ("dagger", {"task": "t"}), ("inference", {})):
        with pytest.raises(ValidationError, match="gello is a gello-mode field"):
            _spec(mode=mode, gello={}, **extra)
        with pytest.raises(ValidationError, match="gello is a gello-mode field"):
            _spec(mode=mode, gello={"viewpoint": "hold"}, **extra)
        plain = _spec(mode=mode, gello=None, **extra)
        assert plain.gello is None
        body = json.loads(plain.model_dump_json())
        assert body["gello"] is None
        body.pop("gello")
        assert SessionSpec.model_validate(body).gello is None
    # the existing policy_source rule is untouched for the other modes
    with pytest.raises(ValidationError, match="requires mode dagger\\|inference"):
        _spec(mode="teleop", gello=None, policy_source="external")
    ext_dagger = _spec(mode="dagger", gello=None, task="t", policy_source="external")
    assert ext_dagger.gello is None and ext_dagger.policy_source == "external"


def test_session_info_echoes_gello_last():
    """16-gello §8.1: ``SessionInfo.gello`` echoes the spec block (None otherwise), appended
    LAST so pre-phase-15 producers and consumers keep parsing."""
    assert list(SessionInfo.model_fields)[-1] == "gello"
    assert SessionInfo.model_fields["gello"].default is None
    base = dict(session_id="s15", epoch="e", mode="gello", arms=["view", "grip"],
                streams=["sim"], state="running", kind="sim", speed_scale=1.0)
    assert SessionInfo(**base).gello is None
    info = SessionInfo(**base, gello=GelloSessionConfig(viewpoint="external"))
    wire = json.loads(info.model_dump_json())
    assert wire["mode"] == "gello" and wire["gello"] == {"viewpoint": "external"}
    assert wire["policy_source"] == "checkpoint" and wire["online_dagger"] is None
    assert SessionInfo.model_validate(wire) == info
    legacy = dict(wire)
    legacy.pop("gello")
    assert SessionInfo.model_validate(legacy).gello is None
    with pytest.raises(ValidationError):
        SessionInfo(**base, gello={"viewpoint": "policy"})
    with pytest.raises(ValidationError):
        SessionInfo(**{**base, "mode": "GELLO"})


# -- actions + sources ---------------------------------------------------------------------------
def test_action_name_gains_gello_pause_resume_without_keys_or_args():
    """16-gello D9 / §8.2: ``gello_pause`` / ``gello_resume`` are the LAST two ActionNames
    (after goto_profile, in this order), argless and deliberately NOT keymap rows (the
    keymap is operator-owned; a Pause key is 16-gello §16 item 3, the operator's call)."""
    names = get_args(ActionName)
    assert names[-3:] == ("goto_profile", "gello_pause", "gello_resume")
    assert names.index("gello_pause") > names.index("goto_profile")  # appended, not inserted
    for name in ("gello_pause", "gello_resume"):
        msg = parse_client_msg(json.dumps({"t": "action", "name": name}))
        assert isinstance(msg, ActionMsg) and msg.name == name and msg.args == {}
        assert validate_action_args(msg) is None  # takes no args
        with pytest.raises(ValueError, match="takes no args"):
            validate_action_args(ActionMsg(name=name, args={"arm_id": "grip"}))
        assert not any(row.action == name for row in KEYMAP)
    for ack in (AckMsg(name="gello_pause", ok=True),  # idempotent: a no-op ack is a plain ok
                AckMsg(name="gello_resume", ok=False, detail="not a GELLO session"),
                AckMsg(name="gello_resume", ok=True, detail="out_of_sync: leader 0.31 rad away")):
        assert AckMsg.model_validate_json(ack.model_dump_json()) == ack
    with pytest.raises(ValidationError):
        parse_client_msg(json.dumps({"t": "action", "name": "gello_toggle"}))
    assert len(KEYMAP) == 24  # the operator's table is untouched
    assert not any(row.code == "KeyG" for row in KEYMAP)  # G stays free (16-gello §16 item 3)


def test_command_source_gains_gello_last():
    """16-gello §6 / §8.2: ``CommandSource.GELLO = "gello"`` appended last; it rides
    ``CollisionEvent.source`` (gate events) like every other source and exports as one
    more enum member (the UI union)."""
    assert [m.value for m in CommandSource] == [
        "teleop", "joint_jog", "policy", "takeover", "planner", "gello",
    ]
    assert CommandSource.GELLO == "gello" and CommandSource("gello") is CommandSource.GELLO
    ev = CollisionEvent(ts=1.0, kind="blocked", pairs=[("grip_link6", "fridge_body")],
                        dists_m=[0.006], min_clearance_m=0.006, source=CommandSource.GELLO,
                        arm_ids=["grip"])
    wire = json.loads(ev.model_dump_json())
    assert wire["source"] == "gello"
    assert CollisionEvent.model_validate(wire) == ev
    assert CollisionEvent.model_json_schema()["$defs"]["CommandSource"]["enum"][-1] == "gello"


# -- telemetry -----------------------------------------------------------------------------------
def test_gello_telemetry_fields_are_exactly_16_gello_8_3():
    """The block's field order is the contract: the device half (shared with GET /api/gello
    through ``GelloDeviceTelemetry``) then the session half; only ``backend`` / ``status``
    / ``joint_signs`` are required, so a runtime without a leader still produces a block."""
    assert list(GelloDeviceTelemetry.model_fields) == _DEVICE_FIELDS
    assert list(GelloTelemetry.model_fields) == _DEVICE_FIELDS + _SESSION_FIELDS
    assert issubclass(GelloTelemetry, GelloDeviceTelemetry)
    required = {n for n, f in GelloTelemetry.model_fields.items() if f.is_required()}
    assert required == {"backend", "status", "joint_signs"}
    assert list(GelloViewpointTelemetry.model_fields) == [
        "mode", "attached", "policy_id", "detail", "paused",
    ]
    vp_required = {n for n, f in GelloViewpointTelemetry.model_fields.items() if f.is_required()}
    assert vp_required == {"mode", "attached"}
    # defaults of the optional fields, exactly as written in §8.3
    bare = GelloTelemetry(backend="none", status="no_backend", joint_signs=_SIGNS)
    assert bare.model_dump() == {
        "backend": "none", "status": "no_backend", "detail": "", "port": "", "baud": None,
        "seq": 0, "rate_hz": 0.0, "age_s": None, "q_raw": None, "q": None,
        "gripper_frac": None, "calibrated": False, "joint_offsets_rad": None,
        "joint_signs": _SIGNS, "state": None, "state_detail": "", "lag_rad": None,
        "max_lag_rad": None, "engaged_arm": None, "viewpoint": None, "paused_latched": None,
    }
    for state in get_args(GelloState):
        assert GelloTelemetry(**_device(), state=state).state == state
    for bad in ("engaged", "idle", "TRACKING", ""):
        with pytest.raises(ValidationError):
            GelloTelemetry(**_device(), state=bad)
    for bad_status in ("live", "searching", "tracking"):  # the tracker's / mic's words
        with pytest.raises(ValidationError):
            GelloTelemetry(**_device(status=bad_status))
    with pytest.raises(ValidationError):
        GelloTelemetry(backend="none", status="no_backend")  # joint_signs required
    with pytest.raises(ValidationError):
        GelloTelemetry(**_device(backend="libsurvive"))


def test_telemetry_gello_block_is_additive_and_round_trips():
    """``TelemetryMsg.gello`` is appended LAST, defaults to None (pre-phase-15 producers and
    a runtime with ``backend: none`` and no reader keep validating) and carries the device
    half session-less like tracker / microphone; the session half rides only during a gello
    session."""
    assert list(TelemetryMsg.model_fields)[-1] == "gello"
    assert TelemetryMsg.model_fields["gello"].default is None
    fields = list(TelemetryMsg.model_fields)
    assert fields.index("gello") == fields.index("datasets") + 1
    assert json.loads(_frame().model_dump_json())["gello"] is None
    # session-less: device half only
    device_only = _frame(active_arm=None, controller_connected=False,
                         gello=GelloTelemetry(**_device()))
    wire = json.loads(device_only.model_dump_json())
    assert wire["gello"]["status"] == "connected" and wire["gello"]["state"] is None
    assert wire["gello"]["viewpoint"] is None and wire["gello"]["q"][0] == 3.1416
    assert wire["arms"] == [] and wire["session"] is None
    assert TelemetryMsg.model_validate(wire) == device_only
    # in a gello session: tracking, external node driving the Perception Arm
    tracking = GelloTelemetry(
        **_device(), state="tracking", state_detail="", lag_rad=[0.01, -0.02, 0, 0, 0, 0, 0.03],
        max_lag_rad=0.03, engaged_arm="grip",
        viewpoint=GelloViewpointTelemetry(mode="auto", attached=True, policy_id="viewpoint-v1",
                                          detail="external node viewpoint-v1 attached"),
    )
    frame = _frame(gello=tracking)
    wire = json.loads(frame.model_dump_json())
    assert list(wire["gello"]) == _DEVICE_FIELDS + _SESSION_FIELDS
    assert wire["gello"]["state"] == "tracking" and wire["gello"]["engaged_arm"] == "grip"
    assert wire["gello"]["viewpoint"] == {
        "mode": "auto", "attached": True, "policy_id": "viewpoint-v1",
        "detail": "external node viewpoint-v1 attached", "paused": False,
    }
    assert TelemetryMsg.model_validate(wire) == frame
    assert TelemetryMsg.model_validate_json(frame.model_dump_json()) == frame
    # out of sync: per-joint deltas for the panel's bars; holding the GELLO posture
    oos = tracking.model_copy(update={
        "state": "out_of_sync", "state_detail": "leader 0.31 rad from the arm - move GELLO "
        "within 0.10 rad", "lag_rad": [0.31, 0, 0, 0, 0, 0, 0], "max_lag_rad": 0.31,
        "engaged_arm": None,
        "viewpoint": GelloViewpointTelemetry(mode="hold", attached=False,
                                             detail="holding the GELLO posture"),
    })
    wire = json.loads(_frame(gello=oos).model_dump_json())["gello"]
    assert wire["state"] == "out_of_sync" and wire["max_lag_rad"] == 0.31
    assert wire["viewpoint"]["attached"] is False and wire["viewpoint"]["policy_id"] is None
    # a pre-phase-15 consumer's frame (no key) still parses
    legacy = json.loads(frame.model_dump_json())
    legacy.pop("gello")
    assert TelemetryMsg.model_validate(legacy).gello is None
    with pytest.raises(ValidationError):
        GelloViewpointTelemetry(mode="auto")  # attached is required
    with pytest.raises(ValidationError):
        GelloViewpointTelemetry(mode="policy", attached=True)


# -- REST models ---------------------------------------------------------------------------------
def test_gello_info_is_the_device_half_plus_the_sheet_facts():
    """16-gello §8.4: ``GET /api/gello`` = the device half of the telemetry block (same
    class, so the two never drift) plus the scene the card launches, the Perception Arm's
    hold posture, the calibration path, the hardware admission flag (D8) and the two
    gripper-endpoint echoes (§4 "the result of every op is echoed")."""
    assert issubclass(GelloInfo, GelloDeviceTelemetry)
    assert list(GelloInfo.model_fields) == _DEVICE_FIELDS + [
        "scene_id", "scene_label", "view_posture_rad", "view_rail_m", "calibration_path",
        "hardware_admitted", "gripper_open_rad", "gripper_closed_rad",
    ]
    assert not set(_SESSION_FIELDS) & set(GelloInfo.model_fields)  # no session half here
    required = {n for n, f in GelloInfo.model_fields.items() if f.is_required()}
    assert required == {
        "backend", "status", "joint_signs", "scene_id", "scene_label", "view_posture_rad",
        "view_rail_m", "calibration_path", "hardware_admitted",
    }
    facts = dict(scene_id="mavis_v2_kitchen", scene_label="APOLLO MAVIS V2 Kitchen (GELLO)",
                 view_posture_rad=_VIEW_HOLD, view_rail_m=0.0,
                 calibration_path="/ws/var/gello_calibration.json", hardware_admitted=True)
    info = GelloInfo(**_device(), **facts, gripper_open_rad=2.9, gripper_closed_rad=1.1)
    wire = json.loads(info.model_dump_json())
    assert wire["scene_id"] == "mavis_v2_kitchen" and wire["view_posture_rad"] == _VIEW_HOLD
    assert wire["hardware_admitted"] is True and wire["gripper_open_rad"] == 2.9
    assert GelloInfo.model_validate(wire) == info
    # no leader configured: the device half says so, the sheet facts are still served
    none = GelloInfo(backend="none", status="no_backend", detail="gello.backend is none",
                     joint_signs=_SIGNS, **facts)
    assert none.calibrated is False and none.q is None and none.gripper_open_rad is None
    assert GelloInfo.model_validate_json(none.model_dump_json()) == none
    with pytest.raises(ValidationError):
        GelloInfo(**_device())  # the sheet facts are required
    schema = GelloInfo.model_json_schema()
    assert "allOf" not in schema and "$defs" not in schema  # a flat object for the UI types
    assert schema["required"] == [
        "backend", "status", "joint_signs", "scene_id", "scene_label", "view_posture_rad",
        "view_rail_m", "calibration_path", "hardware_admitted",
    ]


def test_gello_calibrate_request_and_result():
    """16-gello D10 / §4: four ops, ``kind`` says where the Manipulation Arm's joints come
    from for ``match_arm``; the result echoes the calibration now in force."""
    assert list(GelloCalibrateRequest.model_fields) == ["op", "kind"]
    for op in ("match_arm", "gripper_open", "gripper_closed", "clear"):
        for kind in ("hardware", "sim"):
            req = GelloCalibrateRequest(op=op, kind=kind)
            assert GelloCalibrateRequest.model_validate_json(req.model_dump_json()) == req
    for bad in ("match", "reset", "gripper", "", None):
        with pytest.raises(ValidationError):
            GelloCalibrateRequest(op=bad, kind="sim")
    with pytest.raises(ValidationError):
        GelloCalibrateRequest(op="clear")  # kind required
    with pytest.raises(ValidationError):
        GelloCalibrateRequest(op="clear", kind="twin")
    assert list(GelloCalibrateResult.model_fields) == [
        "ok", "detail", "joint_offsets_rad", "gripper_open_rad", "gripper_closed_rad",
    ]
    cleared = GelloCalibrateResult(ok=True, detail="calibration cleared")
    assert (cleared.joint_offsets_rad, cleared.gripper_open_rad, cleared.gripper_closed_rad) == (
        None, None, None)
    matched = GelloCalibrateResult(ok=True, joint_offsets_rad=[0.0, 1.5708, 0.0, -1.5708, 0, 0, 0],
                                   gripper_open_rad=2.9, gripper_closed_rad=1.1)
    wire = json.loads(matched.model_dump_json())
    assert wire["joint_offsets_rad"][1] == 1.5708 and wire["detail"] == ""
    assert GelloCalibrateResult.model_validate(wire) == matched
    refused = GelloCalibrateResult(ok=False, detail="no fresh leader sample (status stale)")
    assert GelloCalibrateResult.model_validate_json(refused.model_dump_json()) == refused
    with pytest.raises(ValidationError):
        GelloCalibrateResult(detail="x")  # ok required


def test_gello_preview_request_result_and_pairs():
    """16-gello §5.4: the session-less launch check. ``ok`` is True iff ``status == clear``
    (validated, like CollisionReport's blocked / severity); ``pairs`` tightest first, the
    PNG base64 or None, ``camera`` defaults to ``cam_kitchen``; never a 409 for a bad
    posture — the status says."""
    assert list(GelloPreviewRequest.model_fields) == ["kind", "scene", "speed_scale"]
    req = GelloPreviewRequest(kind="sim")
    assert req.scene is None and req.speed_scale is None
    assert GelloPreviewRequest(kind="hardware", scene="mavis_v2_kitchen",
                               speed_scale=0.5).speed_scale == 0.5
    for bad in (0, 1.5, -0.1):
        with pytest.raises(ValidationError):
            GelloPreviewRequest(kind="sim", speed_scale=bad)
    with pytest.raises(ValidationError):
        GelloPreviewRequest(scene="mavis_v2_kitchen")  # kind required
    assert list(GelloPairInfo.model_fields) == ["a", "b", "dist_m"]
    pair = GelloPairInfo(a="grip_link6", b="fridge_body", dist_m=0.003)
    with pytest.raises(ValidationError):
        GelloPairInfo(a="x", b="y")  # dist_m required
    assert list(GelloPreviewResult.model_fields) == [
        "status", "ok", "detail", "pairs", "q_goal", "leader_q", "image_png_b64", "camera",
    ]
    q_goal = {"grip": _LEADER_Q + [0.32], "view": _VIEW_HOLD + [0.0]}
    clear = GelloPreviewResult(status="clear", ok=True, q_goal=q_goal, leader_q=_LEADER_Q,
                               image_png_b64="iVBORw0KGgo=")
    assert clear.pairs == [] and clear.camera == "cam_kitchen" and clear.detail == ""
    wire = json.loads(clear.model_dump_json())
    assert wire["q_goal"]["grip"][7] == 0.32 and len(wire["q_goal"]["view"]) == 8
    assert GelloPreviewResult.model_validate(wire) == clear
    collision = GelloPreviewResult(
        status="collision", ok=False,
        detail="collides: fridge_body / grip_link6 at 3 mm - move GELLO and retry",
        pairs=[pair, GelloPairInfo(a="grip_link7", b="fridge_door_handle", dist_m=0.007)],
        q_goal=q_goal, leader_q=_LEADER_Q, image_png_b64="iVBORw0KGgo=",
    )
    wire = json.loads(collision.model_dump_json())
    assert wire["pairs"][0] == {"a": "grip_link6", "b": "fridge_body", "dist_m": 0.003}
    assert GelloPreviewResult.model_validate_json(collision.model_dump_json()) == collision
    for status in ("joint_limit", "no_leader", "not_calibrated", "no_workcell", "scene_error"):
        res = GelloPreviewResult(status=status, ok=False, detail=status)
        assert res.leader_q is None and res.image_png_b64 is None and res.q_goal == {}
        assert GelloPreviewResult.model_validate_json(res.model_dump_json()) == res
    # the ok / status invariant
    with pytest.raises(ValidationError, match="ok must be True iff status == 'clear'"):
        GelloPreviewResult(status="collision", ok=True)
    with pytest.raises(ValidationError, match="ok must be True iff status == 'clear'"):
        GelloPreviewResult(status="clear", ok=False)
    with pytest.raises(ValidationError):
        GelloPreviewResult(status="blocked", ok=False)
    # two results never share the default containers
    a, b = GelloPreviewResult(status="clear", ok=True), GelloPreviewResult(status="clear", ok=True)
    assert a.pairs is not b.pairs and a.q_goal is not b.q_goal


# -- dora wire -----------------------------------------------------------------------------------
def test_session_announce_external_arms_is_last_and_additive():
    """16-gello §7 / D5: ``SessionAnnounce.external_arms`` appended after ``online_dagger``;
    empty = every session arm (today's behaviour), ``["view"]`` in a gello session with an
    external viewpoint. ``EVENT_KINDS`` / ``RUNTIME_INPUTS`` / ``POLICY_OUTPUTS`` /
    ``MAVIS_SCHEMA`` are unchanged (the goldens in both repos move together)."""
    assert list(ext.SessionAnnounce.model_fields)[-2:] == ["online_dagger", "external_arms"]
    field = ext.SessionAnnounce.model_fields["external_arms"]
    assert field.default_factory is list and not field.is_required()
    idle = ext.SessionAnnounce(epoch="e", session_id=None, state="idle")
    assert idle.external_arms == []
    a, b = ext.SessionAnnounce(epoch="e", session_id=None, state="idle"), idle
    assert a.external_arms is not b.external_arms  # no shared default list
    spec = _spec()
    gello_ann = ext.SessionAnnounce(
        epoch="ep0", session_id="s15", state="running", spec=spec, kind="sim",
        arm_ids=["view", "grip"], has_rail={"view": True, "grip": True},
        frames={"view": "arm_base:view", "grip": "arm_base:grip"}, action_space="delta_ee",
        action_names=["view_ee.dx", "view_ee.dy", "view_ee.dz", "view_ee.drx", "view_ee.dry",
                      "view_ee.drz", "view_gripper.pos", "view_rail.dpos"],
        state_names=["view_joint1.pos"], policy_source="checkpoint", external_arms=["view"],
    )
    wire = json.loads(gello_ann.model_dump_json())
    assert list(wire)[-2:] == ["online_dagger", "external_arms"]
    assert wire["external_arms"] == ["view"] and wire["online_dagger"] is None
    assert wire["spec"]["mode"] == "gello" and wire["spec"]["gello"] == {"viewpoint": "auto"}
    assert wire["arm_ids"] == ["view", "grip"]  # both arms still announced (obs_state)
    assert wire["mavis_schema"] == 1
    assert ext.SessionAnnounce.model_validate(wire) == gello_ann
    legacy = dict(wire)
    legacy.pop("external_arms")  # a phase-14 announce
    assert ext.SessionAnnounce.model_validate(legacy).external_arms == []
    hold = gello_ann.model_copy(update={"spec": _spec(gello={"viewpoint": "hold"}),
                                        "external_arms": []})
    assert json.loads(hold.model_dump_json())["external_arms"] == []
    with pytest.raises(ValidationError):
        ext.SessionAnnounce(epoch="e", session_id=None, state="idle", external_arms="view")
    # nothing else moved
    assert ext.MAVIS_SCHEMA == 1 and len(ext.EVENT_KINDS) == 10
    assert ext.RUNTIME_INPUTS[-1] == "policy_trainer_status"
    assert ext.POLICY_OUTPUTS == ("action", "spec", "status", "trainer_status")
    assert list(ext.PolicySpecAnnounce.model_fields)[-1] == "capabilities"
    assert list(ext.OnlineDaggerAnnounce.model_fields) == [
        "session_name", "session_dir", "rollouts_dir",
    ]


# -- exported schemas ----------------------------------------------------------------------------
def test_gello_models_are_exported_schemas(tmp_path):
    """16-gello §8.4 / 01-core §14: the SessionSpec block and the six REST models are
    exported (50 files: 48 models + keymap.json + index.json); the telemetry block rides
    TelemetryMsg's $defs, GelloPairInfo GelloPreviewResult's."""
    for name in ("GelloSessionConfig", "GelloInfo", "GelloCalibrateRequest",
                 "GelloCalibrateResult", "GelloPreviewRequest", "GelloPreviewResult",
                 "GelloPairInfo"):
        assert name in EXPORTED_MODELS and EXPORTED_MODELS[name].__name__ == name
    assert "GelloTelemetry" not in EXPORTED_MODELS  # rides TelemetryMsg $defs only
    assert len(EXPORTED_MODELS) == 48
    written = {p.name: p for p in export(tmp_path / "schemas")}
    assert len(written) == 50
    index = json.loads(written["index.json"].read_text())
    assert "GelloInfo" in index["models"] and len(index["models"]) == 48
    tele = json.loads(written["TelemetryMsg.json"].read_text())
    assert {"GelloTelemetry", "GelloViewpointTelemetry"} <= set(tele["$defs"])
    assert tele["properties"]["gello"]["default"] is None
    assert tele["$defs"]["GelloTelemetry"]["required"] == ["backend", "status", "joint_signs"]
    assert tele["$defs"]["GelloTelemetry"]["properties"]["state"]["anyOf"][0]["enum"] == list(
        get_args(GelloState))
    # (``properties`` keys are SORTED in the export; field ORDER is pinned by the
    # ``model_fields`` assertions above — here we pin the set and the shapes.)
    spec = json.loads(written["SessionSpec.json"].read_text())
    assert spec["properties"]["mode"]["enum"][-1] == "gello"
    assert spec["$defs"]["GelloSessionConfig"]["additionalProperties"] is False
    assert spec["properties"]["gello"] == {
        "anyOf": [{"$ref": "#/$defs/GelloSessionConfig"}, {"type": "null"}], "default": None,
    }
    assert spec["required"] == ["mode", "kind", "arms", "frames"]  # gello stays optional
    info = json.loads(written["SessionInfo.json"].read_text())
    assert info["properties"]["gello"] == spec["properties"]["gello"]
    assert info["$defs"]["GelloSessionConfig"] == spec["$defs"]["GelloSessionConfig"]
    preview = json.loads(written["GelloPreviewResult.json"].read_text())
    assert "GelloPairInfo" in preview["$defs"]
    assert preview["properties"]["camera"]["default"] == "cam_kitchen"
    ann = json.loads(written["SessionAnnounce.json"].read_text())
    assert "external_arms" not in ann.get("required", [])
    assert ann["properties"]["external_arms"] == {
        "items": {"type": "string"}, "title": "External Arms", "type": "array",
    }
    for name in ("ActionMsg", "AckMsg"):
        enum = json.loads(written[f"{name}.json"].read_text())["properties"]["name"]["enum"]
        assert enum[-2:] == ["gello_pause", "gello_resume"]
    ev = json.loads(written["CollisionEvent.json"].read_text())
    assert ev["$defs"]["CommandSource"]["enum"][-1] == "gello"
