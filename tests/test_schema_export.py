"""Schema-export tests (design doc 01-core §14, §18)."""

from __future__ import annotations

import json

from apollo_mavis_v2_core.protocol.export_schemas import EXPORTED_MODELS, export, main

_EXPECTED_FILES = {f"{name}.json" for name in EXPORTED_MODELS} | {"keymap.json", "index.json"}


def test_export_is_byte_deterministic(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    files_a = export(a)
    files_b = export(b)
    assert {p.name for p in files_a} == _EXPECTED_FILES
    assert [p.name for p in files_a] == [p.name for p in files_b]
    for pa, pb in zip(files_a, files_b, strict=True):
        assert pa.read_bytes() == pb.read_bytes(), pa.name


def test_check_passes_on_fresh_export(tmp_path):
    out = tmp_path / "schemas"
    assert main(["--out", str(out)]) == 0
    assert main(["--out", str(out), "--check"]) == 0


def test_check_fails_on_tampered_file(tmp_path):
    out = tmp_path / "schemas"
    export(out)
    victim = out / "TelemetryMsg.json"
    victim.write_bytes(victim.read_bytes() + b"// tampered\n")
    assert main(["--out", str(out), "--check"]) == 1


def test_check_fails_on_missing_file(tmp_path):
    out = tmp_path / "schemas"
    export(out)
    (out / "KeysMsg.json").unlink()
    assert main(["--out", str(out), "--check"]) == 1


def test_exported_json_is_clean(tmp_path):
    """Every file json.loads cleanly and contains no numpy artifacts."""
    for path in export(tmp_path / "schemas"):
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n"), path.name
        assert "numpy" not in text, path.name
        assert "ndarray" not in text, path.name
        json.loads(text)  # raises on malformed output


def test_index_and_keymap_contents(tmp_path):
    out = tmp_path / "schemas"
    export(out)
    index = json.loads((out / "index.json").read_text())
    assert index["models"] == sorted(EXPORTED_MODELS)
    assert isinstance(index["core_version"], str) and index["core_version"]
    keymap = json.loads((out / "keymap.json").read_text())
    assert isinstance(keymap, list) and len(keymap) == 24
    assert {row["code"] for row in keymap} >= {"KeyW", "Space", "ArrowRight", "KeyC", "KeyZ"}
    # 13-tracker §3: every row carries the (nullable) gamepad field.
    assert all("gamepad" in row for row in keymap)
    gamepads = {row["code"]: row["gamepad"] for row in keymap}
    assert gamepads["KeyC"] == "RT" and gamepads["KeyZ"] == "LB" and gamepads["KeyW"] is None
    # 2026-09-07 (phase-13): the keyboard is a full teleop interface — no
    # ``keyboard`` flag on the wire, codes unique across the table, episode keys
    # on N / Enter / Backspace.
    assert all("keyboard" not in row for row in keymap)
    codes = [row["code"] for row in keymap]
    assert len(set(codes)) == len(codes) == 24
    episode = {row["action"]: row["code"] for row in keymap if row["group"] == "episode"}
    assert episode == {
        "episode_new": "KeyN", "episode_save": "Enter", "episode_discard": "Backspace",
    }
    entry = json.loads((out / "KeymapEntry.json").read_text())
    assert "keyboard" not in entry["properties"]


def test_telemetry_schema_embeds_tracker_block(tmp_path):
    """TrackerTelemetry rides TelemetryMsg's $defs (no top-level export needed)."""
    out = tmp_path / "schemas"
    export(out)
    telemetry = json.loads((out / "TelemetryMsg.json").read_text())
    assert {
        "TrackerTelemetry", "TrackerSettingsMsg", "ControllerTelemetry", "PoseMsg",
        # phase-10: calibration sub-models ride the same $defs (protocol.tracker)
        "TrackerCalibrationStatus", "LighthouseStatus", "CalibrationValidation",
        "YawGesturePoint",
    } <= set(telemetry["$defs"])
    assert "tracker" in telemetry["properties"]
    tracker = telemetry["$defs"]["TrackerTelemetry"]
    assert set(tracker["properties"]) == {
        "backend", "status", "detail", "object_name", "seq", "rate_hz", "age_s",
        "pose_raw", "pose_world", "pose_filtered", "clutch", "engaged_arm", "anchor_tcp",
        "target_tcp", "settings", "controller", "device_held", "device_action",
        "charging", "calibration",
        # Controller link (13-tracker §3.5 item 7b, 2026-09-07).
        "controller_age_s", "objects", "dongle_present",
    }
    # phase-10: calibration is a nullable, defaulted sub-model (additive).
    calibration = tracker["properties"]["calibration"]
    assert {"$ref": "#/$defs/TrackerCalibrationStatus"} in calibration["anyOf"]
    assert {"type": "null"} in calibration["anyOf"]
    assert calibration["default"] is None
    assert not {"charging", "calibration"} & set(tracker["required"])
    # The link fields are optional on the wire too, so the UI reads a producer
    # that omits them as UNKNOWN rather than as "unplugged" / "not paired".
    assert not {"controller_age_s", "objects", "dongle_present"} & set(tracker["required"])
    assert tracker["properties"]["objects"]["type"] == "array"
    assert {"type": "null"} in tracker["properties"]["dongle_present"]["anyOf"]
    assert {"type": "null"} in tracker["properties"]["controller_age_s"]["anyOf"]
    assert set(tracker["properties"]["status"]["enum"]) == {
        "no_backend", "starting", "searching", "tracking", "stale", "error",
    }
    # 13-tracker §1.1: controller block is nullable, device_held a string list.
    controller_ref = tracker["properties"]["controller"]
    assert {"$ref": "#/$defs/ControllerTelemetry"} in controller_ref["anyOf"]
    assert {"type": "null"} in controller_ref["anyOf"]
    assert controller_ref["default"] is None
    device_held = tracker["properties"]["device_held"]
    assert device_held["type"] == "array" and device_held["items"] == {"type": "string"}
    # default_factory=list -> optional on the wire (no "default" key is emitted).
    assert not {"controller", "device_held"} & set(tracker["required"])
    # 13-tracker §4: pose_filtered is a nullable PoseMsg; device_action a nullable string.
    pose_filtered = tracker["properties"]["pose_filtered"]
    assert {"$ref": "#/$defs/PoseMsg"} in pose_filtered["anyOf"]
    assert {"type": "null"} in pose_filtered["anyOf"]
    assert pose_filtered["default"] is None
    device_action = tracker["properties"]["device_action"]
    assert {"type": "string"} in device_action["anyOf"]
    assert {"type": "null"} in device_action["anyOf"]
    assert device_action["default"] is None
    assert not {"pose_filtered", "device_action"} & set(tracker["required"])
    # Effective filter settings ride TrackerSettingsMsg with defaults (additive).
    settings = telemetry["$defs"]["TrackerSettingsMsg"]
    assert set(settings["properties"]) == {
        "yaw_deg", "pos_scale", "follow_rotation",
        "filter_enabled", "filter_min_cutoff_hz", "filter_beta",
    }
    assert set(settings["required"]) == {"yaw_deg", "pos_scale", "follow_rotation"}
    assert settings["properties"]["filter_enabled"] == {
        "type": "boolean", "default": True, "title": "Filter Enabled",
    }
    assert settings["properties"]["filter_min_cutoff_hz"] == {
        "type": "number", "default": 1.0, "title": "Filter Min Cutoff Hz",
    }
    assert settings["properties"]["filter_beta"] == {
        "type": "number", "default": 5.0, "title": "Filter Beta",  # 2026-09-07 retune
    }
    controller = telemetry["$defs"]["ControllerTelemetry"]
    assert set(controller["properties"]) == {
        "trigger", "trigger_pressed", "trackpad_touch", "trackpad_click",
        "trackpad_x", "trackpad_y", "grip", "menu", "system",
    }
    assert "required" not in controller  # every field defaults to released
    assert controller["properties"]["trigger"] == {
        "type": "number", "default": 0.0, "title": "Trigger",
    }
    assert controller["properties"]["grip"] == {
        "type": "boolean", "default": False, "title": "Grip",
    }
    settings_args = json.loads((out / "TrackerSettingsArgs.json").read_text())
    assert set(settings_args["properties"]) == {
        "yaw_deg", "pos_scale", "follow_rotation",
        "filter_enabled", "filter_min_cutoff_hz", "filter_beta",
    }
    assert "required" not in settings_args  # every field optional (= unchanged)

    def _number_branch(prop):
        return next(v for v in prop["anyOf"] if v.get("type") == "number")

    pos_scale = settings_args["properties"]["pos_scale"]
    assert {"minimum": 0.1, "maximum": 3.0}.items() <= _number_branch(pos_scale).items()
    min_cutoff = settings_args["properties"]["filter_min_cutoff_hz"]
    assert {"minimum": 0.05, "maximum": 50.0}.items() <= _number_branch(min_cutoff).items()
    assert {"type": "null"} in min_cutoff["anyOf"] and min_cutoff["default"] is None
    beta = settings_args["properties"]["filter_beta"]
    assert {"minimum": 0.0, "maximum": 200.0}.items() <= _number_branch(beta).items()
    assert {"type": "null"} in beta["anyOf"] and beta["default"] is None
    enabled = settings_args["properties"]["filter_enabled"]
    assert {"type": "boolean"} in enabled["anyOf"] and {"type": "null"} in enabled["anyOf"]
    assert enabled["default"] is None


def test_tracker_calibration_schemas(tmp_path):
    """phase-10: REST models export top-level; literals become enums (13-tracker §3)."""
    out = tmp_path / "schemas"
    export(out)
    status = json.loads((out / "TrackerCalibrationStatus.json").read_text())
    assert set(status["properties"]) == {
        "kind", "phase", "detail", "started_at", "elapsed_s",
        "scenes", "lighthouses", "stations_visible", "controller_still", "validation",
        "installed_path", "backup_path",
        "yaw_points", "next_point", "fitted_yaw_deg", "fit_residual_deg", "fit_checks",
        "applied_yaw_deg",
        "yaw_valid", "yaw_calibrated_at", "base_station_installed_at",
    }
    assert "required" not in status  # every field defaults (idle snapshot = {})
    assert status["properties"]["kind"] == {
        "type": "string", "enum": ["none", "base_station", "yaw"],
        "default": "none", "title": "Kind",
    }
    assert status["properties"]["phase"]["enum"] == [
        "idle", "starting", "capturing", "validating", "fitting",
        "installing", "done", "failed", "aborted",
    ]
    assert status["properties"]["phase"]["default"] == "idle"
    assert status["properties"]["yaw_valid"] == {
        "type": "boolean", "default": True, "title": "Yaw Valid",
    }
    # Sub-models ride $defs; PoseMsg is shared with telemetry (same class name).
    assert {
        "LighthouseStatus", "CalibrationValidation", "YawGesturePoint", "PoseMsg",
    } <= set(status["$defs"])
    assert status["properties"]["lighthouses"]["items"] == {"$ref": "#/$defs/LighthouseStatus"}
    assert status["properties"]["yaw_points"]["items"] == {"$ref": "#/$defs/YawGesturePoint"}
    assert status["properties"]["fit_checks"]["items"] == {"type": "string"}
    next_point = status["properties"]["next_point"]
    assert {"type": "string", "enum": [
        "start", "left", "forward", "right", "back", "up", "down",
    ]} in next_point["anyOf"]
    assert {"type": "null"} in next_point["anyOf"] and next_point["default"] is None
    validation = status["$defs"]["CalibrationValidation"]
    assert set(validation["properties"]) == {
        "samples", "std_mm", "max_step_mm", "threshold_std_mm", "threshold_step_mm", "passed",
    }
    assert "required" not in validation
    std_mm = validation["properties"]["std_mm"]  # fixed 3-tuple -> prefixItems
    assert std_mm["type"] == "array" and std_mm["minItems"] == 3 and std_mm["maxItems"] == 3
    assert std_mm["prefixItems"] == [{"type": "number"}] * 3
    assert std_mm["default"] == [0.0, 0.0, 0.0]
    assert validation["properties"]["threshold_std_mm"]["default"] == 5.0
    assert validation["properties"]["threshold_step_mm"]["default"] == 20.0
    lighthouse = status["$defs"]["LighthouseStatus"]
    assert set(lighthouse["properties"]) == {
        "index", "channel", "serial", "pose", "scenes", "reference",
    }
    assert lighthouse["required"] == ["index"]
    point = status["$defs"]["YawGesturePoint"]
    assert set(point["required"]) == {"label", "pose"}
    assert point["properties"]["pose"] == {"$ref": "#/$defs/PoseMsg"}

    command = json.loads((out / "TrackerCalibrationCommand.json").read_text())
    assert set(command["properties"]) == {"kind", "op", "point"}
    assert set(command["required"]) == {"kind", "op"}
    assert command["properties"]["kind"]["enum"] == ["base_station", "yaw"]  # no "none"
    assert command["properties"]["op"]["enum"] == [
        "start", "capture", "validate", "install", "apply", "abort",
    ]
    assert command["properties"]["point"]["default"] is None
    assert "$defs" not in command  # flat body


def test_exported_models_cover_spec_sections():
    """§14 model list, spelled exactly."""
    assert set(EXPORTED_MODELS) == {
        # control
        "HelloMsg", "KeysMsg", "ActionMsg", "AckMsg",
        "JointTargetArgs", "SaveProfileArgs", "SetInitialConditionArgs",
        "SwitchArmArgs", "TrackerSettingsArgs",
        # goto_profile (2026-09-08; profile row "go to" button)
        "GotoProfileArgs",
        # telemetry
        "TelemetryMsg",
        # tracker calibration (REST; sub-models ride $defs)
        "TrackerCalibrationStatus", "TrackerCalibrationCommand",
        # session
        "SessionSpec", "SessionInfo", "WorkcellStatus", "ArmStatusInfo",
        "CameraInfo", "SceneInfo", "ProfileInfo", "PolicyInfo",
        # return-to-initial (REST POST /api/session/return_home; 2026-09-08)
        "ReturnHomeResult",
        # datasets (REST /api/datasets; 2026-09-07 data collection)
        "DatasetInfo", "EpisodeInfo", "DatasetExportInfo", "DatasetExportRequest",
        # episode playback (2026-09-10; operator request, 04-runtime §10.8, 05-ui §8.1
        # item 7): the dialog reads Info before anything moves, Request carries the action
        "EpisodePlaybackInfo", "EpisodePlaybackRequest",
        # dataset layout (REST GET /api/datasets/layout; phase-14, 15-online-dagger §7)
        "DatasetLayoutInfo", "DatasetNamespaceInfo",
        # Online DAgger (phase-14; 15-online-dagger §5-§7): the spec block, the sessions
        # row, the generic trainer_status payload and the SessionAnnounce paths block
        "OnlineDaggerConfig", "OnlineDaggerSessionInfo",
        "TrainerStatusAnnounce", "OnlineDaggerAnnounce",
        # microphone (REST /api/microphones; phase-11)
        "MicrophoneInfo",
        # arm maintenance (REST POST /api/hardware/arms/{arm_id}/maintenance; phase-09b)
        "ArmMaintenanceRequest", "ArmMaintenanceResult",
        # external interface over dora (phase-12; 14-dora §13)
        "SessionAnnounce", "PolicySpecAnnounce", "DoraInfo",
        # misc
        "StateProfile", "KeymapEntry", "CollisionEvent",
    }


def test_telemetry_schema_embeds_microphone_block(tmp_path):
    """phase-11: MicrophoneTelemetry rides TelemetryMsg's $defs; every field defaults."""
    out = tmp_path / "schemas"
    export(out)
    telemetry = json.loads((out / "TelemetryMsg.json").read_text())
    assert "MicrophoneTelemetry" in telemetry["$defs"]
    microphone = telemetry["properties"]["microphone"]
    assert {"$ref": "#/$defs/MicrophoneTelemetry"} in microphone["anyOf"]
    assert {"type": "null"} in microphone["anyOf"]
    assert microphone["default"] is None
    assert "microphone" not in telemetry["required"]
    # Additive: the block sits after ``tracker`` in the model (schema keys are sorted).
    assert {"session", "tracker", "microphone"} <= set(telemetry["properties"])
    mic = telemetry["$defs"]["MicrophoneTelemetry"]
    assert set(mic["properties"]) == {
        "mic_id", "status", "detail", "seq", "age_s", "rate_hz", "sample_rate",
        "rms_dbfs", "peak_dbfs", "clipping", "env_min", "env_max", "overruns",
    }
    assert "required" not in mic  # every field defaults (no-microphone producers validate)
    assert mic["properties"]["mic_id"] == {
        "type": "string", "default": "mic_view", "title": "Mic Id",
    }
    assert mic["properties"]["status"] == {
        "type": "string",
        "enum": ["no_backend", "starting", "absent", "live", "stalled", "error"],
        "default": "no_backend",
        "title": "Status",
    }
    assert mic["properties"]["sample_rate"] == {
        "type": "integer", "default": 48000, "title": "Sample Rate",
    }
    assert mic["properties"]["clipping"] == {
        "type": "boolean", "default": False, "title": "Clipping",
    }
    assert mic["properties"]["overruns"] == {
        "type": "integer", "default": 0, "title": "Overruns",
    }
    for key in ("env_min", "env_max"):
        env = mic["properties"][key]
        assert env["type"] == "array" and env["items"] == {"type": "integer"}, key
        assert env["default"] == [], key
    for key in ("age_s", "rms_dbfs", "peak_dbfs"):
        prop = mic["properties"][key]
        assert {"type": "number"} in prop["anyOf"] and {"type": "null"} in prop["anyOf"], key
        assert prop["default"] is None, key


def test_microphone_info_schema(tmp_path):
    """phase-11: GET /api/microphones row exports top-level as a flat body."""
    out = tmp_path / "schemas"
    export(out)
    info = json.loads((out / "MicrophoneInfo.json").read_text())
    assert set(info["properties"]) == {
        "mic_id", "label", "kind", "source", "sample_rate", "channels", "live", "status",
        "detail",
    }
    assert set(info["required"]) == {
        "mic_id", "label", "kind", "source", "sample_rate", "live", "status",
    }
    assert info["properties"]["kind"] == {
        "type": "string", "enum": ["pulse", "fake", "none"], "title": "Kind",
    }
    assert info["properties"]["status"] == {
        "type": "string",
        "enum": ["no_backend", "starting", "absent", "live", "stalled", "error"],
        "title": "Status",
    }
    source = info["properties"]["source"]
    assert {"type": "string"} in source["anyOf"] and {"type": "null"} in source["anyOf"]
    assert info["properties"]["channels"] == {
        "type": "integer", "default": 1, "title": "Channels",
    }
    assert info["properties"]["detail"] == {"type": "string", "default": "", "title": "Detail"}
    assert "$defs" not in info  # flat body
    index = json.loads((out / "index.json").read_text())
    assert "MicrophoneInfo" in index["models"]


def test_workcell_schemas_gain_phase11_fields(tmp_path):
    """phase-11: ArmStatusInfo.reachable / WorkcellStatus.hardware_ready are additive."""
    out = tmp_path / "schemas"
    export(out)
    arm = json.loads((out / "ArmStatusInfo.json").read_text())
    assert set(arm["properties"]) == {
        "arm_id", "ip", "connected", "reachable", "has_rail", "gripper",
        "gripper_force_capable", "error_code", "joint_limits",
    }
    assert "reachable" not in arm["required"]
    assert arm["properties"]["reachable"] == {
        "type": "string",
        "enum": ["open", "refused", "unreachable", "unknown"],
        "default": "unknown",
        "title": "Reachable",
    }
    assert arm["properties"]["connected"] == {"type": "boolean", "title": "Connected"}
    workcell = json.loads((out / "WorkcellStatus.json").read_text())
    assert set(workcell["properties"]) == {
        "kind", "available_kinds", "arms", "cameras", "policies_available", "hardware_ready",
    }
    assert set(workcell["required"]) == {"kind", "available_kinds", "arms", "cameras"}
    assert workcell["properties"]["hardware_ready"] == {
        "type": "boolean", "default": False, "title": "Hardware Ready",
    }
    # The nested arm rows carry the probe field too (same $defs class).
    assert "reachable" in workcell["$defs"]["ArmStatusInfo"]["properties"]


def test_telemetry_schema_embeds_hardware_monitor_block(tmp_path):
    """phase-09a: HardwareMonitorTelemetry + its two row models ride TelemetryMsg's $defs."""
    out = tmp_path / "schemas"
    export(out)
    telemetry = json.loads((out / "TelemetryMsg.json").read_text())
    assert {
        "HardwareMonitorTelemetry", "ArmMonitorTelemetry", "TwinOverlayTelemetry",
    } <= set(telemetry["$defs"])
    block = telemetry["properties"]["hardware_monitor"]
    assert {"$ref": "#/$defs/HardwareMonitorTelemetry"} in block["anyOf"]
    assert {"type": "null"} in block["anyOf"]
    assert block["default"] is None
    assert "hardware_monitor" not in telemetry["required"]
    # Additive: sits after ``microphone`` in the model (schema keys are sorted).
    assert {"session", "tracker", "microphone", "hardware_monitor"} <= set(
        telemetry["properties"]
    )
    monitor = telemetry["$defs"]["HardwareMonitorTelemetry"]
    assert set(monitor["properties"]) == {"enabled", "paused", "arms", "overlays"}
    assert "required" not in monitor  # every field defaults (no-hardware producers validate)
    for key in ("enabled", "paused"):
        assert monitor["properties"][key] == {
            "type": "boolean", "default": False, "title": key.capitalize(),
        }, key
    arms = monitor["properties"]["arms"]
    assert arms["type"] == "array" and arms["items"] == {"$ref": "#/$defs/ArmMonitorTelemetry"}
    assert arms["default"] == []
    overlays = monitor["properties"]["overlays"]
    assert overlays["type"] == "array"
    assert overlays["items"] == {"$ref": "#/$defs/TwinOverlayTelemetry"}
    assert overlays["default"] == []

    arm = telemetry["$defs"]["ArmMonitorTelemetry"]
    assert set(arm["properties"]) == {
        "arm_id", "status", "detail", "seq", "age_s", "q", "tcp_pose",
        "rail_present", "rail_homed", "rail_enabled", "rail_pos_m", "rail_raw_mm",
        "gripper_open_frac", "gripper_raw", "error_code", "warn_code", "state", "mode",
        # phase-09b read-back + maintenance flag
        "collision_sensitivity", "tcp_load_kg", "tcp_load_cog_mm", "backstops_match",
        "maintenance_busy",
        # phase-09d async maintenance job progress
        "maintenance",
    }
    assert arm["required"] == ["arm_id"]
    assert arm["properties"]["arm_id"] == {"type": "string", "title": "Arm Id"}
    assert arm["properties"]["status"] == {
        "type": "string",
        "enum": ["off", "connecting", "running", "stale", "paused", "error"],
        "default": "off",
        "title": "Status",
    }
    assert arm["properties"]["detail"] == {"type": "string", "default": "", "title": "Detail"}
    for key, title in (("seq", "Seq"), ("error_code", "Error Code"), ("warn_code", "Warn Code")):
        assert arm["properties"][key] == {"type": "integer", "default": 0, "title": title}, key
    for key in ("q", "tcp_pose", "tcp_load_cog_mm"):
        prop = arm["properties"][key]
        assert prop["type"] == "array" and prop["items"] == {"type": "number"}, key
        assert prop["default"] == [], key
    for key in ("age_s", "rail_pos_m", "rail_raw_mm", "gripper_open_frac", "gripper_raw",
                "tcp_load_kg"):
        prop = arm["properties"][key]
        assert {"type": "number"} in prop["anyOf"] and {"type": "null"} in prop["anyOf"], key
        assert prop["default"] is None, key
    for key in ("rail_present", "rail_homed", "rail_enabled", "backstops_match"):
        prop = arm["properties"][key]
        assert {"type": "boolean"} in prop["anyOf"] and {"type": "null"} in prop["anyOf"], key
        assert prop["default"] is None, key
    for key in ("state", "mode", "collision_sensitivity"):
        prop = arm["properties"][key]
        assert {"type": "integer"} in prop["anyOf"] and {"type": "null"} in prop["anyOf"], key
        assert prop["default"] is None, key
    # phase-09b: the read-back is unbounded here (it is what the controller reports; the
    # 0..5 bound lives on ArmConfig), the busy flag a plain defaulted boolean.
    assert "minimum" not in arm["properties"]["collision_sensitivity"]
    assert arm["properties"]["maintenance_busy"] == {
        "type": "boolean", "default": False, "title": "Maintenance Busy",
    }
    # phase-09d: the async job progress is a nullable, defaulted sub-model on the same $defs.
    progress = arm["properties"]["maintenance"]
    assert {"$ref": "#/$defs/MaintenanceProgress"} in progress["anyOf"]
    assert {"type": "null"} in progress["anyOf"] and progress["default"] is None
    assert "MaintenanceProgress" in telemetry["$defs"]

    overlay = telemetry["$defs"]["TwinOverlayTelemetry"]
    assert set(overlay["properties"]) == {
        "stream_id", "camera_id", "arm_id", "status", "detail", "fps",
        "rail_fallback_m", "joint1_offset_rad", "mask_fraction",
    }
    assert set(overlay["required"]) == {"stream_id", "camera_id", "arm_id"}
    for key, title in (("stream_id", "Stream Id"), ("camera_id", "Camera Id"),
                       ("arm_id", "Arm Id")):
        assert overlay["properties"][key] == {"type": "string", "title": title}, key
    assert overlay["properties"]["status"] == {
        "type": "string",
        "enum": ["off", "waiting", "live", "stale", "error"],
        "default": "off",
        "title": "Status",
    }
    assert overlay["properties"]["detail"] == {"type": "string", "default": "", "title": "Detail"}
    for key, title in (
        ("fps", "Fps"), ("joint1_offset_rad", "Joint1 Offset Rad"),
        ("mask_fraction", "Mask Fraction"),
    ):
        assert overlay["properties"][key] == {
            "type": "number", "default": 0.0, "title": title,
        }, key
    fallback = overlay["properties"]["rail_fallback_m"]
    assert {"type": "number"} in fallback["anyOf"] and {"type": "null"} in fallback["anyOf"]
    assert fallback["default"] is None
    # None of the four exports top-level (EXPORTED_MODELS unchanged; contract §1).
    index = json.loads((out / "index.json").read_text())
    assert not {
        "HardwareMonitorTelemetry", "ArmMonitorTelemetry", "TwinOverlayTelemetry",
        "MaintenanceProgress",
    } & set(index["models"])
    assert not {
        "HardwareMonitorTelemetry.json", "ArmMonitorTelemetry.json", "TwinOverlayTelemetry.json",
        "MaintenanceProgress.json",
    } & {p.name for p in out.iterdir()}


def test_camera_info_schema_kind_gains_twin(tmp_path):
    """phase-09a: ``CameraInfo.kind`` enum gains "twin" (overlay rows in /api/cameras)."""
    out = tmp_path / "schemas"
    export(out)
    camera = json.loads((out / "CameraInfo.json").read_text())
    fields = {"camera_id", "kind", "label", "resolution", "fps", "live"}
    assert set(camera["properties"]) == fields
    assert set(camera["required"]) == fields  # no defaults: every row spells its kind
    assert camera["properties"]["kind"] == {
        "type": "string", "enum": ["v4l2", "realsense", "sim", "twin"], "title": "Kind",
    }
    assert camera["properties"]["live"] == {"type": "boolean", "title": "Live"}
    assert "$defs" not in camera  # flat body
    # The nested /api/workcell rows share the class (same enum in WorkcellStatus $defs).
    workcell = json.loads((out / "WorkcellStatus.json").read_text())
    nested = workcell["$defs"]["CameraInfo"]["properties"]["kind"]
    assert nested["enum"] == ["v4l2", "realsense", "sim", "twin"]


def test_telemetry_schema_arm_telemetry_gains_fault_fields(tmp_path):
    """phase-09b: ``ArmTelemetry.fault_detail`` / ``.recovering`` are defaulted (additive)."""
    out = tmp_path / "schemas"
    export(out)
    telemetry = json.loads((out / "TelemetryMsg.json").read_text())
    arm = telemetry["$defs"]["ArmTelemetry"]
    assert set(arm["properties"]) == {
        "arm_id", "connected", "q", "rail_pos_m", "ee_pose", "gripper_open_frac",
        "error_code", "warn_code", "stale", "goto", "fault_detail", "recovering",
        "collision_sensitivity",  # 2026-09-11: the operator's in-session level (additive)
    }
    assert set(arm["required"]) == {
        "arm_id", "connected", "q", "rail_pos_m", "ee_pose", "gripper_open_frac", "error_code",
    }
    assert arm["properties"]["fault_detail"] == {
        "type": "string", "default": "", "title": "Fault Detail",
    }
    assert arm["properties"]["collision_sensitivity"] == {
        "anyOf": [{"type": "integer"}, {"type": "null"}], "default": None,
        "title": "Collision Sensitivity",
    }
    assert arm["properties"]["recovering"] == {
        "type": "boolean", "default": False, "title": "Recovering",
    }
    # The pre-09b additive fields keep their shapes.
    assert arm["properties"]["warn_code"] == {"type": "integer", "default": 0, "title": "Warn Code"}
    assert arm["properties"]["stale"] == {"type": "boolean", "default": False, "title": "Stale"}
    assert arm["properties"]["error_code"] == {"type": "integer", "title": "Error Code"}


def test_arm_maintenance_schemas(tmp_path):
    """phase-09b: the maintenance body + result export top-level; the result nests
    ArmMonitorTelemetry (same class as TelemetryMsg's $defs). phase-09c adds the
    ``home_rail`` op, the ``dry_run`` flag and the nested ``RailSweepVerdict``; phase-09d
    ``status`` / ``job_id`` and the nested ``PrePositionPlan`` / ``MaintenanceProgress``."""
    out = tmp_path / "schemas"
    export(out)
    request = json.loads((out / "ArmMaintenanceRequest.json").read_text())
    assert set(request["properties"]) == {"op", "dry_run", "collision_sensitivity"}
    assert request["required"] == ["op"]
    assert request["properties"]["op"] == {
        "type": "string",
        "enum": ["clear_errors", "apply_backstops", "recover", "home_rail",
                 "set_collision_sensitivity"],  # 2026-09-11
        "title": "Op",
    }
    assert request["properties"]["dry_run"] == {
        "type": "boolean", "default": False, "title": "Dry Run",
    }
    # 2026-09-11: the level rides the body, bounded 1..3 IN THE SCHEMA (the UI's dropdown
    # offers exactly those; the runtime answers 422 for anything else), nullable + defaulted
    # so every other op's body is unchanged. The "required for set_collision_sensitivity"
    # rule is a pydantic after-validator, invisible to JSON Schema.
    assert request["properties"]["collision_sensitivity"] == {
        "anyOf": [{"type": "integer", "minimum": 1, "maximum": 3}, {"type": "null"}],
        "default": None, "title": "Collision Sensitivity",
    }
    assert "$defs" not in request  # flat body

    result = json.loads((out / "ArmMaintenanceResult.json").read_text())
    assert set(result["properties"]) == {
        "arm_id", "op", "path", "ok", "detail", "sdk_codes", "warnings", "before", "after",
        "rail_sweep",
        "status", "job_id",  # phase-09d
        "collision_sensitivity",  # 2026-09-11: the level written (unbounded here: read-back)
    }
    assert result["properties"]["collision_sensitivity"] == {
        "anyOf": [{"type": "integer"}, {"type": "null"}],
        "default": None, "title": "Collision Sensitivity",
    }
    assert set(result["required"]) == {"arm_id", "op", "path", "ok"}
    assert result["properties"]["arm_id"] == {"type": "string", "title": "Arm Id"}
    assert result["properties"]["op"] == request["properties"]["op"]
    assert result["properties"]["path"] == {
        "type": "string", "enum": ["monitor", "session"], "title": "Path",
    }
    assert result["properties"]["ok"] == {"type": "boolean", "title": "Ok"}
    assert result["properties"]["detail"] == {"type": "string", "default": "", "title": "Detail"}
    assert result["properties"]["sdk_codes"] == {
        "type": "object", "additionalProperties": {"type": "integer"}, "default": {},
        "title": "Sdk Codes",
    }
    assert result["properties"]["warnings"] == {
        "type": "array", "items": {"type": "string"}, "default": [], "title": "Warnings",
    }
    for key in ("before", "after"):
        prop = result["properties"][key]
        assert {"$ref": "#/$defs/ArmMonitorTelemetry"} in prop["anyOf"], key
        assert {"type": "null"} in prop["anyOf"] and prop["default"] is None, key
    # The nested monitor row is the phase-09a class plus the 09b read-back, byte-identical
    # to the one riding TelemetryMsg.
    telemetry = json.loads((out / "TelemetryMsg.json").read_text())
    assert result["$defs"]["ArmMonitorTelemetry"] == telemetry["$defs"]["ArmMonitorTelemetry"]
    assert {"collision_sensitivity", "tcp_load_kg", "backstops_match", "maintenance_busy"} <= set(
        result["$defs"]["ArmMonitorTelemetry"]["properties"]
    )
    # phase-09c: the sweep verdict rides the result's $defs (not a top-level export).
    sweep = result["properties"]["rail_sweep"]
    assert {"$ref": "#/$defs/RailSweepVerdict"} in sweep["anyOf"]
    assert {"type": "null"} in sweep["anyOf"] and sweep["default"] is None
    assert set(result["$defs"]) == {
        "ArmMonitorTelemetry", "RailSweepVerdict",
        "PrePositionPlan", "MaintenanceProgress",  # phase-09d (via the verdict / monitor row)
    }
    verdict = result["$defs"]["RailSweepVerdict"]
    assert set(verdict["properties"]) == {
        "scene_id", "inflation_m", "step_m", "travel_m", "clear",
        "first_blocked_m", "first_blocked_pair",
        "min_clearance_m", "min_clearance_at_m", "min_clearance_pair",
        "q_checked", "other_arms", "assumptions", "sample_seq",
        "pre_position",  # phase-09d
    }
    assert set(verdict["required"]) == {"scene_id", "inflation_m", "step_m", "clear"}
    assert verdict["properties"]["travel_m"] == {
        "type": "number", "default": 0.65, "title": "Travel M",
    }
    assert verdict["properties"]["clear"] == {"type": "boolean", "title": "Clear"}
    for key, title in (("first_blocked_m", "First Blocked M"),
                       ("min_clearance_m", "Min Clearance M"),
                       ("min_clearance_at_m", "Min Clearance At M")):
        prop = verdict["properties"][key]
        assert {"type": "number"} in prop["anyOf"] and {"type": "null"} in prop["anyOf"], key
        assert prop["default"] is None and prop["title"] == title, key
    for key in ("first_blocked_pair", "min_clearance_pair", "assumptions"):
        assert verdict["properties"][key]["type"] == "array", key
        assert verdict["properties"][key]["items"] == {"type": "string"}, key
        assert verdict["properties"][key]["default"] == [], key
    assert verdict["properties"]["q_checked"] == {
        "type": "array", "items": {"type": "number"}, "default": [], "title": "Q Checked",
    }
    assert verdict["properties"]["other_arms"] == {
        "type": "object", "additionalProperties": {"type": "array", "items": {"type": "number"}},
        "default": {}, "title": "Other Arms",
    }
    assert verdict["properties"]["sample_seq"] == {
        "type": "integer", "default": 0, "title": "Sample Seq",
    }
    index = json.loads((out / "index.json").read_text())
    assert {"ArmMaintenanceRequest", "ArmMaintenanceResult"} <= set(index["models"])
    assert "RailSweepVerdict" not in index["models"]
    assert not (out / "RailSweepVerdict.json").exists()


def test_home_rail_planning_schemas(tmp_path):
    """phase-09d: the result gains ``status`` / ``job_id``; PrePositionPlan rides
    RailSweepVerdict.pre_position and MaintenanceProgress rides ArmMonitorTelemetry.maintenance -
    both nested (never top-level), both nullable with default None."""
    out = tmp_path / "schemas"
    export(out)
    result = json.loads((out / "ArmMaintenanceResult.json").read_text())
    assert result["properties"]["status"] == {
        "type": "string", "enum": ["done", "accepted", "refused"], "default": "done",
        "title": "Status",
    }
    job_id = result["properties"]["job_id"]
    assert {"type": "string"} in job_id["anyOf"] and {"type": "null"} in job_id["anyOf"]
    assert job_id["default"] is None and job_id["title"] == "Job Id"
    assert not {"status", "job_id"} & set(result["required"])  # additive
    verdict = result["$defs"]["RailSweepVerdict"]
    pre = verdict["properties"]["pre_position"]
    assert {"$ref": "#/$defs/PrePositionPlan"} in pre["anyOf"]
    assert {"type": "null"} in pre["anyOf"] and pre["default"] is None
    assert "pre_position" not in verdict["required"]
    plan = result["$defs"]["PrePositionPlan"]
    assert set(plan["properties"]) == {
        "needed", "source", "target_q", "waypoints", "duration_s", "checked_rail_positions",
        "clear", "detail",
    }
    assert plan["required"] == ["needed"]
    assert plan["properties"]["needed"] == {"type": "boolean", "title": "Needed"}
    assert plan["properties"]["source"] == {
        "type": "string", "enum": ["current", "keyframe", "home", "search"],
        "default": "current", "title": "Source",
    }
    assert plan["properties"]["target_q"] == {
        "type": "array", "items": {"type": "number"}, "default": [], "title": "Target Q",
    }
    for key, title in (("waypoints", "Waypoints"),
                       ("checked_rail_positions", "Checked Rail Positions")):
        assert plan["properties"][key] == {"type": "integer", "default": 0, "title": title}, key
    assert plan["properties"]["duration_s"] == {
        "type": "number", "default": 0.0, "title": "Duration S",
    }
    assert plan["properties"]["clear"] == {"type": "boolean", "default": True, "title": "Clear"}
    assert plan["properties"]["detail"] == {"type": "string", "default": "", "title": "Detail"}
    # MaintenanceProgress: one class, byte-identical in the result's nested monitor row and in
    # TelemetryMsg; op shares the request's vocabulary, phase is the job's ordered enum.
    telemetry = json.loads((out / "TelemetryMsg.json").read_text())
    progress = telemetry["$defs"]["MaintenanceProgress"]
    assert result["$defs"]["MaintenanceProgress"] == progress
    assert set(progress["properties"]) == {
        "op", "job_id", "phase", "detail", "progress", "started_at",
    }
    assert progress["required"] == ["op", "job_id", "phase"]
    assert progress["properties"]["op"] == result["properties"]["op"]
    assert progress["properties"]["phase"] == {
        "type": "string",
        "enum": ["queued", "sweeping", "planning", "connecting", "positioning",
                 "homing", "verifying", "done", "failed"],
        "title": "Phase",
    }
    assert progress["properties"]["job_id"] == {"type": "string", "title": "Job Id"}
    assert progress["properties"]["detail"] == {"type": "string", "default": "", "title": "Detail"}
    assert progress["properties"]["progress"] == {
        "type": "number", "default": 0.0, "title": "Progress",
    }
    started = progress["properties"]["started_at"]
    assert {"type": "number"} in started["anyOf"] and {"type": "null"} in started["anyOf"]
    assert started["default"] is None
    # The nested monitor row in the result carries the field too (same class as TelemetryMsg).
    row = result["$defs"]["ArmMonitorTelemetry"]["properties"]["maintenance"]
    assert {"$ref": "#/$defs/MaintenanceProgress"} in row["anyOf"] and row["default"] is None
    # Neither exports top-level (EXPORTED_MODELS unchanged; contract §1).
    index = json.loads((out / "index.json").read_text())
    assert not {"PrePositionPlan", "MaintenanceProgress"} & set(index["models"])
    assert not {"PrePositionPlan.json", "MaintenanceProgress.json"} & {
        p.name for p in out.iterdir()
    }


def test_session_spec_and_info_speed_scale_schema(tmp_path):
    """phase-09c (D2): SessionSpec.speed_scale is (0, 1] with default 1.0; SessionInfo echoes
    it and the workcell kind, both defaulted (additive)."""
    out = tmp_path / "schemas"
    export(out)
    spec = json.loads((out / "SessionSpec.json").read_text())
    assert spec["properties"]["speed_scale"] == {
        "type": "number", "default": 1.0, "exclusiveMinimum": 0, "maximum": 1,
        "title": "Speed Scale",
    }
    assert "speed_scale" not in spec["required"]
    assert spec["required"] == ["mode", "kind", "arms", "frames"]
    info = json.loads((out / "SessionInfo.json").read_text())
    assert set(info["properties"]) == {
        "session_id", "epoch", "mode", "arms", "streams", "state", "kind", "speed_scale",
        "policy_source",  # phase-12 echo (additive)
        "fault_detail",  # 2026-09-08 (additive): the session-level notice
        "online_dagger",  # phase-14 echo (additive; 15-online-dagger §5)
    }
    assert info["properties"]["fault_detail"] == {
        "type": "string", "default": "", "title": "Fault Detail",
    }
    assert set(info["required"]) == {"session_id", "epoch", "mode", "arms", "streams", "state"}
    assert info["properties"]["kind"] == {
        "type": "string", "enum": ["hardware", "sim"], "default": "sim", "title": "Kind",
    }
    assert info["properties"]["kind"]["enum"] == spec["properties"]["kind"]["enum"]
    assert info["properties"]["speed_scale"] == spec["properties"]["speed_scale"]
    # phase-14: the echoed Online DAgger block is the only nested model of the response.
    assert set(info["$defs"]) == {"OnlineDaggerConfig"}


def test_telemetry_schema_session_bringup_rows(tmp_path):
    """phase-09c (D5): ArmBringupTelemetry rides TelemetryMsg's $defs via
    SessionTelemetry.bringup (nullable list, default None)."""
    out = tmp_path / "schemas"
    export(out)
    telemetry = json.loads((out / "TelemetryMsg.json").read_text())
    assert "ArmBringupTelemetry" in telemetry["$defs"]
    session = telemetry["$defs"]["SessionTelemetry"]
    assert set(session["properties"]) == {
        "state", "start_from_progress", "plan_status", "trainer_alive", "bringup",
        "translate_frame",  # additive 2026-09-08 (04-runtime §6)
        "fault_detail",  # additive 2026-09-08 (04-runtime §13.3): session-level notice
        # additive 2026-09-09 evening (04-runtime §13.2 orphaned session / §13.3): the live
        # session's identity for the Welcome page + the runtime's auto-end notice
        "session_id", "mode", "kind", "auto_ended",
    }
    assert "SessionAutoEndNotice" in telemetry["$defs"]
    notice = telemetry["$defs"]["SessionAutoEndNotice"]
    assert set(notice["required"]) == {"session_id", "mode", "kind", "ended_at", "reason"}
    assert session["properties"]["auto_ended"]["default"] is None
    assert {"$ref": "#/$defs/SessionAutoEndNotice"} in session["properties"]["auto_ended"]["anyOf"]
    assert session["properties"]["fault_detail"] == {
        "type": "string", "default": "", "title": "Fault Detail",
    }
    assert session["required"] == ["state"]
    frame = session["properties"]["translate_frame"]
    assert {"type": "string", "enum": ["camera", "world", "base"]} in frame["anyOf"]
    assert {"type": "null"} in frame["anyOf"] and frame["default"] is None
    bringup = session["properties"]["bringup"]
    assert {"type": "array", "items": {"$ref": "#/$defs/ArmBringupTelemetry"}} in bringup["anyOf"]
    assert {"type": "null"} in bringup["anyOf"] and bringup["default"] is None
    row = telemetry["$defs"]["ArmBringupTelemetry"]
    assert set(row["properties"]) == {"arm_id", "step", "status", "detail"}
    assert row["required"] == ["arm_id", "step", "status"]
    assert row["properties"]["status"] == {
        "type": "string", "enum": ["pending", "ok", "warning", "error"], "title": "Status",
    }
    assert row["properties"]["step"] == {"type": "string", "title": "Step"}
    assert row["properties"]["detail"] == {"type": "string", "default": "", "title": "Detail"}
    # Not a top-level export (EXPORTED_MODELS unchanged; contract §1).
    index = json.loads((out / "index.json").read_text())
    assert "ArmBringupTelemetry" not in index["models"]
    assert not (out / "ArmBringupTelemetry.json").exists()


def test_online_dagger_schemas(tmp_path):
    """phase-14 (15-online-dagger §5-§7): OnlineDaggerConfig rides SessionSpec / SessionInfo
    $defs AND exports top-level (the sheet validates against it); TrainerStatusAnnounce,
    OnlineDaggerAnnounce and the REST rows export top-level; OnlineDaggerStatus rides
    TelemetryMsg's $defs through DaggerStatus.online_dagger (never top-level). Nothing of
    the v1.0 shell's algorithm-specific models is exported any more."""
    out = tmp_path / "schemas"
    export(out)
    index = json.loads((out / "index.json").read_text())
    new_models = {
        "OnlineDaggerConfig", "OnlineDaggerSessionInfo", "DatasetLayoutInfo",
        "DatasetNamespaceInfo", "TrainerStatusAnnounce", "OnlineDaggerAnnounce",
    }
    assert new_models <= set(index["models"])
    assert {f"{m}.json" for m in new_models} <= {p.name for p in out.iterdir()}
    assert "OnlineDaggerStatus" not in index["models"]
    assert not (out / "OnlineDaggerStatus.json").exists()
    # every exported model that mentions DAgger is one of the shell's; the v1.0 names
    # (algorithm config / paths / reference-gradient status) are gone with their files
    assert {m for m in index["models"] if "Dagger" in m} == {
        "OnlineDaggerConfig", "OnlineDaggerSessionInfo", "OnlineDaggerAnnounce",
    }
    assert not [p.name for p in out.iterdir() if "Grad" in p.name]

    # SessionSpec.online_dagger: nullable $ref, default None, not required; field order kept.
    spec = json.loads((out / "SessionSpec.json").read_text())
    block = spec["properties"]["online_dagger"]
    assert {"$ref": "#/$defs/OnlineDaggerConfig"} in block["anyOf"]
    assert {"type": "null"} in block["anyOf"] and block["default"] is None
    assert spec["required"] == ["mode", "kind", "arms", "frames"]
    # (``properties`` keys are sorted in the export; field ORDER is pinned by the
    # ``model_fields`` assertions in test_protocol.py — here we pin the SET and the shapes.)
    assert {"action_filter", "return_to_start", "online_dagger"} <= set(spec["properties"])
    assert set(spec["$defs"]) == {"ActionFilterConfig", "OnlineDaggerConfig"}
    cfg = spec["$defs"]["OnlineDaggerConfig"]
    cfg_top = json.loads((out / "OnlineDaggerConfig.json").read_text())
    assert cfg["properties"] == cfg_top["properties"]  # same class, nested and top-level
    assert cfg["required"] == cfg_top["required"] == ["session_name"]
    assert "$defs" not in cfg_top  # flat body
    assert set(cfg["properties"]) == {
        "session_name", "resume", "pause_while_training", "wait_for_trainer_ready",
    }
    # the slug pattern rides the schema (the UI slugs against the same regex), with the
    # length cap (64 — the name becomes a directory)
    assert cfg["properties"]["session_name"] == {
        "type": "string", "pattern": r"^[A-Za-z0-9][A-Za-z0-9_\-]*$", "maxLength": 64,
        "title": "Session Name",
    }
    # unknown keys are a 422 (extra="forbid") — the sheet emits every key by name
    assert cfg["additionalProperties"] is False and cfg_top["additionalProperties"] is False
    assert cfg["properties"]["resume"] == {
        "type": "boolean", "default": False, "title": "Resume",
    }
    for key, title in (("pause_while_training", "Pause While Training"),
                       ("wait_for_trainer_ready", "Wait For Trainer Ready")):
        assert cfg["properties"][key] == {"type": "boolean", "default": True, "title": title}
    # SessionInfo echoes the same class (byte-identical $defs entry).
    info = json.loads((out / "SessionInfo.json").read_text())
    assert info["$defs"]["OnlineDaggerConfig"] == cfg
    assert info["properties"]["online_dagger"]["default"] is None

    # Telemetry: DaggerStatus.online_dagger -> OnlineDaggerStatus -> TrainerStatusAnnounce,
    # all in one $defs; no iteration / history-row model.
    telemetry = json.loads((out / "TelemetryMsg.json").read_text())
    assert {"OnlineDaggerStatus", "TrainerStatusAnnounce"} <= set(telemetry["$defs"])
    assert {n for n in telemetry["$defs"] if "Dagger" in n} == {
        "DaggerStatus", "OnlineDaggerStatus",
    }
    dagger = telemetry["$defs"]["DaggerStatus"]
    od = dagger["properties"]["online_dagger"]
    assert {"$ref": "#/$defs/OnlineDaggerStatus"} in od["anyOf"]
    assert {"type": "null"} in od["anyOf"] and od["default"] is None
    assert "online_dagger" not in dagger["required"]
    assert {"policy_stale", "online_dagger"} <= set(dagger["properties"])
    status = telemetry["$defs"]["OnlineDaggerStatus"]
    assert set(status["properties"]) == {
        "session_name", "phase", "rollouts_saved", "detail", "trainer_alive", "trainer_age_s",
        "trainer", "policy_version_acting", "expert_frames_session", "novice_frames_session",
        "session_dir",
    }
    assert status["required"] == ["session_name", "phase", "rollouts_saved"]
    assert status["properties"]["phase"] == {
        "type": "string", "enum": ["waiting_trainer", "rollout", "training", "error"],
        "title": "Phase",
    }
    assert status["properties"]["rollouts_saved"] == {
        "type": "integer", "title": "Rollouts Saved",
    }
    trainer = status["properties"]["trainer"]
    assert {"$ref": "#/$defs/TrainerStatusAnnounce"} in trainer["anyOf"]
    assert {"type": "null"} in trainer["anyOf"] and trainer["default"] is None
    assert status["properties"]["session_dir"] == {
        "type": "string", "default": "", "title": "Session Dir",
    }
    for key, title in (("expert_frames_session", "Expert Frames Session"),
                       ("novice_frames_session", "Novice Frames Session")):
        assert status["properties"][key] == {"type": "integer", "default": 0, "title": title}
    # The trainer payload is the SAME class top-level and nested (byte-identical schema body).
    trainer_top = json.loads((out / "TrainerStatusAnnounce.json").read_text())
    nested = telemetry["$defs"]["TrainerStatusAnnounce"]
    assert nested["properties"] == trainer_top["properties"]
    assert nested["required"] == trainer_top["required"] == ["trainer_id", "node_version"]
    assert set(trainer_top["properties"]) == {
        "mavis_schema", "trainer_id", "node_version", "state", "session_id", "policy_version",
        "progress", "metrics", "detail", "uptime_s",
    }
    assert trainer_top["properties"]["state"] == {
        "type": "string", "enum": ["idle", "preparing", "training", "ready", "error"],
        "default": "idle", "title": "State",
    }
    assert trainer_top["properties"]["progress"] == {
        "type": "number", "default": 0.0, "minimum": 0, "maximum": 1, "title": "Progress",
    }
    assert trainer_top["properties"]["metrics"] == {
        "type": "object", "additionalProperties": {"type": "number"}, "default": {},
        "title": "Metrics",
    }
    assert trainer_top["properties"]["uptime_s"] == {
        "type": "number", "default": 0.0, "minimum": 0, "title": "Uptime S",
    }
    session_id = trainer_top["properties"]["session_id"]
    assert {"type": "string"} in session_id["anyOf"] and {"type": "null"} in session_id["anyOf"]
    assert session_id["default"] is None
    assert "$defs" not in trainer_top  # a flat payload: no nested block

    # External: SessionAnnounce.online_dagger + OnlineDaggerAnnounce,
    # PolicySpecAnnounce.capabilities.
    announce = json.loads((out / "SessionAnnounce.json").read_text())
    assert {"deprecated_keys", "online_dagger"} <= set(announce["properties"])
    oda = announce["properties"]["online_dagger"]
    assert {"$ref": "#/$defs/OnlineDaggerAnnounce"} in oda["anyOf"]
    assert {"type": "null"} in oda["anyOf"] and oda["default"] is None
    assert {"OnlineDaggerAnnounce", "OnlineDaggerConfig", "SessionSpec"} <= set(announce["$defs"])
    paths = json.loads((out / "OnlineDaggerAnnounce.json").read_text())
    assert announce["$defs"]["OnlineDaggerAnnounce"]["properties"] == paths["properties"]
    assert paths["required"] == [  # ``required`` is a list: field order survives the export
        "session_name", "session_dir", "rollouts_dir",
    ]
    assert set(paths["properties"]) == set(paths["required"])  # nothing optional
    assert "$defs" not in paths  # flat body
    policy_spec = json.loads((out / "PolicySpecAnnounce.json").read_text())
    assert "capabilities" in policy_spec["properties"]
    assert policy_spec["properties"]["capabilities"] == {
        "type": "array", "items": {"type": "string"}, "default": [], "title": "Capabilities",
    }
    assert "capabilities" not in policy_spec["required"]
    # 14-dora v1.3 (2026-09-11): PolicySpecModel.arms / action_frames, appended last, defaulted
    model = policy_spec["$defs"]["PolicySpecModel"]  # (properties are sort_keys-ordered here;
    #   the field ORDER is pinned by test_protocol)
    assert model["properties"]["arms"] == {
        "type": "array", "items": {"type": "string"}, "default": [], "title": "Arms",
    }
    assert model["properties"]["action_frames"] == {
        "type": "object", "additionalProperties": {"type": "string"}, "default": {},
        "title": "Action Frames",
    }
    assert "arms" not in model["required"] and "action_frames" not in model["required"]

    # REST rows: GET /api/online_dagger/sessions and GET /api/datasets/layout; DatasetInfo
    # gains namespace / path (defaulted).
    row = json.loads((out / "OnlineDaggerSessionInfo.json").read_text())
    assert set(row["properties"]) == {
        "session_name", "path", "created_at", "task", "rollouts", "last_used_at",
    }
    assert row["required"] == ["session_name", "path", "created_at", "task", "rollouts"]
    task = row["properties"]["task"]
    assert {"type": "string"} in task["anyOf"] and {"type": "null"} in task["anyOf"]
    assert "default" not in task  # required-but-nullable, like SessionSpec.task
    assert row["properties"]["rollouts"] == {"type": "integer", "title": "Rollouts"}
    assert row["properties"]["last_used_at"]["default"] is None
    assert "$defs" not in row
    layout = json.loads((out / "DatasetLayoutInfo.json").read_text())
    assert layout["required"] == ["default_namespace", "generic_root", "namespaces"]
    assert set(layout["properties"]) == set(layout["required"])
    assert layout["properties"]["namespaces"] == {
        "type": "object", "additionalProperties": {"$ref": "#/$defs/DatasetNamespaceInfo"},
        "title": "Namespaces",
    }
    ns = json.loads((out / "DatasetNamespaceInfo.json").read_text())
    assert layout["$defs"]["DatasetNamespaceInfo"]["properties"] == ns["properties"]
    assert set(ns["properties"]) == {"root", "subdir"} and ns["required"] == ["root"]
    assert ns["properties"]["subdir"]["default"] is None
    dataset = json.loads((out / "DatasetInfo.json").read_text())
    assert {"namespace", "path"} <= set(dataset["properties"])
    for key, title in (("namespace", "Namespace"), ("path", "Path")):
        assert dataset["properties"][key] == {"type": "string", "default": "", "title": title}
        assert key not in dataset["required"]


def test_goto_profile_args_schema(tmp_path):
    """2026-09-08: ``GotoProfileArgs`` exports top-level, closed, with ``profile_id``
    required and its ProfileStore-charset ``pattern`` exposed so the UI can validate
    before sending; ``goto_profile`` is the last ``ActionMsg.name`` enum member."""
    out = tmp_path / "schemas"
    export(out)
    schema = json.loads((out / "GotoProfileArgs.json").read_text())
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["profile_id"]
    assert set(schema["properties"]) == {"profile_id"}
    assert schema["properties"]["profile_id"] == {
        "type": "string", "pattern": r"^[A-Za-z0-9_\-]+$", "title": "Profile Id",
    }
    assert "$defs" not in schema
    for name in ("ActionMsg", "AckMsg"):
        enum = json.loads((out / f"{name}.json").read_text())["properties"]["name"]["enum"]
        assert enum[-1] == "goto_profile" and enum[-4:-1] == ["takeover", "handback", "train_now"]
    index = json.loads((out / "index.json").read_text())
    assert "GotoProfileArgs" in index["models"]
