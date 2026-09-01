"""Config + safety schema tests (design doc 01-core §6, §7, §18)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from apollo_xarm7_core.errors import ConfigError
from apollo_xarm7_core.schemas.config import PoseModel, load_workcell_config
from apollo_xarm7_core.schemas.safety import CollisionReport, SafetyConfig
from apollo_xarm7_core.types import Pose

FIXTURES = Path(__file__).parent / "fixtures"


# -- valid fixtures ------------------------------------------------------------


def test_sim_1arm_fixture_loads():
    cfg = load_workcell_config(FIXTURES / "sim_1arm.yaml")
    assert cfg.kind == "sim"
    assert cfg.sim_scene == "table_single"
    assert [a.id for a in cfg.arms] == ["arm0"]
    assert cfg.arms[0].ip is None
    assert [c.kind for c in cfg.cameras] == ["sim"]
    assert cfg.safety.enabled is True  # default SafetyConfig


def test_hardware_2arm_fixture_loads():
    cfg = load_workcell_config(FIXTURES / "hardware_2arm.yaml")
    assert cfg.kind == "hardware"
    assert cfg.digital_twin_scene == "twin_dual"
    assert [a.id for a in cfg.arms] == ["arm0", "arm1"]
    assert cfg.arms[0].expect_rail == "yes"  # quoted in YAML; not a bool
    assert cfg.arms[1].expect_rail == "no"
    assert cfg.arms[0].ip == "192.168.1.201"
    assert cfg.arms[1].gripper == "xarm_g2"
    assert cfg.safety.warn_clearance_m == 0.03


def test_hardware_3arm_4cam_fixture_loads():
    cfg = load_workcell_config(FIXTURES / "hardware_3arm_4cam.yaml")
    assert len(cfg.arms) == 3
    assert len(cfg.cameras) == 4
    kinds = {c.id: c.kind for c in cfg.cameras}
    assert kinds == {
        "wrist0": "v4l2",
        "wrist1": "v4l2",
        "wrist2": "realsense",
        "env": "realsense",
    }
    frames = {c.id: c.extrinsics_frame for c in cfg.cameras}
    assert frames["wrist0"] == "ee:arm0"
    assert frames["wrist2"] == "arm_base:arm2"
    assert frames["env"] == "world"
    assert cfg.cameras[2].intrinsics is not None
    assert cfg.cameras[2].intrinsics.fx == 615.0
    assert cfg.safety.allowed_pairs_extra == [("arm0/link7", "arm1/link7")]


# -- PoseModel bridge ------------------------------------------------------------


def test_pose_model_bridges_and_normalizes():
    pm = PoseModel(position=(1.0, 2.0, 3.0), orientation_wxyz=(-2.0, 0.0, 0.0, 0.0))
    p = pm.to_pose()
    assert isinstance(p, Pose)
    np.testing.assert_allclose(p.position, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(p.orientation, [1.0, 0.0, 0.0, 0.0])  # unit, w >= 0
    back = PoseModel.from_pose(p)
    assert back.position == (1.0, 2.0, 3.0)
    np.testing.assert_allclose(back.orientation_wxyz, (1.0, 0.0, 0.0, 0.0))


# -- failing cross-field validators ---------------------------------------------


def _sim_dict() -> dict:
    return {
        "kind": "sim",
        "sim_scene": "table_single",
        "arms": [{"id": "arm0", "base_in_world": {}}],
        "cameras": [{"id": "cam0", "kind": "sim"}],
    }


def _hardware_dict() -> dict:
    return {
        "kind": "hardware",
        "digital_twin_scene": "twin_dual",
        "arms": [
            {"id": "arm0", "ip": "192.168.1.201", "base_in_world": {}},
            {"id": "arm1", "ip": "192.168.1.202", "base_in_world": {}},
        ],
        "cameras": [
            {"id": "cam0", "kind": "v4l2", "device_path": "/dev/video0"},
        ],
    }


def _drop_sim_scene(d: dict) -> None:
    d.pop("sim_scene")


def _drop_arm_ip(d: dict) -> None:
    d["arms"][0].pop("ip")


def _drop_twin_scene(d: dict) -> None:
    d.pop("digital_twin_scene")


def _disable_safety(d: dict) -> None:
    d["safety"] = {"enabled": False}


def _dup_arm_ids(d: dict) -> None:
    d["arms"][1]["id"] = d["arms"][0]["id"]


def _dup_camera_ids(d: dict) -> None:
    d["cameras"].append(dict(d["cameras"][0]))


def _drop_device_path(d: dict) -> None:
    d["cameras"][0].pop("device_path")


def _realsense_no_serial(d: dict) -> None:
    d["cameras"][0] = {"id": "cam0", "kind": "realsense"}


def _sim_camera_on_hardware(d: dict) -> None:
    d["cameras"][0] = {"id": "cam0", "kind": "sim"}


def _bad_extrinsics_frame(d: dict) -> None:
    d["cameras"][0]["extrinsics_frame"] = "arm_base:"


def _undeclared_extrinsics_ref(d: dict) -> None:
    d["cameras"][0]["extrinsics_frame"] = "camera:nope"


@pytest.mark.parametrize(
    ("base", "mutate"),
    [
        (_sim_dict, _drop_sim_scene),
        (_hardware_dict, _drop_arm_ip),
        (_hardware_dict, _drop_twin_scene),
        (_hardware_dict, _disable_safety),
        (_hardware_dict, _dup_arm_ids),
        (_hardware_dict, _dup_camera_ids),
        (_hardware_dict, _drop_device_path),
        (_hardware_dict, _realsense_no_serial),
        (_hardware_dict, _sim_camera_on_hardware),
        (_hardware_dict, _bad_extrinsics_frame),
        (_hardware_dict, _undeclared_extrinsics_ref),
    ],
    ids=lambda f: getattr(f, "__name__", str(f)),
)
def test_cross_field_validators_reject(tmp_path, base, mutate):
    data = base()
    mutate(data)
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ConfigError) as ei:
        load_workcell_config(path)
    err = ei.value
    assert err.path == str(path)  # carries the file path
    assert err.loc is not None and err.loc.startswith("/")  # JSON-pointer-ish loc
    assert str(path) in str(err)


def test_missing_file_and_bad_yaml_raise_config_error(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(ConfigError) as ei:
        load_workcell_config(missing)
    assert ei.value.path == str(missing)

    bad = tmp_path / "broken.yaml"
    bad.write_text("kind: [unclosed")
    with pytest.raises(ConfigError):
        load_workcell_config(bad)

    scalar = tmp_path / "scalar.yaml"
    scalar.write_text("just a string\n")
    with pytest.raises(ConfigError) as ei:
        load_workcell_config(scalar)
    assert ei.value.loc == "/"


# -- safety model validators (§6) -------------------------------------------------


def test_collision_report_blocked_iff_severity():
    with pytest.raises(ValueError):
        CollisionReport(blocked=True, severity="ok")
    with pytest.raises(ValueError):
        CollisionReport(blocked=False, severity="blocked")
    report = CollisionReport(blocked=True, severity="blocked", min_clearance_m=-0.001)
    assert report.blocked


def test_collision_report_ok_constant_is_benign_and_frozen():
    report = CollisionReport.ok()
    assert report.blocked is False
    assert report.severity == "ok"
    assert report.pairs == [] and report.violations == []
    with pytest.raises(ValidationError):
        report.blocked = True  # frozen (§16)


def test_safety_config_validators():
    with pytest.raises(ValueError):
        SafetyConfig(geom_inflation_m=0.0)
    with pytest.raises(ValueError):
        SafetyConfig(geom_inflation_m=-0.01)
    with pytest.raises(ValueError):
        SafetyConfig(enabled=False, safety_debug=True)
    assert SafetyConfig(enabled=False).safety_debug is False  # sim may disable
