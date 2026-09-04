"""State dataclass validation (design doc 01-core §4, §18)."""

from __future__ import annotations

import numpy as np
import pytest

from apollo_mavis_v2_core.errors import CommandError
from apollo_mavis_v2_core.state import ArmState, CameraFrame, GripperCommand, GripperState
from apollo_mavis_v2_core.types import Pose


def _arm_state(**overrides) -> ArmState:
    kwargs = dict(
        arm_id="arm0",
        q=np.zeros(7),
        dq=np.zeros(7),
        ee_pose=Pose.identity(),
        gripper=GripperState(open_frac=1.0),
        rail_pos_m=None,
        error_code=0,
        warn_code=0,
        mode=1,
        state=0,
        stale=False,
        t_mono=0.0,
        wallclock_ns=0,
    )
    kwargs.update(overrides)
    return ArmState(**kwargs)


# --- shape / dtype rejection ---------------------------------------------------


@pytest.mark.parametrize("shape", [(6,), (9,), (7, 1), ()])
def test_arm_state_rejects_bad_q_shape(shape) -> None:
    with pytest.raises(ValueError):
        _arm_state(q=np.zeros(shape), dq=np.zeros(shape))


def test_arm_state_rejects_dq_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        _arm_state(q=np.zeros(7), dq=np.zeros(8))


def test_arm_state_rejects_non_finite() -> None:
    q = np.zeros(7)
    q[3] = np.nan
    with pytest.raises(ValueError):
        _arm_state(q=q)
    dq = np.zeros(7)
    dq[0] = np.inf
    with pytest.raises(ValueError):
        _arm_state(dq=dq)


def test_camera_frame_rejects_bad_shape_and_dtype() -> None:
    with pytest.raises(ValueError):
        CameraFrame("cam0", rgb=np.zeros((4, 4), dtype=np.uint8), t_mono=0.0, wallclock_ns=0)
    with pytest.raises(ValueError):
        CameraFrame("cam0", rgb=np.zeros((4, 4, 4), dtype=np.uint8), t_mono=0.0, wallclock_ns=0)
    with pytest.raises(ValueError):
        CameraFrame("cam0", rgb=np.zeros((4, 4, 3), dtype=np.float64), t_mono=0.0, wallclock_ns=0)
    frame = CameraFrame("cam0", rgb=np.zeros((4, 4, 3), dtype=np.uint8), t_mono=0.0, wallclock_ns=0)
    assert frame.seq == 0


# --- rail coupling ----------------------------------------------------------------


def test_rail_slot_couples_to_rail_pos_m() -> None:
    q = np.zeros(8)
    q[7] = 0.25
    state = _arm_state(q=q, dq=np.zeros(8), rail_pos_m=0.25)
    assert state.rail_pos_m == pytest.approx(q[7])
    assert state.has_rail and state.dof == 8


def test_rail_mismatch_rejected() -> None:
    q = np.zeros(8)
    q[7] = 0.25
    with pytest.raises(ValueError):
        _arm_state(q=q, dq=np.zeros(8), rail_pos_m=0.30)


def test_eight_dof_requires_rail_pos() -> None:
    with pytest.raises(ValueError):
        _arm_state(q=np.zeros(8), dq=np.zeros(8), rail_pos_m=None)


def test_seven_dof_forbids_rail_pos() -> None:
    with pytest.raises(ValueError):
        _arm_state(q=np.zeros(7), dq=np.zeros(7), rail_pos_m=0.1)


# --- gripper ranges ------------------------------------------------------------------


@pytest.mark.parametrize("open_frac", [-0.1, 1.1, float("nan"), float("inf")])
def test_gripper_command_open_frac_range(open_frac: float) -> None:
    with pytest.raises(CommandError):
        GripperCommand(open_frac=open_frac)


@pytest.mark.parametrize("field", ["force", "speed"])
@pytest.mark.parametrize("value", [-0.5, 1.5, float("nan")])
def test_gripper_command_force_speed_range(field: str, value: float) -> None:
    with pytest.raises(CommandError):
        GripperCommand(open_frac=0.5, **{field: value})


def test_gripper_command_valid() -> None:
    cmd = GripperCommand(open_frac=0.5, force=1.0, speed=0.0)
    assert cmd.open_frac == 0.5 and cmd.force == 1.0 and cmd.speed == 0.0
    assert GripperCommand(open_frac=1.0).force is None


def test_gripper_state_open_frac_range() -> None:
    with pytest.raises(ValueError):
        GripperState(open_frac=2.0)
    assert GripperState(open_frac=0.0).open_frac == 0.0
