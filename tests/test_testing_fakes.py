"""Canonical fakes behave per design doc 01-core §18."""

from __future__ import annotations

import numpy as np
import pytest

from apollo_xarm7_core.errors import CommandError, RailUnavailableError
from apollo_xarm7_core.interfaces.arm import ArmInterface
from apollo_xarm7_core.interfaces.camera import CameraInterface
from apollo_xarm7_core.interfaces.teleop import HeldState, TeleopInput
from apollo_xarm7_core.interfaces.workcell import WorkcellInterface
from apollo_xarm7_core.se3 import RAIL_TRAVEL_M
from apollo_xarm7_core.state import GripperCommand
from apollo_xarm7_core.testing import FakeArm, FakeCamera, FakeWorkcell


def test_fake_arm_slews_toward_target() -> None:
    arm = FakeArm("arm0", max_joint_speed_rad_s=1.0)
    arm.connect()
    target = np.full(7, 0.5)
    arm.command_joints(target)
    state0 = arm.get_state()
    assert np.allclose(state0.q, 0.0)  # command_joints is non-blocking

    arm.step(0.1)
    s1 = arm.get_state()
    assert np.allclose(s1.q, 0.1)  # slew-limited: 1.0 rad/s * 0.1 s
    assert np.allclose(s1.dq, 1.0)
    assert s1.t_mono == pytest.approx(0.1)

    for _ in range(10):
        arm.step(0.1)
    s2 = arm.get_state()
    assert np.allclose(s2.q, target)  # converged, no overshoot
    assert s2.mode == 1 and s2.state == 0 and not s2.stale


def test_fake_arm_latest_wins_and_stop() -> None:
    arm = FakeArm()
    arm.command_joints(np.full(7, 1.0))
    arm.command_joints(np.full(7, -1.0))  # overwrites
    arm.step(0.1)
    assert np.all(arm.get_state().q < 0.0)
    arm.stop()
    arm.stop()  # idempotent
    q = arm.get_state().q.copy()
    arm.step(1.0)
    assert np.allclose(arm.get_state().q, q)


def test_fake_arm_command_validation() -> None:
    arm = FakeArm()
    with pytest.raises(CommandError):
        arm.command_joints(np.zeros(8))  # wrong shape for a 7-dof arm
    with pytest.raises(CommandError):
        arm.command_joints(np.array([np.nan] + [0.0] * 6))


def test_fake_arm_rail_clamp() -> None:
    arm = FakeArm("rail_arm", has_rail=True, max_rail_speed_m_s=10.0)
    assert arm.dof == 8 and arm.has_rail
    arm.command_rail(10.0)  # clamped to [0, RAIL_TRAVEL_M]
    arm.step(1.0)
    state = arm.get_state()
    assert state.rail_pos_m == pytest.approx(RAIL_TRAVEL_M)
    assert state.q[7] == pytest.approx(RAIL_TRAVEL_M)
    assert state.dq[7] == 0.0  # rail velocity unobservable
    arm.command_rail(-3.0)
    arm.step(1.0)
    assert arm.get_state().rail_pos_m == pytest.approx(0.0)
    # command_joints rail slot clamps too
    arm.command_joints(np.concatenate([np.zeros(7), [99.0]]))
    arm.step(100.0)
    assert arm.get_state().rail_pos_m == pytest.approx(RAIL_TRAVEL_M)


def test_fake_arm_rail_unavailable() -> None:
    arm = FakeArm(has_rail=False)
    with pytest.raises(RailUnavailableError):
        arm.command_rail(0.1)


def test_fake_arm_gripper_and_error_injection() -> None:
    arm = FakeArm(gripper_force_capable=True)
    assert arm.gripper_force_capable
    cmd = GripperCommand(open_frac=0.25, force=0.5)
    arm.command_gripper(cmd)
    assert arm.gripper_commands == [cmd]
    assert arm.get_state().gripper.open_frac == pytest.approx(0.25)

    arm.inject_error(31)
    assert arm.get_state().error_code == 31
    arm.clear_errors()
    assert arm.get_state().error_code == 0


def test_fake_camera_seq_increments() -> None:
    cam = FakeCamera("cam0", resolution=(32, 24), fps=10.0)
    assert cam.latest() is None  # never started
    cam.start()
    f0, f1, f2 = cam.latest(), cam.latest(), cam.latest()
    assert [f.seq for f in (f0, f1, f2)] == [0, 1, 2]
    assert f0.rgb.shape == (24, 32, 3) and f0.rgb.dtype == np.uint8
    assert f1.t_mono == pytest.approx(f0.t_mono + 0.1)
    assert not np.array_equal(f0.rgb, f1.rgb)  # deterministic gradient shifts
    cam.stop()
    cam.stop()  # idempotent
    assert cam.latest() is None


def test_fake_workcell_start_stop_idempotent() -> None:
    cell = FakeWorkcell(
        arms={"a": FakeArm("a"), "b": FakeArm("b", has_rail=True)},
        cameras={"cam0": FakeCamera("cam0")},
    )
    assert cell.kind == "sim"
    cell.stop()  # safe before start
    cell.start()
    cell.start()  # idempotent
    assert cell.started and cell.start_calls == 2
    assert all(arm.connected for arm in cell.arms.values())  # type: ignore[attr-defined]
    states = cell.states()
    assert set(states) == {"a", "b"}
    assert states["b"].rail_pos_m == pytest.approx(0.0)
    cell.stop()
    cell.stop()
    assert not cell.started and cell.stop_calls == 3
    assert not any(arm.connected for arm in cell.arms.values())  # type: ignore[attr-defined]


def test_fakes_satisfy_interfaces() -> None:
    assert isinstance(FakeArm(), ArmInterface)
    assert isinstance(FakeCamera(), CameraInterface)
    assert isinstance(FakeWorkcell(), WorkcellInterface)

    class StubTeleop:
        def latest(self) -> HeldState | None:
            return HeldState(held=frozenset({"KeyW"}), seq=1, rx_mono=0.0)

    assert isinstance(StubTeleop(), TeleopInput)  # runtime_checkable Protocol
    assert not isinstance(object(), TeleopInput)
    held = StubTeleop().latest()
    assert held is not None and held.held == frozenset({"KeyW"})
