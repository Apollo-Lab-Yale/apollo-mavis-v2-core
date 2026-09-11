"""SE3 / quaternion property tests (design doc 01-core §18), hypothesis-driven."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from apollo_mavis_v2_core import se3
from apollo_mavis_v2_core.types import Pose, Twist

# --- strategies (finite floats, non-degenerate quats) -------------------------

_unit = st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False)
_coord = st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False)

vec3 = st.tuples(_coord, _coord, _coord).map(np.array)
directions = (
    st.tuples(_unit, _unit, _unit)
    .map(np.array)
    .filter(lambda v: float(np.linalg.norm(v)) > 1e-3)
)


@st.composite
def rotvecs(draw, max_angle: float = np.pi - 0.1) -> np.ndarray:
    d = draw(directions)
    angle = draw(st.floats(0.0, max_angle, allow_nan=False, allow_infinity=False))
    return d / np.linalg.norm(d) * angle


quats = rotvecs().map(se3.rotvec_to_quat)
raw_quats = (
    st.tuples(_unit, _unit, _unit, _unit)
    .map(np.array)
    .filter(lambda q: float(np.linalg.norm(q)) > 1e-2)
)
poses = st.builds(Pose, vec3, quats)


def _geodesic_to(a: np.ndarray, b: np.ndarray) -> float:
    return se3.quat_geodesic(a, b)


# --- round trips ---------------------------------------------------------------


@settings(deadline=None)
@given(quats)
def test_quat_mat_roundtrip(q: np.ndarray) -> None:
    assert np.allclose(se3.mat_to_quat(se3.quat_to_mat(q)), q, atol=1e-9)


@settings(deadline=None)
@given(quats)
def test_quat_rpy_roundtrip(q: np.ndarray) -> None:
    assert _geodesic_to(se3.rpy_to_quat(se3.quat_to_rpy(q)), q) < 1e-6


def _rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_rpy_composition_is_extrinsic_xyz() -> None:
    """xArm RPY = Rz(yaw) @ Ry(pitch) @ Rx(roll) (2026-09-11; the old Rx.Ry.Rz was wrong).

    The roundtrip test above is composition-agnostic - this pin is what makes the
    convention testable. Two non-zero angles are what tells the two orders apart.
    """
    r, p, y = 0.4, 0.5, -0.6
    expected = se3.mat_to_quat(_rz(y) @ _ry(p) @ _rx(r))
    assert np.allclose(se3.rpy_to_quat((r, p, y)), expected, atol=1e-12)
    wrong = se3.mat_to_quat(_rx(r) @ _ry(p) @ _rz(y))
    assert _geodesic_to(se3.rpy_to_quat((r, p, y)), wrong) > 0.1
    assert np.allclose(se3.quat_to_rpy(expected), [r, p, y], atol=1e-12)


@pytest.mark.parametrize("pitch", [np.pi / 2, -np.pi / 2])
@pytest.mark.parametrize(("roll", "yaw"), [(0.3, 0.7), (-1.2, 0.4), (2.0, -2.5)])
def test_quat_rpy_gimbal_lock_recomposes(pitch: float, roll: float, yaw: float) -> None:
    q = se3.rpy_to_quat((roll, pitch, yaw))
    back = se3.quat_to_rpy(q)
    assert back[2] == 0.0  # yaw folded into roll
    assert _geodesic_to(se3.rpy_to_quat(back), q) < 1e-6


@settings(deadline=None)
@given(rotvecs())
def test_rotvec_roundtrip(r: np.ndarray) -> None:
    assert np.allclose(se3.quat_to_rotvec(se3.rotvec_to_quat(r)), r, atol=1e-8)


@settings(deadline=None)
@given(quats)
def test_quat_xyzw_roundtrip(q: np.ndarray) -> None:
    assert np.allclose(se3.xyzw_to_wxyz(se3.wxyz_to_xyzw(q)), q, atol=1e-12)


# --- rot6d (Zhou et al. 2019) ----------------------------------------------------


@settings(deadline=None)
@given(quats)
def test_rot6d_roundtrip(q: np.ndarray) -> None:
    r6 = se3.quat_to_rot6d(q)
    assert r6.shape == (6,)
    assert _geodesic_to(se3.rot6d_to_quat(r6), q) < 1e-9
    assert se3.rot6d_to_quat(r6)[0] >= 0.0


def test_rot6d_is_first_two_columns_column_major() -> None:
    m = se3.quat_to_mat(se3.rotvec_to_quat(np.array([0.3, -0.7, 1.1])))
    r6 = se3.mat_to_rot6d(m)
    assert np.allclose(r6, [m[0, 0], m[1, 0], m[2, 0], m[0, 1], m[1, 1], m[2, 1]])
    assert np.allclose(se3.rot6d_to_mat(r6), m, atol=1e-12)  # idempotent on a clean pair


def test_rot6d_gram_schmidt_orthonormalizes_a_perturbed_pair() -> None:
    m = se3.quat_to_mat(se3.rotvec_to_quat(np.array([-0.2, 0.9, 0.4])))
    r6 = se3.mat_to_rot6d(m) + np.array([0.05, -0.03, 0.02, 0.04, 0.06, -0.01])
    out = se3.rot6d_to_mat(r6)
    assert np.allclose(out.T @ out, np.eye(3), atol=1e-12)
    assert np.linalg.det(out) == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(out[:, 0], r6[:3] / np.linalg.norm(r6[:3]))  # b1 = normalize(c1)
    assert se3.quat_geodesic(se3.mat_to_quat(out), se3.mat_to_quat(m)) < 0.1


@pytest.mark.parametrize(
    "r6",
    [
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # zero first column
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # zero second column
        [1.0, 0.0, 0.0, 2.0, 0.0, 0.0],  # parallel
        [1.0, 0.0, 0.0, -1.0, 1e-9, 0.0],  # anti-parallel within 1e-6
    ],
)
def test_rot6d_degenerate_raises(r6: list[float]) -> None:
    with pytest.raises(ValueError):
        se3.rot6d_to_mat(r6)


# --- flange <-> TCP ---------------------------------------------------------------


def test_flange_to_tcp_gripper_is_rz_pi_plus_172mm() -> None:
    tcp = se3.flange_to_tcp(Pose.identity(), gripper=True)
    assert np.allclose(tcp.position, [0.0, 0.0, se3.TCP_OFFSET_M])
    assert np.allclose(tcp.orientation, [0.0, 0.0, 0.0, 1.0])  # Rz(pi), wxyz
    assert tuple(se3.FLANGE_TO_TCP_QUAT) == (0.0, 0.0, 0.0, 1.0)


def test_flange_to_tcp_gripperless_is_identity() -> None:
    flange = Pose(np.array([0.3, -0.05, 0.22]), se3.rpy_to_quat((np.pi, 0.0, 0.0)))
    tcp = se3.flange_to_tcp(flange, gripper=False)
    assert np.allclose(tcp.position, flange.position)
    assert np.allclose(tcp.orientation, flange.orientation)


@settings(deadline=None)
@given(poses, st.booleans())
def test_flange_tcp_roundtrip(flange: Pose, gripper: bool) -> None:
    back = se3.tcp_to_flange(se3.flange_to_tcp(flange, gripper=gripper), gripper=gripper)
    pos_err, rot_err = se3.pose_error(back, flange)
    assert pos_err < 1e-9
    assert rot_err < 1e-7


def test_flange_to_tcp_tool_down_goes_down() -> None:
    # roll pi = tool z pointing DOWN: the 0.172 m offset lowers the TCP
    flange = Pose(np.array([0.3, -0.05, 0.22]), se3.rpy_to_quat((np.pi, 0.0, 0.0)))
    tcp = se3.flange_to_tcp(flange, gripper=True)
    assert np.allclose(tcp.position, [0.3, -0.05, 0.048], atol=1e-12)
    # Rx(pi) . Rz(pi) = Ry(pi) -> (0, 0, 1, 0) up to sign; canonical w >= 0 keeps +
    assert np.allclose(np.abs(tcp.orientation), [0.0, 0.0, 1.0, 0.0], atol=1e-12)


# --- pose interpolation ------------------------------------------------------------


@settings(deadline=None)
@given(poses, poses)
def test_pose_interp_endpoints_and_clip(a: Pose, b: Pose) -> None:
    for t, want in ((0.0, a), (1.0, b), (-3.0, a), (7.0, b)):
        pos_err, rot_err = se3.pose_error(se3.pose_interp(a, b, t), want)
        assert pos_err < 1e-9
        assert rot_err < 1e-7


def test_pose_interp_midpoint() -> None:
    a = Pose(np.array([0.0, 0.0, 0.0]), se3.rotvec_to_quat(np.zeros(3)))
    b = Pose(np.array([0.2, -0.4, 0.6]), se3.rotvec_to_quat(np.array([0.0, 0.0, 1.0])))
    mid = se3.pose_interp(a, b, 0.5)
    assert np.allclose(mid.position, [0.1, -0.2, 0.3])
    assert np.allclose(se3.quat_to_rotvec(mid.orientation), [0.0, 0.0, 0.5], atol=1e-9)


# --- canonical form ------------------------------------------------------------


@settings(deadline=None)
@given(raw_quats, raw_quats, st.floats(0.0, 1.0, allow_nan=False))
def test_canonical_w_nonneg_after_every_op(a: np.ndarray, b: np.ndarray, t: float) -> None:
    assert se3.quat_normalize(a)[0] >= 0.0
    assert se3.quat_mul(a, b)[0] >= 0.0
    assert se3.quat_conj(a)[0] >= 0.0
    assert se3.mat_to_quat(se3.quat_to_mat(a))[0] >= 0.0
    assert se3.quat_slerp(a, b, t)[0] >= 0.0
    assert se3.xyzw_to_wxyz(a)[0] >= 0.0


@settings(deadline=None)
@given(rotvecs(max_angle=np.pi - 1e-3))
def test_canonical_w_nonneg_rotvec(r: np.ndarray) -> None:
    assert se3.rotvec_to_quat(r)[0] >= 0.0


# --- pose algebra ---------------------------------------------------------------


@settings(deadline=None)
@given(poses)
def test_pose_mul_inverse_is_identity(a: Pose) -> None:
    ident = se3.pose_mul(a, se3.pose_inv(a))
    pos_err, rot_err = se3.pose_error(ident, Pose.identity())
    assert pos_err < 1e-8
    assert rot_err < 1e-7


@settings(deadline=None)
@given(poses, poses)
def test_pose_between_recomposes(a: Pose, b: Pose) -> None:
    pos_err, rot_err = se3.pose_error(se3.pose_mul(a, se3.pose_between(a, b)), b)
    assert pos_err < 1e-7
    assert rot_err < 1e-6


# --- twist integration vs finite differences -------------------------------------


def _hamilton_raw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


@settings(deadline=None)
@given(
    poses,
    st.tuples(_unit, _unit, _unit).map(np.array),
    st.tuples(_unit, _unit, _unit).map(np.array),
)
def test_integrate_twist_matches_finite_differences(p: Pose, v: np.ndarray, w: np.ndarray) -> None:
    dt = 1e-4
    p1 = se3.integrate_twist(p, Twist(v, w), dt)
    # translation: exact space-frame step along the twist's linear axes
    assert np.allclose((p1.position - p.position) / dt, v, atol=1e-9)
    # rotation: q(dt) ~= q + dt * qdot with qdot = 0.5 * [0, w] (x) q  (space frame)
    qdot = 0.5 * _hamilton_raw(np.concatenate(([0.0], w)), p.orientation)
    assert np.allclose(p1.orientation, p.orientation + dt * qdot, atol=1e-6)


def test_integrate_twist_zero_is_noop() -> None:
    p = Pose(np.array([0.1, -0.2, 0.3]), se3.rotvec_to_quat(np.array([0.4, 0.0, -0.9])))
    p1 = se3.integrate_twist(p, Twist.zero(), 0.01)
    pos_err, rot_err = se3.pose_error(p1, p)
    assert pos_err == 0.0 and rot_err < 1e-12


# --- slerp endpoints --------------------------------------------------------------


@settings(deadline=None)
@given(quats, quats)
def test_slerp_endpoints(a: np.ndarray, b: np.ndarray) -> None:
    assert _geodesic_to(se3.quat_slerp(a, b, 0.0), a) < 1e-7
    assert _geodesic_to(se3.quat_slerp(a, b, 1.0), b) < 1e-7


# --- leash clamp -------------------------------------------------------------------


@settings(deadline=None)
@given(
    poses,
    poses,
    st.floats(0.01, 1.0, allow_nan=False),
    st.floats(0.01, 1.0, allow_nan=False),
)
def test_leash_clamp_bounded_and_idempotent(
    target: Pose, anchor: Pose, max_pos_m: float, max_rot_rad: float
) -> None:
    clamped = se3.clamp_pose_to_leash(target, anchor, max_pos_m, max_rot_rad)
    pos_err, rot_err = se3.pose_error(clamped, anchor)
    assert pos_err <= max_pos_m + 1e-9
    assert rot_err <= max_rot_rad + 1e-6
    again = se3.clamp_pose_to_leash(clamped, anchor, max_pos_m, max_rot_rad)
    pos_drift, rot_drift = se3.pose_error(again, clamped)
    assert pos_drift < 1e-9
    assert rot_drift < 1e-6
