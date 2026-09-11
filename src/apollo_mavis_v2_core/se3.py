"""Pure-numpy SE3 / quaternion math (design doc 01-core §3).

THE quaternion convention module: every repo in the stack uses these
functions instead of rolling its own. Quaternions are ``(w, x, y, z)`` unit
vectors, canonical ``w >= 0``. Rotations follow the Hamilton convention.
"""

from __future__ import annotations

import numpy as np

from .types import Pose, Quat, Twist, Vec3

# --- Stack-wide geometric constants -----------------------------------------

TCP_OFFSET_M = 0.172
"""link7 flange -> link_tcp along the tool +Z axis (matches the MJCF assets)."""

FLANGE_TO_TCP_QUAT = (0.0, 0.0, 0.0, 1.0)
"""Rz(pi), wxyz: the flange -> ``link_tcp`` ROTATION of a gripper arm (2026-09-11).

The xArm Gripper base is mounted 180 deg about the tool z axis under link7 (MJCF
``xarm_gripper_base_link`` ``quat="0 0 0 1"``) and ``link_tcp`` is its child,
``TCP_OFFSET_M`` further along tool z. A gripper-less arm's ``link_tcp`` IS the flange
(the sim builder adds the site at link7 with no offset), so the rotation applies to
gripper arms only - see :func:`flange_to_tcp`. Not to be confused with the legacy
180-deg-about-X constant below.
"""

LEGACY_FLANGE_QUAT_OFFSET = (0.0, 1.0, 0.0, 0.0)
"""Legacy xarm7-ik 180°-about-X flange convention ("identity = gripper down").

A DIFFERENT thing from :data:`FLANGE_TO_TCP_QUAT` (180° about tool Z, the real
gripper mount): this one only maps legacy datasets. See 10-frames-and-data.md for
the full compatibility mapping.
"""

RAIL_TRAVEL_M = 0.65
"""Usable linear-rail travel. The SDK does NOT clamp (legacy solver used 0.74)."""

_EPS = 1e-12


# --- Quaternion primitives ---------------------------------------------------

def quat_normalize(q: Quat) -> Quat:
    """Unit norm AND canonical sign (negate whole quaternion if w < 0)."""
    q = np.asarray(q, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < _EPS:
        raise ValueError("cannot normalize a ~zero quaternion")
    q = q / norm
    if q[0] < 0.0:
        q = -q
    return q


def quat_mul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = np.asarray(a, dtype=np.float64)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64)
    return quat_normalize(
        np.array(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ]
        )
    )


def quat_conj(q: Quat) -> Quat:
    q = np.asarray(q, dtype=np.float64)
    return quat_normalize(np.array([q[0], -q[1], -q[2], -q[3]]))


def quat_rotate(q: Quat, v: Vec3) -> Vec3:
    """Rotate vector v by quaternion q (q v q*)."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    qv = q[1:]
    t = 2.0 * np.cross(qv, v)
    return v + q[0] * t + np.cross(qv, t)


def quat_to_mat(q: Quat) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat(m: np.ndarray) -> Quat:
    """Rotation matrix -> quaternion via Shepperd's method; canonical w >= 0."""
    m = np.asarray(m, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"expected (3, 3) rotation matrix, got {m.shape}")
    t = np.trace(m)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q = np.array(
            [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        )
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        q = np.array(
            [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        )
    elif m[1, 1] >= m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        q = np.array(
            [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        )
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        q = np.array(
            [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
        )
    return quat_normalize(q)


# --- Euler / rotation-vector conversions -------------------------------------

def rpy_to_quat(rpy) -> Quat:
    """xArm SDK RPY -> quat. Extrinsic XYZ == intrinsic ZYX:
    ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

    Verified against 15 hardware episodes on 2026-09-11 (FK of the recorded joints vs
    the controller's reported RPY agree to 0.0001 deg); the pre-2026-09-11
    ``Rx @ Ry @ Rz`` composition was WRONG for any pose with two non-zero angles.
    Only ``apollo_mavis_v2_hardware.units`` should call this (02-hardware §2).
    """
    r, p, y = (float(v) for v in rpy)
    qx = np.array([np.cos(r / 2), np.sin(r / 2), 0.0, 0.0])
    qy = np.array([np.cos(p / 2), 0.0, np.sin(p / 2), 0.0])
    qz = np.array([np.cos(y / 2), 0.0, 0.0, np.sin(y / 2)])
    return quat_mul(quat_mul(qz, qy), qx)


def quat_to_rpy(q: Quat) -> np.ndarray:
    """Exact inverse of :func:`rpy_to_quat` (extrinsic XYZ), gimbal-safe at |pitch|=pi/2."""
    m = quat_to_mat(q)
    # R = Rz @ Ry @ Rx  =>  m[2,0] = -sin(pitch)
    sp = float(np.clip(-m[2, 0], -1.0, 1.0))
    pitch = np.arcsin(sp)
    if abs(sp) < 1.0 - 1e-9:
        roll = np.arctan2(m[2, 1], m[2, 2])
        yaw = np.arctan2(m[1, 0], m[0, 0])
    else:  # gimbal lock: fold yaw into roll
        # at cos(pitch) = 0: m[1,1] = cos(yaw -/+ roll), m[1,2] = -/+ sin(yaw -/+ roll)
        roll = np.arctan2(-m[1, 2], m[1, 1])
        yaw = 0.0
    return np.array([roll, pitch, yaw])


# --- 6-D rotation representation (Zhou et al. 2019) -----------------------------

_ROT6D_EPS = 1e-6


def mat_to_rot6d(m: np.ndarray) -> np.ndarray:
    """Rotation matrix -> the first two COLUMNS, column-major ``[R00,R10,R20,R01,R11,R21]``.

    The stack-wide continuous rotation codec for absolute-pose actions
    (``abs_ee``, 10-frames §3.1); decode with :func:`rot6d_to_mat`.
    """
    m = np.asarray(m, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"expected (3, 3) rotation matrix, got {m.shape}")
    return np.concatenate([m[:, 0], m[:, 1]])


def rot6d_to_mat(r6) -> np.ndarray:
    """Gram-Schmidt decode of :func:`mat_to_rot6d`: ``b1 = normalize(c1)``,
    ``b2 = normalize(c2 - (b1 . c2) b1)``, ``b3 = b1 x b2``; ``R = [b1 b2 b3]``.

    Tolerates a perturbed (non-orthonormal) pair; raises ``ValueError`` when a column
    norm is below 1e-6 or the two columns are parallel within 1e-6.
    """
    r6 = np.asarray(r6, dtype=np.float64).reshape(-1)
    if r6.shape != (6,):
        raise ValueError(f"expected 6 values, got {r6.shape}")
    c1, c2 = r6[:3], r6[3:]
    n1 = float(np.linalg.norm(c1))
    if n1 < _ROT6D_EPS or float(np.linalg.norm(c2)) < _ROT6D_EPS:
        raise ValueError("rot6d: a column is ~zero")
    b1 = c1 / n1
    u2 = c2 - float(np.dot(b1, c2)) * b1
    n2 = float(np.linalg.norm(u2))
    if n2 < _ROT6D_EPS:
        raise ValueError("rot6d: the two columns are parallel")
    b2 = u2 / n2
    b3 = np.cross(b1, b2)
    return np.column_stack([b1, b2, b3])


def quat_to_rot6d(q: Quat) -> np.ndarray:
    return mat_to_rot6d(quat_to_mat(q))


def rot6d_to_quat(r6) -> Quat:
    """Canonical ``w >= 0`` (via :func:`mat_to_quat`)."""
    return mat_to_quat(rot6d_to_mat(r6))


def rotvec_to_quat(r) -> Quat:
    """Exponential map: rotation vector (axis * angle, rad) -> quaternion."""
    r = np.asarray(r, dtype=np.float64)
    angle = float(np.linalg.norm(r))
    if angle < 1e-9:
        # Taylor: sin(a/2)/a ~= 1/2 - a^2/48
        half_sinc = 0.5 - angle * angle / 48.0
        return quat_normalize(np.concatenate(([np.cos(angle / 2)], r * half_sinc)))
    axis = r / angle
    return quat_normalize(
        np.concatenate(([np.cos(angle / 2)], axis * np.sin(angle / 2)))
    )


def quat_to_rotvec(q: Quat) -> np.ndarray:
    """Log map: quaternion -> rotation vector (axis * angle, rad), |angle| <= pi."""
    q = quat_normalize(q)
    vec_norm = float(np.linalg.norm(q[1:]))
    angle = 2.0 * np.arctan2(vec_norm, q[0])  # in [0, pi] since w >= 0
    if vec_norm < 1e-9:
        return q[1:] * 2.0  # small-angle: r ~= 2 * vec
    return q[1:] / vec_norm * angle


def quat_geodesic(a: Quat, b: Quat) -> float:
    """Geodesic angle between two orientations, in [0, pi]."""
    d = quat_mul(quat_conj(a), b)
    return float(2.0 * np.arctan2(np.linalg.norm(d[1:]), abs(d[0])))


def quat_slerp(a: Quat, b: Quat, t: float) -> Quat:
    """Shortest-path spherical interpolation; t in [0, 1]."""
    a = quat_normalize(a)
    b = quat_normalize(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b, dot = -b, -dot
    if dot > 1.0 - 1e-10:
        return quat_normalize(a + t * (b - a))
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(theta)
    return quat_normalize(
        a * (np.sin((1.0 - t) * theta) / s) + b * (np.sin(t * theta) / s)
    )


def xyzw_to_wxyz(q) -> Quat:
    """The ONLY sanctioned order swap for scipy/ROS interop."""
    q = np.asarray(q, dtype=np.float64)
    return quat_normalize(np.array([q[3], q[0], q[1], q[2]]))


def wxyz_to_xyzw(q) -> np.ndarray:
    q = quat_normalize(q)
    return np.array([q[1], q[2], q[3], q[0]])


# --- Pose algebra ------------------------------------------------------------

def pose_mul(a: Pose, b: Pose) -> Pose:
    """Compose transforms: (a ⊕ b) maps b-child coords through a."""
    return Pose(
        a.position + quat_rotate(a.orientation, b.position),
        quat_mul(a.orientation, b.orientation),
    )


def pose_inv(a: Pose) -> Pose:
    q_inv = quat_conj(a.orientation)
    return Pose(-quat_rotate(q_inv, a.position), q_inv)


def pose_between(a: Pose, b: Pose) -> Pose:
    """Relative transform a⁻¹ ⊕ b."""
    return pose_mul(pose_inv(a), b)


def pose_interp(a: Pose, b: Pose, t: float) -> Pose:
    """Lerp position + slerp orientation from ``a`` (t=0) to ``b`` (t=1); t clipped to [0, 1]."""
    t = min(1.0, max(0.0, float(t)))
    return Pose(
        a.position + (b.position - a.position) * t,
        quat_slerp(a.orientation, b.orientation, t),
    )


# --- flange <-> TCP ----------------------------------------------------------------

_FLANGE_TO_TCP = Pose(
    np.array([0.0, 0.0, TCP_OFFSET_M]), np.array(FLANGE_TO_TCP_QUAT, dtype=np.float64)
)


def flange_to_tcp(pose: Pose, *, gripper: bool) -> Pose:
    """link7 flange pose -> the twin's ``link_tcp`` pose, same parent frame.

    ``gripper=True``: ``pose ⊕ (Trans(0, 0, TCP_OFFSET_M), Rz(pi))`` - the gripper base
    is mounted 180 deg about tool z under link7 and ``link_tcp`` sits 0.172 m further
    along tool z. ``gripper=False``: unchanged - a gripper-less arm's ``link_tcp`` IS the
    flange (sim builder adds the site at link7 with no offset). 10-frames §2.4.
    """
    return pose_mul(pose, _FLANGE_TO_TCP) if gripper else pose


def tcp_to_flange(pose: Pose, *, gripper: bool) -> Pose:
    """Inverse of :func:`flange_to_tcp`."""
    return pose_mul(pose, pose_inv(_FLANGE_TO_TCP)) if gripper else pose


def pose_error(a: Pose, b: Pose) -> tuple[float, float]:
    """(position error m, geodesic orientation error rad)."""
    return (
        float(np.linalg.norm(a.position - b.position)),
        quat_geodesic(a.orientation, b.orientation),
    )


def integrate_twist(p: Pose, tw: Twist, dt: float) -> Pose:
    """Space-frame left-composition increment (04-runtime §6 teleop semantics).

    Translation moves along the twist's linear axes; rotation is applied
    about the pose origin in the parent (space) frame:
    ``p' = (exp(w dt), v dt) ⊕_space p``. Rail/grip rates are NOT integrated
    here — the runtime handles those channels separately.
    """
    dq = rotvec_to_quat(tw.w * dt)
    return Pose(p.position + tw.v * dt, quat_mul(dq, p.orientation))


def clamp_pose_to_leash(
    target: Pose, anchor: Pose, max_pos_m: float, max_rot_rad: float
) -> Pose:
    """Clamp target into a ball (position) and geodesic cone (orientation)
    around anchor. Idempotent; used to keep integrated teleop targets from
    running away from the measured pose."""
    delta = target.position - anchor.position
    dist = float(np.linalg.norm(delta))
    position = (
        anchor.position + delta * (max_pos_m / dist)
        if dist > max_pos_m and dist > _EPS
        else target.position
    )
    angle = quat_geodesic(anchor.orientation, target.orientation)
    orientation = (
        quat_slerp(anchor.orientation, target.orientation, max_rot_rad / angle)
        if angle > max_rot_rad and angle > _EPS
        else target.orientation
    )
    return Pose(position, orientation)
