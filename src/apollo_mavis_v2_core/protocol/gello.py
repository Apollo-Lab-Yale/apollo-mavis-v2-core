"""GELLO leader-arm vocabulary and REST models (design doc 01-core §11/§12; 16-gello §8;
phase-15, 2026-09-09).

GELLO is the passive xArm7-shaped leader arm (Dynamixel servos over one USB serial adapter)
that drives the Manipulation Arm in joint space in the ``gello`` session mode. This module is
a dependency-free leaf (pure pydantic + ``typing``), like ``protocol.microphone``: it holds
the shared spelling of every GELLO literal, the DEVICE half of the telemetry block (shared
by ``TelemetryMsg.gello`` and ``GET /api/gello``) and the session-less REST bodies:

* ``GET /api/gello -> GelloInfo``
* ``POST /api/gello/calibrate  GelloCalibrateRequest -> GelloCalibrateResult`` (16-gello D10)
* ``POST /api/gello/preview    GelloPreviewRequest -> GelloPreviewResult``   (16-gello §5.4)

``protocol.telemetry`` imports the literals and :class:`GelloDeviceTelemetry` from here and
adds the session half (``GelloTelemetry``); ``protocol.session`` imports
:data:`GelloViewpointMode` for ``GelloSessionConfig.viewpoint``. Nothing here imports from
either, so the package stays cycle-free. Core gets no new interface for the leader (it is a
runtime device, like the Vive controller — 16-gello D2).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

GelloBackend = Literal["dynamixel", "fake", "none"]
# dynamixel: the real bus over the FT232H adapter (runtime extra ``[gello]``)
# fake:      scripted posture at poll_hz (tests, sim); ``reader.fake_set(q, gripper)`` moves it
# none:      leader disabled (repo default)

GelloDeviceStatus = Literal["no_backend", "starting", "connected", "stale", "error"]
# no_backend: backend none / dynamixel_sdk not importable
# starting:   opening the port / scanning the baud rates
# connected:  samples arriving within stale_s
# stale:      no sample for > stale_s
# error:      open / read failure (the reader restarts with backoff)

GelloState = Literal["no_leader", "out_of_sync", "tracking", "paused", "motion"]
# The engagement state machine (16-gello D3 / §6.1). Only ``tracking`` follows the leader;
# every other state HOLDS the last command — never a move:
#   no_leader:   sample missing / stale / invalid
#   out_of_sync: leader farther than engage_tol_rad from the measured arm (or the leash
#                exceeded while tracking); re-engages automatically once within tolerance
#   tracking:    the Manipulation Arm streams the leader's joints + gripper
#   paused:      operator Pause, a fault / RECOVERING, or forced before a planned motion;
#                leaves only on the operator's Resume (``gello_resume``)
#   motion:      a twin-planned motion owns the arm (launch, R, Go to profile, return)

GelloViewpointMode = Literal["auto", "external", "hold"]
# How the Perception Arm is driven in a gello session (16-gello D5):
#   auto:     attach the external viewpoint node whenever a compatible policy_spec is fresh,
#             hold the GELLO hold posture when it is stale / absent (default)
#   external: the node must be attached at launch (409 otherwise)
#   hold:     ignore the bus; hold the GELLO hold posture

GelloCalibrateOp = Literal["match_arm", "gripper_open", "gripper_closed", "clear"]
# match_arm:      offsets = round((raw - sign * q_arm) / (pi/2)) * pi/2 per joint, from the
#                 Manipulation Arm's CURRENT joints (monitor sample on hardware, parked posture
#                 in sim) -> var/gello_calibration.json
# gripper_open / gripper_closed: store the raw gripper reading at the two endpoints
# clear:          delete the calibration file

GelloPreviewStatus = Literal[
    "clear", "collision", "joint_limit", "no_leader", "not_calibrated", "no_workcell",
    "scene_error",
]
# clear:          the GELLO posture is collision-free and inside the joint limits (Start enabled)
# collision:      ``pairs`` lists the violating pairs, tightest first (16-gello §5.1 item 4)
# joint_limit:    ``detail`` names the joint and the limit (item 3)
# no_leader:      no fresh valid sample from the reader
# not_calibrated: the reader has no joint offsets yet (POST /api/gello/calibrate match_arm)
# no_workcell:    the requested kind is not configured / not reachable
# scene_error:    the kitchen twin could not be built for the requested scene


class GelloDeviceTelemetry(BaseModel):
    """The DEVICE half of the GELLO block (16-gello §8.3): what the leader reader reports
    whether or not a session exists. ``TelemetryMsg.gello`` (``GelloTelemetry``) and
    ``GET /api/gello`` (``GelloInfo``) both extend this class, so the two surfaces spell the
    device state identically (the microphone precedent: ``MicStatus`` shared by
    ``MicrophoneTelemetry`` and ``MicrophoneInfo``).

    ``q_raw`` is the servo reading in rad (ticks * 2pi / 4096) BEFORE ``joint_signs`` /
    ``joint_offsets_rad``; ``q`` is the mapped value ``sign * (raw - offset)`` the follower
    would track; ``gripper_frac`` is 0 closed .. 1 open and None until both gripper endpoints
    are calibrated. ``calibrated`` is False until ``joint_offsets_rad`` exist (config or the
    calibration file). ``joint_signs`` is the operator-owned config echo.
    """

    backend: GelloBackend
    status: GelloDeviceStatus
    detail: str = ""  # human-readable reason for error / no_backend / stale
    port: str = ""  # serial node actually opened (e.g. /dev/ttyUSB0); "" when none
    baud: int | None = None  # bus rate in use (scanned or configured); None until connected
    seq: int = 0  # last sample sequence number
    rate_hz: float = 0.0  # measured sample rate
    age_s: float | None = None  # now - rx_mono of the last sample
    q_raw: list[float] | None = None  # rad, before signs / offsets (7 joints)
    q: list[float] | None = None  # rad, mapped: sign * (raw - offset)
    gripper_frac: float | None = None  # 0 closed .. 1 open; None until both endpoints exist
    calibrated: bool = False  # joint offsets known (config or var/gello_calibration.json)
    joint_offsets_rad: list[float] | None = None  # the offsets in force (7), None uncalibrated
    joint_signs: list[int]  # operator-owned config echo (7 x +-1)


class GelloInfo(GelloDeviceTelemetry):
    """``GET /api/gello`` (16-gello §8.4 / §9.2): the device half plus what the GELLO launch
    sheet needs before a session exists — the twin scene the GELLO card launches
    (``scene_id`` = ``gello.scene_id``, the hidden ``mavis_v2_kitchen``; ``scene_label`` its
    title), the Perception Arm's GELLO hold posture (``view_posture_rad`` J1-J7 rad +
    ``view_rail_m``), where the calibration file lives and whether the hardware tab may
    launch gello (16-gello D8: ``hardware_admitted`` is True on this runtime; the flag lets
    the UI follow a later operator decision without a code change).

    ``gripper_open_rad`` / ``gripper_closed_rad`` echo the two gripper-endpoint calibration
    ops (16-gello §4 "the result of every op is echoed in ``GET /api/gello``"); None until
    calibrated.
    """

    scene_id: str  # the twin the GELLO card launches (sim_scene / digital_twin_scene)
    scene_label: str  # its title, e.g. "APOLLO MAVIS V2 Kitchen (GELLO)"
    view_posture_rad: list[float]  # Perception Arm GELLO hold posture, J1-J7 rad
    view_rail_m: float  # ... and its rail slot (twin convention)
    calibration_path: str  # var/gello_calibration.json (absolute)
    hardware_admitted: bool  # the hardware tab may launch gello (16-gello D8)
    gripper_open_rad: float | None = None  # raw reading stored by op gripper_open
    gripper_closed_rad: float | None = None  # raw reading stored by op gripper_closed


class GelloCalibrateRequest(BaseModel):
    """``POST /api/gello/calibrate`` body (16-gello D10 / §4). Session-less; 409 while a
    session runs or when the leader has no fresh sample. ``kind`` picks where the
    Manipulation Arm's current joints come from for ``match_arm`` (the hardware monitor
    sample, or the parked sim posture)."""

    op: GelloCalibrateOp
    kind: Literal["hardware", "sim"]


class GelloCalibrateResult(BaseModel):
    """``POST /api/gello/calibrate`` response: the calibration now in force (all None after
    ``clear``)."""

    ok: bool
    detail: str = ""  # operator-facing outcome / reason
    joint_offsets_rad: list[float] | None = None  # 7 offsets (multiples of pi/2)
    gripper_open_rad: float | None = None
    gripper_closed_rad: float | None = None


class GelloPreviewRequest(BaseModel):
    """``POST /api/gello/preview`` body (16-gello §5.4). ``scene`` None = the runtime's
    ``gello.scene_id``; ``speed_scale`` None = the kind's default (it only annotates the
    check — the preview never moves anything)."""

    kind: Literal["hardware", "sim"]
    scene: str | None = None
    speed_scale: float | None = Field(default=None, gt=0, le=1)


class GelloPairInfo(BaseModel):
    """One violating geometry pair of a GELLO preview (``GelloPreviewResult.pairs``)."""

    a: str  # geom label, e.g. "grip_link6"
    b: str  # geom label, e.g. "fridge_body"
    dist_m: float  # signed clearance (negative = penetration) at the goal posture


class GelloPreviewResult(BaseModel):
    """``POST /api/gello/preview`` response (16-gello §5.4): the launch check of §5.1 run
    session-less on a cached kitchen twin, plus a PNG of the virtual cell rendered from
    ``camera`` with the colliding bodies tinted red. Never a 409 for a bad posture — the
    ``status`` says. ``ok`` is True iff ``status == "clear"`` (the sheet enables Start on it;
    the invariant is validated so the two can never disagree). ``q_goal`` is the full joint
    goal per arm (``{grip: [8], view: [8]}``, rail slot last) the launch would plan to;
    ``leader_q`` the unwrapped leader joints (None without a leader)."""

    status: GelloPreviewStatus
    ok: bool
    detail: str = ""  # verdict line, e.g. "collides: fridge_body / grip_link6 at 3 mm"
    pairs: list[GelloPairInfo] = []  # violating pairs, tightest first (collision only)
    q_goal: dict[str, list[float]] = {}  # per-arm full q incl. rail slot
    leader_q: list[float] | None = None  # unwrapped leader joints (7), None without a leader
    image_png_b64: str | None = None  # the preview render; None when rendering failed
    camera: str = "cam_kitchen"  # the scene camera the PNG was rendered from

    @model_validator(mode="after")
    def _ok_iff_clear(self) -> GelloPreviewResult:
        if self.ok != (self.status == "clear"):
            raise ValueError("ok must be True iff status == 'clear'")
        return self


__all__ = [
    "GelloBackend",
    "GelloDeviceStatus",
    "GelloState",
    "GelloViewpointMode",
    "GelloCalibrateOp",
    "GelloPreviewStatus",
    "GelloDeviceTelemetry",
    "GelloInfo",
    "GelloCalibrateRequest",
    "GelloCalibrateResult",
    "GelloPreviewRequest",
    "GelloPairInfo",
    "GelloPreviewResult",
]
