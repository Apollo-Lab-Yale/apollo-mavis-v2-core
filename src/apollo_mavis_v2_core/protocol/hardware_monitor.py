"""Hardware-monitor + twin-overlay telemetry blocks (design doc 01-core §11; phase-09a).

Session-less blocks for ``TelemetryMsg.hardware_monitor``: while no hardware
session owns the control boxes the runtime polls both xArm7 controllers
READ-ONLY (joint angles, flange pose, linear-track and gripper registers,
error/warn codes) and renders the ``mavis_v2`` digital twin from each wrist
camera's viewpoint as a tinted overlay on the real frame (streams
``<camera_id>_align``, listed in ``/api/cameras`` with ``CameraInfo.kind ==
"twin"``, §12). The monitor never sends motion commands; a hardware session
PAUSES it (connections released) instead of sharing a box between two SDK
clients. This module is a dependency-free leaf (pure pydantic) so
``protocol.telemetry`` can import it without cycles; the models ride
``TelemetryMsg``'s ``$defs`` (§14) and none exports top-level.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ArmMonitorStatus = Literal["off", "connecting", "running", "stale", "paused", "error"]
# off:        monitor disabled / hardware package not importable
# connecting: SDK client opening the control box (also during reconnect backoff)
# running:    samples arriving within stale_s
# stale:      connected, but the last sample is older than stale_s
# paused:     a hardware session owns the box - connection released (hand-over)
# error:      connect / read failure (runtime retries with exponential backoff)


class ArmMonitorTelemetry(BaseModel):
    """One arm as seen by the read-only state monitor.

    Every field but ``arm_id`` defaults so an arm the monitor never reached
    still validates. ``q`` is the controller's 7 joint angles in radians,
    controller order - an IDENTITY mapping onto the twin's ``<arm>_joint1..7``
    (verified 2026-09-04: no pi offset). ``tcp_pose`` is the controller flange
    pose (``tcp_offset`` zero) in the arm base frame, ``[x, y, z]`` m followed
    by ``[roll, pitch, yaw]`` rad (mm/deg converted by the hardware package).
    ``rail_pos_m`` is filled only while the track reports homed (``on_zero ==
    1``) AND enabled - the raw register is meaningless otherwise - whereas
    ``rail_raw_mm`` is always reported when the registers are readable.
    ``gripper_open_frac`` is 0 closed .. 1 open (``None`` for gripper
    ``"none"``); ``gripper_raw`` is the SDK reading for diagnosis.
    ``error_code``/``warn_code`` are the controller's codes (e.g. 19 = End
    Module Communication Error); ``state`` 4 = stopped / not enabled.
    """

    arm_id: str
    status: ArmMonitorStatus = "off"
    detail: str = ""  # e.g. "controller error 19: End Module Communication Error"
    seq: int = 0  # last sample sequence number
    age_s: float | None = None  # now - t_mono of the last sample
    q: list[float] = []  # 7 joint angles, rad, controller order (identity to the twin)
    tcp_pose: list[float] = []  # flange pose [x,y,z m, roll,pitch,yaw rad] in base frame
    rail_present: bool | None = None  # linear-track registers readable
    rail_homed: bool | None = None  # on_zero == 1
    rail_enabled: bool | None = None
    rail_pos_m: float | None = None  # None unless homed AND enabled (meaningless otherwise)
    rail_raw_mm: float | None = None  # raw register, always reported when present
    gripper_open_frac: float | None = None  # 0 closed .. 1 open; None for gripper "none"
    gripper_raw: float | None = None  # raw SDK reading for diagnosis
    error_code: int = 0  # controller error code (0 = ok)
    warn_code: int = 0
    state: int | None = None  # controller state (4 = stopped / not enabled)
    mode: int | None = None  # controller mode


TwinOverlayStatus = Literal["off", "waiting", "live", "stale", "error"]
# off:     overlay disabled (config) or the twin scene failed to build
# waiting: no real camera frame yet (nothing published)
# live:    composited frames flowing at ~cfg.fps
# stale:   monitor sample for this arm stale / error (twin drawn in the stale tint)
# error:   render / composite failure


class TwinOverlayTelemetry(BaseModel):
    """One digital-twin overlay stream (``<camera_id>_align``; 04-runtime §13.4).

    ``rail_fallback_m`` is set while the track is not homed and the twin
    assumes a configured rail position instead (``detail`` spells it out for
    the tile caption); ``joint1_offset_rad`` echoes the diagnostic knob (0 =
    the verified identity convention); ``mask_fraction`` is robot pixels /
    image pixels of the last composited frame (0 = the twin sees no robot from
    this camera).
    """

    stream_id: str  # grip_wrist_align / view_wrist_align
    camera_id: str  # grip_wrist / view_wrist (the real frame underneath)
    arm_id: str
    status: TwinOverlayStatus = "off"
    detail: str = ""  # e.g. "rail not homed - twin assumes 0.65 m"
    fps: float = 0.0  # measured publish rate (1 s window)
    rail_fallback_m: float | None = None  # set when the fallback is in use
    joint1_offset_rad: float = 0.0
    mask_fraction: float = 0.0  # robot pixels / image pixels of the last frame


class HardwareMonitorTelemetry(BaseModel):
    """``TelemetryMsg.hardware_monitor`` block (phase-09a; 04-runtime §13.3), additive.

    Every field defaults so a runtime without the hardware package still emits
    a valid (``enabled: false``) block. ``paused`` is true while a hardware
    session owns the control boxes: the monitor releases its connections
    instead of sharing a box between two SDK clients.
    """

    enabled: bool = False
    paused: bool = False  # a hardware session owns the boxes
    arms: list[ArmMonitorTelemetry] = []
    overlays: list[TwinOverlayTelemetry] = []


__all__ = [
    "ArmMonitorStatus",
    "ArmMonitorTelemetry",
    "TwinOverlayStatus",
    "TwinOverlayTelemetry",
    "HardwareMonitorTelemetry",
]
