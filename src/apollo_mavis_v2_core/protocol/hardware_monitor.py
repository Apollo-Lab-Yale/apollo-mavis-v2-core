"""Hardware-monitor + twin-overlay telemetry blocks (design doc 01-core §11; phase-09a).

Session-less blocks for ``TelemetryMsg.hardware_monitor``: while no hardware
session owns the control boxes the runtime polls both xArm7 controllers
READ-ONLY (joint angles, flange pose, linear-track and gripper registers,
error/warn codes) and renders the ``mavis_v2`` digital twin from each wrist
camera's viewpoint as a tinted overlay on the real frame (streams
``<camera_id>_align``, listed in ``/api/cameras`` with ``CameraInfo.kind ==
"twin"``, §12). The monitor never sends motion commands and writes nothing
to a box unless the operator issues an explicit maintenance request
(``protocol.maintenance``, phase-09b: clear errors / apply the controller-side
safety parameters; phase-09c/09d: home the linear track, possibly after a
twin-planned pre-positioning motion); a hardware session PAUSES it
(connections released) instead of sharing a box between two SDK clients.

This module is a dependency-free leaf (pure pydantic) so ``protocol.telemetry``
and ``protocol.maintenance`` can import it without cycles. That is also why
the maintenance-op vocabulary :data:`ArmMaintenanceOp` and the asynchronous
job progress block :class:`MaintenanceProgress` / :data:`MaintenancePhase`
are DEFINED here (phase-09d) and merely re-exported by ``protocol.maintenance``:
the progress rides :attr:`ArmMonitorTelemetry.maintenance` while
``protocol.maintenance`` embeds ``ArmMonitorTelemetry`` in its result, so
defining them there would make the two modules import each other. The models
ride ``TelemetryMsg``'s ``$defs`` (§14) and none exports top-level.
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

ArmMaintenanceOp = Literal[
    "clear_errors", "apply_backstops", "recover", "home_rail", "set_collision_sensitivity"
]
# The maintenance-op vocabulary of ``POST /api/hardware/arms/{arm_id}/maintenance``
# (``protocol.maintenance`` documents each op and re-exports the name; it lives here
# because MaintenanceProgress.op below needs it in the leaf - see the module docstring).
# ``set_collision_sensitivity`` (2026-09-11): the operator's level 1..3 override of the
# controller's collision sensitivity, one write, no motion; volatile - the config value
# is re-applied at the next connect.

MaintenancePhase = Literal[
    "queued",
    "sweeping",
    "planning",
    "connecting",
    "positioning",
    "homing",
    "verifying",
    "done",
    "failed",
]
# Phases of an asynchronous maintenance job (phase-09d ``RailHomingJob``, one arm at a
# time), in execution order:
# queued       accepted (202) - the job thread has not started yet
# sweeping     full-travel twin sweep at the arm's CURRENT posture (RailSweepChecker)
# planning     twin RRT-Connect to a rail-safe posture + position-agnostic path check
#              (every waypoint clear for EVERY rail position - the carriage is unknown)
# connecting   monitor paused + joined; this arm's driver connected with the rail still
#              unhomed, speed_scale 0.1; the other arm frozen at its last sample (09c D1)
# positioning  the planned joint path executes under the gate (ControlLoop._op_execute_plan)
# homing       driver.home_rail(): the carriage drives to the homing end, joints held
# verifying    registers homed + enabled + no error and a monitor sample; the driver is
#              torn down (state 4 + brakes -> the posture is HELD) and the monitor resumed
# done/failed  terminal - the final ArmMaintenanceResult is at GET .../maintenance/last


class MaintenanceProgress(BaseModel):
    """Live progress of an asynchronous maintenance job on one arm (phase-09d).

    Rides :attr:`ArmMonitorTelemetry.maintenance` while a ``RailHomingJob`` runs
    (``None`` when no job exists) so the UI's Home-rail sheet can list the
    phases as they happen. ``job_id`` matches the ``202`` response's
    ``ArmMaintenanceResult.job_id``; ``progress`` is a coarse 0..1 estimate
    (phase index, plus the waypoint fraction while ``positioning``);
    ``started_at`` is unix seconds. How long a terminal ``done`` / ``failed``
    stays visible is runtime territory (04-runtime §13.3); the final
    ``ArmMaintenanceResult`` is fetched from ``GET .../maintenance/last``.
    """

    op: ArmMaintenanceOp
    job_id: str
    phase: MaintenancePhase
    detail: str = ""  # operator-facing progress / failure reason
    progress: float = 0.0  # 0..1 coarse estimate
    started_at: float | None = None  # unix s


class ArmMonitorTelemetry(BaseModel):
    """One arm as seen by the read-only state monitor.

    Every field but ``arm_id`` defaults so an arm the monitor never reached
    still validates. ``q`` is the controller's 7 joint angles in radians,
    controller order - an IDENTITY mapping onto the twin's ``<arm>_joint1..7``
    (verified 2026-09-04: no pi offset). ``tcp_pose`` is the controller FLANGE
    pose (``tcp_offset`` zero) in the arm base frame, ``[x, y, z]`` m followed
    by ``[roll, pitch, yaw]`` rad (mm/deg converted by the hardware package); the
    RPY is the xArm extrinsic-XYZ convention (``Rz(yaw) . Ry(pitch) . Rx(roll)``).
    The twin's ``link_tcp`` = flange (+) (Rz(pi), +0.172 m along tool z) on a
    gripper arm - that is ``ArmState.ee_pose``, NOT this field.
    ``rail_pos_m`` is filled only while the track reports homed (``on_zero ==
    1``) AND enabled - the raw register is meaningless otherwise - whereas
    ``rail_raw_mm`` is always reported when the registers are readable.
    ``gripper_open_frac`` is 0 closed .. 1 open (``None`` for gripper
    ``"none"``); ``gripper_raw`` is the SDK reading for diagnosis.
    ``error_code``/``warn_code`` are the controller's codes (e.g. 19 = End
    Module Communication Error); ``state`` 4 = stopped / not enabled.

    phase-09b read-back (slow poll, additive): ``collision_sensitivity`` /
    ``tcp_load_kg`` / ``tcp_load_cog_mm`` are the controller's CURRENT
    safety parameters; ``backstops_match`` is the runtime's comparison against
    the arm's ``ArmConfig`` (sensitivity equal, load within 0.05 kg, centre of
    gravity within 10 mm; ``None`` = not compared) and ``maintenance_busy`` is
    true while a maintenance op executes on this arm. Since 2026-09-11 the
    sensitivity term of ``backstops_match`` compares against the level the
    operator last REQUESTED through the ``set_collision_sensitivity``
    maintenance op when one is set (the runtime remembers it per arm until the
    next driver connect re-applies the config value), so an intentional
    override does not read as a mismatch; ``collision_sensitivity`` itself
    stays the raw controller read-back - the UI's sensitivity control shows
    exactly this value, never an optimistic one.

    phase-09d (additive): ``maintenance`` is the live :class:`MaintenanceProgress`
    of an asynchronous job (rail homing that first needs a planned
    pre-positioning motion) on this arm, ``None`` when no job exists;
    ``maintenance_busy`` stays true for the job's whole life.
    """

    arm_id: str
    status: ArmMonitorStatus = "off"
    detail: str = ""  # e.g. "controller error 19: End Module Communication Error"
    seq: int = 0  # last sample sequence number
    age_s: float | None = None  # now - t_mono of the last sample
    q: list[float] = []  # 7 joint angles, rad, controller order (identity to the twin)
    tcp_pose: list[float] = []  # FLANGE pose [x,y,z m, roll,pitch,yaw rad] in base frame
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
    # phase-09b read-back of the controller-side safety parameters + maintenance flag
    collision_sensitivity: int | None = None  # controller collision sensitivity 0..5
    tcp_load_kg: float | None = None  # controller tcp_load mass
    tcp_load_cog_mm: list[float] = []  # controller tcp_load centre of gravity [x, y, z] mm
    backstops_match: bool | None = None  # read-back == ArmConfig (runtime); None = not compared
    maintenance_busy: bool = False  # a maintenance op is executing on this arm
    maintenance: MaintenanceProgress | None = None  # phase-09d async job progress (None = none)


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
    "ArmMaintenanceOp",
    "MaintenancePhase",
    "MaintenanceProgress",
    "ArmMonitorTelemetry",
    "TwinOverlayStatus",
    "TwinOverlayTelemetry",
    "HardwareMonitorTelemetry",
]
