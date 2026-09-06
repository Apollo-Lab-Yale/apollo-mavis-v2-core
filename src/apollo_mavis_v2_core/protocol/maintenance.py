"""Arm maintenance channel (design doc 01-core §12; phase-09b, phase-09c, phase-09d).

``POST /api/hardware/arms/{arm_id}/maintenance`` (body
:class:`ArmMaintenanceRequest`, response :class:`ArmMaintenanceResult`) lets
the operator clear xArm controller errors, (re)apply the controller-side
safety parameters and home the linear track from the UI. Three of the four ops
produce no motion (measured 2026-09-04 on the Perception Arm: ``clean_error``
moved no joint by more than 5e-5 rad). ``home_rail`` is the ONE motion op:
operator-triggered, twin-gated, session-less — the carriage drives to the
homing end, so the runtime first sweeps the digital twin over the full rail
travel at the arm's CURRENT joint posture; the verdict rides
:attr:`ArmMaintenanceResult.rail_sweep`. Since phase-09d a posture that is NOT
sweep-clear is no longer a flat refusal: the runtime plans (twin RRT-Connect) a
joint path to a rail-safe posture whose every waypoint is clear for EVERY rail
position (the carriage is unknown) and offers it as
:attr:`RailSweepVerdict.pre_position` (:class:`PrePositionPlan`); the operator
confirms in the UI, the op then runs as an asynchronous ``RailHomingJob``
(``202``, :attr:`ArmMaintenanceResult.status` ``"accepted"`` + ``job_id``) whose
phases ride ``ArmMonitorTelemetry.maintenance`` (:class:`MaintenanceProgress`)
and whose final result is ``GET .../maintenance/last``. Only when no plan exists
is the op ``"refused"`` (zero writes, a suggestion in ``detail``).

* ``clear_errors`` — session-less: ``clean_error`` + ``clean_warn`` only, never
  ``motion_enable`` (via the read-only monitor's maintenance queue; the
  monitor's zero-write guarantee becomes "zero writes unless an explicit
  maintenance request"). Inside a hardware session the runtime routes the same
  op to the driver's user-initiated recovery, i.e. it is then equivalent to
  ``recover`` and DOES enable (04-runtime §13.1); the UI never posts it then.
* ``apply_backstops`` — ``backstops.apply_backstops`` (tcp_load -> gravity ->
  collision sensitivity -> self-collision + tool model -> optional Reduced-mode
  boundary -> rebound off) from the arm's :class:`ArmConfig`; session-less only
  (409 while a hardware session owns the box - volatile settings are re-applied
  at connect anyway).
* ``recover`` — the driver's full user-initiated recovery (clean errors ->
  enable -> servo mode -> ready -> reseed from the MEASURED position); hardware
  session only (409 otherwise: "no hardware session - use clear_errors").
* ``home_rail`` — ``set_linear_track_back_origin`` (+ track enable + speed);
  session-less only (409 "end the session first" while a hardware session
  exists). Gated by a full-travel (0-0.65 m) twin sweep at the arm's current
  posture against the other arm's last monitor sample; ``dry_run`` returns the
  verdict alone (zero writes, ``pre_position`` included). Posture sweep-clear
  (``pre_position.needed == False``) -> synchronous on the monitor's polling
  thread, joints untouched, 200. Not clear but planned (``needed == True``) ->
  asynchronous ``RailHomingJob``: connect this arm's driver alone (rail
  unhomed, 10 % speed, the other arm frozen), execute the planned path under
  the gate, home the rail with the joints held, verify, tear down (brakes on,
  posture HELD - no automatic return), resume the monitor; ``202`` +
  ``status: accepted`` + ``job_id``. A hardware session is refused while any
  arm's rail is unhomed (position unknown -> the twin cannot gate) and while a
  job runs ("rail homing in progress"), so this op is how the operator makes a
  session possible.

``path`` records which thread executed the op; ``before``/``after`` are the
monitor samples taken around it so the UI can show the error code going to 0.
The module is pure pydantic; the request and result export top-level (§14)
while :class:`RailSweepVerdict`, :class:`PrePositionPlan` and (via the embedded
monitor row) :class:`MaintenanceProgress` ride the result's ``$defs``.
:data:`ArmMaintenanceOp`, :data:`MaintenancePhase` and
:class:`MaintenanceProgress` are DEFINED in ``protocol.hardware_monitor`` (the
dependency-free leaf) and re-exported here - the progress block rides
``ArmMonitorTelemetry.maintenance`` while this module embeds
``ArmMonitorTelemetry`` in the result, so defining them here would make the two
modules import each other. Import them from either module.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from apollo_mavis_v2_core.protocol.hardware_monitor import (
    ArmMaintenanceOp,
    ArmMonitorTelemetry,
    MaintenancePhase,
    MaintenanceProgress,
)

# ArmMaintenanceOp = Literal["clear_errors", "apply_backstops", "recover", "home_rail"]
# (defined in hardware_monitor.py, re-exported above - see the module docstring)
# clear_errors:    clean_error + clean_warn, no enable (monitor path, no session)
# apply_backstops: backstops.apply_backstops(api, cfg) (monitor path; 409 in a session)
# recover:         driver _recover(user_initiated=True) (session path; 409 without one)
# home_rail:       twin-gated set_linear_track_back_origin + enable + speed (monitor path;
#                  409 in a session) — the ONE op that moves a mechanical part; since
#                  phase-09d it may first run a planned pre-positioning motion as an
#                  asynchronous RailHomingJob (status "accepted", 202)

MaintenancePath = Literal["monitor", "session"]
# monitor: executed on the read-only monitor's polling thread (no hardware session)
# session: executed on the session driver's monitor thread (request_recovery)

MaintenanceStatus = Literal["done", "accepted", "refused"]
# done:     the op ran to completion synchronously (``ok`` says whether it succeeded;
#           dry-run verdicts are "done" too) - HTTP 200
# accepted: an asynchronous job was started (phase-09d RailHomingJob: the posture needs a
#           planned pre-positioning motion first) - HTTP 202, ``job_id`` set, progress on
#           ArmMonitorTelemetry.maintenance, the final result at GET .../maintenance/last
# refused:  nothing ran, nothing written (e.g. the sweep is blocked and no rail-safe plan
#           was found) - ``ok`` False, an operator suggestion in ``detail``


class ArmMaintenanceRequest(BaseModel):
    """``POST /api/hardware/arms/{arm_id}/maintenance`` body."""

    op: ArmMaintenanceOp
    dry_run: bool = False  # home_rail only: sweep verdict alone, zero writes (additive)


class PrePositionPlan(BaseModel):
    """Twin-planned pre-positioning motion that makes ``home_rail`` possible (phase-09d).

    Rides :attr:`RailSweepVerdict.pre_position`. ``needed`` is ``False`` when
    the arm's CURRENT posture is already sweep-clear (the rail homes with the
    joints untouched, 09c path). Otherwise the runtime tries candidate postures
    in order (``source``: the scene keyframe's 7 joints for this arm, then the
    ``<arm>_home`` keyframe; ``"search"`` is reserved for a sampled posture
    around a candidate and is not produced in phase-09d) and keeps the first
    that is sweep-clear over the full travel AND reachable by a twin RRT-Connect
    plan from the current posture whose EVERY waypoint is collision-free for
    EVERY rail position (``checked_rail_positions`` = 131 at 5 mm steps - the
    carriage is unknown, so the path must be position-agnostic; this check is
    the ONLY safety basis of the motion). ``clear`` is that verdict;
    ``needed and not clear`` means no plan was found and the op is refused
    (``detail`` tells the operator what to do, e.g. fold the arm toward the
    factory-zero posture in Studio and retry). ``duration_s`` is the estimated
    execution time at ``speed_scale`` 0.1 so the UI can say "~X s".
    """

    needed: bool  # False = current posture already sweep-clear (no motion before homing)
    source: Literal["current", "keyframe", "home", "search"] = "current"
    # current:  no plan needed (target_q = the current posture, or empty)
    # keyframe: the scene keyframe's 7 joints for this arm (factory-zero folded posture)
    # home:     the <arm>_home keyframe
    # search:   sampled around a candidate (reserved for later; not produced in phase-09d)
    target_q: list[float] = []  # 7 joints, rad - the posture the arm HOLDS after homing
    waypoints: int = 0  # planned joint-space waypoints (0 when not needed)
    duration_s: float = 0.0  # estimated execution time at speed_scale 0.1 (s)
    checked_rail_positions: int = 0  # 131 when the position-agnostic validation ran
    clear: bool = True  # path clear for EVERY rail position (the only safety basis)
    detail: str = ""  # operator-facing summary / why no plan was found


class RailSweepVerdict(BaseModel):
    """Digital-twin rail sweep that gates ``home_rail`` (phase-09c).

    The carriage position is UNKNOWN while the track is unhomed, so the runtime
    sweeps the full travel: it poses the twin at the arm's current 7 joints
    (``q_checked``), the other arm(s) at their last monitor sample
    (``other_arms``: q7 + rail position, ``rail_fallback_m`` when the rail is
    unknown — recorded in ``assumptions``), and steps the target arm's rail
    slot from 0 to ``travel_m`` in ``step_m`` increments, checking every
    monitored geometry pair at ``inflation_m``. ``clear`` iff no step violates;
    otherwise ``first_blocked_m`` / ``first_blocked_pair`` name the first
    blocking position and pair. ``min_clearance_*`` report the tightest pair
    over the whole sweep (``None`` when no pair was measured). ``sample_seq``
    is the monitor sample the posture came from; the executing monitor
    re-samples and refuses if the joints moved since. ``pre_position``
    (phase-09d, additive) carries the planned pre-positioning motion when the
    posture is not clear (``needed`` True) or says none is needed; ``None`` =
    a pre-09d producer / planning not evaluated.
    """

    scene_id: str
    inflation_m: float  # geometry inflation used by the sweep (m)
    step_m: float  # rail step between checks (m)
    travel_m: float = 0.65  # rail travel swept, 0..travel_m (m)
    clear: bool
    first_blocked_m: float | None = None  # rail position of the first violation (m)
    first_blocked_pair: list[str] = []  # [geom_a, geom_b] at first_blocked_m
    min_clearance_m: float | None = None  # tightest pair distance over the sweep (m)
    min_clearance_at_m: float | None = None  # rail position of that minimum (m)
    min_clearance_pair: list[str] = []  # [geom_a, geom_b] of that minimum
    q_checked: list[float] = []  # the 7 joints the sweep assumed (must match at execution)
    other_arms: dict[str, list[float]] = {}  # arm_id -> q7 + rail used for the other arm(s)
    assumptions: list[str] = []  # e.g. "view rail unknown - used fallback 0.00 m"
    sample_seq: int = 0  # monitor sample the posture came from
    pre_position: PrePositionPlan | None = None  # phase-09d plan (None = not evaluated)


class ArmMaintenanceResult(BaseModel):
    """Outcome of one maintenance op (REST response; 200 whether or not ``ok``).

    ``sdk_codes`` maps every SDK call the op made to its return code, in call
    order (``{"clean_error": 0, "clean_warn": 0}``); ``warnings`` lists the
    non-fatal ``apply_backstops`` codes (e.g. a self-collision tool model the
    firmware rejected). ``before``/``after`` are ``None`` when the executing
    path had no monitor sample (session path, or the monitor never connected).
    ``rail_sweep`` is the twin verdict for ``home_rail`` (dry-run or real;
    ``None`` for the other ops) — a refused sweep is ``ok=False`` with the
    verdict and an empty ``sdk_codes`` (zero writes).

    phase-09d (additive): ``status`` says how the op ran — ``"done"`` (the
    default; synchronous, 200), ``"accepted"`` (an asynchronous ``RailHomingJob``
    started because the posture needs a planned pre-positioning motion; 202,
    ``job_id`` set, ``ok`` True, ``sdk_codes`` empty so far — progress rides
    ``ArmMonitorTelemetry.maintenance`` and the job's final result, again an
    ``ArmMaintenanceResult`` with ``status: done`` and the same ``job_id``,
    replaces it at ``GET .../maintenance/last``) or ``"refused"`` (nothing ran,
    ``ok`` False, suggestion in ``detail``).
    """

    arm_id: str
    op: ArmMaintenanceOp
    path: MaintenancePath
    ok: bool
    detail: str = ""  # human-readable outcome
    sdk_codes: dict[str, int] = {}  # SDK call -> return code, in call order
    warnings: list[str] = []  # apply_backstops non-fatal codes
    before: ArmMonitorTelemetry | None = None
    after: ArmMonitorTelemetry | None = None
    rail_sweep: RailSweepVerdict | None = None  # home_rail twin verdict (additive)
    status: MaintenanceStatus = "done"  # accepted = async job started (202) (phase-09d)
    job_id: str | None = None  # RailHomingJob id when status == "accepted" (phase-09d)


__all__ = [
    "ArmMaintenanceOp",
    "MaintenancePath",
    "MaintenanceStatus",
    "MaintenancePhase",
    "ArmMaintenanceRequest",
    "PrePositionPlan",
    "RailSweepVerdict",
    "ArmMaintenanceResult",
    "MaintenanceProgress",
]
