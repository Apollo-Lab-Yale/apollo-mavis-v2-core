"""Arm maintenance channel (design doc 01-core §12; phase-09b).

``POST /api/hardware/arms/{arm_id}/maintenance`` (body
:class:`ArmMaintenanceRequest`, response :class:`ArmMaintenanceResult`) lets
the operator clear xArm controller errors and (re)apply the controller-side
safety parameters from the UI. None of the three ops produces motion (measured
2026-09-04 on the Perception Arm: ``clean_error`` moved no joint by more than
5e-5 rad); the rail homing command IS motion and is not a maintenance op.

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

``path`` records which thread executed the op; ``before``/``after`` are the
monitor samples taken around it so the UI can show the error code going to 0.
The module is a leaf like ``hardware_monitor`` (pure pydantic) and both models
export top-level (§14).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from apollo_mavis_v2_core.protocol.hardware_monitor import ArmMonitorTelemetry

ArmMaintenanceOp = Literal["clear_errors", "apply_backstops", "recover"]
# clear_errors:    clean_error + clean_warn, no enable (monitor path, no session)
# apply_backstops: backstops.apply_backstops(api, cfg) (monitor path; 409 in a session)
# recover:         driver _recover(user_initiated=True) (session path; 409 without one)

MaintenancePath = Literal["monitor", "session"]
# monitor: executed on the read-only monitor's polling thread (no hardware session)
# session: executed on the session driver's monitor thread (request_recovery)


class ArmMaintenanceRequest(BaseModel):
    """``POST /api/hardware/arms/{arm_id}/maintenance`` body."""

    op: ArmMaintenanceOp


class ArmMaintenanceResult(BaseModel):
    """Outcome of one maintenance op (REST response; 200 whether or not ``ok``).

    ``sdk_codes`` maps every SDK call the op made to its return code, in call
    order (``{"clean_error": 0, "clean_warn": 0}``); ``warnings`` lists the
    non-fatal ``apply_backstops`` codes (e.g. a self-collision tool model the
    firmware rejected). ``before``/``after`` are ``None`` when the executing
    path had no monitor sample (session path, or the monitor never connected).
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


__all__ = [
    "ArmMaintenanceOp",
    "MaintenancePath",
    "ArmMaintenanceRequest",
    "ArmMaintenanceResult",
]
