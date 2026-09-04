"""Tracker calibration wire models (design doc 01-core §12; 13-tracker §3/§4).

Two calibrations share one REST endpoint, ``/api/tracker/calibration``
(04-runtime §13.1): ``GET`` returns :class:`TrackerCalibrationStatus`, ``POST``
takes a :class:`TrackerCalibrationCommand` and returns the new status; the
same status also rides telemetry as ``TrackerTelemetry.calibration`` (§11) so
the Devices-page wizard never holds flow state of its own.

* ``base_station`` re-solves the Lighthouse poses with libsurvive's global
  scene solver (capture -> validate -> install).
* ``yaw`` fits ``tracker_settings.yaw_deg`` from the 7-click gesture
  (start, left, forward, right, back, up, down) and applies it.

Calibration is a session-less device-management flow, so it deliberately adds
no ``ActionName`` (§10). Core is the spelling authority for every literal here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from apollo_mavis_v2_core.protocol.telemetry import PoseMsg

CalibrationKind = Literal["none", "base_station", "yaw"]
CalibrationPhase = Literal[
    "idle",
    "starting",
    "capturing",
    "validating",
    "fitting",
    "installing",
    "done",
    "failed",
    "aborted",
]
CalibrationOp = Literal["start", "capture", "validate", "install", "apply", "abort"]
YawPointLabel = Literal["start", "left", "forward", "right", "back", "up", "down"]


class LighthouseStatus(BaseModel):
    """One Lighthouse base station as seen during base-station calibration."""

    index: int  # libsurvive LH index
    channel: int | None = None  # OOTX channel (mode); None until reported
    serial: str | None = None  # base-station serial; None until reported
    pose: PoseMsg | None = None  # lighthouse-world pose (m, wxyz)
    scenes: int = 0  # GSS scenes solved for this station
    reference: bool = False  # "Using LH i as reference lighthouse"


class CalibrationValidation(BaseModel):
    """Stationary-controller validation result (13-tracker §4).

    Acceptance (2026-09-03 measurements): every axis ``std_mm`` below
    ``threshold_std_mm`` and ``max_step_mm`` below ``threshold_step_mm``.
    """

    samples: int = 0  # valid samples after the skip window
    std_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)  # per-axis position std
    max_step_mm: float = 0.0  # largest jump between adjacent samples
    threshold_std_mm: float = 5.0
    threshold_step_mm: float = 20.0
    passed: bool = False


class YawGesturePoint(BaseModel):
    """One captured point of the yaw gesture."""

    label: YawPointLabel
    pose: PoseMsg  # RAW lighthouse-world pose at the click


class TrackerCalibrationStatus(BaseModel):
    """Calibration state machine snapshot (REST response + telemetry block).

    ``kind``/``phase`` describe the flow in progress; the base-station and yaw
    field groups are only meaningful for their own kind. The persisted-state
    group is always filled from ``calibration_dir/tracker_calibration.json``.
    """

    kind: CalibrationKind = "none"
    phase: CalibrationPhase = "idle"
    detail: str = ""  # operator-facing progress / failure reason
    started_at: float | None = None  # unix s
    elapsed_s: float | None = None
    # base-station
    scenes: int = 0  # max over stations
    lighthouses: list[LighthouseStatus] = Field(default_factory=list)
    stations_visible: int = 0
    controller_still: bool | None = None  # None = no fresh samples
    validation: CalibrationValidation | None = None
    installed_path: str | None = None  # libsurvive config replaced on install
    backup_path: str | None = None  # <installed_path>.bak-YYYYMMDD-HHMMSS
    # yaw
    yaw_points: list[YawGesturePoint] = Field(default_factory=list)
    next_point: YawPointLabel | None = None  # None once all seven are captured
    fitted_yaw_deg: float | None = None
    fit_residual_deg: float | None = None
    fit_checks: list[str] = Field(default_factory=list)  # failed checks; empty = ok
    applied_yaw_deg: float | None = None
    # persisted calibration state (always filled)
    yaw_valid: bool = True  # False after a base-station install until yaw is redone
    yaw_calibrated_at: float | None = None  # unix s
    base_station_installed_at: float | None = None  # unix s


class TrackerCalibrationCommand(BaseModel):
    """POST /api/tracker/calibration body (illegal transitions -> 409)."""

    kind: Literal["base_station", "yaw"]
    op: CalibrationOp
    point: YawPointLabel | None = None  # yaw 'capture' label; None = next_point


__all__ = [
    "CalibrationKind",
    "CalibrationPhase",
    "CalibrationOp",
    "YawPointLabel",
    "LighthouseStatus",
    "CalibrationValidation",
    "YawGesturePoint",
    "TrackerCalibrationStatus",
    "TrackerCalibrationCommand",
]
