"""Safety wire schemas (design doc 01-core §6).

All pydantic (wire-visible). Semantics live in ``11-safety-collision.md``;
the runtime safety gate and the sim NullGate both speak these models.
Event-like models are frozen (§16).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CommandSource(str, Enum):
    """Origin of the command stream a safety event refers to."""

    TELEOP = "teleop"
    JOINT_JOG = "joint_jog"
    POLICY = "policy"
    TAKEOVER = "takeover"
    PLANNER = "planner"


class CollisionEvent(BaseModel):
    """Rich per-transition collision-gate event (additive detail)."""

    model_config = ConfigDict(frozen=True)

    t: Literal["collision_event"] = "collision_event"
    ts: float  # server monotonic, s
    kind: Literal["blocked", "cleared", "warn", "penetration", "stale_twin"]
    pairs: list[tuple[str, str]]  # body names ("arm0/link5", "arm1/link3")
    dists_m: list[float]  # signed clearance per pair (inflated geoms)
    min_clearance_m: float
    source: CommandSource | None = None
    arm_ids: list[str] = []  # offending arms


class CollisionReport(BaseModel):
    """Gate verdict for one check; the UI shape (05-ui §2)."""

    model_config = ConfigDict(frozen=True)

    blocked: bool  # gate clamping/holding
    severity: Literal["ok", "warn", "blocked"]
    pairs: list[tuple[str, str]] = []
    min_clearance_m: float = 1.0
    violations: list[CollisionEvent] = []  # rich detail; additive
    ts: float = 0.0

    @model_validator(mode="after")
    def _blocked_iff_severity(self) -> CollisionReport:
        if self.blocked != (self.severity == "blocked"):
            raise ValueError("blocked must be True iff severity == 'blocked'")
        return self

    @classmethod
    def ok(cls) -> CollisionReport:
        """A benign report (NullGate / sim default).

        Returns a FRESH instance each call: ``frozen=True`` does not freeze
        list contents, so a shared singleton could be corrupted stack-wide by
        one misbehaving consumer.
        """
        return cls(blocked=False, severity="ok")


class SafetyConfig(BaseModel):
    """Workcell safety-gate tuning (``WorkcellConfig.safety``)."""

    enabled: bool = True  # hardware REFUSES to start if False (§7 validator)
    safety_debug: bool = False  # sim only: run the FULL hardware-mode stack
    geom_inflation_m: float = Field(default=0.008, gt=0)  # TOTAL pair inflation δ
    min_clearance_m: float = 0.0  # extra block threshold above inflation
    warn_clearance_m: float = 0.025  # UI amber
    hysteresis_m: float = 0.002  # unblock needs dist >= δ + this
    max_active_constraint_rows: int = 12  # IK CollisionAvoidanceLimit row cap
    twin_staleness_s: float = 0.15  # gate fails closed past either staleness
    rail_staleness_s: float = 0.5
    input_deadman_s: float = 0.2
    input_ramp_s: float = 0.1
    allowed_pairs_extra: list[tuple[str, str]] = []

    @model_validator(mode="after")
    def _debug_requires_enabled(self) -> SafetyConfig:
        if self.safety_debug and not self.enabled:
            raise ValueError("safety_debug=True requires enabled=True")
        return self


class PlanRequest(BaseModel):
    """Joint-space RRT-Connect request for ``DigitalTwinInterface.plan``."""

    q_start: dict[str, list[float]]  # measured, full q incl. rail slot
    q_goal: dict[str, list[float]]  # e.g. from StateProfile (joint space)
    arm_order: list[str] | None = None  # None -> planner heuristic
    timeout_s: float = 5.0  # per arm
    max_step_rad: float = 0.05  # edge-check resolution (rail 0.01 m)


class PlanResult(BaseModel):
    """Planner outcome; waypoints per arm on success."""

    ok: bool
    waypoints: dict[str, list[list[float]]] = {}
    failure: Literal["goal_in_collision", "start_in_collision", "timeout"] | None = None
    failing_pair: tuple[str, str] | None = None


__all__ = [
    "CommandSource",
    "CollisionEvent",
    "CollisionReport",
    "SafetyConfig",
    "PlanRequest",
    "PlanResult",
]
