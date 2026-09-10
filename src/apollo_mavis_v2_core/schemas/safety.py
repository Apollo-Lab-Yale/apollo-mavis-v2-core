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
    # Range of the 25 Hz clearance sweep behind ``telemetry.clearances`` (pairs at or
    # beyond it are not reported). 0.10 m since 2026-09-07 so the Cockpit's
    # proximity frame can fade in before the 0.05 m "close" grade; was 0.05.
    clearance_sweep_m: float = Field(default=0.10, gt=0)
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
    # The ``SessionSpec.speed_scale`` the plan will be executed at (additive, 2026-09-09).
    # The gate's escape rule demands a strict opening on EVERY executor tick of an arm
    # inside the inflation shell, and a slower session has smaller ticks, so the planner
    # judges an escape from a pinched start at this speed; a plan judged at a slower speed
    # stays valid at any faster one. Default: the slowest speed the runtime offers.
    speed_scale: float = Field(default=0.1, gt=0, le=1)


class PlanResult(BaseModel):
    """Planner outcome; waypoints per arm on success.

    ``arm_order`` is the order the sequential planner validated: arm k was
    planned with arms < k frozen at their GOALS and arms > k at their STARTS,
    so the per-arm paths are collision-free ONLY when executed in this order,
    one arm after another (never simultaneously; 11-safety §9). Executors MUST
    honour it. Empty on failure. Additive (2026-09-08): a ``reset_to_initial``
    on the real cell executed both arms' waypoints at once through combinations
    the planner never validated and sat gate-blocked until the budget.

    ``failure == "no_escape"`` (additive, 2026-09-09): the arm starts inside the
    inflation shell and the planner's escape phase — which mirrors the gate's
    strict-opening rule (11-safety §7.1 step 6 / §9) — found no step that opens
    every pinched pair without closing another; ``failing_pair`` names the
    tightest pair. The arm is boxed in: nothing is safe to execute.
    """

    ok: bool
    waypoints: dict[str, list[list[float]]] = {}
    failure: (
        Literal["goal_in_collision", "start_in_collision", "no_escape", "timeout"] | None
    ) = None
    failing_pair: tuple[str, str] | None = None
    arm_order: list[str] = []


__all__ = [
    "CommandSource",
    "CollisionEvent",
    "CollisionReport",
    "SafetyConfig",
    "PlanRequest",
    "PlanResult",
]
