"""Digital-twin / collision consumer contract (design doc 01-core §5.2).

Implemented by the sim package over MuJoCo; consumed by the runtime safety
gate. Semantics live in 11-safety-collision.md.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from ..state import ArmState, CameraFrame

if TYPE_CHECKING:
    from ..schemas.safety import CollisionReport, PlanRequest, PlanResult


@dataclass(frozen=True, eq=False)
class PairClearance:
    """Signed clearance between one monitored geom pair."""

    geom1: str
    geom2: str
    body_pair: tuple[str, str]
    dist_m: float  # signed; <= 0 = hulls touching
    fromto: np.ndarray | None = None  # (6,) witness segment (mj_geomDistance)

    def __post_init__(self) -> None:
        if self.fromto is not None:
            fromto = np.asarray(self.fromto, dtype=np.float64)
            if fromto.shape != (6,):
                raise ValueError(f"fromto must have shape (6,), got {fromto.shape}")
            object.__setattr__(self, "fromto", fromto)


@runtime_checkable
class DigitalTwinInterface(Protocol):
    """Kinematic mirror of the workcell for collision checking and planning."""

    def sync(self, states: Mapping[str, ArmState]) -> None:
        """Write measured q into the twin's qpos."""
        ...

    def check(self, q_by_arm: Mapping[str, np.ndarray]) -> CollisionReport:
        """Check the COMMANDED config, all arms jointly; kinematic-only
        (0.24-0.75 ms for 3 arms)."""
        ...

    def check_config(self, q_full: np.ndarray) -> bool:
        """Planner fast path: collision-free predicate for one full config."""
        ...

    def clearance(self, distmax: float = 0.05) -> list[PairClearance]:
        """Measured-config clearances, ascending; ~1 µs/pair. Telemetry rate,
        not gate rate."""
        ...

    def plan(self, req: PlanRequest) -> PlanResult:
        """RRT-Connect joint-space plan (§6 schemas)."""
        ...

    def render(self, view: str) -> CameraFrame | None:
        """Offscreen render; call from the render thread only."""
        ...

    def set_grasp_whitelist(self, arm_id: str, body_names: list[str]) -> None:
        """Suppress collision pairs with bodies the arm is grasping."""
        ...
