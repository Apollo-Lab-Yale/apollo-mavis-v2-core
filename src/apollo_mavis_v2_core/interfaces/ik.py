"""IK solver consumer contract (design doc 01-core §5.2).

Implemented by the sim package (mink QP over the digital twin); consumed by
the runtime control loop.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..state import ArmState
from ..types import Pose


@dataclass(frozen=True, eq=False)
class IKResult:
    """Outcome of one differential IK solve for a single arm."""

    q: np.ndarray  # full config (7|8, incl. rail slot)
    pos_err_m: float
    rot_err_rad: float
    diverged: bool  # velocity IK converges silently to the nearest reachable
    #   pose; runtime re-anchors the teleop target on True
    active_collision_rows: int
    solve_time_s: float

    def __post_init__(self) -> None:
        q = np.asarray(self.q, dtype=np.float64)
        if q.shape not in ((7,), (8,)):
            raise ValueError(f"q must have shape (7,) or (8,), got {q.shape}")
        object.__setattr__(self, "q", q)


@runtime_checkable
class IKSolver(Protocol):
    """Differential IK over the shared kinematic scene."""

    def solve(self, arm_id: str, target: Pose, q_seed: np.ndarray) -> IKResult:
        """One differential step (mink QP measured ~0.12 ms/arm, ~8.6 kHz)."""
        ...

    def solve_to_convergence(
        self,
        arm_id: str,
        target: Pose,
        q_seed: np.ndarray,
        max_steps: int = 50,
        n_restarts: int = 4,
    ) -> IKResult:
        """One-shot far targets (goto / planner endpoints); 2-4 ms measured."""
        ...

    def sync_passive(self, states: Mapping[str, ArmState]) -> None:
        """Pin non-active arms to their measured configurations."""
        ...

    def reset(self, arm_id: str, q_measured: np.ndarray) -> None:
        """Re-seed + clear accel/jerk history; MANDATORY after recovery/takeover."""
        ...
