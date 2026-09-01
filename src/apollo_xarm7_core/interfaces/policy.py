"""Policy consumer contract (design doc 01-core §5.2).

Implemented by the runtime's policy wrappers (torch lives THERE, never in
core); consumed by inference and DAgger sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, eq=False)
class Observation:
    """One policy input frame."""

    state: np.ndarray  # float32, layout per PolicySpec.state_names
    images: dict[str, np.ndarray]  # camera_id -> (H, W, 3) uint8
    t_mono: float
    wallclock_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", np.asarray(self.state, dtype=np.float32))


@dataclass(frozen=True, eq=False)
class PolicyOutput:
    """One policy action frame."""

    actions: np.ndarray  # float32; per-arm blocks in WorkcellConfig arm order,
    #   layout per PolicySpec.action_names
    version: int
    t_mono: float
    chunk_remaining: int = 0  # > 0 for chunked policies (ACT / diffusion)

    def __post_init__(self) -> None:
        object.__setattr__(self, "actions", np.asarray(self.actions, dtype=np.float32))


@dataclass(frozen=True)
class PolicySpec:
    """Metadata carried by every checkpoint."""

    action_space: Literal["delta_ee", "abs_ee", "joint"]
    action_frame: str  # full FrameRef: "arm_base:<arm_id>" | "world" |
    #   "camera:<camera_id>" (per-arm; never bare "base" — 10-frames §1.3)
    action_names: list[str]
    state_names: list[str]
    camera_keys: list[str]
    version: int


@runtime_checkable
class Policy(Protocol):
    """A loaded policy checkpoint."""

    spec: PolicySpec  # property on implementations

    def reset(self) -> None:
        """Drop chunks/history (takeover handback)."""
        ...

    def act(self, obs: Observation) -> PolicyOutput: ...

    def load_weights(self, path: str) -> None:
        """state_dict hot-swap; episode boundaries only."""
        ...
