"""DAgger core types (design doc 01-core §9).

Shared by DAgger AND inference: the identical TakeoverGate gates Space in
both; the *presence of a recorder* — not a flag — distinguishes them
(12-dagger §3, binding). Hot-path payloads are frozen dataclasses;
``TrainerStatus`` is pydantic because it rides telemetry (§11).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from pydantic import BaseModel


class ControlMode(str, Enum):
    POLICY = "policy"
    HUMAN = "human"
    TAKEOVER_TRANSITION = "takeover_transition"

    def to_int8(self) -> int:
        """Dataset encoding (12-dagger §4): POLICY 0, HUMAN 1, TRANSITION 2."""
        return _MODE_TO_INT8[self]


_MODE_TO_INT8: dict[ControlMode, int] = {
    ControlMode.POLICY: 0,
    ControlMode.HUMAN: 1,
    ControlMode.TAKEOVER_TRANSITION: 2,
}


@dataclass(frozen=True)
class GateEvent:
    """Emitted on every mode change, per arm."""

    arm_id: str
    mode: ControlMode  # the NEW mode
    t_mono: float
    seq: int  # monotonic per session
    source: str  # "keyboard" | "auto_advance" | "episode_reset"


@dataclass(frozen=True, eq=False)
class FrameAnnotations:
    """Extra per-frame dataset columns."""

    control_mode: ControlMode
    executed_action: np.ndarray  # post-twin-gate value, canonical frame = LABEL
    policy_action: np.ndarray | None  # counterfactual; None -> NaN row in dataset
    policy_version: int
    action_frame: str  # FrameRef: "arm_base:<id>" | "world" | "camera:<id>"


@dataclass(frozen=True)
class CheckpointInfo:
    """manifest.json payload (12-dagger §7)."""

    run_id: str
    version: int  # monotonic within run
    path: str  # checkpoints/{run_id}/v{n:06d}/
    parent_version: int | None
    trained_on_frames: int  # dataset watermark (label frames consumed)
    trained_on_episodes: list[int]
    action_frame: str  # must match session; FrameRef
    action_space: str  # "delta_ee" | "abs_ee" | "joint"
    sanity_ok: bool
    mean_loss: float
    sha256: str  # of state_dict.pt (verified before swap)
    created_wallclock_ns: int


@dataclass(frozen=True)
class EpisodeSummary:
    """Runtime -> trainer at every save_episode."""

    episode_index: int
    n_frames: int
    n_intervention_frames: int  # control_mode != POLICY
    n_label_frames: int  # control_mode == HUMAN (HG-DAgger Eq. 2)
    takeover_segments: int  # maximal runs of control_mode != POLICY
    segment_doubts: list[float]  # policy-action variance at takeover instants
    success: bool | None
    episode_id: str = ""  # 10-frames §11.3 directory id (2026-09-07); the spool is
    #   ``trainer_spool/ep_<episode_id>.parquet``; ``episode_index`` stays the
    #   capture-order ordinal the trainer watermark uses (12-dagger §7)
    # Online DAgger actor split (additive, phase-14; 15-online-dagger §4): frames with the
    # ``actor`` column 1 (expert: control_mode != POLICY) / 0 (novice: the policy drove).
    # ``n_expert_frames`` counts transition frames too, unlike ``n_label_frames``.
    n_expert_frames: int = 0
    n_novice_frames: int = 0


class TrainerStatus(BaseModel):
    """Trainer health; pydantic — rides telemetry (§11)."""

    state: Literal["starting", "idle", "training", "dead"]
    steps_total: int = 0
    last_burst_loss: float | None = None
    last_checkpoint_version: int | None = None
    last_checkpoint_ts: float | None = None
    new_label_frames: int = 0  # progress toward the 100-frame trigger


__all__ = [
    "ControlMode",
    "GateEvent",
    "FrameAnnotations",
    "CheckpointInfo",
    "EpisodeSummary",
    "TrainerStatus",
]
