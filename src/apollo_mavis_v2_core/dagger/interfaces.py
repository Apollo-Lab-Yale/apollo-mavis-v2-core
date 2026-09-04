"""DAgger consumer contracts (design doc 01-core §9).

All ``@runtime_checkable`` Protocols; implementations live in the runtime
package (gate, DaggerRecorder, reloader, ZMQ trainer client).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .types import (
    CheckpointInfo,
    ControlMode,
    EpisodeSummary,
    FrameAnnotations,
    GateEvent,
    TrainerStatus,
)


@runtime_checkable
class TakeoverGate(Protocol):
    """Space-toggle takeover state machine (impl: runtime/dagger/gate.py)."""

    def mode(self, arm_id: str) -> ControlMode: ...

    def engaged_arm(self) -> str | None:
        """Arm currently in HUMAN/TRANSITION, else None."""
        ...

    def on_toggle(self, arm_id: str, t_mono: float) -> GateEvent | None:
        """ActionMsg{takeover_toggle} handler; None -> caller Nacks."""
        ...

    def tick(self, t_mono: float) -> list[GateEvent]:
        """100 Hz; auto-advances TRANSITION -> HUMAN after T_blend (0.3 s)."""
        ...

    def reset(self) -> None:
        """Episode boundary -> all arms back to POLICY."""
        ...


@runtime_checkable
class InterventionRecorder(Protocol):
    """Annotation-aware episode recorder (impl: runtime DaggerRecorder, LeRobot v3)."""

    def start_episode(self, meta: dict[str, object]) -> str: ...

    def add_frame(self, obs: dict[str, object], ann: FrameAnnotations) -> None: ...

    def end_episode(self, success: bool | None, rerecord: bool) -> EpisodeSummary: ...


@runtime_checkable
class PolicyReloader(Protocol):
    """Checkpoint hot-swap coordinator (impl: runtime reloader, 12-dagger §8)."""

    def stage(self, ckpt: CheckpointInfo) -> None:
        """Stage a verified checkpoint; keeps only the newest."""
        ...

    def maybe_swap(
        self, at_episode_boundary: bool, current_mode: ControlMode
    ) -> int | None:
        """Swap if safe; returns the new version, else None."""
        ...

    def rollback(self) -> int:
        """Revert to LAST_KNOWN_GOOD; allowed mid-episode."""
        ...

    def mark_good(self) -> None:
        """Clean episode completed on the current version."""
        ...


@runtime_checkable
class AsyncTrainerClient(Protocol):
    """Trainer-process client (impl: runtime, ZMQ + filesystem)."""

    def submit_episode(self, episode_path: str, summary: EpisodeSummary) -> None: ...

    def poll_checkpoint(self) -> CheckpointInfo | None:
        """Newest sanity_ok checkpoint above the current version, else None."""
        ...

    def status(self) -> TrainerStatus | None:
        """None = unreachable/dead."""
        ...

    def request_stop(self) -> None:
        """Graceful stop; implementation escalates."""
        ...


__all__ = [
    "TakeoverGate",
    "InterventionRecorder",
    "PolicyReloader",
    "AsyncTrainerClient",
]
