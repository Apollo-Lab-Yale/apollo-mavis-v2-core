"""DAgger / interactive-learning core types and Protocols (design doc 01-core §9)."""

from .interfaces import (
    AsyncTrainerClient,
    InterventionRecorder,
    PolicyReloader,
    TakeoverGate,
)
from .types import (
    CheckpointInfo,
    ControlMode,
    EpisodeSummary,
    FrameAnnotations,
    GateEvent,
    TrainerStatus,
)

__all__ = [
    "ControlMode",
    "GateEvent",
    "FrameAnnotations",
    "CheckpointInfo",
    "EpisodeSummary",
    "TrainerStatus",
    "TakeoverGate",
    "InterventionRecorder",
    "PolicyReloader",
    "AsyncTrainerClient",
]
