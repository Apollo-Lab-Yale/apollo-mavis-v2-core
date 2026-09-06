"""apollo-mavis-v2-core: interfaces, schemas, and protocols for the apollo-mavis-v2 stack.

Stable public API re-exports (design doc 01-core §2). Wire-protocol models
live under :mod:`apollo_mavis_v2_core.protocol` and are imported explicitly.
"""

from . import se3
from .bus import Command, CommandBus, CommandResult, LatestSlot
from .dagger import ControlMode
from .errors import (
    ApolloError,
    ArmConnectError,
    ArmIdentityError,
    BringupError,
    CameraInitError,
    CommandError,
    ConfigError,
    FrameRefError,
    GripperInitError,
    ProfileError,
    ProfileNotFoundError,
    RailExpectedError,
    RailNotHomedError,
    RailUnavailableError,
    SchemaExportError,
    VideoFramingError,
    WorkcellBringupError,
)
from .interfaces import (
    ArmInterface,
    CameraInterface,
    DigitalTwinInterface,
    EpisodeRecorder,
    HeldState,
    IKResult,
    IKSolver,
    Observation,
    PairClearance,
    Policy,
    PolicyOutput,
    PolicySpec,
    TeleopInput,
    WorkcellInterface,
)
from .profiles import ProfileStore
from .schemas import (
    ArmConfig,
    ArmPosture,
    CameraConfig,
    CollisionEvent,
    CollisionReport,
    CommandSource,
    PlanRequest,
    PlanResult,
    SafetyConfig,
    StateProfile,
    WorkcellConfig,
    load_workcell_config,
)
from .state import ArmState, CameraFrame, GripperCommand, GripperState
from .types import FrameRef, ParsedFrame, Pose, Transform, Twist, frame_ref, parse_frame

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "se3",
    # geometry
    "Pose",
    "Transform",
    "Twist",
    "FrameRef",
    "ParsedFrame",
    "parse_frame",
    "frame_ref",
    # state
    "ArmState",
    "GripperState",
    "GripperCommand",
    "CameraFrame",
    # interfaces
    "ArmInterface",
    "CameraInterface",
    "WorkcellInterface",
    "IKResult",
    "IKSolver",
    "PairClearance",
    "DigitalTwinInterface",
    "Observation",
    "PolicyOutput",
    "PolicySpec",
    "Policy",
    "EpisodeRecorder",
    "HeldState",
    "TeleopInput",
    # schemas
    "WorkcellConfig",
    "ArmConfig",
    "CameraConfig",
    "SafetyConfig",
    "CommandSource",
    "CollisionEvent",
    "CollisionReport",
    "PlanRequest",
    "PlanResult",
    "StateProfile",
    "ArmPosture",
    "load_workcell_config",
    # profiles
    "ProfileStore",
    # dagger
    "ControlMode",
    # bus
    "Command",
    "CommandResult",
    "CommandBus",
    "LatestSlot",
    # errors
    "ApolloError",
    "ConfigError",
    "FrameRefError",
    "CommandError",
    "RailUnavailableError",
    "ProfileError",
    "ProfileNotFoundError",
    "VideoFramingError",
    "SchemaExportError",
    "BringupError",
    "ArmConnectError",
    "ArmIdentityError",
    "RailExpectedError",
    "RailNotHomedError",
    "GripperInitError",
    "CameraInitError",
    "WorkcellBringupError",
]
