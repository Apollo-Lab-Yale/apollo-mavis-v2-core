"""Runtime <-> UI wire protocol (design doc 01-core §10-§14)."""

from .control import (
    AckMsg,
    ActionMsg,
    ActionName,
    ControlClientMsg,
    ControlServerMsg,
    HelloMsg,
    JointTargetArgs,
    KeysMsg,
    SaveProfileArgs,
    SetInitialConditionArgs,
    parse_client_msg,
    validate_action_args,
)
from .keymap import DISCRETE_CODES, HELD_CODES, KEYMAP, KeymapEntry, axis_map
from .session import (
    START_FROM_RE,
    ArmStatusInfo,
    CameraInfo,
    Mode,
    PolicyInfo,
    ProfileInfo,
    SceneInfo,
    SessionInfo,
    SessionSpec,
    WorkcellStatus,
)
from .telemetry import (
    ArmTelemetry,
    ClearanceItem,
    DaggerStatus,
    EpisodeStatus,
    InferenceStatus,
    PoseMsg,
    SessionTelemetry,
    TelemetryMsg,
)
from .video import (
    HEADER_FMT,
    HEADER_SIZE,
    RESERVED_STREAM_IDS,
    is_reserved_stream,
    pack_frame,
    unpack_header,
)

__all__ = [
    # control
    "ActionName",
    "HelloMsg",
    "KeysMsg",
    "ActionMsg",
    "AckMsg",
    "JointTargetArgs",
    "SaveProfileArgs",
    "SetInitialConditionArgs",
    "ControlClientMsg",
    "ControlServerMsg",
    "parse_client_msg",
    "validate_action_args",
    # telemetry
    "PoseMsg",
    "ArmTelemetry",
    "ClearanceItem",
    "EpisodeStatus",
    "DaggerStatus",
    "InferenceStatus",
    "SessionTelemetry",
    "TelemetryMsg",
    # session
    "Mode",
    "START_FROM_RE",
    "SessionSpec",
    "SessionInfo",
    "ArmStatusInfo",
    "CameraInfo",
    "WorkcellStatus",
    "SceneInfo",
    "ProfileInfo",
    "PolicyInfo",
    # video
    "HEADER_FMT",
    "HEADER_SIZE",
    "RESERVED_STREAM_IDS",
    "pack_frame",
    "unpack_header",
    "is_reserved_stream",
    # keymap
    "KeymapEntry",
    "KEYMAP",
    "HELD_CODES",
    "DISCRETE_CODES",
    "axis_map",
]
