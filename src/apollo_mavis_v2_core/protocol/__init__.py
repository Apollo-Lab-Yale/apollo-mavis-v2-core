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
    TrackerSettingsArgs,
    parse_client_msg,
    validate_action_args,
)
from .hardware_monitor import (
    ArmMonitorStatus,
    ArmMonitorTelemetry,
    HardwareMonitorTelemetry,
    TwinOverlayStatus,
    TwinOverlayTelemetry,
)
from .keymap import (
    DISCRETE_CODES,
    HELD_CODES,
    HELD_MODIFIER_ACTIONS,
    KEYMAP,
    KeymapEntry,
    axis_map,
)
from .maintenance import (
    ArmMaintenanceOp,
    ArmMaintenanceRequest,
    ArmMaintenanceResult,
    MaintenancePath,
    MaintenancePhase,
    MaintenanceProgress,
    MaintenanceStatus,
    PrePositionPlan,
    RailSweepVerdict,
)
from .microphone import MicrophoneInfo, MicStatus
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
    ArmBringupTelemetry,
    ArmTelemetry,
    ClearanceItem,
    ControllerTelemetry,
    DaggerStatus,
    EpisodeStatus,
    InferenceStatus,
    MicrophoneTelemetry,
    PoseMsg,
    SessionTelemetry,
    TelemetryMsg,
    TrackerSettingsMsg,
    TrackerTelemetry,
)

# Keep ``.telemetry`` above ``.tracker``: tracker reuses PoseMsg and telemetry
# embeds TrackerCalibrationStatus (telemetry imports tracker once PoseMsg exists).
from .tracker import (
    CalibrationKind,
    CalibrationOp,
    CalibrationPhase,
    CalibrationValidation,
    LighthouseStatus,
    TrackerCalibrationCommand,
    TrackerCalibrationStatus,
    YawGesturePoint,
    YawPointLabel,
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
    "TrackerSettingsArgs",
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
    "ArmBringupTelemetry",
    "SessionTelemetry",
    "TrackerSettingsMsg",
    "ControllerTelemetry",
    "TrackerTelemetry",
    "MicrophoneTelemetry",
    "TelemetryMsg",
    # tracker calibration
    "CalibrationKind",
    "CalibrationPhase",
    "CalibrationOp",
    "YawPointLabel",
    "LighthouseStatus",
    "CalibrationValidation",
    "YawGesturePoint",
    "TrackerCalibrationStatus",
    "TrackerCalibrationCommand",
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
    # microphone (phase-11)
    "MicStatus",
    "MicrophoneInfo",
    # hardware monitor + twin overlay (phase-09a)
    "ArmMonitorStatus",
    "ArmMonitorTelemetry",
    "TwinOverlayStatus",
    "TwinOverlayTelemetry",
    "HardwareMonitorTelemetry",
    # arm maintenance (phase-09b; home_rail + RailSweepVerdict phase-09c; the async
    # RailHomingJob models phase-09d - MaintenanceProgress / MaintenancePhase /
    # ArmMaintenanceOp are defined in .hardware_monitor and re-exported by .maintenance)
    "ArmMaintenanceOp",
    "MaintenancePath",
    "MaintenanceStatus",
    "MaintenancePhase",
    "ArmMaintenanceRequest",
    "PrePositionPlan",
    "RailSweepVerdict",
    "ArmMaintenanceResult",
    "MaintenanceProgress",
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
    "HELD_MODIFIER_ACTIONS",
    "axis_map",
]
