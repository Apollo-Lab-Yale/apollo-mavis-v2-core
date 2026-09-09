"""External (dora) interface spellings and JSON payload models (14-dora §4/§5/§13).

This module is the SPELLING AUTHORITY for every dora node / stream / command
id the runtime publishes or consumes and for the JSON payloads that ride the
low-rate streams (``session``, ``policy_spec``, ``policy_reset``, ``events``).
It is plain pydantic + stdlib: core stays dora- and pyarrow-free (the ruff
``banned-api`` list and ``tests/test_import_guard.py`` enforce it); the runtime's
``dora_bridge`` package and the independent ``apollo-mavis-v2-policy-node``
repo (its ``contract.py`` copies these spellings, never imports them) are the
only places that touch dora.

Wire conventions (14-dora §3): typed data is a flat Arrow array + metadata;
structured data is one ``Utf8`` scalar holding the JSON of one of the models
below; units are metres / radians / seconds; quaternions are **wxyz** (every
quaternion field name ends in ``_wxyz`` or the metadata says ``quat_order:
"wxyz"``); ``mavis_schema`` rides every metadata dict and every JSON payload.

Phase-14 (15-online-dagger §6) adds the Online DAgger trainer contract: the generic
``trainer_status`` stream, ``PolicySpecAnnounce.capabilities``,
``SessionAnnounce.online_dagger`` and the ``train_now`` event. History: the v1.0
PRO-DAgger shell superseded 2026-09-08 carried algorithm-specific payloads here;
none of it shipped and none of it survives — the runtime is algorithm-agnostic.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from apollo_mavis_v2_core.protocol.session import SessionSpec
from apollo_mavis_v2_core.types import FrameRef

# -- schema + node ids ------------------------------------------------------------------
MAVIS_SCHEMA = 1  # bump ONLY on a breaking wire change (14-dora §3.3)
EXTERNAL_NODE_ID = "mavis_runtime"  # the runtime attaches as Node(EXTERNAL_NODE_ID)
DATAFLOW_NAME = "mavis_v2"  # default `dora start --name`

# Placeholder dynamic nodes the runtime's dataflow declares so foreign processes
# can attach under a known id (14-dora §2.3). Ids use [a-zA-Z0-9_] only.
PLACEHOLDER_POLICY = "policy"  # the policy repo attaches as Node("policy")
PLACEHOLDER_VIEWER = "viewer"  # fixed-viewpoint consumer; inputs only, no outputs
PLACEHOLDER_OBSERVER = "observer"  # loggers / rerun bridges; receives everything
PLACEHOLDERS: tuple[str, ...] = (PLACEHOLDER_POLICY, PLACEHOLDER_VIEWER, PLACEHOLDER_OBSERVER)
PROBE_NODE_ID = "probe"  # the ONLY spawned node: dataflow keepalive + liveness canary
# Per-machine placeholders for LAN subscribers (v0.3, §9): ``viewer_<id>`` / ``observer_<id>``.
REMOTE_PLACEHOLDER_KINDS: tuple[str, ...] = (PLACEHOLDER_VIEWER, PLACEHOLDER_OBSERVER)


def remote_placeholder_id(kind: str, machine_id: str) -> str:
    """``viewer`` + ``gpubox`` -> ``viewer_gpubox`` (14-dora §2.3)."""
    return f"{kind}_{machine_id}"


# -- runtime OUTPUT ids (14-dora §4.1) ------------------------------------------------------
OUT_HEARTBEAT = "heartbeat"
OUT_SESSION = "session"
OUT_TELEMETRY = "telemetry"
OUT_EVENTS = "events"
OUT_ARM_STATE = "arm_state"
OUT_ARM_CMD = "arm_cmd"
OUT_OBS_STATE = "obs_state"
OUT_POLICY_RESET = "policy_reset"
RUNTIME_FIXED_OUTPUTS: tuple[str, ...] = (
    OUT_HEARTBEAT,
    OUT_SESSION,
    OUT_TELEMETRY,
    OUT_EVENTS,
    OUT_ARM_STATE,
    OUT_ARM_CMD,
    OUT_OBS_STATE,
    OUT_POLICY_RESET,
)
CAMERA_OUTPUT_PREFIX = "cam_"
DEPTH_OUTPUT_SUFFIX = "_depth"
MIC_OUTPUT_PREFIX = "mic_"


def camera_output_id(camera_id: str) -> str:
    """``view_wrist_cam`` -> ``cam_view_wrist_cam``."""
    return f"{CAMERA_OUTPUT_PREFIX}{camera_id}"


def depth_output_id(camera_id: str) -> str:
    """``view_wrist_cam`` -> ``cam_view_wrist_cam_depth``."""
    return f"{CAMERA_OUTPUT_PREFIX}{camera_id}{DEPTH_OUTPUT_SUFFIX}"


def mic_output_id(mic_id: str) -> str:
    """``mic_view`` -> ``mic_mic_view``."""
    return f"{MIC_OUTPUT_PREFIX}{mic_id}"


# -- runtime INPUT ids (14-dora §5) ---------------------------------------------------------
IN_TICK = "tick"  # dora/timer/hz/10 - bridge watchdog
IN_PROBE_HEARTBEAT = "probe_heartbeat"  # probe/heartbeat - dataflow liveness
IN_POLICY_ACTION = "policy_action"  # policy/action
IN_POLICY_SPEC = "policy_spec"  # policy/spec (1 Hz heartbeat)
IN_POLICY_STATUS = "policy_status"  # policy/status (free text)
IN_POLICY_TRAINER_STATUS = "policy_trainer_status"  # policy/trainer_status (JSON
#   TrainerStatusAnnounce, 1 Hz + on change; dataflow queue_size 8) — phase-14,
#   15-online-dagger §6
RUNTIME_INPUTS: tuple[str, ...] = (  # append-only: the order is the contract golden's
    IN_TICK,
    IN_PROBE_HEARTBEAT,
    IN_POLICY_ACTION,
    IN_POLICY_SPEC,
    IN_POLICY_STATUS,
    IN_POLICY_TRAINER_STATUS,
)
TICK_TIMER = "dora/timer/hz/10"
PROBE_TIMER = "dora/timer/hz/1"

# Outputs of the POLICY placeholder node (the policy repo's side of §5).
POLICY_OUT_ACTION = "action"
POLICY_OUT_SPEC = "spec"
POLICY_OUT_STATUS = "status"
POLICY_OUT_TRAINER_STATUS = "trainer_status"  # Online DAgger trainer role (15-online-dagger §6)
POLICY_OUTPUTS: tuple[str, ...] = (  # append-only: the order is the contract golden's
    POLICY_OUT_ACTION,
    POLICY_OUT_SPEC,
    POLICY_OUT_STATUS,
    POLICY_OUT_TRAINER_STATUS,
)
PROBE_OUT_HEARTBEAT = "heartbeat"

# Reserved (declared in the compatibility ledger, NOT implemented in v1; 14-dora §5).
RESERVED_IDS: tuple[str, ...] = ("weights_reload", "weights_ack", "cmd_request", "cmd_response")

# -- metadata keys carried by every message (14-dora §3.2) ------------------------------------
META_SCHEMA = "mavis_schema"
META_EPOCH = "epoch"
META_SESSION_ID = "session_id"
META_SEQ = "seq"
META_T_MONO = "t_mono"
META_WALLCLOCK_NS = "wallclock_ns"
META_CLIENT = "client"  # inbound: free-form id of the sending process
COMMON_OUTPUT_METADATA: tuple[str, ...] = (
    META_SCHEMA,
    META_EPOCH,
    META_SESSION_ID,
    META_SEQ,
    META_T_MONO,
    META_WALLCLOCK_NS,
)
# Required metadata on ``policy_action`` beyond the common inbound keys (§5).
POLICY_ACTION_REQUIRED_METADATA: tuple[str, ...] = (
    "observation_id",
    "chunk_len",
    "action_dim",
    "chunk_dt_s",
    "policy_id",
    "policy_version",
)

# -- arm_state block (14-dora §4.2): 32 values per arm, in this order ----------------------------
ARM_STATE_LAYOUT: tuple[str, ...] = (
    *(f"q{i}" for i in range(1, 8)),
    "rail_pos",
    *(f"dq{i}" for i in range(1, 8)),
    "drail",
    "ee_base.x",
    "ee_base.y",
    "ee_base.z",
    "ee_base.qw",
    "ee_base.qx",
    "ee_base.qy",
    "ee_base.qz",
    "ee_world.x",
    "ee_world.y",
    "ee_world.z",
    "ee_world.qw",
    "ee_world.qx",
    "ee_world.qy",
    "ee_world.qz",
    "gripper_open_frac",
    "rail_pos_m",
)
ARM_STATE_BLOCK = len(ARM_STATE_LAYOUT)  # 32
assert ARM_STATE_BLOCK == 32  # noqa: S101 - spelling invariant (14-dora §4.2)

PoseSource = Literal["loop", "idle"]
ExternalState = Literal["disabled", "unavailable", "attached", "detached", "closed"]
# ``episode_boundary`` is the reason the runtime publishes at every episode boundary
# (episode_new / save / discard); ``handback`` ONLY when the operator hands control back
# to the policy inside an episode (15-online-dagger §6 — before phase-14 both spelled
# "handback").
PolicyResetReason = Literal[
    "handback", "episode_boundary", "session_start", "anomaly", "session_stop"
]
# Append-only (the order is the contract golden's). Payload keys (``EventEnvelope.payload``;
# 14-dora §4.2, 15-online-dagger §3/§6):
#   gate                 {arm_id, mode, seq, source, episode_id}  (every take-over /
#                        hand-back instant: Space, the takeover / handback actions,
#                        auto-advance, episode reset)
#   episode_saved        {episode_index, summary, dataset_root, spool_path, run_id}
#                        + online_dagger: {episode_id, rollouts_saved, actor_counts:
#                        {novice, expert}, policy_version, spool_path} (Online DAgger only)
#   episode_discarded    {episode_index, episode_id, reason}  (NOTHING is persisted for a
#                        discarded rollout; a trainer that watched it live drops it)
#   train_now            {rollouts_saved, requested_by: "operator"}  (the Cockpit button;
#                        the trainer decides whether to honour it)
# The shell never publishes iteration- or algorithm-level events: the trainer counts
# rollouts and decides when to train; the runtime reports what the trainer says.
EventKind = Literal[
    "collision",
    "gate",
    "episode_saved",
    "episode_discarded",
    "policy_anomaly",
    "policy_swap",
    "policy_version_changed",
    "reset_watermark",
    "session_error",
    "train_now",  # phase-14 (15-online-dagger §3): the operator asked the trainer to train
]
EVENT_KINDS: tuple[str, ...] = EventKind.__args__  # type: ignore[attr-defined]


# -- JSON payload models --------------------------------------------------------------------
class PolicySpecModel(BaseModel):
    """Pydantic mirror of core ``interfaces.policy.PolicySpec`` (the dataclass is
    not a wire model); the layout the policy node declares."""

    action_space: Literal["delta_ee", "abs_ee", "joint"]
    action_frame: str  # FrameRef: "arm_base:<id>" | "world" | "camera:<id>"
    action_names: list[str]
    state_names: list[str]
    camera_keys: list[str] = []
    version: int = 0


class CameraAnnounce(BaseModel):
    """One camera of the session (``SessionAnnounce.cameras``)."""

    resolution: tuple[int, int]
    fps: float
    frame_ref: str  # "camera:<id>"
    mount: str  # "ee:<arm_id>" (wrist) | "world" (static)
    intrinsics: list[float] | None = None  # [fx, fy, cx, cy] when known
    distortion: list[float] = []
    T_E_C: list[float] | None = None  # [x,y,z,qw,qx,qy,qz] camera in the arm's TCP frame
    T_W_C: list[float] | None = None  # [x,y,z,qw,qx,qy,qz] static camera in world
    depth: bool = False  # a ``cam_<id>_depth`` sibling is actually produced


class OnlineDaggerAnnounce(BaseModel):
    """``SessionAnnounce.online_dagger`` (15-online-dagger §6): the paths the trainer node
    needs — the session directory and the rollouts dataset it reads (a MAVIS
    episode-directory dataset with the ``actor`` column). The trainer keeps its own
    artefacts wherever it likes (the skill suggests ``<session_dir>/trainer/``; the
    runtime never reads them). Field order is the contract (both goldens pin it)."""

    session_name: str
    session_dir: str  # ~/data/online_dagger/<session_name>
    rollouts_dir: str  # <session_dir>/rollouts — a MAVIS episode-directory dataset


class SessionAnnounce(BaseModel):
    """The ``session`` stream payload (14-dora §4.2): the contract message a late
    joiner learns everything from (sent on every state change + 1 Hz)."""

    mavis_schema: int = MAVIS_SCHEMA
    epoch: str
    session_id: str | None
    state: str  # SessionState value or "idle"
    spec: SessionSpec | None = None
    kind: Literal["hardware", "sim"] | None = None
    arm_ids: list[str] = []  # WorkcellConfig order (block order everywhere)
    has_rail: dict[str, bool] = {}
    frames: dict[str, FrameRef] = {}  # recording frames actually in force
    action_space: str | None = None  # "delta_ee" (v1 requirement, 12-dagger §6)
    action_names: list[str] = []
    state_names: list[str] = []
    camera_ids: list[str] = []
    cameras: dict[str, CameraAnnounce] = {}
    policy_source: Literal["checkpoint", "external"] | None = None
    dataset_root: str | None = None  # collect / dagger
    run_id: str | None = None  # dagger
    deprecated_keys: list[str] = []
    online_dagger: OnlineDaggerAnnounce | None = None  # phase-14 (15-online-dagger §6):
    #   non-null iff spec.online_dagger is set; appended last (additive)


class PolicySpecAnnounce(BaseModel):
    """The ``policy_spec`` input payload (14-dora §6.2), heartbeated at 1 Hz by the
    policy node."""

    mavis_schema: int = MAVIS_SCHEMA
    policy_id: str  # e.g. "act-pick-2026-09-03"
    policy_version: int
    node_version: str  # package version of the node
    spec: PolicySpecModel
    rate_hz: float  # the node's own act() rate (10-30)
    chunk_len: int = 1
    chunk_dt_s: float | None = None
    loader: str = "custom"  # "mlp_bundle_v1" | "lerobot_pretrained" | "custom"
    device: str = ""
    supports_reload: bool = False  # reserved (weights_reload, §11.3)
    health: Literal["ok", "degraded", "error"] = "ok"
    detail: str = ""
    uptime_s: float = 0.0
    acts_total: int = 0
    last_compute_ms: float | None = None
    extrinsics_sha: str | None = None  # camera-frame policies: 10-frames §5.3 check
    capabilities: list[str] = []  # phase-14 (15-online-dagger §6): a trainer-capable node
    #   lists "online_dagger"; the runtime's launch 409 reads the absence. Appended last


class TrainerStatusAnnounce(BaseModel):
    """The ``policy_trainer_status`` input payload (15-online-dagger §6): JSON on the
    policy node's ``trainer_status`` output, heartbeated at 1 Hz (and on change) while
    an Online DAgger session is announced. Generic by design — the runtime knows no
    DAgger variant. ``state`` walks idle -> preparing -> ready (the shell lets rollouts
    start) -> training (``episode_new`` refused while ``pause_while_training``) ->
    ready (weights swapped, ``policy_version`` bumped); ``error`` carries ``detail``.
    ``session_id`` MUST echo the served ``SessionAnnounce.session_id``: another
    session's id is ignored outright, ``None`` counts as "trainer alive" only.
    ``metrics`` is a free-form dict of finite scalars (``loss``, ``proj_rate``, ...)
    the Cockpit lists verbatim (a ``loss`` key gets the sparkline). Metadata: the
    common inbound keys (``client``, ``seq``, ``t_mono``, ``wallclock_ns``,
    ``mavis_schema``), same ``seq`` counter as the node's other outputs. Field order
    is the contract (both goldens pin it). Every float refuses inf / nan
    (``allow_inf_nan=False``; a diverged loss is reported as ``state: "error"`` +
    ``detail``, never as a non-finite number) — that rule leaves the JSON schema
    alone."""

    mavis_schema: int = MAVIS_SCHEMA
    trainer_id: str  # e.g. "my-policy-repo/online_dagger"
    node_version: str
    state: Literal["idle", "preparing", "training", "ready", "error"] = "idle"
    session_id: str | None = None  # echo of the announce it is serving
    policy_version: int = 0  # version the acting policy runs after the last swap
    progress: float = Field(0.0, ge=0, le=1, allow_inf_nan=False)  # 0..1 of the current
    #   preparing / training step (the TRAINING pill's bar)
    metrics: dict[str, Annotated[float, Field(allow_inf_nan=False)]] = {}  # free-form
    #   finite scalars (loss, proj_rate, ...); the trainer picks the keys
    detail: str = ""
    uptime_s: float = Field(0.0, ge=0, allow_inf_nan=False)


class PolicyResetMsg(BaseModel):
    """The ``policy_reset`` output payload (14-dora §4.2): the runtime drops every
    ``policy_action`` whose ``observation_id <= after_observation_id``."""

    mavis_schema: int = MAVIS_SCHEMA
    reason: PolicyResetReason
    after_observation_id: int
    session_id: str
    t_mono: float = 0.0


class EventEnvelope(BaseModel):
    """The ``events`` output payload (14-dora §4.2). Unknown ``kind`` values are
    additive for consumers; ``payload`` is the event-specific JSON object."""

    mavis_schema: int = MAVIS_SCHEMA
    kind: str  # one of EVENT_KINDS (additive)
    t_mono: float
    wallclock_ns: int
    session_id: str | None
    epoch: str = ""
    payload: dict = {}


class ExternalStatus(BaseModel):
    """``telemetry.external`` (14-dora §13): the bridge + external-policy state."""

    enabled: bool = False
    state: ExternalState = "disabled"
    detail: str = ""
    node_id: str = EXTERNAL_NODE_ID
    dataflow_id: str | None = None
    reattach_count: int = 0
    dataflow_restarts: int = 0  # rescan / join-triggered dataflow restarts (v0.3)
    publish_hz: dict[str, float] = {}  # measured per topic
    dropped_inputs: int = 0
    actions_late: int = 0
    policy_attached: bool = False
    policy_id: str | None = None
    policy_version: int | None = None
    policy_rate_hz: float | None = None
    action_age_s: float | None = None
    spec_age_s: float | None = None  # age of the newest policy_spec heartbeat
    version_changes_mid_episode: int = 0
    idle_reader: Literal["off", "running", "paused", "stale"] = "off"  # §4.2
    capabilities: list[str] = []  # phase-14 (15-online-dagger §6/§8): the FRESH spec's
    #   PolicySpecAnnounce.capabilities (e.g. ["online_dagger"]) so the launcher can gate
    #   "Start Online DAgger" before a session exists; [] when no spec is fresh. Appended last
    trainer_status: TrainerStatusAnnounce | None = None  # phase-14: the newest
    #   policy_trainer_status (session-less; the pre-launch trainer pill), None until one
    #   arrives or after the node detaches. Appended last


class DoraMachineInfo(BaseModel):
    """One configured remote consumer machine (``dora.machines``; 14-dora §2.6/§9, §16.1).

    ``registered`` = its daemon is registered at the coordinator (rescan); ``joined`` = its
    placeholders are rendered in the running dataflow (after ``POST /api/dora/machines/
    {id}/join`` and the remote consumer's attach cleared dora's multi-machine start barrier);
    ``detail`` explains a refused / expired join.
    """

    id: str
    registered: bool = False  # its daemon is registered at the coordinator
    joined: bool = False  # its placeholders are live in the dataflow (join protocol)
    placeholders: list[str] = []  # nodes rendered for it (empty until joined)
    detail: str = ""  # e.g. "join timed out: attach viewer_<id> within 30 s"


class DoraInfo(BaseModel):
    """``GET /api/dora`` (14-dora §2.6): connection facts for foreign clients. The
    auth token is deliberately NOT part of this model (§9)."""

    enabled: bool = False
    state: ExternalState = "disabled"
    detail: str = ""
    bind_host: str = "127.0.0.1"  # resolved IPv4 the private control plane listens on
    machine_id: str = "lab"
    auth: bool = False
    coordinator_addr: str = "127.0.0.1"
    coordinator_port: int = 6113
    daemon_port: int = 53391
    zenoh_port: int = 7447
    zenoh_connect: str = "tcp/127.0.0.1:7447"
    dataflow_name: str = DATAFLOW_NAME
    dataflow_id: str | None = None
    node_id: str = EXTERNAL_NODE_ID
    placeholders: list[str] = Field(default_factory=lambda: list(PLACEHOLDERS))
    machines: list[DoraMachineInfo] = []
    dataflow_restarts: int = 0
    reattach_count: int = 0
    dataflow_yaml: str | None = None
    mavis_schema: int = MAVIS_SCHEMA


__all__ = [
    "MAVIS_SCHEMA",
    "EXTERNAL_NODE_ID",
    "DATAFLOW_NAME",
    "PLACEHOLDER_POLICY",
    "PLACEHOLDER_VIEWER",
    "PLACEHOLDER_OBSERVER",
    "PLACEHOLDERS",
    "PROBE_NODE_ID",
    "REMOTE_PLACEHOLDER_KINDS",
    "remote_placeholder_id",
    "OUT_HEARTBEAT",
    "OUT_SESSION",
    "OUT_TELEMETRY",
    "OUT_EVENTS",
    "OUT_ARM_STATE",
    "OUT_ARM_CMD",
    "OUT_OBS_STATE",
    "OUT_POLICY_RESET",
    "RUNTIME_FIXED_OUTPUTS",
    "CAMERA_OUTPUT_PREFIX",
    "DEPTH_OUTPUT_SUFFIX",
    "MIC_OUTPUT_PREFIX",
    "camera_output_id",
    "depth_output_id",
    "mic_output_id",
    "IN_TICK",
    "IN_PROBE_HEARTBEAT",
    "IN_POLICY_ACTION",
    "IN_POLICY_SPEC",
    "IN_POLICY_STATUS",
    "IN_POLICY_TRAINER_STATUS",
    "RUNTIME_INPUTS",
    "TICK_TIMER",
    "PROBE_TIMER",
    "POLICY_OUT_ACTION",
    "POLICY_OUT_SPEC",
    "POLICY_OUT_STATUS",
    "POLICY_OUT_TRAINER_STATUS",
    "POLICY_OUTPUTS",
    "PROBE_OUT_HEARTBEAT",
    "RESERVED_IDS",
    "META_SCHEMA",
    "META_EPOCH",
    "META_SESSION_ID",
    "META_SEQ",
    "META_T_MONO",
    "META_WALLCLOCK_NS",
    "META_CLIENT",
    "COMMON_OUTPUT_METADATA",
    "POLICY_ACTION_REQUIRED_METADATA",
    "ARM_STATE_LAYOUT",
    "ARM_STATE_BLOCK",
    "PoseSource",
    "ExternalState",
    "PolicyResetReason",
    "EventKind",
    "EVENT_KINDS",
    "PolicySpecModel",
    "CameraAnnounce",
    "OnlineDaggerAnnounce",
    "SessionAnnounce",
    "PolicySpecAnnounce",
    "TrainerStatusAnnounce",
    "PolicyResetMsg",
    "EventEnvelope",
    "ExternalStatus",
    "DoraMachineInfo",
    "DoraInfo",
]
