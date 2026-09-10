"""Session and REST body models (design doc 01-core §12).

Bodies for the ``/api`` surface (04-runtime §13.1); runtime defines no wire
model of its own.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from apollo_mavis_v2_core.types import FrameRef, parse_frame

Mode = Literal["teleop", "collect", "dagger", "inference"]

START_FROM_RE = r"^(keep_current|profile:[A-Za-z0-9_\-]+)$"
_START_FROM_PATTERN = re.compile(START_FROM_RE)

# Dataset repo id a collect session records into (2026-09-07; 10-frames §8.1):
# ``<namespace>/<name>`` or a bare ``<name>`` (the runtime prefixes ``apollo/``).
# Names are slugs — the operator types "pick red cube", the UI slugs it — so a
# repo id is always a safe directory name under ``datasets_root``.
DATASET_RE = r"^(?:[A-Za-z0-9][A-Za-z0-9_\-]*/)?[A-Za-z0-9][A-Za-z0-9_\-]*$"
_DATASET_PATTERN = re.compile(DATASET_RE)

# One bare slug (2026-09-08; 15-online-dagger §5): the name half of ``DATASET_RE`` with
# no namespace. An Online DAgger session name is a directory under the ``online_dagger``
# dataset root (``~/data/online_dagger/<session_name>/``), so it must be a safe path
# component — the UI slugs against the same regex (it rides the schema ``pattern``).
SLUG_RE = r"^[A-Za-z0-9][A-Za-z0-9_\-]*$"


class ActionFilterConfig(BaseModel):
    """Idle-frame filter parameters (10-frames §11.4; 04-runtime §10.5; 2026-09-07,
    operator): the pro-dagger hesitation heuristic ported to the frame stream. A
    candidate frame whose commanded TCP (Chebyshev over xyz / geodesic angle),
    gripper fraction and rail slot are all within the epsilons of the LAST KEPT
    frame — and that has no gripper change within ±``gripper_context_s`` — is not
    recorded. Defaults = pro-dagger's 1 mm / 1e-3 / 1 mm per step and its one
    H=16 chunk at 10 Hz of gripper context; ``enabled: false`` records every
    frame. DAgger filters human-controlled frames only."""

    enabled: bool = True
    pos_eps_m: float = Field(0.001, ge=0)  # Chebyshev over the commanded TCP xyz
    rot_eps_rad: float = Field(0.001, ge=0)  # geodesic angle of the commanded TCP orientation
    gripper_eps_frac: float = Field(0.01, ge=0)  # open fraction (~1 mm of the G2's 86 mm stroke)
    rail_eps_m: float = Field(0.001, ge=0)
    gripper_context_s: float = Field(1.6, ge=0)  # keep idle frames within +- this of a
    #   gripper change (look-ahead buffer)


class OnlineDaggerConfig(BaseModel):
    """Online DAgger session parameters (15-online-dagger §5; phase-14, 2026-09-08).

    The runtime is the algorithm-agnostic SHELL (operator decision 2026-09-08): it
    performs rollouts, exposes take-over / hand-back, labels every step novice /
    expert, saves the kept rollouts and reports what the trainer says. Which DAgger
    variant runs, its hyper-parameters and every training artefact belong to the
    trainer node in the policy repo, so this block carries NO algorithm settings
    — only what the shell itself needs: the session directory name, whether an
    existing one is continued, and the two generic gates on ``episode_new``.

    ``extra="forbid"``: an unknown key is a 422 at POST, never silently dropped —
    a hyper-parameter typed here by mistake would otherwise vanish while the
    operator believes it travelled (the trainer configures itself; the runtime
    never forwards hyper-parameters).
    """

    model_config = ConfigDict(extra="forbid")

    session_name: str = Field(pattern=SLUG_RE, max_length=64)  # ~/data/online_dagger/<name>/
    resume: bool = False  # an existing session dir: continue it (else 409 "already exists")
    pause_while_training: bool = True  # refuse episode_new while the trainer reports
    #   ``training`` (15-online-dagger D2)
    wait_for_trainer_ready: bool = True  # refuse episode_new until the trainer has reported
    #   ``ready`` once for THIS session (15-online-dagger D2)


class SessionSpec(BaseModel):
    """POST /api/session body."""

    mode: Mode
    kind: Literal["hardware", "sim"]  # honored iff config available (binding; else 409)
    arms: list[str]  # participating arm ids
    frames: dict[str, FrameRef]  # per-arm RECORDING frame; never affects control math
    sim_scene: str | None = None  # required when kind == "sim"
    digital_twin_scene: str | None = None  # required when kind == "hardware"
    start_from: str = "keep_current"  # START_FROM_RE
    task: str | None = None  # dataset task string (collect/dagger: required)
    policy: str | None = None  # checkpoint id (dagger/inference); None = latest /
    #   promoted deploy ckpt (409 if none promoted — 12-dagger §9)
    speed_scale: float = Field(1.0, gt=0, le=1)  # additive (phase-09c): multiplies the
    #   host-side teleop / target-rate / jog / dq_max limits AND the driver-side servo /
    #   Cartesian-step / rail-speed caps; 0 and > 1 are 422. Hardware tab default 0.1
    policy_source: Literal["checkpoint", "external"] = "checkpoint"  # additive (phase-12,
    #   14-dora §6.1): "external" = the policy drives through the dora bus (policy_action
    #   from the independent policy node); dagger/inference only, and `policy` must be None
    # Dataset naming (additive, 2026-09-07; collect only, 04-runtime §10.5): the repo
    # id to record into (``DATASET_RE``, exposed as the schema ``pattern`` so the UI
    # slugs against the same regex); None keeps the phase-07 behaviour (derived from
    # ``task``). ``dataset_resume`` False = the dataset must NOT exist yet (409
    # "already exists"), True = it MUST exist and its feature schema must match this
    # session's (409 otherwise) — the UI's "New dataset" / "Continue existing" panes.
    dataset: str | None = Field(None, pattern=DATASET_RE)
    dataset_resume: bool = False
    # Idle-frame filter (collect / dagger; 2026-09-07, 04-runtime §10.5): the defaults
    # are the operator's pro-dagger heuristic; a non-default config on another mode is
    # a validation error (the default itself is inert there, like return_to_start).
    action_filter: ActionFilterConfig = ActionFilterConfig()
    # Return-to-start (collect / dagger; DEFAULT ON, operator decision 2026-09-07;
    # 04-runtime §10.5): after every save / discard the arms are driven back to the
    # start_from profile (else the kind's initial-condition profile) — twin-planned,
    # gated, cancellable by any operator input. With neither profile and the flag on
    # the POST is 409. Collect-only until 2026-09-08, when Online DAgger rollouts got the
    # same between-rollout return (15-online-dagger D6): a non-default value (False) on
    # teleop / inference is a validation error; the default True is inert there (and
    # survives a JSON round trip, which is why the rule is "non-default", not
    # "explicitly set").
    return_to_start: bool = True
    # Online DAgger (additive, 2026-09-08; 15-online-dagger §5): non-null = this dagger
    # session is an online rollout loop against an external trainer node. Requires
    # ``mode == "dagger"`` and ``policy_source == "external"``; ``dataset`` /
    # ``dataset_resume`` must stay unset because the rollouts repo id is DERIVED
    # (``online_dagger/<session_name>``, resumed iff ``online_dagger.resume``).
    online_dagger: OnlineDaggerConfig | None = None

    @field_validator("start_from")
    @classmethod
    def _start_from_matches(cls, v: str) -> str:
        if not _START_FROM_PATTERN.fullmatch(v):
            raise ValueError(f"start_from must match {START_FROM_RE}, got {v!r}")
        return v

    @field_validator("dataset")
    @classmethod
    def _dataset_matches(cls, v: str | None) -> str | None:
        if v is not None and not _DATASET_PATTERN.fullmatch(v):
            raise ValueError(f"dataset must match {DATASET_RE}, got {v!r}")
        return v

    @model_validator(mode="after")
    def _cross_field(self) -> SessionSpec:
        extra = set(self.frames) - set(self.arms)
        if extra:
            raise ValueError(f"frames keys must be a subset of arms: {sorted(extra)}")
        for arm_id, ref in self.frames.items():
            parsed = parse_frame(ref)  # raises FrameRefError (a ValueError)
            if parsed.kind == "ee":
                raise ValueError(f"frames[{arm_id!r}] may not be an ee: frame: {ref!r}")
        if self.mode in ("collect", "dagger") and not self.task:
            raise ValueError(f"mode {self.mode!r} requires a task string")
        if self.online_dagger is not None:  # 15-online-dagger §5 (checked before the
            #   dataset rule so a dagger body with both gets the specific message)
            if self.mode != "dagger":
                raise ValueError("online_dagger requires mode dagger")
            if self.policy_source != "external":
                raise ValueError("online_dagger requires policy_source 'external'")
            if self.dataset is not None or self.dataset_resume:
                raise ValueError(
                    "online_dagger derives the rollouts dataset - leave dataset unset"
                )
        if self.dataset is not None and self.mode != "collect":
            raise ValueError("dataset is a collect-mode field")
        if self.return_to_start is False and self.mode not in ("collect", "dagger"):
            raise ValueError("return_to_start is a collect / dagger-mode field")
        if self.action_filter != ActionFilterConfig() and self.mode not in ("collect", "dagger"):
            raise ValueError("action_filter is a collect / dagger-mode field")
        if self.policy_source == "external":
            if self.mode not in ("dagger", "inference"):
                raise ValueError(
                    f"policy_source 'external' requires mode dagger|inference, got {self.mode!r}"
                )
            if self.policy is not None:
                raise ValueError(
                    "policy_source 'external' takes no checkpoint: policy must be null"
                )
        return self


class SessionInfo(BaseModel):
    """POST/GET /api/session response."""

    session_id: str
    epoch: str
    mode: Mode
    arms: list[str]
    streams: list[str]  # video ids: camera ids + "sim" and/or "twin"
    state: str  # SessionState value
    kind: Literal["hardware", "sim"] = "sim"  # additive (phase-09c): which workcell the
    #   session drives; defaults to "sim" because every pre-09c session was one (hardware
    #   was 409), so the UI no longer infers it from hardware_monitor.paused
    speed_scale: float = Field(1.0, gt=0, le=1)  # additive (phase-09c): echo of
    #   SessionSpec.speed_scale (pre-09c producers ran unscaled)
    policy_source: Literal["checkpoint", "external"] = "checkpoint"  # additive (phase-12):
    #   echo of SessionSpec.policy_source
    fault_detail: str = ""  # additive (2026-09-08): the same session-level notice as
    #   ``SessionTelemetry.fault_detail`` (a refused / unplannable ``start_from``, a
    #   "Go to profile" / `R` return that did not arrive, else the manager's fault text);
    #   "" = nothing to say
    online_dagger: OnlineDaggerConfig | None = None  # additive (phase-14, 15-online-dagger
    #   §5): echo of SessionSpec.online_dagger; None for every other session


class ArmStatusInfo(BaseModel):
    """Landing-page arm card."""

    arm_id: str
    ip: str | None
    connected: bool  # "a session exists" (semantics unchanged; phase-11 note)
    reachable: Literal["open", "refused", "unreachable", "unknown"] = "unknown"
    # hardware probe (TCP 502 connect-and-close): open = box up / refused = box
    #   booting / unreachable = no route or timeout / unknown = not probed (sim);
    #   additive (phase-11)
    has_rail: bool
    gripper: Literal["xarm", "xarm_g2", "none"]
    gripper_force_capable: bool
    error_code: int
    joint_limits: list[tuple[float, float]]  # 7 rad pairs; + [0.0, 0.65] m for rail arms


class CameraInfo(BaseModel):
    """Landing-page camera card."""

    camera_id: str
    kind: Literal["v4l2", "realsense", "sim", "twin"]
    # twin = digital-twin overlay stream (``<camera_id>_align``, phase-09a): the
    #   twin rendered from the real wrist camera's viewpoint, tinted over the
    #   real frame; additive
    label: str
    resolution: tuple[int, int]
    fps: int
    live: bool  # pre-session preview available (~15 fps)


class WorkcellStatus(BaseModel):
    """GET /api/workcell response."""

    kind: Literal["hardware", "sim"]
    available_kinds: list[Literal["hardware", "sim"]]
    arms: list[ArmStatusInfo]
    cameras: list[CameraInfo]
    policies_available: bool = False  # enables DAgger/Inference launch (05-ui §8.1)
    hardware_ready: bool = False  # every configured hardware arm reachable == "open"
    #   (gates the Hardware-tab mode launchers; additive, phase-11)


class SceneInfo(BaseModel):
    """GET /api/scenes?kind=sim|twin row."""

    scene_id: str
    label: str
    num_arms: int
    rail_flags: list[bool]
    cameras: list[str]
    kind: Literal["sim", "twin"]


class ProfileInfo(BaseModel):
    """GET /api/profiles row (full posture via GET /api/profiles/{id})."""

    profile_id: str
    name: str
    arms: list[str]
    notes: str
    created_at: str
    is_initial_condition: bool
    workcell_kind: Literal["hardware", "sim"] | None = None  # additive (2026-09-07): the
    #   kind the profile was saved on — the LaunchSheet gates return-to-start on an
    #   initial condition of the SELECTED tab's kind; None = an older runtime


class DatasetExportInfo(BaseModel):
    """``manifest.last_export`` as the REST sees it (10-frames §11.5; 04-runtime §10.6).

    ``state``: ``none`` = never exported, ``fresh`` = the export matches the episode
    set, ``stale`` = an episode was added / deleted since, ``running`` = the export
    job is on it right now (process-local), ``failed`` = the last job failed
    (``detail`` carries the reason, until the next success)."""

    state: Literal["none", "stale", "fresh", "running", "failed"]
    format: str = "lerobot_v3"
    path: str | None = None  # relative to the dataset root (``exports/lerobot_v3``)
    at: str | None = None  # ISO-8601 UTC of the last successful export
    episodes: int = 0  # episodes in that export
    detail: str = ""


class DatasetInfo(BaseModel):
    """``GET /api/datasets`` row (2026-09-07; 04-runtime §10.6, 10-frames §11): one
    dataset under ``datasets_root``, read from ``manifest.json`` + the per-episode
    ``episode.json`` sidecars — no lerobot import, so the list is cheap. ``layout``
    ``episode_dirs`` is the primary per-episode store; ``lerobot_v3`` is a legacy
    phase-07 tree (``meta/info.json``), listed read-only. ``kind`` / ``task`` come
    from the most recent ``sessions/session_*.json`` sidecar (None without one).
    ``in_use`` = the active session records into it (deleting the OPEN episode is
    then 409; every other episode can go).

    ``namespace`` / ``path`` (additive, 2026-09-08; 15-online-dagger D5): dataset roots
    are per-namespace now (``bc_demo/<name>`` -> ``~/data/bc_demo/<name>``,
    ``online_dagger/<s>`` -> ``~/data/online_dagger/<s>/rollouts``, everything else under
    ``datasets_root/<ns>/<name>``), so the row spells its namespace and the REAL
    folder for the UI to show; both default to "" so an older runtime still
    validates (``root`` keeps meaning the absolute dataset directory)."""

    repo_id: str
    root: str  # absolute dataset directory
    layout: Literal["episode_dirs", "lerobot_v3"] = "episode_dirs"
    total_episodes: int
    total_frames: int
    fps: int
    robot_type: str | None = None
    kind: Literal["hardware", "sim"] | None = None
    task: str | None = None
    cameras: list[str] = []  # camera ids (video feature keys minus ``observation.images.``)
    arms: list[str] = []
    modified_at: str  # ISO-8601 UTC of the newest metadata write
    in_use: bool = False
    export: DatasetExportInfo | None = None
    namespace: str = ""  # the ``<ns>`` half of repo_id (additive, 2026-09-08)
    path: str = ""  # the folder the operator sees (== root; additive, 2026-09-08)


class EpisodeInfo(BaseModel):
    """``GET /api/datasets/{ns}/{name}/episodes`` row: one saved episode directory
    (10-frames §11.3 / §11.4), read from its ``episode.json`` only."""

    episode_id: str  # directory name; the API key (never reused, never renumbered)
    index: int  # position in capture order (UI label only: "#12 of 40")
    frames: int
    duration_s: float
    task: str | None = None
    session_id: str | None = None
    recorded_at: str | None = None  # ISO-8601 UTC wallclock of the first frame
    frames_dropped: int = 0
    audio: bool = False  # an ``audio.wav`` sidecar exists in the directory
    export_ok: bool = True  # False + ``export_note`` = kept but excluded from exports
    export_note: str | None = None
    open: bool = False  # currently being recorded (delete -> 409)


class EpisodePlaybackArm(BaseModel):
    """One arm's state at an episode's FIRST recorded frame (2026-09-10).

    Read out of ``frames.parquet``'s ``observation.state`` row 0 by the per-dim names
    the dataset's own manifest carries (10-frames §6.1 / §7), so an episode recorded
    with a different arm set or without a track still reads correctly.
    """

    arm_id: str
    q: list[float]  # the 7 joint angles, rad (the rail is NOT in here)
    rail_pos_m: float | None = None  # None = this arm has no track in the recording
    gripper_open_frac: float | None = None  # None = no gripper column (Perception Arm)


class EpisodePlaybackInfo(BaseModel):
    """``GET /api/datasets/{ns}/{name}/episodes/{id}/playback`` (2026-09-10; operator
    request: a **Playback** button on every episode row of the Welcome page's Datasets
    panel, 05-ui §8.1 item 7).

    What a playback of this episode WOULD do, read from the episode directory only — no
    session needed, so the dialog can open and explain itself before anything moves. The
    two motion buttons ride ``POST /api/session/playback`` and do need one.

    ``playable`` false + ``reason`` covers every case the runtime would refuse: no
    session, a session whose arms or workcell kind do not match the recording, a legacy
    tree, an episode whose parquet is missing or unreadable. The UI shows ``reason``
    verbatim instead of letting the operator meet a 409.
    """

    repo_id: str
    episode_id: str
    frames: int
    fps: float
    duration_s: float
    arms: list[EpisodePlaybackArm]  # the INITIAL state, one row per recorded arm
    playable: bool
    reason: str = ""  # operator-facing; "" when playable


class EpisodePlaybackRequest(BaseModel):
    """``POST /api/session/playback`` body (2026-09-10).

    ``goto_initial`` walks the arms to the episode's first frame and is SYNCHRONOUS,
    like ``return_home``: the dialog awaits it and only then enables **Playback**, which
    is the operator's rule — you cannot replay a trajectory from the wrong place.
    ``play`` streams the recorded trajectory through the same twin-planned, gated
    executor; ``stop`` cancels whatever is in flight.
    """

    repo_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_\-]*/[A-Za-z0-9][A-Za-z0-9_\-]*$")
    #   `<ns>/<name>`, the repo-id grammar of 10-frames §11 — validated HERE and not only
    #   in the path routes, because this one arrives in a JSON body and is joined onto a
    #   filesystem root: no empty segment, no `.`, no `..`, no third slash.
    episode_id: str = Field(pattern=r"^[0-9A-Za-z][0-9A-Za-z.\-]*$")
    #   A capture-time stamp (10-frames §11.3, `20260907T141203.512Z-3f9a1c`): the `.` and
    #   `Z` travel verbatim, but it must start with an alphanumeric so no `..` or `.hidden`
    #   can reach the directory join.
    action: Literal["goto_initial", "play", "stop"]

    @field_validator("episode_id")
    @classmethod
    def _no_traversal(cls, v: str) -> str:
        if ".." in v:
            raise ValueError("episode_id must not contain '..'")
        return v


class DatasetExportRequest(BaseModel):
    """``POST /api/datasets/{ns}/{name}/export`` body (04-runtime §13.1)."""

    format: Literal["lerobot_v3"] = "lerobot_v3"
    out: str | None = None  # None = ``<root>/exports/lerobot_v3``


class DatasetNamespaceInfo(BaseModel):
    """One mapped dataset namespace (``DatasetLayoutInfo.namespaces[ns]``; 15-online-dagger
    §7 / D5): its datasets live at ``<root>/<name>`` or, with ``subdir``, at
    ``<root>/<name>/<subdir>`` (``online_dagger`` -> ``~/data/online_dagger/<s>/rollouts``)."""

    root: str  # absolute directory the namespace's datasets live under
    subdir: str | None = None  # per-dataset sub-folder holding the dataset (None = the dir)


class DatasetLayoutInfo(BaseModel):
    """``GET /api/datasets/layout`` (15-online-dagger §7; 2026-09-08): where datasets live,
    so the UI shows the REAL folder in its previews and never hard-codes a
    namespace. A bare ``dataset: "<name>"`` resolves into ``default_namespace``;
    a namespace absent from ``namespaces`` lives at ``<generic_root>/<ns>/<name>``."""

    default_namespace: str
    generic_root: str  # ``datasets_root``: <generic_root>/<ns>/<name> for unmapped namespaces
    namespaces: dict[str, DatasetNamespaceInfo]


class OnlineDaggerSessionInfo(BaseModel):
    """``GET /api/online_dagger/sessions`` row (15-online-dagger §3/§5; 2026-09-08): one
    ``session.json`` under the ``online_dagger`` root, for the launch sheet's resume
    pill. ``rollouts`` is the session's kept-rollout count (``current.rollouts_saved``,
    what a resume continues from); rows come newest ``last_used_at`` first. The runtime
    reads nothing of the trainer's own artefacts."""

    session_name: str
    path: str  # absolute session directory (~/data/online_dagger/<session_name>)
    created_at: str  # ISO-8601 UTC
    task: str | None
    rollouts: int  # kept rollouts of the session
    last_used_at: str | None = None  # ISO-8601 UTC of the last session that ran it


class ReturnHomeResult(BaseModel):
    """``POST /api/session/return_home`` response (04-runtime §10.5; 2026-09-08).

    The synchronous "walk the workcell back to its designated initial condition"
    op the Cockpit runs BEFORE it tears a session down, and the same motion the
    ``reset_to_initial`` key fires. ``ok`` is what the UI branches on: false ⇒
    the arms are NOT at the initial condition and the operator has to be told
    (the Cockpit shows a dialog and offers to end the session anyway, after
    which the arms may be moved from UFACTORY Studio — never during a session).

    ``status`` distinguishes *why*: ``done`` — arrived; ``skipped`` — nothing to
    do (no initial-condition profile for this workcell kind, or already there),
    which is a success; ``failed`` — the twin could not plan a collision-free
    path (or an arm is faulted); ``cancelled`` — operator input interrupted the
    motion; ``timeout`` — the gate held the motion past its budget, the arms
    were stopped where they are; ``refused`` — a precondition said no (an
    episode is still recording, no session).
    """

    ok: bool
    status: Literal["done", "skipped", "failed", "cancelled", "timeout", "refused"]
    detail: str = ""  # operator-facing reason; "" only when status == "done"
    arms: list[str] = []  # arms the motion covered (empty when it never started)
    profile_id: str | None = None  # the initial-condition profile aimed at, if any


class PolicyInfo(BaseModel):
    """GET /api/policies row (04-runtime §13.1)."""

    policy_id: str  # "{run_id}/v{n:06d}" | "{run_id}/deploy/v{k:03d}"
    path: str
    action_space: Literal["delta_ee", "abs_ee", "joint"]
    action_frame: str  # FrameRef: "arm_base:<id>" | "world" | "camera:<id>"
    policy_version: int
    promoted: bool = False  # deploy checkpoints only (12-dagger §9)


__all__ = [
    "Mode",
    "START_FROM_RE",
    "DATASET_RE",
    "SLUG_RE",
    "ActionFilterConfig",
    "OnlineDaggerConfig",
    "DatasetInfo",
    "DatasetExportInfo",
    "DatasetExportRequest",
    "DatasetLayoutInfo",
    "DatasetNamespaceInfo",
    "EpisodeInfo",
    "OnlineDaggerSessionInfo",
    "SessionSpec",
    "SessionInfo",
    "ArmStatusInfo",
    "CameraInfo",
    "WorkcellStatus",
    "SceneInfo",
    "ProfileInfo",
    "PolicyInfo",
    "ReturnHomeResult",
]
