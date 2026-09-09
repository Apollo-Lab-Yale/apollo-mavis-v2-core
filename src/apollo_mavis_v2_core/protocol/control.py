"""Control WebSocket messages (design doc 01-core §10).

Binding decisions mirrored exactly (05-ui §2 / 04-runtime §13.2); the ``t``
discriminators let one :class:`~pydantic.TypeAdapter` parse the channel.
Space is a discrete takeover toggle — one ``ActionMsg{takeover_toggle}`` per
physical press, never in ``KeysMsg.held``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

ActionName = Literal[
    "switch_arm",
    "switch_arm_prev",
    "takeover_toggle",
    "episode_new",
    "episode_save",
    "episode_discard",
    "reset_to_initial",
    "save_profile",
    "set_initial_condition",
    "joint_target",
    "tracker_settings",
    # phase-14 (15-online-dagger D3 / §3): the Online DAgger shell's three Cockpit buttons.
    # None has a key binding (not in KEYMAP — the keymap is operator-owned; Space keeps
    # ``takeover_toggle``) and none takes args.
    "takeover",  # explicit take-over of the active arm; idempotent (a no-op ack in HUMAN /
    #   TRANSITION)
    "handback",  # hand control back to the policy; idempotent (a no-op ack in POLICY)
    "train_now",  # ask the trainer to train on the rollouts saved so far (events.train_now;
    #   the trainer may ignore it)
    # 2026-09-08: drive the session's arms to a SAVED profile on the operator's request
    # (the profile row's "go to" button). Args: GotoProfileArgs. Motion is planned on the
    # twin and runs through the same gated, cancellable execute_plan path as
    # return-to-initial — never implicit. No key binding (not in KEYMAP — the keymap is
    # operator-owned).
    "goto_profile",
    # phase-15 (16-gello D3 / D9 / §8.2; 2026-09-09): the GELLO Manipulation session's two
    # Cockpit buttons. Pause stops the follower from following the leader (state ``paused``,
    # the last command is HELD — never a move); Resume re-runs the engage rule (within
    # ``engage_tol_rad`` -> ``tracking``, else ``out_of_sync``). Both idempotent (a no-op ack
    # when already in that state), argless, NO key binding (not in KEYMAP — the keymap is
    # operator-owned; 16-gello §16 item 3), nacked "not a GELLO Manipulation session"
    # elsewhere (the base ControlLoop handler; 2026-09-09 review).
    "gello_pause",
    "gello_resume",
]


class HelloMsg(BaseModel):
    """Server -> client, immediately after WS accept."""

    t: Literal["hello"] = "hello"
    epoch: str  # runtime process UUID (client detects restarts)
    session_id: str | None
    role: Literal["controller", "observer"]  # first connection = controller (binding)


class KeysMsg(BaseModel):
    """Controller -> server: immediate on every key transition + 25 Hz heartbeat."""

    t: Literal["keys"] = "keys"
    seq: int  # monotonic; server drops seq <= last_seq
    ts: float  # client wall clock s — latency metric ONLY; watchdog feeds on rx time
    held: list[str]  # held movement KeyboardEvent.codes; rail codes always sent


class ActionMsg(BaseModel):
    """Client -> server, once per press/click."""

    t: Literal["action"] = "action"
    name: ActionName
    args: dict[str, Any] = {}  # validated per-name via validate_action_args()


class AckMsg(BaseModel):
    """Server -> client, one per ActionMsg."""

    t: Literal["ack"] = "ack"
    name: ActionName
    ok: bool
    detail: str = ""  # "observer", "takeover active", planner reason


class JointTargetArgs(BaseModel):
    """Args for ``name == "joint_target"``."""

    arm_id: str
    positions: list[float]  # FULL q incl. rail slot (rad; rail m); len == dof
    mode: Literal["jog", "goto"]  # jog: slew-limited streaming; goto: twin planner


class SwitchArmArgs(BaseModel):
    """Args for ``name == "switch_arm"`` (optional; empty args = cycle).

    ``arm_id`` makes the switch EXPLICIT and idempotent, which is what a UI
    control needs: the Cockpit's arm rows are clickable (2026-09-07) and a
    click must land on the arm the operator clicked whatever the session's arm
    order is. Keyboard Tab / gamepad RB keep sending no args and keep cycling.
    An unknown id is refused (``ack.ok == false``) — the active arm never
    changes silently. ``switch_arm_prev`` takes no args.

    ``extra="forbid"`` keeps the guarantee this action had before it grew an
    args model: anything but ``arm_id`` is a client bug and is rejected rather
    than silently ignored (pydantic's default).
    """

    model_config = ConfigDict(extra="forbid")

    arm_id: str | None = None


class SaveProfileArgs(BaseModel):
    """Args for ``name == "save_profile"``.

    ``set_initial`` designates the saved profile as the workcell's initial
    condition in the same round trip (2026-09-07): the Cockpit has ONE profile
    button now — "Save current state as profile" with a name field and an
    optional "use as initial condition" switch — instead of the two buttons
    ("Save profile…" / "Set current state as initial condition") whose
    difference nobody could see. ``set_initial_condition`` stays on the wire
    for designating an EXISTING profile by id.
    """

    name: str
    notes: str = ""
    set_initial: bool = False


class SetInitialConditionArgs(BaseModel):
    """Args for ``name == "set_initial_condition"``."""

    profile_id: str | None = None  # None: save current state first, then designate


class GotoProfileArgs(BaseModel):
    """Args for ``name == "goto_profile"`` (2026-09-08).

    Move the session's arms to the SAVED profile ``profile_id`` — operator-
    requested motion only: the runtime plans it on the twin and executes it
    through the same gated, cancellable path as return-to-initial (04-runtime
    §10.5). Unlike ``set_initial_condition`` there is no "current state"
    default, so ``profile_id`` is REQUIRED. Its pattern is the ProfileStore id
    charset (uuid4 hex; also the ``profile:<id>`` half of ``START_FROM_RE``),
    so a malformed id is refused at the wire (``ack.ok == false``) instead of
    surfacing as a store error later.

    ``extra="forbid"``: anything but ``profile_id`` is a client bug and is
    rejected rather than silently ignored (pydantic's default).
    """

    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(pattern=r"^[A-Za-z0-9_\-]+$")


class TrackerSettingsArgs(BaseModel):
    """Args for ``name == "tracker_settings"`` (13-tracker §3.4, §4 "Pose filter").

    Every field is optional; omitted (``None``) fields leave the live runtime
    setting unchanged. The ``filter_*`` fields tune the runtime's One Euro pose
    filter live (debug page); a change while the clutch is engaged re-anchors
    instead of moving the arm (13-tracker §4 "Anchor and re-seed rules").
    """

    yaw_deg: float | None = None  # lighthouse-world -> MJCF-world yaw alignment
    pos_scale: float | None = Field(default=None, ge=0.1, le=3.0)  # tracker->EE gain
    follow_rotation: bool | None = None  # apply tracker orientation deltas
    filter_enabled: bool | None = None  # False bypasses the One Euro pose filter
    filter_min_cutoff_hz: float | None = Field(default=None, ge=0.05, le=50.0)
    # One Euro min cutoff (Hz): lower = smoother at rest, more lag
    filter_beta: float | None = Field(default=None, ge=0.0, le=200.0)
    # One Euro speed coefficient in Hz per (m/s): higher = less lag during fast
    # motion. Ceiling raised 5 -> 200 on 2026-09-07: the useful range for a hand
    # in METRES starts around 1 (the paper's ~0.007 is per pixel/s), so the old
    # ceiling only just reached the first usable value. Runtime default is 5.0.


ControlClientMsg = Annotated[KeysMsg | ActionMsg, Field(discriminator="t")]
ControlServerMsg = Annotated[HelloMsg | AckMsg, Field(discriminator="t")]

_CLIENT_MSG_ADAPTER: TypeAdapter[KeysMsg | ActionMsg] = TypeAdapter(ControlClientMsg)

# ActionNames whose args model exists; every other action requires empty args.
_ARGS_MODELS: dict[str, type[BaseModel]] = {
    "switch_arm": SwitchArmArgs,
    "joint_target": JointTargetArgs,
    "save_profile": SaveProfileArgs,
    "set_initial_condition": SetInitialConditionArgs,
    "tracker_settings": TrackerSettingsArgs,
    "goto_profile": GotoProfileArgs,
}


def parse_client_msg(raw: str | bytes) -> KeysMsg | ActionMsg:
    """Parse one /ws/control client frame via the module-level TypeAdapter."""
    return _CLIENT_MSG_ADAPTER.validate_json(raw)


def validate_action_args(msg: ActionMsg) -> BaseModel | None:
    """Validate ``msg.args`` against the per-name model.

    Returns the parsed args model for switch_arm / joint_target /
    save_profile / set_initial_condition / tracker_settings / goto_profile;
    for every other action (incl. switch_arm_prev, takeover, handback,
    train_now, gello_pause and gello_resume), requires ``args == {}`` and
    returns None.
    Every field of ``SwitchArmArgs`` is optional, so the keyboard / gamepad
    ``switch_arm`` with no args still validates and still cycles.
    Raises pydantic ``ValidationError`` / ``ValueError``.
    """
    model = _ARGS_MODELS.get(msg.name)
    if model is None:
        if msg.args:
            raise ValueError(f"action {msg.name!r} takes no args, got {msg.args!r}")
        return None
    return model.model_validate(msg.args)


__all__ = [
    "ActionName",
    "HelloMsg",
    "KeysMsg",
    "ActionMsg",
    "AckMsg",
    "JointTargetArgs",
    "SaveProfileArgs",
    "SetInitialConditionArgs",
    "SwitchArmArgs",
    "TrackerSettingsArgs",
    "GotoProfileArgs",
    "ControlClientMsg",
    "ControlServerMsg",
    "parse_client_msg",
    "validate_action_args",
]
