"""Control WebSocket messages (design doc 01-core §10).

Binding decisions mirrored exactly (05-ui §2 / 04-runtime §13.2); the ``t``
discriminators let one :class:`~pydantic.TypeAdapter` parse the channel.
Space is a discrete takeover toggle — one ``ActionMsg{takeover_toggle}`` per
physical press, never in ``KeysMsg.held``.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

ActionName = Literal[
    "switch_arm",
    "takeover_toggle",
    "episode_new",
    "episode_save",
    "episode_discard",
    "save_profile",
    "set_initial_condition",
    "joint_target",
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


class SaveProfileArgs(BaseModel):
    """Args for ``name == "save_profile"``."""

    name: str
    notes: str = ""


class SetInitialConditionArgs(BaseModel):
    """Args for ``name == "set_initial_condition"``."""

    profile_id: str | None = None  # None: save current state first, then designate


ControlClientMsg = Annotated[KeysMsg | ActionMsg, Field(discriminator="t")]
ControlServerMsg = Annotated[HelloMsg | AckMsg, Field(discriminator="t")]

_CLIENT_MSG_ADAPTER: TypeAdapter[KeysMsg | ActionMsg] = TypeAdapter(ControlClientMsg)

# ActionNames whose args model exists; every other action requires empty args.
_ARGS_MODELS: dict[str, type[BaseModel]] = {
    "joint_target": JointTargetArgs,
    "save_profile": SaveProfileArgs,
    "set_initial_condition": SetInitialConditionArgs,
}


def parse_client_msg(raw: str | bytes) -> KeysMsg | ActionMsg:
    """Parse one /ws/control client frame via the module-level TypeAdapter."""
    return _CLIENT_MSG_ADAPTER.validate_json(raw)


def validate_action_args(msg: ActionMsg) -> BaseModel | None:
    """Validate ``msg.args`` against the per-name model.

    Returns the parsed args model for joint_target / save_profile /
    set_initial_condition; for every other action, requires ``args == {}``
    and returns None. Raises pydantic ``ValidationError`` / ``ValueError``.
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
    "ControlClientMsg",
    "ControlServerMsg",
    "parse_client_msg",
    "validate_action_args",
]
