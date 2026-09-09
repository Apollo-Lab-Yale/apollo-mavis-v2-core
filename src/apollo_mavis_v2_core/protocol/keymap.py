"""Canonical keymap (design doc 01-core §13; spine 00-overview §5; 13-tracker §3).

``GET /api/keymap`` serves exactly this table; the UI builds its bound-key
set from it — no hardcoded duplicate (binding). ``axis_map`` is the single
source of signs for runtime's ``held_to_twist``. The optional ``gamepad``
field carries the XInput control label so the UI never hard-codes the pad
mapping either.

2026-09-07 (operator decision, overview §5 "Input interfaces"): the keyboard is
a full teleop interface alongside the Vive controller, so every held row here IS
a key; codes are unique across the whole table (no code is both a held and a
discrete row), which is what keeps the episode keys N / Enter / Backspace off
the movement keys.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Group = Literal["translate", "rotate", "gripper", "rail", "session", "episode", "tracker"]


class KeymapEntry(BaseModel):
    """One keyboard binding row (optionally mirrored on the gamepad)."""

    code: str  # KeyboardEvent.code
    action: str  # held: axis name or held modifier; discrete: ActionName
    kind: Literal["held", "discrete"]
    label: str  # overlay text
    group: Group
    requires_rail: bool = False
    gamepad: str | None = None  # XInput control: DpadLeft/DpadRight/A/B/LB/RB/RT


def _e(
    code: str,
    action: str,
    kind: Literal["held", "discrete"],
    label: str,
    group: Group,
    requires_rail: bool = False,
    gamepad: str | None = None,
) -> KeymapEntry:
    return KeymapEntry(
        code=code,
        action=action,
        kind=kind,
        label=label,
        group=group,
        requires_rail=requires_rail,
        gamepad=gamepad,
    )


_SPACE_LABEL = (
    "takeover toggle (DAgger: recorded; inference: safety escape, never recorded)"
)

KEYMAP: tuple[KeymapEntry, ...] = (
    # held / translate (spine §5: W/S = forward/back, A/D = left/right, E/Q = up/down).
    # The axis names below are the KEY axes (x forward, y left, z up); which physical
    # frame they land on is the runtime's `control.translate_frame` (default the
    # operator-fixed WORLD frame — decision of 2026-09-08 evening, superseding that
    # morning's wrist-camera default; `camera` / `base` stay selectable) — see
    # runtime control/teleop.py.
    _e("KeyW", "translate_x_pos", "held", "forward", "translate"),
    _e("KeyS", "translate_x_neg", "held", "back", "translate"),
    _e("KeyA", "translate_y_pos", "held", "left", "translate"),
    _e("KeyD", "translate_y_neg", "held", "right", "translate"),
    _e("KeyE", "translate_z_pos", "held", "up", "translate"),
    _e("KeyQ", "translate_z_neg", "held", "down", "translate"),
    # held / rotate (about TCP axes; spine §5: I/K roll, J/L pitch, U/O yaw)
    _e("KeyI", "roll_pos", "held", "roll +", "rotate"),
    _e("KeyK", "roll_neg", "held", "roll -", "rotate"),
    _e("KeyJ", "pitch_pos", "held", "pitch +", "rotate"),
    _e("KeyL", "pitch_neg", "held", "pitch -", "rotate"),
    _e("KeyU", "yaw_pos", "held", "yaw +", "rotate"),
    _e("KeyO", "yaw_neg", "held", "yaw -", "rotate"),
    # held / gripper (gamepad A / B; 13-tracker §1)
    _e("KeyF", "gripper_close", "held", "gripper close", "gripper", gamepad="B"),
    _e("KeyH", "gripper_open", "held", "gripper open", "gripper", gamepad="A"),
    # held / rail (only if rail detected; 0-0.65 m; gamepad D-pad)
    _e("ArrowLeft", "rail_neg", "held", "rail left", "rail", requires_rail=True,
       gamepad="DpadLeft"),
    _e("ArrowRight", "rail_pos", "held", "rail right", "rail", requires_rail=True,
       gamepad="DpadRight"),
    # held / tracker (modifier, not an axis; gamepad RT held >= 0.5)
    _e("KeyC", "tracker_clutch", "held", "tracker clutch (hold)", "tracker", gamepad="RT"),
    # discrete / session (gamepad LB / RB)
    _e("KeyZ", "switch_arm_prev", "discrete", "previous arm", "session", gamepad="LB"),
    _e("Tab", "switch_arm", "discrete", "switch active arm", "session", gamepad="RB"),
    _e("Space", "takeover_toggle", "discrete", _SPACE_LABEL, "session"),
    # 2026-09-08 (operator request): one key that walks the workcell back to the
    # designated initial-condition profile (twin-planned, gated, cancelled by any
    # movement input). A no-op — nacked with a reason — when no initial condition
    # is designated for this workcell kind.
    _e("KeyR", "reset_to_initial", "discrete", "return to the initial condition", "session"),
    # discrete / episode (always listed; runtime nacks without a recorder)
    _e("KeyN", "episode_new", "discrete", "start new episode", "episode"),
    _e("Enter", "episode_save", "discrete", "save current episode", "episode"),
    _e("Backspace", "episode_discard", "discrete", "discard current episode", "episode"),
)

HELD_CODES: frozenset[str] = frozenset(e.code for e in KEYMAP if e.kind == "held")
DISCRETE_CODES: dict[str, str] = {e.code: e.action for e in KEYMAP if e.kind == "discrete"}

# Held actions that are NOT axes (13-tracker §3.3): they gate behaviour while
# held instead of driving a twist. ``axis_map`` excludes them and runtime's
# ``held_to_twist`` ignores them.
HELD_MODIFIER_ACTIONS: frozenset[str] = frozenset({"tracker_clutch"})

# Held axis action -> (axis, sign). Sign conventions (documented, binding here).
# These are the KEY axes, resolved into a physical frame by the runtime
# (`control.translate_frame`; default = the operator-fixed world frame since
# 2026-09-08 evening; the arm's wrist-camera frame and the arm-base axes are the
# selectable alternatives):
#   x: +1 forward          y: +1 left            z: +1 up
#   roll/pitch/yaw: +1 = the *_pos action, about TCP axes
#   gripper: axis is open-fraction rate — open = +1, close = -1 (toward closed)
#   rail: +1 = rightward (ArrowRight / rail_pos), -1 = leftward
_AXIS_MAP: dict[str, tuple[str, float]] = {
    "translate_x_pos": ("x", +1.0),
    "translate_x_neg": ("x", -1.0),
    "translate_y_pos": ("y", +1.0),
    "translate_y_neg": ("y", -1.0),
    "translate_z_pos": ("z", +1.0),
    "translate_z_neg": ("z", -1.0),
    "roll_pos": ("roll", +1.0),
    "roll_neg": ("roll", -1.0),
    "pitch_pos": ("pitch", +1.0),
    "pitch_neg": ("pitch", -1.0),
    "yaw_pos": ("yaw", +1.0),
    "yaw_neg": ("yaw", -1.0),
    "gripper_close": ("gripper", -1.0),
    "gripper_open": ("gripper", +1.0),
    "rail_neg": ("rail", -1.0),
    "rail_pos": ("rail", +1.0),
}


def axis_map() -> dict[str, tuple[str, float]]:
    """Held axis action -> ``(axis, sign)``; the single source of teleop signs.

    Held modifiers (:data:`HELD_MODIFIER_ACTIONS`) are deliberately absent.
    """
    return dict(_AXIS_MAP)


__all__ = [
    "KeymapEntry",
    "KEYMAP",
    "HELD_CODES",
    "DISCRETE_CODES",
    "HELD_MODIFIER_ACTIONS",
    "axis_map",
]
