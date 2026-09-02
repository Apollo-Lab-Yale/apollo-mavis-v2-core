"""Keymap invariants and literal spine-§5 table equality (01-core §13, §18).

Any keymap edit is a conscious spine change: the code->action assertions
below are the spine table verbatim (23 entries after 13-tracker §3: 7 held
axis pairs + arrows + the ``tracker_clutch`` modifier + 6 discrete keys).
"""

from __future__ import annotations

import typing

from apollo_xarm7_core.protocol.control import ActionName
from apollo_xarm7_core.protocol.keymap import (
    DISCRETE_CODES,
    HELD_CODES,
    HELD_MODIFIER_ACTIONS,
    KEYMAP,
    axis_map,
)

# Spine §5 / 13-tracker §1, verbatim: code -> (action, kind, group, gamepad).
_SPINE_TABLE = {
    "KeyW": ("translate_x_pos", "held", "translate", None),
    "KeyS": ("translate_x_neg", "held", "translate", None),
    "KeyA": ("translate_y_pos", "held", "translate", None),
    "KeyD": ("translate_y_neg", "held", "translate", None),
    "KeyE": ("translate_z_pos", "held", "translate", None),
    "KeyQ": ("translate_z_neg", "held", "translate", None),
    "KeyI": ("roll_pos", "held", "rotate", None),
    "KeyK": ("roll_neg", "held", "rotate", None),
    "KeyJ": ("pitch_pos", "held", "rotate", None),
    "KeyL": ("pitch_neg", "held", "rotate", None),
    "KeyU": ("yaw_pos", "held", "rotate", None),
    "KeyO": ("yaw_neg", "held", "rotate", None),
    "KeyF": ("gripper_close", "held", "gripper", "B"),
    "KeyH": ("gripper_open", "held", "gripper", "A"),
    "ArrowLeft": ("rail_neg", "held", "rail", "DpadLeft"),
    "ArrowRight": ("rail_pos", "held", "rail", "DpadRight"),
    "KeyC": ("tracker_clutch", "held", "tracker", "RT"),
    "KeyZ": ("switch_arm_prev", "discrete", "session", "LB"),
    "Tab": ("switch_arm", "discrete", "session", "RB"),
    "Space": ("takeover_toggle", "discrete", "session", None),
    "KeyN": ("episode_new", "discrete", "episode", None),
    "Enter": ("episode_save", "discrete", "episode", None),
    "Backspace": ("episode_discard", "discrete", "episode", None),
}


def test_entry_count_and_unique_codes():
    assert len(KEYMAP) == 23  # the full enumerated §13 table (== spine §5)
    codes = [e.code for e in KEYMAP]
    assert len(set(codes)) == len(codes)


def test_literal_table_equality_against_spine():
    assert {e.code for e in KEYMAP} == set(_SPINE_TABLE)
    for entry in KEYMAP:
        action, kind, group, gamepad = _SPINE_TABLE[entry.code]
        assert (entry.action, entry.kind, entry.group) == (action, kind, group), entry.code
        assert entry.gamepad == gamepad, entry.code


def test_translate_labels_match_spec():
    labels = {e.code: e.label for e in KEYMAP}
    assert labels["KeyW"] == "+x (forward)"
    assert labels["KeyS"] == "-x (back)"
    assert labels["KeyA"] == "left"
    assert labels["KeyD"] == "right"
    assert labels["KeyE"] == "up"
    assert labels["KeyQ"] == "down"
    assert labels["KeyC"] == "tracker clutch (hold)"
    assert labels["KeyZ"] == "previous arm"
    assert labels["Space"] == (
        "takeover toggle (DAgger: recorded; inference: safety escape, never recorded)"
    )


def test_discrete_actions_are_valid_action_names():
    valid = set(typing.get_args(ActionName))
    for entry in KEYMAP:
        if entry.kind == "discrete":
            assert entry.action in valid, entry.action


def test_exactly_rail_entries_require_rail():
    rail_required = {e.code for e in KEYMAP if e.requires_rail}
    assert rail_required == {"ArrowLeft", "ArrowRight"}
    for entry in KEYMAP:
        # group "rail" <=> requires_rail; the tracker modifier never drives the rail
        assert entry.requires_rail == (entry.group == "rail")


def test_derived_views_match_table():
    assert HELD_CODES == {e.code for e in KEYMAP if e.kind == "held"}
    assert DISCRETE_CODES == {
        e.code: e.action for e in KEYMAP if e.kind == "discrete"
    }
    assert DISCRETE_CODES == {
        "KeyZ": "switch_arm_prev",
        "Tab": "switch_arm",
        "Space": "takeover_toggle",
        "KeyN": "episode_new",
        "Enter": "episode_save",
        "Backspace": "episode_discard",
    }


def test_held_modifiers_are_held_non_axis_actions():
    """13-tracker §3.3: modifiers are held rows that never appear in axis_map."""
    assert HELD_MODIFIER_ACTIONS == {"tracker_clutch"}
    held_actions = {e.action for e in KEYMAP if e.kind == "held"}
    assert HELD_MODIFIER_ACTIONS <= held_actions
    assert not (HELD_MODIFIER_ACTIONS & set(axis_map()))
    for entry in KEYMAP:
        if entry.action in HELD_MODIFIER_ACTIONS:
            assert entry.kind == "held" and entry.group == "tracker", entry.code


def test_gamepad_labels_match_tracker_doc():
    """13-tracker §1: exactly these seven rows carry a gamepad control."""
    by_gamepad = {e.gamepad: e.action for e in KEYMAP if e.gamepad is not None}
    assert by_gamepad == {
        "DpadLeft": "rail_neg",
        "DpadRight": "rail_pos",
        "A": "gripper_open",
        "B": "gripper_close",
        "LB": "switch_arm_prev",
        "RB": "switch_arm",
        "RT": "tracker_clutch",
    }
    # RT is the only held modifier on the pad; every other pad row is an axis or discrete.
    for entry in KEYMAP:
        if entry.gamepad is None:
            continue
        if entry.kind == "held":
            assert (entry.action in axis_map()) != (entry.action in HELD_MODIFIER_ACTIONS)


def test_axis_map_covers_every_held_axis_entry():
    amap = axis_map()
    held_actions = {e.action for e in KEYMAP if e.kind == "held"}
    # Every held row is either an axis or a modifier — never both, never neither.
    assert set(amap) == held_actions - HELD_MODIFIER_ACTIONS
    for axis, sign in amap.values():
        assert axis in {"x", "y", "z", "roll", "pitch", "yaw", "gripper", "rail"}
        assert sign in (+1.0, -1.0)
    # Documented sign choices (single source for held_to_twist).
    assert amap["translate_x_pos"] == ("x", +1.0)
    assert amap["translate_y_pos"] == ("y", +1.0)  # +y = left
    assert amap["translate_z_neg"] == ("z", -1.0)
    assert amap["gripper_close"] == ("gripper", -1.0)  # close = -1 toward closed
    assert amap["gripper_open"] == ("gripper", +1.0)
    assert amap["rail_pos"] == ("rail", +1.0)  # +1 = rightward (ArrowRight)
    assert amap["rail_neg"] == ("rail", -1.0)
    # Every +/- axis pair shares an axis with opposite signs (axis actions only;
    # held modifiers have no sign and are excluded above).
    by_axis: dict[str, set[float]] = {}
    for axis, sign in amap.values():
        by_axis.setdefault(axis, set()).add(sign)
    assert all(signs == {+1.0, -1.0} for signs in by_axis.values())
