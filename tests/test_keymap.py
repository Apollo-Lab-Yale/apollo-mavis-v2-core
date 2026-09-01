"""Keymap invariants and literal spine-§5 table equality (01-core §13, §18).

Any keymap edit is a conscious spine change: the code->action assertions
below are the spine table verbatim. Note: §13's prose says "25 entries" but
enumerates exactly 21, matching spine §5 (7 held pairs + arrows + 5 discrete
keys = 21); the enumerated table is authoritative.
"""

from __future__ import annotations

import typing

from apollo_xarm7_core.protocol.control import ActionName
from apollo_xarm7_core.protocol.keymap import (
    DISCRETE_CODES,
    HELD_CODES,
    KEYMAP,
    axis_map,
)

# Spine §5, verbatim: code -> (action, kind, group).
_SPINE_TABLE = {
    "KeyW": ("translate_x_pos", "held", "translate"),
    "KeyS": ("translate_x_neg", "held", "translate"),
    "KeyA": ("translate_y_pos", "held", "translate"),
    "KeyD": ("translate_y_neg", "held", "translate"),
    "KeyE": ("translate_z_pos", "held", "translate"),
    "KeyQ": ("translate_z_neg", "held", "translate"),
    "KeyI": ("roll_pos", "held", "rotate"),
    "KeyK": ("roll_neg", "held", "rotate"),
    "KeyJ": ("pitch_pos", "held", "rotate"),
    "KeyL": ("pitch_neg", "held", "rotate"),
    "KeyU": ("yaw_pos", "held", "rotate"),
    "KeyO": ("yaw_neg", "held", "rotate"),
    "KeyF": ("gripper_close", "held", "gripper"),
    "KeyH": ("gripper_open", "held", "gripper"),
    "ArrowLeft": ("rail_neg", "held", "rail"),
    "ArrowRight": ("rail_pos", "held", "rail"),
    "Tab": ("switch_arm", "discrete", "session"),
    "Space": ("takeover_toggle", "discrete", "session"),
    "KeyN": ("episode_new", "discrete", "episode"),
    "Enter": ("episode_save", "discrete", "episode"),
    "Backspace": ("episode_discard", "discrete", "episode"),
}


def test_entry_count_and_unique_codes():
    assert len(KEYMAP) == 21  # the full enumerated §13 table (== spine §5)
    codes = [e.code for e in KEYMAP]
    assert len(set(codes)) == len(codes)


def test_literal_table_equality_against_spine():
    assert {e.code for e in KEYMAP} == set(_SPINE_TABLE)
    for entry in KEYMAP:
        action, kind, group = _SPINE_TABLE[entry.code]
        assert (entry.action, entry.kind, entry.group) == (action, kind, group), entry.code


def test_translate_labels_match_spec():
    labels = {e.code: e.label for e in KEYMAP}
    assert labels["KeyW"] == "+x (forward)"
    assert labels["KeyS"] == "-x (back)"
    assert labels["KeyA"] == "left"
    assert labels["KeyD"] == "right"
    assert labels["KeyE"] == "up"
    assert labels["KeyQ"] == "down"
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
        assert entry.requires_rail == (entry.group == "rail")


def test_derived_views_match_table():
    assert HELD_CODES == {e.code for e in KEYMAP if e.kind == "held"}
    assert DISCRETE_CODES == {
        e.code: e.action for e in KEYMAP if e.kind == "discrete"
    }
    assert DISCRETE_CODES == {
        "Tab": "switch_arm",
        "Space": "takeover_toggle",
        "KeyN": "episode_new",
        "Enter": "episode_save",
        "Backspace": "episode_discard",
    }


def test_axis_map_covers_every_held_entry():
    amap = axis_map()
    held_actions = {e.action for e in KEYMAP if e.kind == "held"}
    assert set(amap) == held_actions
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
    # Every +/- action pair shares an axis with opposite signs.
    by_axis: dict[str, set[float]] = {}
    for axis, sign in amap.values():
        by_axis.setdefault(axis, set()).add(sign)
    assert all(signs == {+1.0, -1.0} for signs in by_axis.values())
