"""Typed exception hierarchy (design doc 01-core §16)."""

from __future__ import annotations

import pytest

import apollo_mavis_v2_core
from apollo_mavis_v2_core.errors import (
    ApolloError,
    ArmConnectError,
    BringupError,
    RailExpectedError,
    RailNotHomedError,
    WorkcellBringupError,
)


def test_bringup_error_step_convention():
    err = ArmConnectError("connect", "box unreachable")
    assert err.step == "connect" and str(err) == "[connect] box unreachable"
    assert str(RailExpectedError("rail")) == "[rail]"
    assert isinstance(err, BringupError) and isinstance(err, ApolloError)


def test_rail_not_homed_error_is_a_rail_bringup_error():
    """phase-09c: connect never homes; an unhomed track fails the ``rail`` step so the
    hardware session is refused and the operator homes from the UI (``home_rail``)."""
    err = RailNotHomedError()
    assert isinstance(err, BringupError) and isinstance(err, ApolloError)
    assert err.step == "rail" and str(err) == "[rail]"
    err = RailNotHomedError(message="grip: linear track not homed (on_zero 0)")
    assert err.step == "rail"
    assert str(err) == "[rail] grip: linear track not homed (on_zero 0)"
    # The sibling (step, message) positional convention still holds.
    err = RailNotHomedError("rail", "view: not homed")
    assert (err.step, str(err)) == ("rail", "[rail] view: not homed")
    # Distinct from the detection mismatch: runtime maps them to different statuses.
    assert not isinstance(err, RailExpectedError)
    assert not isinstance(RailExpectedError("rail"), RailNotHomedError)
    with pytest.raises(BringupError) as info:
        raise RailNotHomedError(message="unhomed")
    assert info.value.step == "rail"


def test_rail_not_homed_error_aggregates_like_its_siblings():
    agg = WorkcellBringupError({"grip": RailNotHomedError(), "view": None})
    assert agg.statuses["grip"].step == "rail" and agg.statuses["view"] is None
    assert str(agg) == "bring-up failed for: grip"


def test_rail_not_homed_error_exported_at_package_top_level():
    assert apollo_mavis_v2_core.RailNotHomedError is RailNotHomedError
    assert "RailNotHomedError" in apollo_mavis_v2_core.__all__
