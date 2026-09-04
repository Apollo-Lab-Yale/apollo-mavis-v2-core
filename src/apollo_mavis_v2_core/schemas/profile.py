"""State-profile models (design doc 01-core §8).

Persisted by :class:`~apollo_mavis_v2_core.profiles.store.ProfileStore` as one
JSON file per profile. Loading a profile is always runtime-side twin-planned
motion — never xArm native gohome.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ArmPosture(BaseModel):
    """One arm's stored posture; ``q`` NEVER includes the rail slot."""

    q: list[float] = Field(min_length=7, max_length=7)  # rad, exactly 7 joints
    rail_pos_m: float | None = None  # None = no rail
    gripper_open_frac: float = Field(default=1.0, ge=0.0, le=1.0)

    @field_validator("q")
    @classmethod
    def _finite(cls, q: list[float]) -> list[float]:
        if not all(math.isfinite(v) for v in q):
            raise ValueError("q must be finite")
        return q


class StateProfile(BaseModel):
    """Named workcell posture snapshot; at most one initial condition per kind."""

    schema_version: int = 1  # store migrates old files on read
    profile_id: str = ""  # uuid4 hex; assigned by the store when empty
    name: str
    notes: str = ""
    workcell_kind: Literal["hardware", "sim"]
    arms: dict[str, ArmPosture]  # may cover a subset of the workcell
    created_at: str = ""  # ISO 8601 UTC; assigned by the store when empty
    is_initial_condition: bool = False  # AT MOST one per workcell_kind (invariant)


__all__ = ["ArmPosture", "StateProfile"]
