"""JSON-directory profile store (design doc 01-core §8).

One file per profile ``<profile_id>.json`` under ``root``. Every write is
atomic: ``<id>.json.tmp`` then ``os.replace`` — a crashed save never corrupts.
Thread-safe (§16); unreadable/foreign files are skipped into
``store.errors``, never fatal.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ..errors import ProfileError, ProfileNotFoundError
from ..schemas.profile import StateProfile

SCHEMA_VERSION = 1

Migration = Callable[[dict[str, Any]], dict[str, Any]]
MIGRATIONS: dict[int, Migration] = {}
"""Read hook: from-version -> upgrader returning a dict with a bumped version."""


def _version_of(data: dict[str, Any], default: int) -> int:
    """schema_version as an int; ProfileError on non-integer junk (foreign files)."""
    raw = data.get("schema_version", default)
    try:
        return int(raw)
    except (TypeError, ValueError) as e:
        raise ProfileError(f"invalid schema_version {raw!r}") from e


def _migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a raw profile dict to SCHEMA_VERSION; ProfileError if impossible."""
    version = _version_of(data, 1)
    while version < SCHEMA_VERSION:
        step = MIGRATIONS.get(version)
        if step is None:
            raise ProfileError(f"no migration from profile schema_version {version}")
        data = step(data)
        new_version = _version_of(data, version)
        if new_version <= version:
            raise ProfileError(f"migration from schema_version {version} did not advance")
        version = new_version
    if version > SCHEMA_VERSION:
        raise ProfileError(
            f"profile schema_version {version} is newer than supported {SCHEMA_VERSION}"
        )
    return data


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProfileStore:
    """One ``<profile_id>.json`` per profile under ``root``; atomic writes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.errors: list[tuple[Path, str]] = []
        self._lock = threading.RLock()

    # -- internal helpers -----------------------------------------------------

    _ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")  # uuid4 hex + START_FROM_RE charset

    def _path(self, profile_id: str) -> Path:
        if not self._ID_RE.fullmatch(profile_id):
            # Wire-visible ids must never become path components ("../evil").
            raise ProfileError(f"invalid profile_id {profile_id!r}")
        return self.root / f"{profile_id}.json"

    def _load(self, path: Path) -> StateProfile:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ProfileError(f"{path}: unreadable profile: {e}") from e
        if not isinstance(data, dict):
            raise ProfileError(f"{path}: not a profile object")
        data = _migrate(data)
        try:
            profile = StateProfile.model_validate(data)
        except Exception as e:  # pydantic ValidationError
            raise ProfileError(f"{path}: invalid profile: {e}") from e
        if profile.profile_id != path.stem:
            raise ProfileError(
                f"{path}: profile_id {profile.profile_id!r} does not match filename"
            )
        return profile

    def _scan(self) -> list[StateProfile]:
        """All readable profiles; failures recorded in ``self.errors``."""
        profiles: list[StateProfile] = []
        errors: list[tuple[Path, str]] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                profiles.append(self._load(path))
            except ProfileError as e:
                errors.append((path, str(e)))
        self.errors = errors
        return profiles

    def _write(self, profile: StateProfile) -> None:
        """Atomic write: ``<id>.json.tmp`` then ``os.replace``."""
        path = self._path(profile.profile_id)
        tmp = path.parent / (path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(profile.model_dump_json(indent=2) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    # -- public API (§8) --------------------------------------------------------

    def list(self) -> list[StateProfile]:
        with self._lock:
            return sorted(self._scan(), key=lambda p: (p.created_at, p.profile_id))

    def get(self, profile_id: str) -> StateProfile:
        with self._lock:
            path = self._path(profile_id)
            if not path.exists():
                raise ProfileNotFoundError(profile_id)
            return self._load(path)

    def save(self, profile: StateProfile) -> StateProfile:
        """Assign id/created_at when empty; existing id = atomic overwrite."""
        with self._lock:
            updates: dict[str, Any] = {}
            if not profile.profile_id:
                updates["profile_id"] = uuid.uuid4().hex
            if not profile.created_at:
                updates["created_at"] = _utc_now_iso()
            if updates:
                profile = profile.model_copy(update=updates)
            self._write(profile)
            return profile

    def delete(self, profile_id: str) -> None:
        """Refuses (ProfileError) on the designated initial-condition profile."""
        with self._lock:
            profile = self.get(profile_id)
            if profile.is_initial_condition:
                raise ProfileError(
                    f"refusing to delete initial-condition profile {profile_id!r}; "
                    "designate another profile first"
                )
            self._path(profile_id).unlink()

    def rename(
        self, profile_id: str, name: str, notes: str | None = None
    ) -> StateProfile:
        with self._lock:
            profile = self.get(profile_id)
            updates: dict[str, Any] = {"name": name}
            if notes is not None:
                updates["notes"] = notes
            profile = profile.model_copy(update=updates)
            self._write(profile)
            return profile

    def set_initial(self, profile_id: str) -> None:
        """Clear the flag on all others of the same kind FIRST, write the target
        LAST — a crash mid-op leaves zero flags, never two."""
        with self._lock:
            target = self.get(profile_id)
            for other in self._scan():
                if other.profile_id == target.profile_id:
                    continue
                if other.workcell_kind == target.workcell_kind and other.is_initial_condition:
                    self._write(other.model_copy(update={"is_initial_condition": False}))
            self._write(target.model_copy(update={"is_initial_condition": True}))

    def initial_for(self, kind: Literal["hardware", "sim"]) -> StateProfile | None:
        with self._lock:
            for profile in self.list():
                if profile.workcell_kind == kind and profile.is_initial_condition:
                    return profile
            return None


__all__ = ["ProfileStore", "SCHEMA_VERSION", "MIGRATIONS"]
