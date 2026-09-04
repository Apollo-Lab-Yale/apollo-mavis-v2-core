"""ProfileStore tests (design doc 01-core §8, §18) — tmp_path, no I/O beyond it."""

from __future__ import annotations

import json
import os

import pytest

from apollo_mavis_v2_core.errors import ProfileError, ProfileNotFoundError
from apollo_mavis_v2_core.profiles import store as store_mod
from apollo_mavis_v2_core.profiles.store import ProfileStore
from apollo_mavis_v2_core.schemas.profile import ArmPosture, StateProfile


def _profile(name: str = "home", kind: str = "sim", **kw) -> StateProfile:
    return StateProfile(
        name=name,
        workcell_kind=kind,
        arms={
            "arm0": ArmPosture(q=[0.0, -0.5, 0.0, 0.8, 0.0, 1.2, 0.0], rail_pos_m=0.1),
            "arm1": ArmPosture(q=[0.1] * 7, gripper_open_frac=0.5),
        },
        **kw,
    )


def test_crud_round_trip(tmp_path):
    store = ProfileStore(tmp_path)
    assert store.list() == []

    saved = store.save(_profile(name="home"))
    assert len(saved.profile_id) == 32  # uuid4 hex assigned
    assert saved.created_at  # ISO-8601 UTC assigned
    assert (tmp_path / f"{saved.profile_id}.json").exists()

    got = store.get(saved.profile_id)
    assert got == saved
    assert got.arms["arm0"].rail_pos_m == 0.1
    assert got.arms["arm1"].rail_pos_m is None

    # explicit created_at values drive list() ordering
    older = store.save(_profile(name="older", created_at="2026-01-01T00:00:00+00:00"))
    assert [p.name for p in store.list()][:1] == ["older"]
    assert [p.name for p in store.list()] == ["older", "home"]

    renamed = store.rename(saved.profile_id, "home2", notes="tweaked")
    assert renamed.name == "home2" and renamed.notes == "tweaked"
    assert store.get(saved.profile_id).name == "home2"
    kept = store.rename(saved.profile_id, "home3")  # notes=None keeps notes
    assert kept.notes == "tweaked"

    # explicit id = atomic overwrite, not a new file
    overwritten = store.save(renamed.model_copy(update={"name": "home4"}))
    assert overwritten.profile_id == saved.profile_id
    assert len(store.list()) == 2

    store.delete(saved.profile_id)
    with pytest.raises(ProfileNotFoundError):
        store.get(saved.profile_id)
    with pytest.raises(ProfileNotFoundError):
        store.delete(saved.profile_id)
    assert [p.name for p in store.list()] == [older.name]


def test_atomic_overwrite_keeps_old_file_on_crash(tmp_path, monkeypatch):
    store = ProfileStore(tmp_path)
    saved = store.save(_profile(name="v1"))

    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("injected crash before replace")
        return real_replace(src, dst)

    monkeypatch.setattr(store_mod.os, "replace", flaky)
    with pytest.raises(OSError):
        store.save(saved.model_copy(update={"name": "v2"}))
    monkeypatch.undo()

    assert store.get(saved.profile_id).name == "v1"  # old file intact
    # leftover .json.tmp is ignored by scans
    assert [p.profile_id for p in store.list()] == [saved.profile_id]
    assert store.errors == []


def test_set_initial_uniqueness_per_kind(tmp_path):
    store = ProfileStore(tmp_path)
    a = store.save(_profile(name="a"))
    b = store.save(_profile(name="b"))
    hw = store.save(_profile(name="hw", kind="hardware"))

    store.set_initial(a.profile_id)
    store.set_initial(hw.profile_id)  # different kind: independent flag
    assert store.initial_for("sim").profile_id == a.profile_id
    assert store.initial_for("hardware").profile_id == hw.profile_id

    store.set_initial(b.profile_id)  # steals the sim flag from a
    flagged = {p.profile_id for p in store.list() if p.is_initial_condition}
    assert flagged == {b.profile_id, hw.profile_id}
    assert store.initial_for("sim").profile_id == b.profile_id

    with pytest.raises(ProfileNotFoundError):
        store.set_initial("deadbeef")


def test_set_initial_crash_ordering_leaves_zero_flags(tmp_path, monkeypatch):
    store = ProfileStore(tmp_path)
    a = store.save(_profile(name="a"))
    b = store.save(_profile(name="b"))
    store.set_initial(a.profile_id)

    real_replace = os.replace

    def crash_on_target(src, dst):
        if str(dst).endswith(f"{b.profile_id}.json"):
            raise OSError("injected crash between clear and target write")
        return real_replace(src, dst)

    monkeypatch.setattr(store_mod.os, "replace", crash_on_target)
    with pytest.raises(OSError):
        store.set_initial(b.profile_id)
    monkeypatch.undo()

    # target written LAST: crash mid-op -> zero flags, never two
    assert [p for p in store.list() if p.is_initial_condition] == []
    assert store.initial_for("sim") is None


def test_delete_refuses_designated_initial(tmp_path):
    store = ProfileStore(tmp_path)
    a = store.save(_profile(name="a"))
    b = store.save(_profile(name="b"))
    store.set_initial(a.profile_id)

    with pytest.raises(ProfileError):
        store.delete(a.profile_id)
    assert store.get(a.profile_id).is_initial_condition  # still there

    store.delete(b.profile_id)  # non-initial deletes fine
    store.set_initial(a.profile_id)  # re-designating is idempotent
    assert store.initial_for("sim").profile_id == a.profile_id


def test_foreign_and_unreadable_files_skipped_into_errors(tmp_path):
    store = ProfileStore(tmp_path)
    saved = store.save(_profile())

    (tmp_path / "garbage.json").write_text("{not json at all")
    (tmp_path / "foreign.json").write_text(json.dumps({"hello": "world"}))
    misnamed = tmp_path / "misnamed.json"  # valid profile, filename != profile_id
    misnamed.write_text((tmp_path / f"{saved.profile_id}.json").read_text())
    (tmp_path / "too_new.json").write_text(
        json.dumps({"schema_version": 99, "profile_id": "too_new"})
    )

    profiles = store.list()  # never fatal
    assert [p.profile_id for p in profiles] == [saved.profile_id]
    bad_names = {path.name for path, _ in store.errors}
    assert bad_names == {"garbage.json", "foreign.json", "misnamed.json", "too_new.json"}
    assert all(msg for _, msg in store.errors)


def test_schema_version_migration_hook(tmp_path, monkeypatch):
    store = ProfileStore(tmp_path)
    saved = store.save(_profile(name="old"))
    raw = json.loads((tmp_path / f"{saved.profile_id}.json").read_text())
    raw["schema_version"] = 0
    raw["legacy_field"] = True
    (tmp_path / f"{saved.profile_id}.json").write_text(json.dumps(raw))

    # no registered migration: skipped into errors, not fatal
    assert store.list() == []
    assert len(store.errors) == 1

    def migrate_0_to_1(data):
        data = dict(data)
        data.pop("legacy_field", None)
        data["schema_version"] = 1
        return data

    monkeypatch.setitem(store_mod.MIGRATIONS, 0, migrate_0_to_1)
    migrated = store.get(saved.profile_id)
    assert migrated.schema_version == 1
    assert migrated.name == "old"
    assert store.list() == [migrated] and store.errors == []
