"""Schema-export tests (design doc 01-core §14, §18)."""

from __future__ import annotations

import json

from apollo_xarm7_core.protocol.export_schemas import EXPORTED_MODELS, export, main

_EXPECTED_FILES = {f"{name}.json" for name in EXPORTED_MODELS} | {"keymap.json", "index.json"}


def test_export_is_byte_deterministic(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    files_a = export(a)
    files_b = export(b)
    assert {p.name for p in files_a} == _EXPECTED_FILES
    assert [p.name for p in files_a] == [p.name for p in files_b]
    for pa, pb in zip(files_a, files_b, strict=True):
        assert pa.read_bytes() == pb.read_bytes(), pa.name


def test_check_passes_on_fresh_export(tmp_path):
    out = tmp_path / "schemas"
    assert main(["--out", str(out)]) == 0
    assert main(["--out", str(out), "--check"]) == 0


def test_check_fails_on_tampered_file(tmp_path):
    out = tmp_path / "schemas"
    export(out)
    victim = out / "TelemetryMsg.json"
    victim.write_bytes(victim.read_bytes() + b"// tampered\n")
    assert main(["--out", str(out), "--check"]) == 1


def test_check_fails_on_missing_file(tmp_path):
    out = tmp_path / "schemas"
    export(out)
    (out / "KeysMsg.json").unlink()
    assert main(["--out", str(out), "--check"]) == 1


def test_exported_json_is_clean(tmp_path):
    """Every file json.loads cleanly and contains no numpy artifacts."""
    for path in export(tmp_path / "schemas"):
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n"), path.name
        assert "numpy" not in text, path.name
        assert "ndarray" not in text, path.name
        json.loads(text)  # raises on malformed output


def test_index_and_keymap_contents(tmp_path):
    out = tmp_path / "schemas"
    export(out)
    index = json.loads((out / "index.json").read_text())
    assert index["models"] == sorted(EXPORTED_MODELS)
    assert isinstance(index["core_version"], str) and index["core_version"]
    keymap = json.loads((out / "keymap.json").read_text())
    assert isinstance(keymap, list) and len(keymap) >= 21
    assert {row["code"] for row in keymap} >= {"KeyW", "Space", "ArrowRight"}


def test_exported_models_cover_spec_sections():
    """§14 model list, spelled exactly."""
    assert set(EXPORTED_MODELS) == {
        # control
        "HelloMsg", "KeysMsg", "ActionMsg", "AckMsg",
        "JointTargetArgs", "SaveProfileArgs", "SetInitialConditionArgs",
        # telemetry
        "TelemetryMsg",
        # session
        "SessionSpec", "SessionInfo", "WorkcellStatus", "ArmStatusInfo",
        "CameraInfo", "SceneInfo", "ProfileInfo", "PolicyInfo",
        # misc
        "StateProfile", "KeymapEntry", "CollisionEvent",
    }
