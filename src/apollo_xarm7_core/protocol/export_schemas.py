"""JSON-schema export for TS generation (design doc 01-core §14).

UI pipeline (05-ui §2): core exports JSON Schema -> UI ``pnpm gen:sync``
copies ``schemas/*.json`` -> ``json-schema-to-typescript`` emits
``src/gen/*.ts``. Output is byte-deterministic; ``--check`` is the CI drift
guard against the checked-in ``schemas/`` directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import metadata
from pathlib import Path

from pydantic import BaseModel

from apollo_xarm7_core.errors import SchemaExportError
from apollo_xarm7_core.protocol import control, keymap, session, telemetry
from apollo_xarm7_core.schemas.profile import StateProfile
from apollo_xarm7_core.schemas.safety import CollisionEvent

_REF_TEMPLATE = "#/$defs/{model}"
_FALLBACK_VERSION = "0.1.0"

EXPORTED_MODELS: dict[str, type[BaseModel]] = {
    # control (§10)
    "HelloMsg": control.HelloMsg,
    "KeysMsg": control.KeysMsg,
    "ActionMsg": control.ActionMsg,
    "AckMsg": control.AckMsg,
    "JointTargetArgs": control.JointTargetArgs,
    "SaveProfileArgs": control.SaveProfileArgs,
    "SetInitialConditionArgs": control.SetInitialConditionArgs,
    # telemetry (§11; embeds sub-models via $defs)
    "TelemetryMsg": telemetry.TelemetryMsg,
    # session (§12)
    "SessionSpec": session.SessionSpec,
    "SessionInfo": session.SessionInfo,
    "WorkcellStatus": session.WorkcellStatus,
    "ArmStatusInfo": session.ArmStatusInfo,
    "CameraInfo": session.CameraInfo,
    "SceneInfo": session.SceneInfo,
    "ProfileInfo": session.ProfileInfo,
    "PolicyInfo": session.PolicyInfo,
    # misc (§8, §13, §6)
    "StateProfile": StateProfile,
    "KeymapEntry": keymap.KeymapEntry,
    "CollisionEvent": CollisionEvent,
}


def _core_version() -> str:
    try:
        return metadata.version("apollo-xarm7-core")
    except metadata.PackageNotFoundError:  # pragma: no cover - dev tree fallback
        return _FALLBACK_VERSION


def _dumps(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, indent=2) + "\n"


def _render_files() -> dict[str, str]:
    """All export outputs as ``{filename: content}`` (byte-deterministic)."""
    try:
        files = {
            f"{name}.json": _dumps(model.model_json_schema(ref_template=_REF_TEMPLATE))
            for name, model in EXPORTED_MODELS.items()
        }
        files["keymap.json"] = _dumps([entry.model_dump() for entry in keymap.KEYMAP])
        files["index.json"] = _dumps(
            {"core_version": _core_version(), "models": sorted(EXPORTED_MODELS)}
        )
    except Exception as exc:  # noqa: BLE001 - re-typed for callers (§16)
        raise SchemaExportError(f"schema generation failed: {exc!r}") from exc
    return files


def export(out_dir: Path) -> list[Path]:
    """Write one ``<Name>.json`` per model plus keymap.json and index.json."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for filename, content in sorted(_render_files().items()):
        path = out_dir / filename
        path.write_bytes(content.encode("utf-8"))
        written.append(path)
    return written


def check(out_dir: Path) -> list[str]:
    """Filenames that would change if regenerated into ``out_dir``."""
    out_dir = Path(out_dir)
    drifted: list[str] = []
    for filename, content in sorted(_render_files().items()):
        path = out_dir / filename
        if not path.is_file() or path.read_bytes() != content.encode("utf-8"):
            drifted.append(filename)
    return drifted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m apollo_xarm7_core.protocol.export_schemas",
        description="Export core wire-model JSON Schemas (01-core §14).",
    )
    parser.add_argument("--out", required=True, type=Path, help="output directory")
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if regeneration would change any file (CI drift guard)",
    )
    args = parser.parse_args(argv)

    try:
        if args.check:
            drifted = check(args.out)
            if drifted:
                print(
                    f"schema drift in {args.out}: {', '.join(drifted)} "
                    "(re-run export to update)",
                    file=sys.stderr,
                )
                return 1
            return 0
        written = export(args.out)
        print(f"wrote {len(written)} files to {args.out}")
        return 0
    except SchemaExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
