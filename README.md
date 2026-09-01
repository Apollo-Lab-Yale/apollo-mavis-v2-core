# apollo-xarm7-core

Foundation package of the apollo-xarm7 stack: geometry & SE3 math, domain
state types, hardware/sim interfaces, config & profile schemas, DAgger core
types, the runtime↔UI wire protocol (with JSON-schema export for TypeScript
generation), and the command-bus/latest-slot threading primitives.

Depends only on `numpy`, `pydantic` v2, and `PyYAML` — never on MuJoCo, the
xArm SDK, FastAPI, torch, or lerobot (enforced by an import-guard test and
ruff banned-imports).

Design document: `apollo-xarm7-ws/docs/design/01-core.md` (binding).
System contract: `apollo-xarm7-ws/docs/design/00-overview.md`.

## Conventions

- Units: meters, radians. Quaternions `(w, x, y, z)`, canonical `w >= 0`.
- Joint vectors: 7 arm joints first; rail position (m) last at `q[7]`.
- Frames: `world` | `arm_base:<arm_id>` | `camera:<camera_id>` | `ee:<arm_id>`.

## Development

```bash
uv sync
uv run pytest
uv run ruff check src tests
uv run python -m apollo_xarm7_core.protocol.export_schemas --out schemas/
```
