"""Workcell configuration models, YAML-loadable (design doc 01-core §7).

Config files (``configs/hardware.yaml``, ``configs/sim.yaml``) live in the
runtime deployment dir; :func:`load_workcell_config` is the single loader and
wraps every failure in :class:`~apollo_xarm7_core.errors.ConfigError` carrying
the file path plus a JSON-pointer-ish location.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field, ValidationError, model_validator

from ..errors import ConfigError, FrameRefError
from ..types import FrameRef, Pose, parse_frame
from .safety import SafetyConfig


class PoseModel(BaseModel):
    """JSON/YAML-friendly :class:`~apollo_xarm7_core.types.Pose`."""

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    def to_pose(self) -> Pose:
        """Convert to a numpy Pose; normalizes the quaternion, canonical w >= 0."""
        return Pose(
            np.asarray(self.position, dtype=np.float64),
            np.asarray(self.orientation_wxyz, dtype=np.float64),
        )

    @classmethod
    def from_pose(cls, p: Pose) -> PoseModel:
        return cls(
            position=tuple(float(x) for x in p.position),
            orientation_wxyz=tuple(float(x) for x in p.orientation),
        )


class ArmConfig(BaseModel):
    """One xArm7 (optionally rail-mounted) in the workcell."""

    id: str  # unique; MJCF attach prefix ("arm0_")
    ip: str | None = None  # required when kind == hardware
    base_in_world: PoseModel  # rail arms: the RAIL ORIGIN (fixed);
    #   base pose = rail origin ⊕ rail travel
    expect_rail: Literal["auto", "yes", "no"] = "auto"
    gripper: Literal["xarm", "xarm_g2", "none"] = "xarm"
    tcp_load_kg: float = 0.82  # -> set_tcp_load (L3 collision detection)
    tcp_load_cog_mm: tuple[float, float, float] = (0.0, 0.0, 48.0)


class CameraIntrinsics(BaseModel):
    """Pinhole intrinsics (pixels)."""

    fx: float
    fy: float
    cx: float
    cy: float
    distortion: list[float] = []  # OpenCV k1..k5


class CameraConfig(BaseModel):
    """One camera stream in the workcell."""

    id: str
    kind: Literal["v4l2", "realsense", "sim"]
    device_path: str | None = None  # v4l2 (stable by-id path)
    serial: str | None = None  # realsense
    resolution: tuple[int, int] = (640, 480)
    fps: int = 30
    intrinsics: CameraIntrinsics | None = None
    extrinsics_frame: FrameRef | None = None  # frame the calibration is expressed in
    extrinsics_file: str | None = None  # calibration file path (spine §3.1)


class WorkcellConfig(BaseModel):
    """Top-level workcell description (§7 cross-field validators)."""

    kind: Literal["hardware", "sim"]
    arms: list[ArmConfig] = Field(min_length=1, max_length=3)
    cameras: list[CameraConfig] = Field(default=[], max_length=4)
    sim_scene: str | None = None  # scene-registry id (sim mode)
    digital_twin_scene: str | None = None  # scene-registry id (hardware mode)
    safety: SafetyConfig = SafetyConfig()

    @model_validator(mode="after")
    def _cross_field(self) -> WorkcellConfig:
        arm_ids = [a.id for a in self.arms]
        if len(set(arm_ids)) != len(arm_ids):
            raise ValueError(f"arm ids must be unique, got {arm_ids}")
        cam_ids = [c.id for c in self.cameras]
        if len(set(cam_ids)) != len(cam_ids):
            raise ValueError(f"camera ids must be unique, got {cam_ids}")

        if self.kind == "sim" and not self.sim_scene:
            raise ValueError("kind == 'sim' requires sim_scene")
        if self.kind == "hardware":
            if not self.digital_twin_scene:
                raise ValueError("kind == 'hardware' requires digital_twin_scene")
            if not self.safety.enabled:
                raise ValueError("hardware refuses to start with safety.enabled=False")
            for arm in self.arms:
                if not arm.ip:
                    raise ValueError(f"hardware arm {arm.id!r} requires ip")

        for cam in self.cameras:
            if cam.kind == "v4l2" and not cam.device_path:
                raise ValueError(f"v4l2 camera {cam.id!r} requires device_path")
            if cam.kind == "realsense" and not cam.serial:
                raise ValueError(f"realsense camera {cam.id!r} requires serial")
            if cam.kind == "sim" and self.kind != "sim":
                raise ValueError(f"sim camera {cam.id!r} only allowed in sim configs")
            if cam.extrinsics_frame is not None:
                try:
                    parsed = parse_frame(cam.extrinsics_frame)
                except FrameRefError as e:
                    raise ValueError(
                        f"camera {cam.id!r}: bad extrinsics_frame: {e}"
                    ) from e
                if parsed.kind in ("arm_base", "ee") and parsed.ident not in arm_ids:
                    raise ValueError(
                        f"camera {cam.id!r}: extrinsics_frame references "
                        f"undeclared arm {parsed.ident!r}"
                    )
                if parsed.kind == "camera" and parsed.ident not in cam_ids:
                    raise ValueError(
                        f"camera {cam.id!r}: extrinsics_frame references "
                        f"undeclared camera {parsed.ident!r}"
                    )
        return self


def load_workcell_config(path: str | Path) -> WorkcellConfig:
    """Load + validate a workcell YAML; every failure becomes a ConfigError."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(str(e), path=str(p)) from e
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML: {e}", path=str(p)) from e
    if not isinstance(data, dict):
        raise ConfigError(
            f"expected a mapping at the document root, got {type(data).__name__}",
            path=str(p),
            loc="/",
        )
    try:
        return WorkcellConfig.model_validate(data)
    except ValidationError as e:
        first = e.errors()[0]
        loc = "/" + "/".join(str(part) for part in first["loc"])
        raise ConfigError(first["msg"], path=str(p), loc=loc) from e


__all__ = [
    "PoseModel",
    "ArmConfig",
    "CameraIntrinsics",
    "CameraConfig",
    "WorkcellConfig",
    "load_workcell_config",
]
