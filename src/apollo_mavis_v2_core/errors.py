"""Typed exception hierarchy shared by the whole stack (design doc 01-core §16).

Typed exceptions signal *caller* mistakes (bad shapes, ranges, missing
capability). Environment problems (stale reports, dead cameras) surface as
degraded-but-valid data instead, so control loops keep ticking.
"""

from __future__ import annotations


class ApolloError(Exception):
    """Base class for every exception raised by the apollo-mavis-v2 stack."""


class ConfigError(ApolloError):
    """Invalid workcell configuration; carries the file path and location."""

    def __init__(self, message: str, *, path: str | None = None, loc: str | None = None):
        self.path = path
        self.loc = loc
        prefix = "".join(
            part for part in (path and f"{path}: ", loc and f"at {loc}: ") if part
        )
        super().__init__(f"{prefix}{message}")


class FrameRefError(ApolloError, ValueError):
    """Malformed or out-of-vocabulary FrameRef string."""


class CommandError(ApolloError):
    """Invalid command input (NaN, wrong shape, out of range)."""


class RailUnavailableError(CommandError):
    """Rail command sent to an arm without a rail."""


class ProfileError(ApolloError):
    """StateProfile store failure."""


class ProfileNotFoundError(ProfileError, KeyError):
    """Unknown profile_id."""


class VideoFramingError(ApolloError, ValueError):
    """Malformed /ws/video binary frame."""


class SchemaExportError(ApolloError):
    """JSON-schema export failed or drifted from the checked-in files."""


class BringupError(ApolloError):
    """A workcell component failed to come up.

    Raised by hardware/sim implementations; typed here so runtime can catch
    them without importing either implementation package.
    """

    def __init__(self, step: str, message: str = ""):
        self.step = step
        super().__init__(f"[{step}] {message}" if message else f"[{step}]")


class ArmConnectError(BringupError):
    """Arm unreachable / SDK connect failed."""


class ArmIdentityError(BringupError):
    """Connected arm does not match the configured identity (serial swap)."""


class RailExpectedError(BringupError):
    """expect_rail says yes/no but detection disagreed."""


class GripperInitError(BringupError):
    """Gripper enable/initialization failed."""


class CameraInitError(BringupError):
    """Camera failed to open or produce frames."""


class WorkcellBringupError(ApolloError):
    """Aggregate bring-up failure; per-component detail for the landing page."""

    def __init__(self, statuses: dict[str, BringupError | None]):
        self.statuses = statuses
        failed = sorted(k for k, v in statuses.items() if v is not None)
        super().__init__(f"bring-up failed for: {', '.join(failed) or '<none>'}")
