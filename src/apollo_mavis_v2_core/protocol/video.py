"""Video WS binary framing (design doc 01-core §13, binding).

``/ws/video/{stream_id}`` frames are a 12-byte little-endian header (f64
timestamp seconds + u32 JPEG length) followed by the JPEG payload.
"""

from __future__ import annotations

import struct

from apollo_mavis_v2_core.errors import VideoFramingError

HEADER_FMT = "<dI"
HEADER_SIZE = 12
assert struct.calcsize(HEADER_FMT) == HEADER_SIZE  # binding wire invariant

RESERVED_STREAM_IDS = ("sim", "twin")  # session-only renders; real-camera ids
#   are live PRE-session at ~15 fps


def pack_frame(ts: float, jpeg: bytes) -> bytes:
    """Build one wire frame: header + payload in a single buffer."""
    return struct.pack(HEADER_FMT, ts, len(jpeg)) + jpeg


def unpack_header(buf: bytes | memoryview) -> tuple[float, int]:
    """Parse the 12-byte header; returns ``(ts, jpeg_len)``.

    Raises :class:`VideoFramingError` if ``buf`` is shorter than the header,
    or — when the full frame is provided — if the declared JPEG length does
    not match the remaining payload bytes.
    """
    n = len(buf)
    if n < HEADER_SIZE:
        raise VideoFramingError(f"frame too short: {n} < {HEADER_SIZE} header bytes")
    ts, jpeg_len = struct.unpack_from(HEADER_FMT, buf)
    if n > HEADER_SIZE and n - HEADER_SIZE != jpeg_len:
        raise VideoFramingError(
            f"payload length mismatch: header says {jpeg_len}, got {n - HEADER_SIZE}"
        )
    return float(ts), int(jpeg_len)


def is_reserved_stream(stream_id: str) -> bool:
    """True for the session-only render streams ("sim", "twin")."""
    return stream_id in RESERVED_STREAM_IDS


__all__ = [
    "HEADER_FMT",
    "HEADER_SIZE",
    "RESERVED_STREAM_IDS",
    "pack_frame",
    "unpack_header",
    "is_reserved_stream",
]
