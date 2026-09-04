"""Video WS framing tests (design doc 01-core §13, §18)."""

from __future__ import annotations

import struct

import pytest

from apollo_mavis_v2_core.errors import VideoFramingError
from apollo_mavis_v2_core.protocol.video import (
    HEADER_FMT,
    HEADER_SIZE,
    RESERVED_STREAM_IDS,
    is_reserved_stream,
    pack_frame,
    unpack_header,
)


def test_header_format_is_twelve_bytes():
    assert HEADER_FMT == "<dI"
    assert struct.calcsize(HEADER_FMT) == 12
    assert HEADER_SIZE == 12


@pytest.mark.parametrize("jpeg", [b"", b"\xff\xd8\xff\xe0jpegdata\xff\xd9", b"x" * 4096])
def test_pack_unpack_round_trip(jpeg: bytes):
    ts = 1234.5625  # exactly representable in f64
    frame = pack_frame(ts, jpeg)
    assert len(frame) == HEADER_SIZE + len(jpeg)
    got_ts, got_len = unpack_header(frame)
    assert got_ts == ts
    assert got_len == len(jpeg)
    assert frame[HEADER_SIZE:] == jpeg
    # Header-only parse (streaming reader) works and agrees.
    assert unpack_header(frame[:HEADER_SIZE]) == (ts, len(jpeg))
    # memoryview accepted.
    assert unpack_header(memoryview(frame)) == (ts, len(jpeg))


@pytest.mark.parametrize("n", [0, 1, 11])
def test_truncated_header_raises(n: int):
    with pytest.raises(VideoFramingError):
        unpack_header(b"\x00" * n)


@pytest.mark.parametrize("delta", [-1, +1, +100])
def test_length_mismatch_raises(delta: int):
    jpeg = b"j" * 64
    frame = struct.pack(HEADER_FMT, 1.0, len(jpeg) + delta) + jpeg
    with pytest.raises(VideoFramingError):
        unpack_header(frame)


def test_reserved_stream_predicate():
    assert RESERVED_STREAM_IDS == ("sim", "twin")
    assert is_reserved_stream("sim")
    assert is_reserved_stream("twin")
    assert not is_reserved_stream("cam0")
    assert not is_reserved_stream("Sim")
    assert not is_reserved_stream("")
