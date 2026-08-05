"""Reading the text rendering of graph values.

The server renders graph values like this::

    vertex      label[labid.locid]{properties}
    edge        label[labid.locid][start,end]{properties}
    path        [vertex,edge,vertex,...]
    element []  [element,element,...]
    empty path  []
    null slot   NULL

A label is written with no escaping, so it may hold any of ``[ ] { } , "``, and a
property map is JSON, so its string values may hold the same characters.

An element taken alone has more than one reading: ``a[3.1]{"v": "[5.5]{}`` is a label
of ``a`` with an unterminated map, and a label of ``a[3.1]{"v": "`` with a map of
``{}``. So an element list is measured from its start -- each element read from a known
position, its end computed, and the character after it required to be a comma or the
end of the list.

A map ends at the first ``}`` that is followed by a comma or the end of the buffer and
that leaves a complete JSON object behind it. The check is ``msgspec`` against ``Raw``,
which validates without building, so a map nobody reads is never decoded.
"""

from __future__ import annotations

import re
from typing import NamedTuple

import msgspec

__all__ = [
    "NULL_ELEMENT",
    "EdgeParts",
    "VertexParts",
    "parse_edge",
    "parse_vertex",
    "split_elements",
]

NULL_ELEMENT = b"NULL"
"""What the server writes where an element is null."""

_ELEM_ID = re.compile(rb"\[(\d+)\.(\d+)\]")

# Validation only: the result is discarded, so nothing is built from the map. What a
# caller reads later comes from decode_json, which honours the exact-numbers setting.
_validate_json = msgspec.json.Decoder(type=msgspec.Raw).decode


class VertexParts(NamedTuple):
    """The pieces of a rendered vertex, with the property map left undecoded."""

    label: bytes
    labid: int
    locid: int
    properties: bytes


class EdgeParts(NamedTuple):
    """The pieces of a rendered edge, with the property map left undecoded."""

    label: bytes
    labid: int
    locid: int
    start_labid: int
    start_locid: int
    end_labid: int
    end_locid: int
    properties: bytes


def _is_json(buf: bytes) -> bool:
    """Whether the whole of *buf* is one complete JSON value."""
    try:
        _validate_json(buf)
    except msgspec.DecodeError:
        return False
    return True


def parse_vertex(buf: bytes) -> VertexParts:
    """Split ``label[labid.locid]{properties}`` into its parts.

    The property map is returned as the raw slice so that a caller which never reads
    it never pays to decode it.
    """
    for m in _ELEM_ID.finditer(buf):
        label = buf[: m.start()]
        rest = buf[m.end() :]
        if not label or not rest.startswith(b"{"):
            continue
        if not _is_json(rest):
            continue
        return VertexParts(label, int(m.group(1)), int(m.group(2)), rest)
    raise ValueError(f"not a vertex: {buf[:80]!r}")


def parse_edge(buf: bytes) -> EdgeParts:
    """Split ``label[labid.locid][start,end]{properties}`` into its parts."""
    for m in _ELEM_ID.finditer(buf):
        label = buf[: m.start()]
        if not label:
            continue
        rest = buf[m.end() :]
        ends = _ENDPOINTS.match(rest)
        if ends is None:
            continue
        props = rest[ends.end() :]
        if not props.startswith(b"{") or not _is_json(props):
            continue
        return EdgeParts(
            label,
            int(m.group(1)),
            int(m.group(2)),
            int(ends.group(1)),
            int(ends.group(2)),
            int(ends.group(3)),
            int(ends.group(4)),
            props,
        )
    raise ValueError(f"not an edge: {buf[:80]!r}")


_ENDPOINTS = re.compile(rb"\[(\d+)\.(\d+),(\d+)\.(\d+)\]")


def split_elements(buf: bytes) -> list[bytes]:
    """Split a path or element array into its elements.

    ``[]`` yields an empty list. A null slot is returned as the literal ``NULL``, for
    the caller to map onto ``None``; it is not an error, because the server emits it.
    """
    if not (buf.startswith(b"[") and buf.endswith(b"]")):
        raise ValueError(f"not an element list: {buf[:80]!r}")
    inner = buf[1:-1]
    if not inner:
        return []

    return _split_exact(inner)


def _split_exact(inner: bytes) -> list[bytes]:
    """Cut by measuring each element in turn.

    An element has a rigid interior: an id group written as ``[digits.digits]``, for an
    edge a second bracketed group of two ids, then a JSON object. Each element is read
    from a known start, its end computed, and the character that follows required to be
    a comma or the end of the list.
    """
    parts: list[bytes] = []
    pos = 0
    end = len(inner)
    while True:
        stop = _scan_element(inner, pos)
        if stop < 0:
            raise ValueError(f"cannot read an element at offset {pos}")
        parts.append(inner[pos:stop])
        if stop == end:
            return parts
        if inner[stop] != 0x2C:  # ,
            raise ValueError(f"expected a comma at offset {stop}")
        pos = stop + 1
        if pos >= end:
            raise ValueError("element list ends with a comma")


def _scan_element(buf: bytes, pos: int) -> int:
    """Measure one element starting at *pos*, returning the offset just past its end.

    Returns -1 when no element can be read there.
    """
    if buf.startswith(NULL_ELEMENT, pos):
        after = pos + len(NULL_ELEMENT)
        if after == len(buf) or buf[after] == 0x2C:
            return after

    search = pos
    while True:
        m = _ELEM_ID.search(buf, search)
        if m is None:
            return -1
        if m.start() == pos:
            # An element always has a label before its id group.
            search = m.start() + 1
            continue
        rest = m.end()
        ends = _ENDPOINTS.match(buf, rest)
        if ends is not None:
            rest = ends.end()
        if rest < len(buf) and buf[rest] == 0x7B:  # {
            stop = _map_end(buf, rest)
            if stop > 0:
                return stop
        search = m.start() + 1


def _map_end(buf: bytes, start: int) -> int:
    """Measure the property map beginning at *start*, returning the offset just past it.

    The map is the last thing in an element, so it closes at a ``}`` followed by a comma
    or the end of the buffer. Each such ``}`` is tried in turn; the first that leaves a
    complete JSON object behind it is the end.

    Returns -1 when no complete object ends where one could.
    """
    end = len(buf)
    pos = start
    while True:
        close = buf.find(b"}", pos)
        if close < 0:
            return -1
        after = close + 1
        if (after == end or buf[after] == 0x2C) and _is_json(buf[start:after]):
            return after
        pos = after
