"""Reading the text rendering of graph values.

The server renders graph values like this::

    vertex      label[labid.locid]{properties}
    edge        label[labid.locid][start,end]{properties}
    path        [vertex,edge,vertex,...]
    element []  [element,element,...]
    empty path  []
    null slot   NULL

Two properties of that format shape everything here. Label names are written with no
escaping at all, so a label may legally contain any of ``[ ] { } , "``. And the
property map is JSON, so its string values may contain those same characters. Neither
the label nor the map can therefore be located by searching for a delimiter alone.

The way through is that a property map has to be decoded anyway, so a decode failure
is a free correctness check on a guessed boundary. Every function here guesses with a
cheap C-level search, validates by decoding, and falls back to a scanner that tracks
brace depth and string state when the guess does not hold. The fallback is counted so
that a workload paying for it constantly is visible rather than merely slow.
"""

from __future__ import annotations

import re
from typing import NamedTuple

import msgspec

from ..numbers import decode_json

__all__ = [
    "NULL_ELEMENT",
    "EdgeParts",
    "VertexParts",
    "fallback_count",
    "parse_edge",
    "parse_vertex",
    "reset_fallback_count",
    "split_elements",
]

NULL_ELEMENT = b"NULL"
"""What the server writes where an element is null."""

_decode_json = decode_json
_ELEM_ID = re.compile(rb"\[(\d+)\.(\d+)\]")

# A boundary between two elements of a path or an element array: the close of one
# property map, a comma, then the start of the next element's label. Labels can
# defeat this, which is why every use is validated by decoding.
_ELEM_BOUNDARY = re.compile(rb"\},(?=(?:[^\[\]{},\"]*\[\d+\.\d+\]|NULL(?:,|$)))")

_fallbacks = 0


def fallback_count() -> int:
    """How many times a guessed boundary had to be re-derived by scanning."""
    return _fallbacks


def reset_fallback_count() -> None:
    """Set the fallback counter back to zero."""
    global _fallbacks
    _fallbacks = 0


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


def _is_decodable(buf: bytes) -> bool:
    try:
        _decode_json(buf)
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
        if not _is_decodable(rest):
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
        if not props.startswith(b"{") or not _is_decodable(props):
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

    parts = _split_fast(inner)
    if parts is not None:
        return parts

    global _fallbacks
    _fallbacks += 1
    return _split_exact(inner)


def _split_fast(inner: bytes) -> list[bytes] | None:
    """Cut on boundaries found by search, keeping the result only if every piece parses.

    Returns ``None`` when the guess does not hold, which sends the caller to the
    scanner rather than accepting a wrong split.
    """
    cuts = [m.start() + 1 for m in _ELEM_BOUNDARY.finditer(inner)]
    if not cuts:
        return [inner] if _element_is_wellformed(inner) else None
    parts: list[bytes] = []
    prev = 0
    for cut in cuts:
        parts.append(inner[prev:cut])
        prev = cut + 1
    parts.append(inner[prev:])
    if all(_element_is_wellformed(p) for p in parts):
        return parts
    return None


def _element_is_wellformed(part: bytes) -> bool:
    if part == NULL_ELEMENT:
        return True
    return _scan_element(part, 0) == len(part)


def _split_exact(inner: bytes) -> list[bytes]:
    """Cut by measuring each element in turn.

    Tracking brace depth across the whole list cannot work, because a label may contain
    a brace and the depth would never return to zero. Nor can it split on commas at
    depth zero, because a label may contain a comma.

    What is reliable is that an element has a rigid interior: an id group written as
    ``[digits.digits]``, for an edge a second bracketed group of two ids, and then a
    JSON object whose extent can be measured exactly. So each element is parsed from a
    known start, its end computed, and the character that follows required to be a
    comma or the end of the list. Alignment is therefore verified at every step rather
    than assumed.
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
            stop = _scan_json_object(buf, rest)
            if stop > 0 and (stop == len(buf) or buf[stop] == 0x2C):
                return stop
        search = m.start() + 1


def _scan_json_object(buf: bytes, start: int) -> int:
    """Measure the JSON object beginning at *start*, returning the offset just past it.

    Inside JSON the syntax is unambiguous, so brace depth is meaningful here even
    though it is not across a whole element list. Quoted strings are skipped and
    backslash escapes honoured, so a brace or a quote inside a string value does not
    move the depth.

    Returns -1 if the object does not close.
    """
    depth = 0
    in_string = False
    escaped = False
    for pos in range(start, len(buf)):
        ch = buf[pos]
        if in_string:
            if escaped:
                escaped = False
            elif ch == 0x5C:  # backslash
                escaped = True
            elif ch == 0x22:  # quote
                in_string = False
            continue
        if ch == 0x22:  # quote
            in_string = True
        elif ch == 0x7B:  # {
            depth += 1
        elif ch == 0x7D:  # }
            depth -= 1
            if depth == 0:
                return pos + 1
            if depth < 0:
                return -1
    return -1
