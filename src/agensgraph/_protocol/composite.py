"""Reading the binary rendering of graph values.

``vertex``, ``edge`` and ``graphpath`` are composite types, so on the wire they use
PostgreSQL's generic record encoding: a field count, then for each field its type oid,
its length and its payload. A length of -1 marks a null field.

Two things the binary form exposes that the text form hides. Every non-dropped column
is present, so a vertex carries the ``tid`` column that ``vertex_out`` never writes.
And the label *name* is absent, because it is resolved at render time from the session
graph path rather than carried in the value; a binary reader gets the label id inside
the graphid and has to resolve the name itself.
"""

from __future__ import annotations

import struct
from typing import NamedTuple

import msgspec

from .graphid import GraphId
from .graphid import unpack as _unpack_graphid

__all__ = [
    "JSONB_OID",
    "TID_OID",
    "Field",
    "decode_array",
    "decode_jsonb",
    "decode_record",
]

JSONB_OID = 3802
TID_OID = 27

_i32 = struct.Struct(">i")
_unpack_i32_from = _i32.unpack_from
_decode_json = msgspec.json.decode


class Field(NamedTuple):
    """One column of a composite value, with its payload left undecoded."""

    oid: int
    data: bytes | None


def decode_record(buf: bytes) -> list[Field]:
    """Split a composite value into its columns.

    Payloads are returned as slices so that a caller can skip decoding a column it
    does not need.
    """
    if len(buf) < 4:
        raise ValueError("truncated composite header")
    (ncols,) = _unpack_i32_from(buf, 0)
    if ncols < 0:
        raise ValueError(f"negative column count: {ncols}")
    pos = 4
    fields: list[Field] = []
    for _ in range(ncols):
        if pos + 8 > len(buf):
            raise ValueError("truncated composite field header")
        (oid,) = _unpack_i32_from(buf, pos)
        (size,) = _unpack_i32_from(buf, pos + 4)
        pos += 8
        if size == -1:
            fields.append(Field(oid, None))
            continue
        if size < 0:
            raise ValueError(f"negative field length: {size}")
        end = pos + size
        if end > len(buf):
            raise ValueError("field length runs past the end of the value")
        fields.append(Field(oid, buf[pos:end]))
        pos = end
    return fields


def decode_jsonb(buf: bytes) -> object:
    """Decode a binary jsonb payload.

    The payload is a one-byte format version followed by the JSON text, so decoding is
    a version check and then an ordinary JSON parse.
    """
    if not buf:
        raise ValueError("empty jsonb payload")
    version = buf[0]
    if version != 1:
        raise ValueError(f"unsupported jsonb format version: {version}")
    return _decode_json(buf[1:])


def decode_array(buf: bytes) -> tuple[int, list[bytes | None]]:
    """Split a one-dimensional binary array into its element payloads.

    Returns the element type oid alongside the payloads. An empty array reports zero
    dimensions and yields no elements.
    """
    if len(buf) < 12:
        raise ValueError("truncated array header")
    (ndim,) = _unpack_i32_from(buf, 0)
    (elem_oid,) = _unpack_i32_from(buf, 8)
    if ndim == 0:
        return elem_oid, []
    if ndim != 1:
        raise ValueError(f"expected a one-dimensional array, got {ndim} dimensions")
    if len(buf) < 20:
        raise ValueError("truncated array dimension header")
    (length,) = _unpack_i32_from(buf, 12)
    if length < 0:
        raise ValueError(f"negative array length: {length}")
    pos = 20
    out: list[bytes | None] = []
    for _ in range(length):
        if pos + 4 > len(buf):
            raise ValueError("truncated array element header")
        (size,) = _unpack_i32_from(buf, pos)
        pos += 4
        if size == -1:
            out.append(None)
            continue
        if size < 0:
            raise ValueError(f"negative element length: {size}")
        end = pos + size
        if end > len(buf):
            raise ValueError("element length runs past the end of the array")
        out.append(buf[pos:end])
        pos = end
    return elem_oid, out


def graphid_of(field: Field) -> GraphId:
    """Read a graphid column, rejecting a null."""
    if field.data is None:
        raise ValueError("graphid column is null")
    return _unpack_graphid(field.data)
