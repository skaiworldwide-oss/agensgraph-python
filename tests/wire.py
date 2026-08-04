"""Building the binary renderings, so the reader can be tested without a server."""

from __future__ import annotations

import struct

from agensgraph._protocol.composite import JSONB_OID, TID_OID
from agensgraph._protocol.decode import EDGE_OID, GRAPHID_OID, VERTEX_OID

_i32 = struct.Struct(">i").pack
_u64 = struct.Struct(">Q").pack

GRAPHID_ARRAY_OID = 7001
VERTEX_ARRAY_OID = 7011
EDGE_ARRAY_OID = 7021


def graphid(labid: int, locid: int) -> bytes:
    return _u64((labid << 48) | locid)


def jsonb(text: bytes) -> bytes:
    """A jsonb payload is a one-byte format version followed by the JSON text."""
    return b"\x01" + text


def tid(block: int = 0, offset: int = 1) -> bytes:
    return struct.pack(">IH", block, offset)


def record(fields: list[tuple[int, bytes | None]]) -> bytes:
    out = [_i32(len(fields))]
    for oid, data in fields:
        out.append(_i32(oid))
        if data is None:
            out.append(_i32(-1))
        else:
            out.append(_i32(len(data)))
            out.append(data)
    return b"".join(out)


def vertex(labid: int, locid: int, props: bytes) -> bytes:
    """A vertex carries its id, its property map and its tuple id.

    The tuple id is present here and absent from the text rendering, so a reader that
    assumes two columns will misread this.
    """
    return record(
        [
            (GRAPHID_OID, graphid(labid, locid)),
            (JSONB_OID, jsonb(props)),
            (TID_OID, tid()),
        ]
    )


def edge(
    labid: int,
    locid: int,
    start: tuple[int, int],
    end: tuple[int, int],
    props: bytes,
) -> bytes:
    return record(
        [
            (GRAPHID_OID, graphid(labid, locid)),
            (GRAPHID_OID, graphid(*start)),
            (GRAPHID_OID, graphid(*end)),
            (JSONB_OID, jsonb(props)),
            (TID_OID, tid()),
        ]
    )


def array(elem_oid: int, payloads: list[bytes | None]) -> bytes:
    if not payloads:
        return _i32(0) + _i32(0) + _i32(elem_oid)
    out = [_i32(1), _i32(0), _i32(elem_oid), _i32(len(payloads)), _i32(1)]
    for payload in payloads:
        if payload is None:
            out.append(_i32(-1))
        else:
            out.append(_i32(len(payload)))
            out.append(payload)
    return b"".join(out)


def path(vertices: list[bytes], edges: list[bytes]) -> bytes:
    return record(
        [
            (VERTEX_ARRAY_OID, array(VERTEX_OID, list(vertices))),
            (EDGE_ARRAY_OID, array(EDGE_OID, list(edges))),
        ]
    )
