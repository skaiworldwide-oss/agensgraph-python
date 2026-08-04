"""Building graph values from either rendering.

Both entry points produce the same objects, so a caller does not need to know which
rendering it received. Having two independent routes to the same value is also what
makes them checkable against each other: feeding the same value through both must
agree, and a disagreement is a defect in one of them.
"""

from __future__ import annotations

from collections.abc import Callable

from ..types import Edge, Path, Vertex
from . import composite, textfmt
from .graphid import GraphId

__all__ = [
    "EDGE_OID",
    "GRAPHID_OID",
    "GRAPHPATH_OID",
    "VERTEX_OID",
    "edge_from_binary",
    "edge_from_text",
    "path_from_binary",
    "path_from_text",
    "vertex_from_binary",
    "vertex_from_text",
]

GRAPHID_OID = 7002
VERTEX_OID = 7012
EDGE_OID = 7022
GRAPHPATH_OID = 7032

GRAPHID_ARRAY_OID = 7001
VERTEX_ARRAY_OID = 7011
EDGE_ARRAY_OID = 7021
GRAPHPATH_ARRAY_OID = 7031

LabelResolver = Callable[[int], str]


def vertex_from_text(buf: bytes) -> Vertex:
    """Build a vertex from ``label[labid.locid]{properties}``."""
    parts = textfmt.parse_vertex(buf)
    return Vertex(
        GraphId(parts.labid, parts.locid),
        parts.label.decode(),
        parts.properties,
    )


def edge_from_text(buf: bytes) -> Edge:
    """Build an edge from ``label[labid.locid][start,end]{properties}``."""
    p = textfmt.parse_edge(buf)
    return Edge(
        GraphId(p.labid, p.locid),
        p.label.decode(),
        GraphId(p.start_labid, p.start_locid),
        GraphId(p.end_labid, p.end_locid),
        p.properties,
    )


def path_from_text(buf: bytes) -> Path:
    """Build a path from ``[vertex,edge,vertex,...]``.

    An empty rendering is a legal path of no elements. A null slot is rejected here
    rather than silently dropped, because a path with a hole in it is not a path.
    """
    parts = textfmt.split_elements(buf)
    if not parts:
        return Path((), ())
    if len(parts) % 2 == 0:
        raise ValueError(f"a path must have an odd number of elements, got {len(parts)}")
    vertices = []
    edges = []
    for i, part in enumerate(parts):
        if part == textfmt.NULL_ELEMENT:
            raise ValueError(f"null element at position {i} of a path")
        if i % 2 == 0:
            vertices.append(vertex_from_text(part))
        else:
            edges.append(edge_from_text(part))
    return Path(tuple(vertices), tuple(edges))


def elements_from_text(buf: bytes) -> list[Vertex | Edge | None]:
    """Build a vertex or edge array from ``[element,element,...]``.

    Unlike a path, an element array may legitimately hold nulls, which are returned as
    ``None``. Each element is classified by its own shape rather than by position.
    """
    out: list[Vertex | Edge | None] = []
    for part in textfmt.split_elements(buf):
        if part == textfmt.NULL_ELEMENT:
            out.append(None)
        elif _looks_like_edge(part):
            out.append(edge_from_text(part))
        else:
            out.append(vertex_from_text(part))
    return out


def _looks_like_edge(part: bytes) -> bool:
    try:
        textfmt.parse_edge(part)
    except ValueError:
        return False
    return True


def vertex_from_binary(buf: bytes, resolve: LabelResolver) -> Vertex:
    """Build a vertex from its composite encoding.

    The columns are the graphid, the property map and the tuple id. The tuple id is
    present in this rendering and absent from the text one; nothing here needs it.
    """
    fields = composite.decode_record(buf)
    if len(fields) < 2:
        raise ValueError(f"a vertex needs at least 2 columns, got {len(fields)}")
    gid = composite.graphid_of(fields[0])
    props = _jsonb_body(fields[1])
    return Vertex(gid, resolve(gid.labid), props)


def edge_from_binary(buf: bytes, resolve: LabelResolver) -> Edge:
    """Build an edge from its composite encoding.

    The columns are the graphid, the start and end graphids, the property map and the
    tuple id.
    """
    fields = composite.decode_record(buf)
    if len(fields) < 4:
        raise ValueError(f"an edge needs at least 4 columns, got {len(fields)}")
    gid = composite.graphid_of(fields[0])
    start = composite.graphid_of(fields[1])
    end = composite.graphid_of(fields[2])
    props = _jsonb_body(fields[3])
    return Edge(gid, resolve(gid.labid), start, end, props)


def path_from_binary(buf: bytes, resolve: LabelResolver) -> Path:
    """Build a path from its composite encoding of a vertex array and an edge array."""
    fields = composite.decode_record(buf)
    if len(fields) < 2:
        raise ValueError(f"a path needs 2 columns, got {len(fields)}")
    if fields[0].data is None or fields[1].data is None:
        raise ValueError("a path column is null")
    _, vertex_payloads = composite.decode_array(fields[0].data)
    _, edge_payloads = composite.decode_array(fields[1].data)
    vertices = []
    for i, payload in enumerate(vertex_payloads):
        if payload is None:
            raise ValueError(f"null vertex at position {i} of a path")
        vertices.append(vertex_from_binary(payload, resolve))
    edges = []
    for i, payload in enumerate(edge_payloads):
        if payload is None:
            raise ValueError(f"null edge at position {i} of a path")
        edges.append(edge_from_binary(payload, resolve))
    return Path(tuple(vertices), tuple(edges))


def _jsonb_body(field: composite.Field) -> bytes:
    """Strip the format version from a binary jsonb payload, leaving the JSON text."""
    if field.data is None:
        raise ValueError("property map is null")
    if field.oid != composite.JSONB_OID:
        raise ValueError(f"expected jsonb in the property column, got oid {field.oid}")
    data = field.data
    if not data:
        raise ValueError("empty jsonb payload")
    if data[0] != 1:
        raise ValueError(f"unsupported jsonb format version: {data[0]}")
    return data[1:]
