"""Building graph values from either rendering.

Both entry points produce the same objects, so a caller does not need to know which
rendering it received. Having two independent routes to the same value is also what
makes them checkable against each other: feeding the same value through both must
agree, and a disagreement is a defect in one of them.
"""

from __future__ import annotations

import struct
from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar

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
    "edges_from_binary",
    "edges_from_text",
    "path_from_binary",
    "path_from_text",
    "vertex_from_binary",
    "vertex_from_text",
    "vertices_from_binary",
    "vertices_from_text",
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

LABEL_NAMES_MAX = 4096
"""How many distinct label names are remembered before the table starts again."""


class _LabelNames(dict[bytes, str]):
    """Label names by the bytes they arrive as.

    A result holds many rows and few distinct labels, so decoding the same name once per
    element is work with one answer. This says nothing about which label an id belongs to
    -- it is the bytes themselves that are the key -- so it holds for every connection and
    every graph, and cannot go stale the way resolving an id can.
    """

    def __missing__(self, key: bytes) -> str:
        if len(self) >= LABEL_NAMES_MAX:
            self.clear()
        name = key.decode()
        self[key] = name
        return name


_label_names = _LabelNames()


class LabelNames(Protocol):
    """What a connection knows the label ids of the graph it is reading to be called.

    The label table. A table holding nothing is not asked.
    """

    def get(self, labid: int) -> str | None: ...

    def __len__(self) -> int: ...


class _Element(Protocol):
    """The two fields every rendered element carries, whichever kind it is."""

    @property
    def label(self) -> bytes: ...

    @property
    def labid(self) -> int: ...


_Parts = TypeVar("_Parts", bound=_Element)
_Value = TypeVar("_Value")


def _vertex(parts: textfmt.VertexParts) -> Vertex:
    return Vertex(
        GraphId(parts.labid, parts.locid),
        _label_names[parts.label],
        parts.properties,
    )


def _edge(p: textfmt.EdgeParts) -> Edge:
    return Edge(
        GraphId(p.labid, p.locid),
        _label_names[p.label],
        GraphId(p.start_labid, p.start_locid),
        GraphId(p.end_labid, p.end_locid),
        p.properties,
    )


def vertex_from_text(buf: bytes) -> Vertex:
    """Build a vertex from ``label[labid.locid]{properties}``."""
    return _vertex(textfmt.parse_vertex(buf))


def edge_from_text(buf: bytes) -> Edge:
    """Build an edge from ``label[labid.locid][start,end]{properties}``."""
    return _edge(textfmt.parse_edge(buf))


def _accounted(found: Sequence[_Element | None], names: LabelNames) -> bool:
    """Whether the table holds every label of a reading, under the name the reading gives it."""
    return all(
        names.get(item.labid) == _label_names[item.label] for item in found if item is not None
    )


def _prefers(
    parse: Callable[[bytes], _Parts], names: LabelNames
) -> Callable[[int, bytes], bool]:
    """A test for one element: it parses, and the table agrees what its label is called."""

    def prefer(index: int, element: bytes) -> bool:
        try:
            parts = parse(element)
        except ValueError:
            return False
        return names.get(parts.labid) == _label_names[parts.label]

    return prefer


def _prefers_in_a_path(names: LabelNames) -> Callable[[int, bytes], bool]:
    """The same, with the kind of each element known from where it sits in the path."""
    vertex = _prefers(textfmt.parse_vertex, names)
    edge = _prefers(textfmt.parse_edge, names)

    def prefer(index: int, element: bytes) -> bool:
        return (vertex if index % 2 == 0 else edge)(index, element)

    return prefer


def _parsed(parts: list[bytes], parse: Callable[[bytes], _Parts]) -> list[_Parts | None] | None:
    """Every element parsed, or ``None`` where one of them is not an element of this kind."""
    found: list[_Parts | None] = []
    for part in parts:
        if part == textfmt.NULL_ELEMENT:
            found.append(None)
            continue
        try:
            found.append(parse(part))
        except ValueError:
            return None
    return found


def _array_from_text(
    buf: bytes,
    names: LabelNames | None,
    parse: Callable[[bytes], _Parts],
    build: Callable[[_Parts], _Value],
) -> list[_Value | None]:
    """An element array, read the way the label table accounts for.

    The shortest reading first. Where the table does not account for it, the list is read
    again element by element, preferring the labels the table holds.
    """
    parts = textfmt.split_elements(buf)
    found = _parsed(parts, parse)
    if names and (found is None or not _accounted(found, names)):
        other = _parsed(textfmt.split_elements(buf, _prefers(parse, names)), parse)
        if other is not None and _accounted(other, names):
            found = other
    if found is None:
        # Parsed again, to fail on the element that cannot be read.
        found = [None if part == textfmt.NULL_ELEMENT else parse(part) for part in parts]
    return [None if item is None else build(item) for item in found]


def _parsed_path(
    parts: list[bytes],
) -> tuple[list[textfmt.VertexParts], list[textfmt.EdgeParts]] | None:
    """The vertices and the edges of a path, or ``None`` where this reading is not one."""
    if not parts or len(parts) % 2 == 0 or textfmt.NULL_ELEMENT in parts:
        return None
    try:
        return (
            [textfmt.parse_vertex(part) for part in parts[0::2]],
            [textfmt.parse_edge(part) for part in parts[1::2]],
        )
    except ValueError:
        return None


def _path_of(parts: list[bytes]) -> Path:
    """A path built from one reading, refusing what a path cannot hold."""
    if not parts:
        return Path((), ())
    if len(parts) % 2 == 0:
        raise ValueError(f"a path must have an odd number of elements, got {len(parts)}")
    for i, part in enumerate(parts):
        if part == textfmt.NULL_ELEMENT:
            raise ValueError(f"null element at position {i} of a path")
    return Path(
        tuple(vertex_from_text(part) for part in parts[0::2]),
        tuple(edge_from_text(part) for part in parts[1::2]),
    )


def path_from_text(buf: bytes, names: LabelNames | None = None) -> Path:
    """Build a path from ``[vertex,edge,vertex,...]``.

    An empty rendering is a legal path of no elements. A null slot is rejected here
    rather than silently dropped, because a path with a hole in it is not a path.

    Read the way an array is, with the kind of each element known from where it sits: a
    vertex at an even position and an edge at an odd one.
    """
    parts = textfmt.split_elements(buf)
    read = _parsed_path(parts)
    if names and (
        read is None or not (_accounted(read[0], names) and _accounted(read[1], names))
    ):
        other = _parsed_path(textfmt.split_elements(buf, _prefers_in_a_path(names)))
        if other is not None and _accounted(other[0], names) and _accounted(other[1], names):
            read = other
    if read is None:
        return _path_of(parts)
    return Path(tuple(_vertex(v) for v in read[0]), tuple(_edge(e) for e in read[1]))


def vertices_from_text(buf: bytes, names: LabelNames | None = None) -> list[Vertex | None]:
    """Build a vertex array from ``[vertex,vertex,...]``.

    An array carries one kind of element and its type says which, so nothing here has to
    work that out from an element's shape. Unlike a path, an array may hold nulls, which
    are returned as ``None``.

    *names* is the label table the connection filled. Where a label holds an element
    rendering of its own the same bytes read as two elements or as four, and the table says
    which labels exist. Without one the shortest reading is taken.
    """
    return _array_from_text(buf, names, textfmt.parse_vertex, _vertex)


def edges_from_text(buf: bytes, names: LabelNames | None = None) -> list[Edge | None]:
    """Build an edge array from ``[edge,edge,...]``, read as :func:`vertices_from_text` is."""
    return _array_from_text(buf, names, textfmt.parse_edge, _edge)


# A vertex is always (graphid, jsonb, tid) and an edge always (graphid, graphid, graphid,
# jsonb, tid): the arity belongs to the type and not to the label. Read from the engine --
# `Natts_ag_vertex` is 3 and `Natts_ag_edge` is 5, and `makeGraphVertexDatum` asserts the
# descriptor has that many -- and confirmed against a label carrying three promoted columns,
# whose vertex still arrives as three columns with oids 7002, 3802 and 27.
#
# So everything up to the property map is one fixed layout, read in a single unpack instead
# of a loop that builds a tuple and a slice per column and then does it once more for the
# tuple id it throws away.
_VERTEX_HEAD = struct.Struct(">iiiQii")
_EDGE_HEAD = struct.Struct(">iiiQiiQiiQii")
_TID_COLUMN = 8 + 6
"""What follows the property map: the tuple id's own header and its six bytes."""


def _jsonb_at(buf: bytes, start: int, size: int) -> bytes:
    """The JSON text of a jsonb payload known to begin at *start*."""
    if size < 1 or start + size > len(buf):
        raise ValueError("property map runs past the end of the value")
    if buf[start] != 1:
        raise ValueError(f"unsupported jsonb format version: {buf[start]}")
    return buf[start + 1 : start + size]


def vertex_from_binary(buf: bytes, resolve: LabelResolver) -> Vertex:
    """Build a vertex from its composite encoding.

    The columns are the graphid, the property map and the tuple id. The tuple id is
    present in this rendering and absent from the text one; nothing here needs it.

    A value that is not the expected shape falls back to reading it a column at a time, so
    a malformed one still raises rather than being read past.
    """
    try:
        ncols, id_oid, id_size, packed, prop_oid, prop_size = _VERTEX_HEAD.unpack_from(buf, 0)
    except struct.error:
        return _vertex_column_by_column(buf, resolve)
    if (
        ncols != 3
        or id_oid != GRAPHID_OID
        or id_size != 8
        or prop_oid != composite.JSONB_OID
        or len(buf) != _VERTEX_HEAD.size + prop_size + _TID_COLUMN
    ):
        return _vertex_column_by_column(buf, resolve)
    gid = GraphId.from_packed(packed)
    return Vertex(gid, resolve(gid.labid), _jsonb_at(buf, _VERTEX_HEAD.size, prop_size))


def _vertex_column_by_column(buf: bytes, resolve: LabelResolver) -> Vertex:
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
    try:
        (
            ncols,
            id_oid,
            id_size,
            packed,
            start_oid,
            start_size,
            start_packed,
            end_oid,
            end_size,
            end_packed,
            prop_oid,
            prop_size,
        ) = _EDGE_HEAD.unpack_from(buf, 0)
    except struct.error:
        return _edge_column_by_column(buf, resolve)
    if (
        ncols != 5
        or id_oid != GRAPHID_OID
        or start_oid != GRAPHID_OID
        or end_oid != GRAPHID_OID
        or id_size != 8
        or start_size != 8
        or end_size != 8
        or prop_oid != composite.JSONB_OID
        or len(buf) != _EDGE_HEAD.size + prop_size + _TID_COLUMN
    ):
        return _edge_column_by_column(buf, resolve)
    gid = GraphId.from_packed(packed)
    return Edge(
        gid,
        resolve(gid.labid),
        GraphId.from_packed(start_packed),
        GraphId.from_packed(end_packed),
        _jsonb_at(buf, _EDGE_HEAD.size, prop_size),
    )


def _edge_column_by_column(buf: bytes, resolve: LabelResolver) -> Edge:
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


def vertices_from_binary(buf: bytes, resolve: LabelResolver) -> list[Vertex | None]:
    """Build a vertex array from its array encoding."""
    _, payloads = composite.decode_array(buf)
    return [None if p is None else vertex_from_binary(p, resolve) for p in payloads]


def edges_from_binary(buf: bytes, resolve: LabelResolver) -> list[Edge | None]:
    """Build an edge array from its array encoding."""
    _, payloads = composite.decode_array(buf)
    return [None if p is None else edge_from_binary(p, resolve) for p in payloads]


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
