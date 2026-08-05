"""The values a graph query returns.

A vertex or an edge is a value, not a handle: it is immutable, it compares and hashes
on its identity alone, and it can therefore be used as a dict key or a set member.
Comparing two vertices that share an identity but differ in properties reports them
equal, which is what the server does too.

Property maps are decoded on first access rather than at construction. Most callers
read a handful of properties from a result and never touch the rest, and the property
map is where nearly all of the decoding cost lives, so holding the raw slice until
somebody asks is the difference between paying for it and not.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import msgspec
from msgspec import Struct

from ._protocol.graphid import GraphId

__all__ = ["Edge", "GraphId", "Label", "Path", "Vertex"]

_decode_json = msgspec.json.decode
_EMPTY: dict[str, Any] = {}

# Stands in for an identity an edge must be given. An edge always has both, so this is only ever
# what the field defaults to before one is passed.
_NO_ID = GraphId(0, 0)


class Label(str):
    """A label or property name to be placed into a query as an identifier.

    Cypher cannot bind a label or a property key as a parameter, so a query that names
    one dynamically has to build it into the text. Requiring this type rather than a
    plain string at those call sites means the value is always quoted on the way in,
    and an unquoted interpolation cannot be written by accident.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return f"Label({str.__repr__(self)})"


class _ElementBehaviour:
    """What a vertex and an edge both do.

    Not a struct itself, so that each of them keeps the order of its own fields. Mixed in ahead of
    the struct, so these definitions win over the ones a struct generates -- equality in particular,
    which a struct would base on every field.
    """

    __slots__ = ()

    _id: GraphId
    _label: str
    _raw: bytes | dict[str, Any] | None
    _props: dict[str, Any] | None

    def __post_init__(self) -> None:
        # Written through setattr because the names are declared by the struct rather than here.
        raw = self._raw
        if raw is None:
            self._props = _EMPTY  # type: ignore[misc]
        elif type(raw) is dict:
            self._props = raw  # type: ignore[misc]
            self._raw = None  # type: ignore[misc]
        elif type(raw) is not bytes:
            # A property map already turned into the wrong thing -- a string, most often, from
            # asking for the map as text and letting it be decoded on the way in.
            raise TypeError(
                f"properties must be a dict, undecoded bytes or None, not {type(raw).__name__}"
            )

    @property
    def id(self) -> GraphId:
        """The identity of this element."""
        return self._id

    @property
    def label(self) -> str:
        """The label this element was created with."""
        return self._label

    @property
    def properties(self) -> dict[str, Any]:
        """The property map, decoded on first access and kept thereafter."""
        props = self._props
        if props is None:
            raw = self._raw
            assert isinstance(raw, bytes)
            decoded = _decode_json(raw)
            if not isinstance(decoded, dict):
                raise ValueError(f"property map is not an object: {decoded!r}")
            props = decoded
            self._props = props  # type: ignore[misc]
            self._raw = None  # type: ignore[misc]
        return props

    def get(self, key: str, default: Any = None) -> Any:
        """Read one property, without requiring the caller to name the map."""
        return self.properties.get(key, default)

    def __eq__(self, other: object) -> bool:
        if type(other) is type(self):
            return self._id == other._id
        return NotImplemented

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._id))


class Vertex(_ElementBehaviour, Struct, gc=False):
    """A vertex.

    A struct the collector does not track, which is what makes reading a large result cheap: two
    hundred thousand of these are built in 33 milliseconds against 176 for a class with ``__slots__``,
    and the collector then has nothing of ours to walk. Nothing here can take part in a reference
    cycle -- a property map holds only what JSON can -- so going untracked loses nothing.

    The public surface is read-only. The fields behind it are named privately rather than protected
    by a ``__setattr__`` that raises, since routing every write through one is most of what made
    building these expensive.
    """

    _id: GraphId
    _label: str
    _raw: bytes | dict[str, Any] | None = None
    _props: dict[str, Any] | None = None

    def __repr__(self) -> str:
        return f"Vertex({self._label}[{self._id}])"


class Edge(_ElementBehaviour, Struct, gc=False):
    """An edge, carrying the identities of the vertices it connects."""

    _id: GraphId
    _label: str
    _start: GraphId = _NO_ID
    _end: GraphId = _NO_ID
    _raw: bytes | dict[str, Any] | None = None
    _props: dict[str, Any] | None = None

    @property
    def start(self) -> GraphId:
        """The identity of the vertex this edge leaves."""
        return self._start

    @property
    def end(self) -> GraphId:
        """The identity of the vertex this edge enters."""
        return self._end

    def __repr__(self) -> str:
        return f"Edge({self._label}[{self._id}] {self._start}->{self._end})"


class Path(Sequence["Vertex | Edge"]):
    """An alternating run of vertices and edges, starting and ending with a vertex.

    Indexing and iteration walk the elements in the order the server wrote them, so
    ``path[0]`` is the first vertex and ``path[1]`` the first edge. ``len`` counts the
    elements, which means a path of one vertex and no edges has length one and is
    truthy. The number of hops is ``length``.
    """

    __slots__ = ("_edges", "_vertices")

    _vertices: tuple[Vertex, ...]
    _edges: tuple[Edge, ...]

    def __init__(self, vertices: tuple[Vertex, ...], edges: tuple[Edge, ...]) -> None:
        if len(vertices) != len(edges) + 1 and not (not vertices and not edges):
            raise ValueError(
                f"a path needs one more vertex than edges, got {len(vertices)} and {len(edges)}"
            )
        object.__setattr__(self, "_vertices", vertices)
        object.__setattr__(self, "_edges", edges)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Path is immutable")

    @property
    def vertices(self) -> tuple[Vertex, ...]:
        """The vertices, in order."""
        return self._vertices

    @property
    def edges(self) -> tuple[Edge, ...]:
        """The edges, in order."""
        return self._edges

    @property
    def start(self) -> Vertex | None:
        """The first vertex, or ``None`` for an empty path."""
        return self._vertices[0] if self._vertices else None

    @property
    def end(self) -> Vertex | None:
        """The last vertex, or ``None`` for an empty path."""
        return self._vertices[-1] if self._vertices else None

    @property
    def length(self) -> int:
        """The number of hops, which is the number of edges."""
        return len(self._edges)

    @property
    def elements(self) -> tuple[Vertex | Edge, ...]:
        """The vertices and edges interleaved, as the server wrote them."""
        out: list[Vertex | Edge] = []
        for i, vertex in enumerate(self._vertices):
            out.append(vertex)
            if i < len(self._edges):
                out.append(self._edges[i])
        return tuple(out)

    def __len__(self) -> int:
        return len(self._vertices) + len(self._edges)

    def __getitem__(self, index: int) -> Vertex | Edge:  # type: ignore[override]
        return self.elements[index]

    def __iter__(self) -> Iterator[Vertex | Edge]:
        return iter(self.elements)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Path):
            return self._vertices == other._vertices and self._edges == other._edges
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self._vertices, self._edges))

    def __repr__(self) -> str:
        return f"Path({len(self._vertices)} vertices, {len(self._edges)} edges)"
