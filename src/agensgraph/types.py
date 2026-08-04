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

from ._protocol.graphid import GraphId

__all__ = ["Edge", "GraphId", "Label", "Path", "Vertex"]

_decode_json = msgspec.json.decode
_EMPTY: dict[str, Any] = {}


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


class _Element:
    """Shared behaviour of a vertex and an edge."""

    __slots__ = ("_id", "_label", "_props", "_raw")

    _id: GraphId
    _label: str
    _raw: bytes | None
    _props: dict[str, Any] | None

    def __init__(
        self, id: GraphId, label: str, properties: bytes | dict[str, Any] | None
    ) -> None:
        object.__setattr__(self, "_id", id)
        object.__setattr__(self, "_label", label)
        if isinstance(properties, bytes):
            object.__setattr__(self, "_raw", properties)
            object.__setattr__(self, "_props", None)
        else:
            object.__setattr__(self, "_raw", None)
            object.__setattr__(self, "_props", properties if properties is not None else _EMPTY)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable")

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
            assert raw is not None
            props = _decode_json(raw)
            if not isinstance(props, dict):
                raise ValueError(f"property map is not an object: {props!r}")
            object.__setattr__(self, "_props", props)
            object.__setattr__(self, "_raw", None)
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


class Vertex(_Element):
    """A vertex."""

    __slots__ = ()

    def __repr__(self) -> str:
        return f"Vertex({self._label}[{self._id}])"


class Edge(_Element):
    """An edge, carrying the identities of the vertices it connects."""

    __slots__ = ("_end", "_start")

    _start: GraphId
    _end: GraphId

    def __init__(
        self,
        id: GraphId,
        label: str,
        start: GraphId,
        end: GraphId,
        properties: bytes | dict[str, Any] | None,
    ) -> None:
        super().__init__(id, label, properties)
        object.__setattr__(self, "_start", start)
        object.__setattr__(self, "_end", end)

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
