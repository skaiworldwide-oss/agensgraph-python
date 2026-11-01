"""Where a label's vectors are kept: in each vertex's properties, or in a table of their own.

:class:`VectorStorage` names the label, the vectors' shape and the design, and the connection does
the rest -- :meth:`~agensgraph.Connection.ensure_vectors`, ``write_vectors``, ``vectors_of``,
``delete_vectors``, ``nearest``, ``build_vector_index`` and ``drop_vectors`` take one.

**In the properties**, the default: the vector is a property of the vertex, and it is indexed and
searched through a Cypher cast, ``n.embedding::vector(1536)``, with the statements
:func:`~agensgraph.vector.vector_index` and :func:`~agensgraph.vector.nearest` build. The cast needs
AgensGraph 2.17 -- 2.14 and 2.15 refuse it, and 2.16 drops its dimension without saying so, which
leaves the index unusable -- so below 2.17 this design is refused, and the refusal says to use the
other one. A property given a column of its own, ``InProperties(promoted=True)``, needs 2.18.

**In a table**, opt-in, keeps the vectors in an ordinary table keyed by the vertex,
``(id graphid primary key references <graph>.<label> (id) on delete cascade, <property> vector(N))``,
in a schema of its own, since a graph's schema holds labels only. It works on every release this
driver supports, and it is the faster design wherever more than the nearest neighbours is asked
for. Measured on 22,000 vectors of 1536 dimensions, on 2.14 and 2.18 alike:

=====================================  =============  ==============
                                       in a table      in properties
=====================================  =============  ==============
nearest 10                             2.5 ms          3.6 -- 5.2 ms
nearest 10 among 10 % / 1 % of them    11 / 4 ms       375 -- 930 ms
reading the label without its vectors  3 ms            320 -- 364 ms
loading                                10.6 s          18.7 s
=====================================  =============  ==============

A vertex deleted with ``DETACH DELETE`` takes its row with it, through the key. Dropping the label
or the graph with ``CASCADE`` drops only the key and leaves the rows, whose identities a re-created
label then hands out again -- so :meth:`~agensgraph.Connection.ensure_vectors` refuses such a table
rather than attaching old vectors to new vertices.

Choosing is a value rather than a flag on the call: ``storage="table"``, or ``InTable(schema=...)``
for a table named by the caller. An option belongs to the design it is an option of, so a table's
name cannot be given to the other design at all.
"""

from __future__ import annotations

import enum
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, ClassVar, NamedTuple

from .vector import TYPES, Distance

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .types import Vertex

__all__ = [
    "HNSW",
    "IVFFlat",
    "InProperties",
    "InTable",
    "Storage",
    "StorageState",
    "VectorHit",
    "VectorStorage",
]

MAX_NAME_BYTES = 63
"""The longest identifier the server keeps whole. A longer one is cut, silently, so it is refused."""


class Storage(enum.StrEnum):
    """The two designs, by name, for ``storage="..."``."""

    PROPERTIES = "properties"
    TABLE = "table"


class StorageState(enum.StrEnum):
    """What a look at the database found for a :class:`VectorStorage`."""

    ABSENT = "absent"
    """Something the design needs is not there yet; ``ensure_vectors`` makes it."""

    READY = "ready"
    """Everything the design needs is there, in the shape asked for."""

    STALE = "stale"
    """A table whose key no longer refers to the label: its rows may name other vertices now."""

    MISMATCHED = "mismatched"
    """Vectors already kept there in another type or dimension than the one asked for."""


@dataclass(frozen=True, slots=True, kw_only=True)
class InProperties:
    """The vector as a property of each vertex: the default.

    ``promoted`` gives the property a column of its own, typed as the vector, which needs 2.18 and
    a label made with that column.
    """

    kind: ClassVar[Storage] = Storage.PROPERTIES
    promoted: bool = False

    def resolved(self, graph: str, label: str, property: str) -> InProperties:
        """Nothing to fill in: the property is named by the storage."""
        return self


@dataclass(frozen=True, slots=True, kw_only=True)
class InTable:
    """The vectors in a table of their own, keyed by the vertex.

    ``schema`` defaults to ``<graph>_vectors`` and ``name`` to ``<label>_<property>``. Both are
    identifiers of at most 63 bytes, which is checked rather than left to the server to cut.
    """

    kind: ClassVar[Storage] = Storage.TABLE
    schema: str | None = None
    name: str | None = None

    def resolved(self, graph: str, label: str, property: str) -> InTable:
        """The same table, with the default names filled in and checked."""
        schema = self.schema if self.schema is not None else f"{graph}_vectors"
        name = self.name if self.name is not None else f"{label}_{property}"
        _check_name("schema", schema)
        _check_name("table name", name)
        return InTable(schema=schema, name=name)


@dataclass(frozen=True, slots=True, kw_only=True)
class HNSW:
    """An HNSW index: the default, and the one to use unless memory for building it is short."""

    method: ClassVar[str] = "hnsw"
    m: int = 16
    ef_construction: int = 64

    def options(self) -> dict[str, int]:
        """The index's ``WITH`` settings."""
        return {"m": self.m, "ef_construction": self.ef_construction}


@dataclass(frozen=True, slots=True, kw_only=True)
class IVFFlat:
    """An IVFFlat index, built from the rows already there -- so build it after loading."""

    method: ClassVar[str] = "ivfflat"
    lists: int = 100

    def options(self) -> dict[str, int]:
        """The index's ``WITH`` settings."""
        return {"lists": self.lists}


class VectorHit(NamedTuple):
    """One answer to a nearest-neighbour search, the same from either design."""

    vertex: Vertex
    distance: float
    """By the storage's distance; smaller is nearer, and hits come nearest first."""


_DEFAULT_DESIGN: Mapping[Storage, InProperties | InTable] = {
    Storage.PROPERTIES: InProperties(),
    Storage.TABLE: InTable(),
}

_MAX_DIMENSIONS: Mapping[str, int] = {
    "vector": 16_000,
    "halfvec": 16_000,
    "sparsevec": 1_000_000_000,
}
"""The most dimensions pgvector stores in each type."""

_INDEXABLE_DIMENSIONS: Mapping[tuple[str, str], int] = {
    ("hnsw", "vector"): 2_000,
    ("hnsw", "halfvec"): 4_000,
    ("hnsw", "sparsevec"): 1_000_000_000,
    ("ivfflat", "vector"): 2_000,
    ("ivfflat", "halfvec"): 4_000,
}
"""The most dimensions each index method takes for each type; a pair not here has no index."""

_INDEXED_DISTANCES: Mapping[str, frozenset[Distance]] = {
    "hnsw": frozenset({Distance.L2, Distance.INNER_PRODUCT, Distance.COSINE, Distance.L1}),
    "ivfflat": frozenset({Distance.L2, Distance.INNER_PRODUCT, Distance.COSINE}),
}
"""The distances each index method can order by."""


@dataclass(frozen=True, slots=True)
class VectorStorage:
    """Where one label's vectors are kept, and in what shape.

    ``VectorStorage("g", "doc", dimensions=1536)`` keeps them in the vertices' ``embedding``
    property. ``storage="table"`` keeps them in a table of their own instead, named
    ``g_vectors.doc_embedding`` unless ``InTable(schema=..., name=...)`` says otherwise.

    Checked when made, so a storage that exists can be carried out: the names fit, the dimension
    fits the type, and the index can be built for that type, dimension and distance. After
    construction ``storage`` holds the resolved design -- an :class:`InProperties` or an
    :class:`InTable` with its names filled in -- whatever form it was given in.
    """

    graph: str
    label: str
    _: KW_ONLY
    dimensions: int
    property: str = "embedding"
    type: str = "vector"
    distance: Distance = Distance.COSINE
    index: HNSW | IVFFlat | None = HNSW()
    storage: Storage | str | InProperties | InTable = Storage.PROPERTIES

    def __post_init__(self) -> None:
        _check_name("graph", self.graph)
        _check_name("label", self.label)
        _check_name("property", self.property)
        if self.type not in TYPES:
            raise ValueError(f"expected a type of {TYPES}, got {self.type!r}")
        _check_dimensions(self.type, self.dimensions)
        distance = Distance(self.distance)
        if distance.is_for_bits:
            raise ValueError(
                f"{distance.name} measures bit strings, not vectors; use L2, INNER_PRODUCT, "
                f"COSINE or L1"
            )
        object.__setattr__(self, "distance", distance)
        if self.index is not None:
            _check_index(self.index, self.type, self.dimensions, distance)
        design = _design_of(self.storage).resolved(self.graph, self.label, self.property)
        object.__setattr__(self, "storage", design)

    def design(self) -> InProperties | InTable:
        """The resolved design, typed as one."""
        design = self.storage
        assert isinstance(design, InProperties | InTable)
        return design

    def kind(self) -> Storage:
        """Which of the two designs this is."""
        return self.design().kind


def _design_of(value: object) -> InProperties | InTable:
    """The design a ``storage=`` argument names, whichever form it came in."""
    if isinstance(value, InProperties | InTable):
        return value
    if isinstance(value, str):
        try:
            return _DEFAULT_DESIGN[Storage(value)]
        except ValueError:
            raise ValueError(
                f"storage is one of {[str(kind) for kind in Storage]}, got {value!r}"
            ) from None
    raise TypeError(f"storage is a name, InProperties or InTable, got {type(value).__name__}")


def _check_name(what: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"the {what} is a non-empty string, got {value!r}")
    if len(value.encode()) > MAX_NAME_BYTES:
        raise ValueError(
            f"the {what} {value!r} is longer than {MAX_NAME_BYTES} bytes, and the server would "
            f"cut it; choose a shorter one"
        )


def _check_dimensions(type: str, dimensions: object) -> None:
    if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions < 1:
        raise ValueError(f"dimensions is a whole number of at least one, got {dimensions!r}")
    most = _MAX_DIMENSIONS[type]
    if dimensions > most:
        raise ValueError(f"a {type} holds at most {most} dimensions, got {dimensions}")


def _check_index(index: HNSW | IVFFlat, type: str, dimensions: int, distance: Distance) -> None:
    most = _INDEXABLE_DIMENSIONS.get((index.method, type))
    if most is None:
        raise ValueError(f"pgvector builds no {index.method} index on {type}; use HNSW()")
    if dimensions > most:
        raise ValueError(
            f"a {index.method} index on {type} takes at most {most} dimensions, got "
            f"{dimensions}; keep them as halfvec (up to 4,000), or pass index=None to search "
            f"exactly"
        )
    if distance not in _INDEXED_DISTANCES[index.method]:
        raise ValueError(f"a {index.method} index cannot order by {distance.name} distance")
