"""Reading and indexing embedding vectors.

Unlike every other type this driver reads, a vector's oid is **not** fixed. The graph types are
built into the server, so their oids are the same everywhere and are written down; ``vector`` comes
from an extension, so its oid is assigned when the extension is created and differs from database
to database -- measured at 279593 in one and guaranteed to be something else in another. So it is
looked up, per connection, and a map built for one database must never be used against another.

That is also why this is opt-in. Nothing here happens unless :func:`register` is called, and
calling it against a database without the extension reports that rather than failing later on a
type nobody can name.

**Promotion changes what a vector property reads as, and that is the reason this module exists.**
A vector kept in the ordinary property map is JSON, so it arrives as a list of numbers. The same
property given a column of its own arrives as that column's type -- and with no loader for it, as
the *string* ``'[1,2,3,4]'``. Both were measured. So a driver that reads vectors correctly without
promotion silently reads them wrongly with it, which is the promotion warning made concrete.

Two ways to index one, and only one of them is a trap:

* **A generated column carrying the dimension**, ``create vlabel emb (v vector(4) generated)``.
  The dimension lives on the column, so the server refuses a wrong-length value when it is
  written -- verified: four dimensions declared, three offered, *expected 4 dimensions, not 3* --
  and an index binds to a real column.
* **An expression index over a cast that carries the dimension.** ``(properties->>'v')::vector``
  cannot be indexed at all -- *column does not have dimensions* -- while
  ``(properties->>'v')::vector(4)`` can. The dimension in the cast is not decoration; it is the
  whole difference between an index that exists and one that does not.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping
from typing import TYPE_CHECKING

from psycopg.adapt import Dumper, Loader
from psycopg.pq import Format

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from psycopg.abc import AdaptContext, Buffer
    from psycopg.types import TypeInfo

__all__ = [
    "TYPES",
    "SparseVector",
    "accept",
    "expression_index",
    "generated_column",
    "nearest",
    "parse_vector_text",
]

TYPES = ("vector", "halfvec", "sparsevec")
"""The types read here.

The first two are lists of numbers. The third is not, and gets a value of its own -- see
:class:`SparseVector` for why a dense list would have been the wrong answer.
"""

TEXT_OID = 25
"""What a sparse vector is sent as. See :class:`SparseVectorDumper` for why."""

# A dense vector: two bytes of dimension, two the type does not use, then that many floats.
_HEADER = struct.Struct(">HH")

# A sparse one: the dimension, how many entries are not zero, one word the type does not use.
# Then that many indices, then that many values. All big-endian, and the indices are counted
# from zero -- which the text form does not do.
_SPARSE_HEADER = struct.Struct(">iii")


def _elements(data: Buffer, code: str) -> list[float]:
    raw = bytes(data)
    dimensions, _unused = _HEADER.unpack_from(raw, 0)
    return list(struct.unpack_from(f">{dimensions}{code}", raw, _HEADER.size))


def parse_vector_text(text: str | bytes) -> list[float]:
    """Read ``[1,2,3]`` into numbers.

    The whole string has to be a bracketed list, so a value that has been through something which
    quoted or truncated it is refused rather than half-read.
    """
    if isinstance(text, bytes):
        text = text.decode()
    text = text.strip()
    if not text.startswith("[") or not text.endswith("]"):
        raise ValueError(f"not a vector: {text!r}")
    body = text[1:-1].strip()
    if not body:
        return []
    return [float(part) for part in body.split(",")]


class VectorLoader(Loader):
    """Read ``[1,2,3]``."""

    format = Format.TEXT

    def load(self, data: Buffer) -> list[float]:
        return parse_vector_text(bytes(data))


class VectorBinaryLoader(Loader):
    """Read a dimension count and that many single-precision floats."""

    format = Format.BINARY

    def load(self, data: Buffer) -> list[float]:
        return _elements(data, "f")


class HalfVectorBinaryLoader(Loader):
    """The same, in half precision.

    Read back as ordinary floats, because half precision is how the server stores them and not a
    thing Python has -- so the numbers widen, and none of them changes value doing so.
    """

    format = Format.BINARY

    def load(self, data: Buffer) -> list[float]:
        return _elements(data, "e")


def accept(context: AdaptContext, info: TypeInfo) -> None:
    """Read one vector type, whose oid has just been looked up.

    Looking it up is what has to be awaited on an awaiting connection, so that half lives on the
    connection and this half -- which waits for nothing -- lives here in one copy.
    """
    info.register(context)
    adapters = context.adapters
    if info.name == "sparsevec":
        adapters.register_loader(info.oid, SparseVectorLoader)
        adapters.register_loader(info.oid, SparseVectorBinaryLoader)
        adapters.register_dumper(SparseVector, SparseVectorDumper)
        return
    adapters.register_loader(info.oid, VectorLoader)
    adapters.register_loader(
        info.oid,
        HalfVectorBinaryLoader if info.name == "halfvec" else VectorBinaryLoader,
    )


def generated_column(name: str, dimensions: int, *, type: str = "vector") -> str:
    """The column definition that gives a property a column of its own, dimension and all.

    Used as ``create vlabel emb ({generated_column('v', 1024)})``. The dimension has to be here:
    it is what makes the server refuse a wrong-length value at the moment it is written, and what
    an index binds to.

    ``generated`` is not optional -- a promoted column that is not generated is refused outright.
    """
    if dimensions < 1:
        raise ValueError(f"a vector has at least one dimension, got {dimensions}")
    if type not in TYPES:
        raise ValueError(f"expected one of {TYPES}, got {type!r}")
    return f"{name} {type}({dimensions}) generated"


def expression_index(
    graph: str,
    label: str,
    property: str,
    dimensions: int,
    *,
    method: str = "hnsw",
    operator: str = "vector_l2_ops",
) -> str:
    """An index over a vector still living in the property map.

    The dimension in the cast is what makes this work. Without it the server refuses the index
    outright -- *column does not have dimensions* -- so it is required here rather than optional.
    """
    from .cypher import quote_identifier

    if dimensions < 1:
        raise ValueError(f"a vector has at least one dimension, got {dimensions}")
    table = f"{quote_identifier(graph)}.{quote_identifier(label)}"
    cast = f"((properties ->> '{property}')::vector({dimensions}))"
    return f"create index on {table} using {method} ({cast} {operator})"


def nearest(
    graph: str, label: str, column: str, *, limit: int = 10, operator: str = "<->"
) -> str:
    """The statement that finds the closest rows to a given vector.

    Written as SQL against the label's own table, because Cypher has no syntax for a distance
    operator -- and reaching for SQL where Cypher has no words for something is what a graph
    connection is expected to allow.
    """
    from .cypher import quote_identifier

    table = f"{quote_identifier(graph)}.{quote_identifier(label)}"
    name = quote_identifier(column)
    # The parameter is cast, because this driver says a string is text and there is no operator
    # between a vector and text -- which is the same cast a caller writes for a date, met here in
    # the driver's own statement.
    return (
        f"select id, {name} {operator} %s::vector as distance "
        f"from {table} order by distance limit {int(limit)}"
    )


def dimensions_of(values: Sequence[float]) -> int:
    """How many dimensions a value has, for building a column or a cast to match it."""
    return len(values)


class SparseVector:
    """A vector most of whose entries are zero, holding only the ones that are not.

    A dense list would have been the obvious reading and is the wrong one. Measured: three
    non-zero entries in a million dimensions is 36 bytes on the wire and roughly 8 MiB as a list
    of Python floats -- so reading it densely expands it more than two hundred thousand fold and
    throws away the whole reason the type exists. :meth:`to_dense` is there for the caller who
    means it.

    **Indices count from zero here.** The server's text form counts from one -- ``{1:9}/3`` is the
    first entry of three -- while its binary form counts from zero. Two renderings disagreeing
    about the base, read into one representation without noticing, give an answer that is wrong by
    one rather than an error; so the conversion happens once, at the text boundary, and everything
    in Python is zero-based like the language it is in. ``to_dense()[i]`` and ``indices`` agree.

    What the server refuses is refused here: a repeated index, an index outside the dimension, a
    dimension below one, and a value that is not finite. Failing before the round trip is the only
    place a caller learns about it in time to do anything.

    An entry that is explicitly zero is dropped, because the server drops it -- ``{1:0,2:5}/4``
    comes back as ``{2:5}/4``. Keeping it would make a round trip lossy in one direction only.
    """

    __slots__ = ("_dimensions", "_indices", "_values")

    _dimensions: int
    _indices: tuple[int, ...]
    _values: tuple[float, ...]

    def __init__(
        self, entries: Mapping[int, float] | Iterable[tuple[int, float]], dimensions: int
    ) -> None:
        if dimensions < 1:
            raise ValueError(f"a sparse vector has at least one dimension, got {dimensions}")
        pairs = entries.items() if isinstance(entries, Mapping) else entries
        kept: dict[int, float] = {}
        for index, value in pairs:
            index = int(index)
            if not 0 <= index < dimensions:
                raise ValueError(
                    f"index {index} is outside a vector of {dimensions} dimensions "
                    f"(indices count from zero here; the server's text form counts from one)"
                )
            if index in kept:
                raise ValueError(f"index {index} appears more than once")
            value = float(value)
            if value != value or value in (float("inf"), float("-inf")):
                raise ValueError(f"a sparse vector holds no infinity and no nan, got {value}")
            if value != 0.0:
                kept[index] = value
        ordered = sorted(kept)
        object.__setattr__(self, "_dimensions", dimensions)
        object.__setattr__(self, "_indices", tuple(ordered))
        object.__setattr__(self, "_values", tuple(kept[index] for index in ordered))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("SparseVector is immutable")

    @property
    def dimensions(self) -> int:
        """How long the vector is, counting the zeros it does not store."""
        return self._dimensions

    @property
    def indices(self) -> tuple[int, ...]:
        """Where the entries that are not zero sit, counted from zero, ascending."""
        return self._indices

    @property
    def values(self) -> tuple[float, ...]:
        """Those entries, in the same order as :attr:`indices`."""
        return self._values

    def to_dict(self) -> dict[int, float]:
        """The entries as a mapping, for a caller who would rather look one up than scan."""
        return dict(zip(self._indices, self._values, strict=True))

    def to_dense(self) -> list[float]:
        """Every entry, zeros and all.

        Costs one float per dimension, which for the values this type is meant for is the
        expansion it exists to avoid -- so this is asked for rather than done by default.
        """
        dense = [0.0] * self._dimensions
        for index, value in zip(self._indices, self._values, strict=True):
            dense[index] = value
        return dense

    @classmethod
    def from_dense(cls, values: Sequence[float]) -> SparseVector:
        """Keep only what is not zero out of an ordinary list."""
        return cls(
            {index: value for index, value in enumerate(values) if value != 0.0}, len(values)
        )

    def to_text(self) -> str:
        """The server's text form, whose indices count from one."""
        body = ",".join(
            f"{index + 1}:{value:g}"
            for index, value in zip(self._indices, self._values, strict=True)
        )
        return f"{{{body}}}/{self._dimensions}"

    @classmethod
    def from_text(cls, text: str | bytes) -> SparseVector:
        """Read ``{1:1,3:2}/4``, whose indices count from one.

        This is the only place the two bases meet, so it is the only place that can get the
        conversion wrong -- which is why it is one function and not a rule to remember.
        """
        if isinstance(text, bytes):
            text = text.decode()
        text = text.strip()
        brace = text.rfind("}")
        if not text.startswith("{") or brace < 0:
            raise ValueError(f"not a sparse vector: {text!r}")
        tail = text[brace + 1 :].strip()
        if not tail.startswith("/"):
            raise ValueError(f"a sparse vector needs its dimension after a slash: {text!r}")
        dimensions = int(tail[1:])
        body = text[1:brace].strip()
        entries: list[tuple[int, float]] = []
        if body:
            for part in body.split(","):
                index, _, value = part.partition(":")
                if not _:
                    raise ValueError(f"not an index and a value: {part!r}")
                # From one on the wire's text form, to zero here.
                entries.append((int(index.strip()) - 1, float(value.strip())))
        return cls(entries, dimensions)

    def __len__(self) -> int:
        """How many entries are stored, which is not the number of dimensions."""
        return len(self._indices)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SparseVector):
            return NotImplemented
        return (
            self._dimensions == other._dimensions
            and self._indices == other._indices
            and self._values == other._values
        )

    def __hash__(self) -> int:
        return hash((self._dimensions, self._indices, self._values))

    def __repr__(self) -> str:
        return f"SparseVector({self.to_dict()}, {self._dimensions})"


class SparseVectorLoader(Loader):
    """Read ``{1:1,3:2}/4``."""

    format = Format.TEXT

    def load(self, data: Buffer) -> SparseVector:
        return SparseVector.from_text(bytes(data))


class SparseVectorBinaryLoader(Loader):
    """Read the dimension, the count, then that many indices and that many values.

    The indices arrive counted from zero, which is what this driver uses, so the path that matters
    for a large value adjusts nothing.
    """

    format = Format.BINARY

    def load(self, data: Buffer) -> SparseVector:
        raw = bytes(data)
        dimensions, stored, _unused = _SPARSE_HEADER.unpack_from(raw, 0)
        at = _SPARSE_HEADER.size
        indices = struct.unpack_from(f">{stored}i", raw, at)
        values = struct.unpack_from(f">{stored}f", raw, at + 4 * stored)
        return SparseVector(zip(indices, values, strict=True), dimensions)


class SparseVectorDumper(Dumper):
    """Write one, which is the only reasonable way to write one at all.

    A Cypher list cannot go into a sparse column -- the server refuses it, saying the contents must
    start with a brace -- where the same list is accepted by a dense one. So sending the text form
    is not a convenience here; it is the way in.

    Sent as **text**, which is the same choice this driver makes for a string and for the same
    reason. Left untyped it would be worked out as jsonb in a Cypher property position and parsed
    as JSON, which it is not. Declared text it goes both ways: a SQL position casts it explicitly,
    and a property position converts it to a JSON string that the generated column then casts --
    which is exactly what a hand-written literal does.
    """

    format = Format.TEXT
    oid = TEXT_OID

    def dump(self, obj: SparseVector) -> bytes:
        return obj.to_text().encode()
