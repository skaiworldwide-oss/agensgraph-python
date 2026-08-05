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
from typing import TYPE_CHECKING

from psycopg.adapt import Loader
from psycopg.pq import Format

if TYPE_CHECKING:
    from collections.abc import Sequence

    from psycopg.abc import AdaptContext, Buffer
    from psycopg.types import TypeInfo

__all__ = [
    "TYPES",
    "accept",
    "expression_index",
    "generated_column",
    "nearest",
    "parse_vector_text",
]

TYPES = ("vector", "halfvec")
"""The types read here.

``sparsevec`` is left out deliberately: its text form is ``{1:1,3:2}/4`` rather than a list, so it
is a different shape rather than a narrower one, and reading it as if it were the others would
produce a plausible wrong answer.
"""

# Two bytes of dimension, two the type does not use, then that many floats, big-endian.
_HEADER = struct.Struct(">HH")


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
