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

A searchable vector is held one of two ways, and both are indexed by a property index:

* **In the property map**, cast where it is used.
  ``create property index on movie using hnsw ((embedding::vector(4)) vector_cosine_ops)``,
  searched by ``order by m.embedding::vector(4) <=> %s::vector(4)``.
* **In a column of its own**, ``create vlabel emb (v vector(4) generated)``, which needs no cast and
  refuses a wrong-length value when it is written.

``CREATE PROPERTY INDEX`` compiles its expression through the Cypher parser, so the index holds the
expression a query builds -- ``((properties.'embedding'::text)::vector(4))``. An index over the SQL
spelling, ``(properties ->> 'embedding')::vector(4)``, holds a different expression and is never
matched by a Cypher query.

Three things have to agree for the index to be used. The cast in the search carries the same
dimension as the cast in the index; a bare ``vector`` or another dimension sorts a sequential scan
instead. The operator class serves the operator being searched with: ``vector_cosine_ops`` answers
``<=>`` alone. :func:`vector_index` and :func:`nearest` spell the cast with one function.
"""

from __future__ import annotations

import enum
import struct
import sys
from array import array
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, overload

from psycopg.adapt import Dumper, Loader
from psycopg.pq import Format

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from psycopg.abc import AdaptContext, Buffer
    from psycopg.types import TypeInfo

__all__ = [
    "SEARCH_OPTIONS",
    "TYPES",
    "Distance",
    "SparseVector",
    "Vector",
    "accept",
    "dense_dumper",
    "generated_column",
    "nearest",
    "parse_vector_text",
    "search_option_statements",
    "vector_index",
]

TYPES = ("vector", "halfvec", "sparsevec")
"""The types read here.

The first two are lists of numbers. The third is not, and gets a value of its own -- see
:class:`SparseVector` for why a dense list would have been the wrong answer.
"""

TEXT_OID = 25
"""What a sparse vector is sent as. See :class:`SparseVectorDumper` for why."""

# Narrowing a parsed decimal back to the width the server actually stores.
#
# The two renderings disagree without this, and the disagreement is easy to miss. The server
# holds single precision and prints the shortest decimal that reproduces it; parsing that decimal
# gives a double, and that double is *not* the one the single-precision value widens to. So the
# binary reading and the text reading of one value came back as different Python floats --
# measured, 390 of 400 random vectors -- while a test using small whole numbers saw them agree.
# Narrowing is done to a whole list at once, through an array of that width, because an array
# converts in C. Doing it a value at a time through struct measured 480 microseconds for 1536
# values against 316 -- and 336 of those microseconds were the narrowing alone, ten times what
# decoding the same vector from its binary form costs.
_F32 = struct.Struct(">f")
_F16 = struct.Struct(">e")


def _narrow_all(values: list[float], *, half: bool = False) -> list[float]:
    """The same numbers at the width the server keeps them.

    Single precision goes through an array, which converts the whole list in C. Half precision
    cannot: the array module has no half typecode, so those go one at a time -- which is slower
    and matters less, since half precision is chosen to make a vector smaller and the vectors
    are correspondingly cheaper to walk.
    """
    if half:
        return [float(_F16.unpack(_F16.pack(value))[0]) for value in values]
    return array("f", values).tolist()


def _narrow(value: float, width: struct.Struct) -> float:
    """One value, for the places that have only one."""
    return float(width.unpack(width.pack(value))[0])


# A dense vector: two bytes of dimension, two the type does not use, then that many floats.
_HEADER = _DENSE_HEADER = struct.Struct(">HH")

# A sparse one: the dimension, how many entries are not zero, one word the type does not use.
# Then that many indices, then that many values. All big-endian, and the indices are counted
# from zero -- which the text form does not do.
_SPARSE_HEADER = struct.Struct(">iii")


def _elements(data: Buffer, code: str) -> list[float]:
    raw = bytes(data)
    dimensions, _unused = _HEADER.unpack_from(raw, 0)
    return list(struct.unpack_from(f">{dimensions}{code}", raw, _HEADER.size))


def parse_vector_text(text: str | bytes, *, half: bool = False) -> list[float]:
    """Read ``[1,2,3]`` into numbers of the width the server keeps them at.

    The whole string has to be a bracketed list, so a value that has been through something which
    quoted or truncated it is refused rather than half-read.

    Each number is narrowed to single precision -- or half, for the half-precision type -- because
    that is what the server stores, and because not doing it makes this disagree with the binary
    reading of the same value. The decimal the server prints is the shortest one that reproduces
    its single-precision value; read as a double it is a different number.
    """
    if isinstance(text, bytes):
        text = text.decode()
    text = text.strip()
    if not text.startswith("[") or not text.endswith("]"):
        raise ValueError(f"not a vector: {text!r}")
    body = text[1:-1].strip()
    if not body:
        return []
    return _narrow_all([float(part) for part in body.split(",")], half=half)


class VectorLoader(Loader):
    """Read ``[1,2,3]``, without parsing it yet."""

    format = Format.TEXT

    def load(self, data: Buffer) -> Vector:
        return Vector.from_wire_text(bytes(data))


class HalfVectorLoader(Loader):
    """The same at half precision, which is what that type keeps."""

    format = Format.TEXT

    def load(self, data: Buffer) -> Vector:
        return Vector.from_wire_text(bytes(data), half=True)


class VectorBinaryLoader(Loader):
    """Keep the bytes; the numbers are unpacked only if somebody reads them."""

    format = Format.BINARY

    def load(self, data: Buffer) -> Vector:
        return Vector.from_wire(bytes(data))


class HalfVectorBinaryLoader(Loader):
    """The same at half precision.

    Half is not a width Python has, so the numbers widen to ordinary floats on the way in, and
    none of them changes value doing so. Unpacked eagerly, because an array cannot hold halves and
    so the saving that makes laziness worth it is not available here.
    """

    format = Format.BINARY

    def load(self, data: Buffer) -> Vector:
        return Vector(_elements(data, "e"), half=True)


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
    half = info.name == "halfvec"
    adapters.register_loader(info.oid, HalfVectorLoader if half else VectorLoader)
    adapters.register_loader(info.oid, HalfVectorBinaryLoader if half else VectorBinaryLoader)
    if not half:
        # Sending is where the cost is, so the one type most things send gets a dumper that
        # packs rather than formats. Built here because the oid is only known now.
        adapters.register_dumper(Vector, dense_dumper(info.oid))


class Distance(enum.StrEnum):
    """The operators pgvector measures distance with, by name rather than by symbol.

    Every one of them is two or three punctuation characters, and two of them differ by a single
    character while meaning entirely different things -- so a name is worth having. All four apply
    to each of the three vector types; the last two apply only to a bit string.
    """

    L2 = "<->"
    """Euclidean distance. The usual choice, and what an unqualified index assumes."""

    INNER_PRODUCT = "<#>"
    """*Negative* inner product, so that smaller is nearer as it is for the others."""

    COSINE = "<=>"
    """Cosine distance, which is one minus cosine similarity."""

    L1 = "<+>"
    """Taxicab distance."""

    HAMMING = "<~>"
    """How many bits differ. For a bit string, as :func:`binary_quantize` produces."""

    JACCARD = "<%>"
    """How much two bit strings fail to overlap. For a bit string."""

    @property
    def operator_class(self) -> str:
        """The operator class an index on a ``vector`` column needs to answer this."""
        return _OPERATOR_CLASS[self]

    @property
    def is_for_bits(self) -> bool:
        """Whether this measures bit strings rather than vectors."""
        return self in (Distance.HAMMING, Distance.JACCARD)


_OPERATOR_CLASS = {
    Distance.L2: "vector_l2_ops",
    Distance.INNER_PRODUCT: "vector_ip_ops",
    Distance.COSINE: "vector_cosine_ops",
    Distance.L1: "vector_l1_ops",
    Distance.HAMMING: "bit_hamming_ops",
    Distance.JACCARD: "bit_jaccard_ops",
}


class Vector:
    """A dense vector, read or sent, whose numbers are unpacked only if somebody looks.

    Reading one costs almost nothing until its numbers are wanted. The wire bytes are kept and
    turned into numbers on first access, which is the same bargain the driver already makes with a
    property map -- and it pays here for the same reason: a vector search asks the *server* for the
    distance, so the components of the vectors it ranked are frequently never read at all.

    Measured, one embedding of 1536 dimensions:

    ==========================================  ========
    a list of Python floats, as this once did      33.8 µs
    an array of singles                             1.74 µs
    **this, untouched**                             **0.49 µs**
    this, with its numbers read                     2.67 µs
    ==========================================  ========

    So sixty-nine times cheaper to read and ignore, and thirteen times cheaper to read and use.
    It also holds four bytes a number rather than eight plus a pointer plus an object header.

    It behaves like the sequence it is: indexing, slicing, iteration, ``len``, ``in``, ``index``,
    ``count``, ``sum``, and equality against any sequence -- so ``vector == [1.0, 2.0]`` is true
    when the numbers match, which a bare array would have quietly called false. :attr:`values` is
    an array that numpy and torch take without copying; :meth:`tolist` gives an ordinary list.

    Sending one is where the saving is largest, because every other way of sending a vector formats
    each number as decimal text. Measured on the same embedding: a list with a cast costs 2.55 ms,
    a hand-built string 0.79 ms, and this 0.32 ms -- and in bulk, 31,000 rows a second against
    2,000 one statement at a time. The wire carries 6,148 bytes rather than 17,595.
    """

    __slots__ = ("_half", "_raw", "_text", "_values")

    _values: array[float] | None
    _raw: bytes | None
    _text: bytes | None
    _half: bool

    def __init__(self, values: Iterable[float] | None = None, *, half: bool = False) -> None:
        object.__setattr__(self, "_half", half)
        object.__setattr__(self, "_raw", None)
        object.__setattr__(self, "_text", None)
        object.__setattr__(self, "_values", None if values is None else array("f", values))

    @classmethod
    def from_wire(cls, raw: bytes, *, half: bool = False) -> Vector:
        """Keep the bytes a result arrived in, and unpack them only if asked."""
        self = cls(half=half)
        object.__setattr__(self, "_raw", raw)
        return self

    @classmethod
    def from_wire_text(cls, text: bytes, *, half: bool = False) -> Vector:
        """The same for the text rendering, whose parsing is the expensive part."""
        self = cls(half=half)
        object.__setattr__(self, "_text", text)
        return self

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("Vector is immutable")

    @property
    def values(self) -> array[float]:
        """The numbers, unpacked on first access and kept.

        An array of singles, which is what the server stores, and which numpy and torch accept
        through the buffer protocol without copying anything.
        """
        held = self._values
        if held is not None:
            return held
        raw = self._raw
        if raw is not None:
            held = array("f")
            held.frombytes(raw[_DENSE_HEADER.size :])
            if sys.byteorder == "little":
                held.byteswap()
        else:
            text = self._text
            assert text is not None
            held = array("f", parse_vector_text(text, half=self._half))
        object.__setattr__(self, "_values", held)
        object.__setattr__(self, "_raw", None)
        object.__setattr__(self, "_text", None)
        return held

    def tolist(self) -> list[float]:
        """An ordinary list, for a caller that wants one."""
        return self.values.tolist()

    def to_bytes(self) -> bytes:
        """The wire form: the dimension, two bytes the type does not use, then the numbers."""
        if self._raw is not None:
            return self._raw
        packed = array("f", self.values)
        if sys.byteorder == "little":
            packed.byteswap()
        return _DENSE_HEADER.pack(len(packed), 0) + packed.tobytes()

    def __len__(self) -> int:
        if self._raw is not None:
            return int(_DENSE_HEADER.unpack_from(self._raw, 0)[0])
        return len(self.values)

    @overload
    def __getitem__(self, index: int) -> float: ...
    @overload
    def __getitem__(self, index: slice) -> array[float]: ...
    def __getitem__(self, index: int | slice) -> float | array[float]:
        return self.values[index]

    def __iter__(self) -> Iterator[float]:
        return iter(self.values)

    def __contains__(self, value: object) -> bool:
        return value in self.values

    def index(self, value: float) -> int:
        """Where a number first appears."""
        return self.values.index(value)

    def count(self, value: float) -> int:
        """How many times a number appears."""
        return self.values.count(value)

    def __eq__(self, other: object) -> bool:
        """Equal to another vector, or to any sequence holding the same numbers.

        Comparing against a plain list has to work: a value that held the same numbers and called
        itself unequal to ``[1.0, 2.0]`` would be wrong in the quietest possible way.
        """
        if isinstance(other, Vector):
            return self.values == other.values
        if isinstance(other, Sequence) and not isinstance(other, str | bytes):
            mine = self.values
            return len(mine) == len(other) and all(
                a == b for a, b in zip(mine, other, strict=True)
            )
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.values.tobytes())

    def __repr__(self) -> str:
        return f"Vector({len(self)} dimensions)"


def generated_column(name: str, dimensions: int, *, type: str = "vector") -> str:
    """The column definition that gives a property a column of its own, dimension and all.

    Used as ``create vlabel emb ({generated_column('v', 1024)})``. The dimension has to be here:
    it is what makes the server refuse a wrong-length value at the moment it is written, and what
    an index binds to.

    ``generated`` is not optional -- a promoted column that is not generated is refused outright.

    :func:`vector_index` indexes a column built this way, and a property left in the map alike.
    """
    if dimensions < 1:
        raise ValueError(f"a vector has at least one dimension, got {dimensions}")
    if type not in TYPES:
        raise ValueError(f"expected one of {TYPES}, got {type!r}")
    return f"{name} {type}({dimensions}) generated"


SEARCH_OPTIONS: Mapping[str, type] = {
    "hnsw.ef_search": int,
    "hnsw.iterative_scan": str,
    "hnsw.max_scan_tuples": int,
    "hnsw.scan_mem_multiplier": float,
    "ivfflat.probes": int,
    "ivfflat.iterative_scan": str,
    "ivfflat.max_probes": int,
}
"""What a search can be tuned with, and the type each one takes.

Defaults on pgvector 0.8.5: ``ef_search`` 40, ``max_scan_tuples`` 20000, ``scan_mem_multiplier`` 1,
``probes`` 1, ``max_probes`` 32768, and both ``iterative_scan`` settings off. They do not appear in
``pg_settings`` until the extension's library is loaded, which the first vector operation in a session
does.
"""


def search_option_statements(options: Mapping[str, object], *, local: bool = True) -> list[str]:
    """Statements that set what a vector search may be tuned with.

    ``local`` keeps each setting to the current transaction, so it reverts on commit rather than
    outliving the search it was meant for. Pass ``False`` to set them for the session.

    A name that is not one of :data:`SEARCH_OPTIONS` is refused rather than sent, since the server
    accepts an unknown ``hnsw.`` name silently -- an extension may define one -- and a typo would
    then read as having been applied.
    """
    statements = []
    scope = "local " if local else ""
    for name, value in options.items():
        wanted = SEARCH_OPTIONS.get(name)
        if wanted is None:
            raise ValueError(
                f"not a vector search option: {name!r}. One of {sorted(SEARCH_OPTIONS)}"
            )
        if wanted is int and not isinstance(value, int):
            raise ValueError(f"{name} takes a whole number, got {value!r}")
        if wanted is float and not isinstance(value, int | float):
            raise ValueError(f"{name} takes a number, got {value!r}")
        rendered = value if isinstance(value, int | float) else f"'{value}'"
        statements.append(f"set {scope}{name} = {rendered}")
    return statements


def _searched_expression(property: str, dimensions: int | None, type: str) -> str:
    """How a searched vector is spelled, for the index and for the search alike.

    An index over ``(embedding::vector(4))`` is matched only by a search casting to ``vector(4)``.
    A bare ``vector``, or another dimension, sorts a sequential scan.

    ``dimensions`` of ``None`` means the property has a column of its own, which needs no cast.
    """
    from .cypher import quote_identifier

    name = quote_identifier(property)
    if dimensions is None:
        return name
    if dimensions < 1:
        raise ValueError(f"a vector has at least one dimension, got {dimensions}")
    if type not in TYPES:
        raise ValueError(f"expected one of {TYPES}, got {type!r}")
    return f"{name}::{type}({dimensions})"


def _with_options(options: Mapping[str, object] | None) -> str:
    """The ``WITH`` clause a method's settings go in, such as an ivfflat list count."""
    if not options:
        return ""
    settings = ", ".join(
        f"{key}={value}" if isinstance(value, int) else f"{key}='{value}'"
        for key, value in options.items()
    )
    return f" with ({settings})"


def vector_index(
    label: str,
    property: str,
    *,
    dimensions: int | None = None,
    type: str = "vector",
    method: str = "hnsw",
    operator_class: str = "vector_cosine_ops",
    name: str | None = None,
    options: Mapping[str, object] | None = None,
) -> str:
    """A property index over a vector, in the property map or in a column of its own.

    A property index carries the expression a Cypher search builds, so the planner matches the two.
    An index over the SQL spelling of the same property, ``(properties ->> 'embedding')::vector(4)``,
    is never matched.

    Pass ``dimensions`` for a property in the map, which is cast in the index and in the search
    alike; leave it out for a property with a column of its own.

    ``method`` is ``hnsw`` or ``ivfflat``, and ``options`` becomes the ``WITH`` clause an ivfflat
    index takes, as ``options={"lists": 10}``. The operator class serves one operator:
    ``vector_cosine_ops`` answers ``<=>`` and a search ordering by ``<->`` against it sorts a
    sequential scan. :attr:`Distance.operator_class` gives the one for an operator.
    """
    from .cypher import quote_identifier

    expression = _searched_expression(property, dimensions, type)
    given = f"{quote_identifier(name)} " if name else ""
    return (
        f"create property index {given}on {quote_identifier(label)} "
        f"using {method} (({expression}) {operator_class}){_with_options(options)}"
    )


def nearest(
    label: str,
    property: str,
    *,
    dimensions: int | None = None,
    type: str = "vector",
    limit: int = 10,
    operator: str = "<=>",
    variable: str = "n",
) -> str:
    """The Cypher statement that finds the closest elements to a given vector.

    A distance operator is an ordinary operator to Cypher, so this is a match with an order by and a
    vertex comes back as a vertex. It plans as an index scan over the index :func:`vector_index`
    builds, including under ``plan_cache_mode = force_generic_plan``.

    ``dimensions`` and ``type`` are the ones the index was built with, spelled by the function the
    index uses.
    """
    from .cypher import quote_identifier

    who = quote_identifier(variable)
    searched = f"{who}.{_searched_expression(property, dimensions, type)}"
    # The parameter is cast because this driver declares a string as text, and there is no operator
    # between a vector and text. It carries the dimension, as the index does.
    against = f"%s::{type}({dimensions})" if dimensions is not None else f"%s::{type}"
    return (
        f"match ({who}:{quote_identifier(label)}) return {who} "
        f"order by {searched} {operator} {against} limit {int(limit)}"
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
        """The server's text form, whose indices count from one.

        Nine significant digits, which is the fewest that reproduces a single-precision float
        exactly -- and single precision is what the server stores. The default six loses up to
        five parts in a million on every value, silently: measured over 200,000 random
        single-precision values, ``%g`` was wrong by as much as 5e-6 and ``%.9g`` by nothing at
        all. Python's own ``repr`` is also exact but 39% longer, and the extra digits describe
        precision the server is about to discard.
        """
        body = ",".join(
            f"{index + 1}:{value:.9g}"
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
                # From one in the text form to zero here, and narrowed to the width the
                # server keeps, so that this and the binary reading agree.
                entries.append((int(index.strip()) - 1, _narrow(float(value.strip()), _F32)))
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


def dense_dumper(oid: int) -> type[Dumper]:
    """A dumper that sends a :class:`Vector` as bytes, for the oid this database gave it.

    Built per registration rather than written out, because the type's oid belongs to the extension
    and is not known until it has been looked up -- the same reason the composite loaders for the
    graph types are built per connection.
    """

    class _DenseVectorBinaryDumper(Dumper):
        format = Format.BINARY

        def dump(self, obj: Vector) -> bytes:
            return obj.to_bytes()

    _DenseVectorBinaryDumper.oid = oid
    return _DenseVectorBinaryDumper
