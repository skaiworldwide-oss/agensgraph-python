"""Handing a result to something that works in columns, and loading one back.

Arrow is built directly rather than by way of a list per column. A result arrives as rows of Python
objects and the whole of the cost is in what happens to each value on the way out -- so the rows are
turned on their side in one C-level step and each column is handed to Arrow whole. Measured on two
hundred thousand vertices returning five properties:

============================================  ==========  ========
                                              a value at   a column
                                              a time       at a time
============================================  ==========  ========
one list per column                            1107 ms      92 ms
an Arrow table                                 1248 ms     104 ms
a pandas frame                                 1335 ms     150 ms
a polars frame                                 1272 ms     114 ms
============================================  ==========  ========

The left column is one Python-level pass per *value*: an ``isinstance`` chain, an append, and a dict
of lists for Arrow to infer types from. What replaces it is ``zip(*rows)`` -- which transposes in C
and checks that every row is the same width while it is at it -- and one builder per column.

**A vector is where it matters most.** The wire form of a dense vector is a header and then its
numbers as big-endian single-precision floats, which is what Arrow's ``FixedSizeList<float32>``
holds, so no number needs to become a Python float. The bytes of every row are appended to one
``array('f')``, byte-swapped once for the whole column, and handed to Arrow as a buffer. Measured on
twenty thousand embeddings of 384 dimensions:

=========================================  ==================  ========
                                           a list of doubles   this
=========================================  ==================  ========
binary rendering                             991 ms             80 ms
text rendering                              4881 ms           2569 ms
=========================================  ==================  ========

So **ask for the binary rendering when a column holds vectors**: ``binary_=True`` on
:meth:`~agensgraph.Connection.execute_query`, which together with the buffer route is sixty-one times
the text rendering read as Python floats. The text rendering spends its time parsing decimal digits,
which nothing here can avoid; the numbers are the same either way. The column holds 30.7 MB rather
than 61.5 MB, since a vector is single precision and a Python float is not.

**A vertex is a struct** of its identity, its label and its property map -- and the property map is
JSON text taken from the bytes it arrived in, so it is never decoded into a dict at all. A vertex
column of two hundred thousand rows is built in 219 ms against 864 for building a dict per row and
letting Arrow infer, and nothing has to guess what the properties are. Asking for the whole vertex is
also the cheapest way to read those three things: reading and exporting ``return n`` took 2955 ms
where ``return id(n), label(n), properties(n)`` took 5008, because each of those is a jsonb value of
its own to be parsed.

An identity becomes a ``uint64``, which is what a graphid is: a label id in the top sixteen bits and
a serial id in the low forty-eight. It is what joins a vertex column to an edge's endpoints. The text
form costs twice as much to build -- 435 ms against 219 for the same vertex column -- and does not
join as cheaply, so it is asked for rather than assumed.

**Where nothing is copied, and where something is.** The way from the wire to Arrow is not
zero-copy and the buffer route does not make it one: psycopg hands over the bytes of a row, each
value becomes a Python object, and the column that object goes into is a buffer of its own. What the
buffer route removes is the Python object *per number* -- a vector's bytes are copied into an
``array('f')``, swapped in place, and held by Arrow through the buffer protocol without being copied
a second time. Checked by address, two things share memory rather than copying it: that array and
the Arrow column over it, and the Arrow column and the polars column made from it -- for numbers and
for vectors, though not for strings, which polars holds its own way. pandas copies every column
whichever dtypes it is asked for; what the Arrow-backed ones buy is the layout, since
``dtypes="numpy"`` turns a struct, a list or a vector column back into Python objects.

Nothing here is a dependency. Each backend is imported inside the function that uses it, so a program
that does not export loads none of them.

**Loading the other way** goes through binary ``COPY``. The property maps of a chunk are written as
JSON and the copy stream is built from them directly, so no row becomes a Python mapping and no value
goes through a dumper. The client's share of loading two hundred thousand vertices falls from 1024 ms
to 125; end to end it is 1.5 times as fast, 2736 ms against 1879, and no more than that because the
copy itself is 1810 ms of either and is the floor.
"""

from __future__ import annotations

import sys
from array import array
from collections.abc import Sequence
from itertools import chain, islice
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from ._core import Result
from ._protocol.decode import EDGE_OID, GRAPHID_OID, GRAPHPATH_OID, VERTEX_OID
from ._protocol.graphid import GraphId, parse_text
from .numbers import encode_json
from .types import Edge, Path, Vertex
from .vector import SparseVector, Vector

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

__all__ = [
    "CHUNK",
    "Layout",
    "Plan",
    "batches",
    "columns",
    "edge_payloads",
    "reader",
    "to_arrow",
    "to_pandas",
    "to_polars",
    "vertex_payloads",
]

CHUNK = 8192
"""How many rows a record batch holds when one is not asked for by size."""

_LITTLE = sys.byteorder == "little"
_HEADER = 4
"""The bytes of dimension and padding a dense vector carries ahead of its numbers."""

_VERTEX_ARRAY_OID = 7011
_EDGE_ARRAY_OID = 7021
_GRAPHID_ARRAY_OID = 7001

_JSON_OIDS = frozenset({114, 3802})
"""json and jsonb. Every Cypher expression is one of these, so the oid says nothing about what a
column holds and the values have to."""


class Layout(NamedTuple):
    """What a column becomes where more than one answer is defensible."""

    elements: Literal["struct", "columns"] = "struct"
    """Whether a vertex, an edge or a path becomes one struct column or several plain ones, named
    ``n.id``, ``n.label`` and so on."""

    identity: Literal["packed", "text"] = "packed"
    """A graphid as its single 64-bit value, or as the ``labid.locid`` text. The label id is the top
    sixteen bits of the packed value, so ``id >> 48`` recovers it."""

    properties: Any = "json"
    """``"json"`` for the map as JSON text, ``"skip"`` to leave it out, or a ``pyarrow`` struct type
    to pull the named fields out of the decoded map into columns of their own."""

    labels: Literal["dictionary", "text"] = "dictionary"
    """A label column dictionary-encoded, which is what a handful of distinct values over many rows
    wants, or plain text."""

    vectors: Literal["fixed", "list"] = "fixed"
    """``FixedSizeList<float32>``, which needs every vector in the column to have the same number of
    dimensions, or a plain ``List<float32>``, which does not."""

    sparse: Literal["parts", "text", "dense"] = "parts"
    """A sparse vector as a struct of its dimension, its indices and its values; as the text it
    prints as; or expanded to the dense vector it stands for."""


DEFAULT = Layout()
"""The layout used where none is given."""


# -- turning rows on their side ---------------------------------------------------------------


def _transpose(rows: Sequence[Sequence[Any]], width: int) -> list[tuple[Any, ...]]:
    """One tuple per column, and a check that every row is as wide as the column names.

    ``zip`` transposes in C, and ``strict`` is what makes it check: without it a short row would
    silently truncate every column to its length.
    """
    if not rows:
        return [()] * width
    try:
        built = list(zip(*rows, strict=True))
    except ValueError:
        widths = sorted({len(row) for row in rows})
        raise ValueError(f"rows of {widths} widths against {width} column names") from None
    if len(built) != width:
        raise ValueError(f"a row of {len(built)} against {width} column names")
    return built


def columns(records: Sequence[Sequence[Any]], keys: Sequence[str]) -> dict[str, list[Any]]:
    """A result turned on its side, one list per column, as plain Python.

    For a caller that wants neither Arrow nor a frame. A graph value that has no Python counterpart
    is converted -- an identity to its text, a vector to a list of numbers, an element to a mapping
    of its identity, label and properties -- and everything else is left as it arrived. The
    conversion is decided once per column rather than once per value.
    """
    if not keys:
        return {}
    out: dict[str, list[Any]] = {}
    for name, values in zip(keys, _transpose(records, len(keys)), strict=True):
        convert = _plainly(_first(values))
        out[name] = list(values) if convert is None else [convert(value) for value in values]
    return out


def _written(value: Any) -> str | None:
    """One identity as its text."""
    return None if value is None else str(value)


def _numbers_of(value: Any) -> list[float] | None:
    """One vector as a list of numbers."""
    return None if value is None else value.tolist()


def _sparse_text(value: Any) -> str | None:
    """One sparse vector as the text it prints as."""
    return None if value is None else value.to_text()


def _mapped(value: Any) -> dict[str, Any] | None:
    """One element or path as nested mappings."""
    return None if value is None else _mapping_of(value)


def _plainly(sample: Any) -> Callable[[Any], Any] | None:
    """How a column of values like *sample* is written as plain Python, or ``None`` for as it is."""
    if isinstance(sample, GraphId):
        return _written
    if isinstance(sample, Vector):
        return _numbers_of
    if isinstance(sample, SparseVector):
        return _sparse_text
    if isinstance(sample, Vertex | Edge | Path):
        return _mapped
    return None


def _mapping_of(value: Vertex | Edge | Path) -> dict[str, Any]:
    """A graph element or a path as nested mappings."""
    if isinstance(value, Path):
        return {
            "vertices": [_mapping_of(vertex) for vertex in value.vertices],
            "edges": [_mapping_of(edge) for edge in value.edges],
        }
    out: dict[str, Any] = {"id": str(value.id), "label": value.label}
    if isinstance(value, Edge):
        out["start"] = str(value.start)
        out["end"] = str(value.end)
    out["properties"] = value.properties
    return out


def _first(values: Iterable[Any]) -> Any:
    """The first value that is not null, or ``None`` when every one of them is."""
    for value in values:
        if value is not None:
            return value
    return None


# -- what each column is built by --------------------------------------------------------------


def _validity(values: Sequence[Any]) -> Any:
    """The null bitmap for a column, or ``None`` when it holds no nulls.

    ``None in values`` is a C-level scan, so a column without nulls pays only that. The bitmap
    itself is the values buffer of a boolean array, which is the same layout Arrow wants and is
    filled in C.
    """
    if None not in values:
        return None
    import pyarrow

    flags = [value is not None for value in values]
    return pyarrow.array(flags, type=pyarrow.bool_()).buffers()[1]


class _Fixed:
    """A column whose type is already settled, built straight into it."""

    __slots__ = ("_type", "names")

    def __init__(self, name: str, type: Any) -> None:
        self.names: tuple[str, ...] = (name,)
        self._type = type

    def build(self, values: Sequence[Any]) -> tuple[Any, ...]:
        import pyarrow

        return (pyarrow.array(values, type=self._type),)


class _Inferred:
    """A column nothing declared a type for, whose type comes from the first chunk and is held.

    Every later chunk is inferred again and cast to the type the first settled on. The cast is what
    reports a chunk that does not fit: building straight into a settled type would convert 2.5 into
    an int64 column as 2 and say nothing.
    """

    __slots__ = ("_fallback", "_type", "names")

    def __init__(self, name: str) -> None:
        self.names: tuple[str, ...] = (name,)
        self._type: Any = None
        self._fallback: _Json | _Text | None = None

    def build(self, values: Sequence[Any]) -> tuple[Any, ...]:
        import pyarrow

        if self._fallback is not None:
            return self._fallback.build(values)
        try:
            built = pyarrow.array(values)
        except (pyarrow.ArrowInvalid, pyarrow.ArrowTypeError, OverflowError, TypeError):
            if self._type is not None:
                raise
            self._fallback = _as_text(self.names[0], _first(values))
            return self._fallback.build(values)
        if self._type is None:
            self._type = built.type
            return (built,)
        if built.type == self._type:
            return (built,)
        try:
            return (built.cast(self._type),)
        except (pyarrow.ArrowInvalid, pyarrow.ArrowNotImplementedError) as exc:
            raise ValueError(
                f"column {self.names[0]!r} held {self._type} in the first chunk and "
                f"{built.type} in this one; pass a schema to settle it"
            ) from exc


def _as_text(name: str, sample: Any) -> _Json | _Text:
    """What a column Arrow cannot type is written as instead.

    A map or a list is JSON, which is what it came from. Anything else is its text -- an integer
    too large for int64 among them, which jsonb can hold and Arrow cannot.
    """
    if isinstance(sample, dict | list | tuple):
        return _Json(name)
    return _Text(name)


class _Json:
    """A column of JSON text."""

    __slots__ = ("names",)

    def __init__(self, name: str) -> None:
        self.names: tuple[str, ...] = (name,)

    def build(self, values: Sequence[Any]) -> tuple[Any, ...]:
        return (_json_array([None if v is None else encode_json(v) for v in values]),)


class _Text:
    """A column of text, from values written out one at a time."""

    __slots__ = ("names",)

    def __init__(self, name: str) -> None:
        self.names: tuple[str, ...] = (name,)

    def build(self, values: Sequence[Any]) -> tuple[Any, ...]:
        import pyarrow

        written = [None if value is None else str(value) for value in values]
        return (pyarrow.array(written, type=pyarrow.utf8()),)


def _json_array(payloads: Sequence[bytes | None]) -> Any:
    """A string column from JSON already written as bytes.

    Built as binary and cast, which validates the UTF-8 and is measurably cheaper than assembling
    the offsets by hand: 60 ms against 65 for two hundred thousand property maps.
    """
    import pyarrow

    return pyarrow.array(payloads, type=pyarrow.binary()).cast(pyarrow.utf8())


class _Identities:
    """A column of graph identities."""

    __slots__ = ("_text", "names")

    def __init__(self, name: str, layout: Layout) -> None:
        self.names: tuple[str, ...] = (name,)
        self._text = layout.identity == "text"

    def build(self, values: Sequence[Any]) -> tuple[Any, ...]:
        return (_identity_array(values, text=self._text),)


def _identity_array(values: Sequence[GraphId | None], *, text: bool) -> Any:
    """Identities as text, or as the single 64-bit value each one is."""
    import pyarrow

    if text:
        written = [None if value is None else str(value) for value in values]
        return pyarrow.array(written, type=pyarrow.utf8())
    validity = _validity(values)
    if validity is None:
        return _packed_array([held.packed for held in values if held is not None], None)
    return _packed_array([0 if held is None else held.packed for held in values], validity)


def _packed_array(packed: list[int], validity: Any) -> Any:
    """A ``uint64`` column from packed identities, over the buffer they were read into."""
    import pyarrow

    held = array("Q", packed)
    return pyarrow.Array.from_buffers(
        pyarrow.uint64(), len(held), [validity, pyarrow.py_buffer(held)]
    )


class _IdentityLists:
    """A column of arrays of identities."""

    __slots__ = ("_text", "names")

    def __init__(self, name: str, layout: Layout) -> None:
        self.names: tuple[str, ...] = (name,)
        self._text = layout.identity == "text"

    def build(self, values: Sequence[Any]) -> tuple[Any, ...]:
        import pyarrow

        if self._text:
            written = [
                None if row is None else [None if held is None else str(held) for held in row]
                for row in values
            ]
            return (pyarrow.array(written, type=pyarrow.list_(pyarrow.utf8())),)
        packed = [
            None if row is None else [None if held is None else held.packed for held in row]
            for row in values
        ]
        return (pyarrow.array(packed, type=pyarrow.list_(pyarrow.uint64())),)


class _Vectors:
    """A column of dense vectors."""

    __slots__ = ("_dimensions", "_fixed", "names")

    def __init__(self, name: str, dimensions: int, layout: Layout) -> None:
        self.names: tuple[str, ...] = (name,)
        self._dimensions = dimensions
        self._fixed = layout.vectors == "fixed"

    def build(self, values: Sequence[Any]) -> tuple[Any, ...]:
        if self._fixed:
            return (_fixed_vectors(self.names[0], values, self._dimensions),)
        return (_listed_vectors(values),)


def _numbers(value: Vector) -> memoryview:
    """One vector's numbers as they came off the wire: big-endian single precision.

    ``to_bytes`` hands back the bytes a vector arrived in without copying them. A vector built from
    Python numbers, or one whose text has not been parsed yet, is packed here instead.
    """
    return memoryview(value.to_bytes())[_HEADER:]


def _fixed_vectors(name: str, values: Sequence[Any], dimensions: int) -> Any:
    """A ``FixedSizeList<float32>`` column, from one buffer swapped once."""
    import pyarrow

    if dimensions < 1:
        raise ValueError(
            f"column {name!r} holds a vector of no dimensions, which a fixed size list has no "
            f"width for; pass Layout(vectors='list')"
        )
    width = dimensions * 4
    flat = array("f")
    blank = bytes(width)
    for value in values:
        if value is None:
            flat.frombytes(blank)
            continue
        numbers = _numbers(value)
        if len(numbers) != width:
            raise ValueError(
                f"column {name!r} holds vectors of {dimensions} and {len(numbers) // 4} "
                f"dimensions; a fixed size list holds one width, so pass "
                f"Layout(vectors='list')"
            )
        flat.frombytes(numbers)
    if _LITTLE:
        flat.byteswap()
    child = pyarrow.Array.from_buffers(
        pyarrow.float32(), len(flat), [None, pyarrow.py_buffer(flat)]
    )
    validity = _validity(values)
    if validity is None:
        return pyarrow.FixedSizeListArray.from_arrays(child, dimensions)
    return pyarrow.Array.from_buffers(
        pyarrow.list_(pyarrow.float32(), dimensions), len(values), [validity], children=[child]
    )


def _listed_vectors(values: Sequence[Any]) -> Any:
    """A ``List<float32>`` column, for vectors that do not all have the same width."""
    import pyarrow

    flat = array("f")
    offsets = array("i", [0])
    position = 0
    for value in values:
        if value is not None:
            numbers = _numbers(value)
            flat.frombytes(numbers)
            position += len(numbers) // 4
        offsets.append(position)
    if _LITTLE:
        flat.byteswap()
    child = pyarrow.Array.from_buffers(
        pyarrow.float32(), len(flat), [None, pyarrow.py_buffer(flat)]
    )
    return pyarrow.Array.from_buffers(
        pyarrow.list_(pyarrow.float32()),
        len(values),
        [_validity(values), pyarrow.py_buffer(offsets)],
        children=[child],
    )


class _SparseVectors:
    """A column of sparse vectors."""

    __slots__ = ("_dimensions", "_how", "names")

    def __init__(self, name: str, dimensions: int, layout: Layout) -> None:
        self.names: tuple[str, ...] = (name,)
        self._dimensions = dimensions
        self._how = layout.sparse

    def build(self, values: Sequence[Any]) -> tuple[Any, ...]:
        import pyarrow

        if self._how == "text":
            written = [None if value is None else value.to_text() for value in values]
            return (pyarrow.array(written, type=pyarrow.utf8()),)
        if self._how == "dense":
            dense = [None if value is None else Vector(value.to_dense()) for value in values]
            return (_fixed_vectors(self.names[0], dense, self._dimensions),)
        indices = pyarrow.array(
            [None if value is None else value.indices for value in values],
            type=pyarrow.list_(pyarrow.int32()),
        )
        numbers = pyarrow.array(
            [None if value is None else value.values for value in values],
            type=pyarrow.list_(pyarrow.float32()),
        )
        sizes = pyarrow.array(
            [None if value is None else value.dimensions for value in values],
            type=pyarrow.int32(),
        )
        mask = _mask(values)
        return (
            pyarrow.StructArray.from_arrays(
                [sizes, indices, numbers], ["dimensions", "indices", "values"], mask=mask
            ),
        )


def _mask(values: Sequence[Any]) -> Any:
    """Which slots of a struct column are null, or ``None`` when none of them is."""
    if None not in values:
        return None
    import pyarrow

    return pyarrow.array([value is None for value in values], type=pyarrow.bool_())


class _Elements:
    """A column of vertices or of edges."""

    __slots__ = ("_kind", "_layout", "_parts", "names")

    def __init__(self, name: str, kind: type[Vertex] | type[Edge], layout: Layout) -> None:
        self._kind = kind
        self._layout = layout
        self._parts = _element_parts(kind, layout)
        if layout.elements == "columns":
            self.names = tuple(f"{name}.{part}" for part in self._parts)
        else:
            self.names = (name,)

    def build(self, values: Sequence[Any]) -> tuple[Any, ...]:
        import pyarrow

        arrays, mask = _element_arrays(values, self._kind, self._layout)
        if self._layout.elements == "columns":
            return tuple(arrays)
        return (pyarrow.StructArray.from_arrays(arrays, list(self._parts), mask=mask),)


def _element_parts(kind: type[Vertex] | type[Edge], layout: Layout) -> tuple[str, ...]:
    """The fields a vertex or an edge becomes, in order."""
    parts = ["id", "label"]
    if kind is Edge:
        parts += ["start", "end"]
    if layout.properties != "skip":
        parts.append("properties")
    return tuple(parts)


_BLANK_VERTEX = Vertex(GraphId(0, 0), "")
_BLANK_EDGE = Edge(GraphId(0, 0), "")


def _element_arrays(
    values: Sequence[Any], kind: type[Vertex] | type[Edge], layout: Layout
) -> tuple[list[Any], Any]:
    """One array per field of a vertex or an edge, and the null mask of the column.

    A null slot is filled with a blank element first, so every pass below reads an element rather
    than testing for one -- and the bitmap makes those slots null whatever was read from them. One
    test for the column against one per value per field is worth about 7% of the build.
    """
    import pyarrow

    mask = _mask(values)
    validity = None
    if mask is not None:
        validity = _validity(values)
        blank = _BLANK_EDGE if kind is Edge else _BLANK_VERTEX
        values = [blank if value is None else value for value in values]
    text = layout.identity == "text"
    out: list[Any] = []
    if text:
        out.append(_identity_array([held.id for held in values], text=True))
    else:
        out.append(_packed_array([held.id.packed for held in values], validity))
    label_type = (
        pyarrow.dictionary(pyarrow.int32(), pyarrow.utf8())
        if layout.labels == "dictionary"
        else pyarrow.utf8()
    )
    out.append(pyarrow.array([held.label for held in values], type=label_type))
    if kind is Edge and text:
        out.append(_identity_array([held.start for held in values], text=True))
        out.append(_identity_array([held.end for held in values], text=True))
    elif kind is Edge:
        out.append(_packed_array([held.start.packed for held in values], validity))
        out.append(_packed_array([held.end.packed for held in values], validity))
    if layout.properties == "json":
        out.append(_json_array([held.properties_json() for held in values]))
    elif layout.properties != "skip":
        maps = [held.properties for held in values]
        out.append(pyarrow.array(maps, type=layout.properties))
    return out, mask


class _ElementLists:
    """A column of arrays of vertices or of edges, as a list of structs."""

    __slots__ = ("_kind", "_layout", "names")

    def __init__(self, name: str, kind: type[Vertex] | type[Edge], layout: Layout) -> None:
        self.names: tuple[str, ...] = (name,)
        self._kind = kind
        self._layout = layout

    def build(self, values: Sequence[Any]) -> tuple[Any, ...]:
        return (_element_lists(values, self._kind, self._layout),)


def _element_lists(
    values: Sequence[Any], kind: type[Vertex] | type[Edge], layout: Layout
) -> Any:
    """A list-of-structs column, from the elements of every row flattened into one struct array."""
    import pyarrow

    flat: list[Any] = []
    offsets = array("i", [0])
    for value in values:
        if value is not None:
            flat.extend(value)
        offsets.append(len(flat))
    arrays, mask = _element_arrays(flat, kind, layout)
    struct = pyarrow.StructArray.from_arrays(
        arrays, list(_element_parts(kind, layout)), mask=mask
    )
    return pyarrow.Array.from_buffers(
        pyarrow.list_(struct.type),
        len(values),
        [_validity(values), pyarrow.py_buffer(offsets)],
        children=[struct],
    )


class _Paths:
    """A column of paths, as a struct of a list of vertices and a list of edges."""

    __slots__ = ("_layout", "names")

    def __init__(self, name: str, layout: Layout) -> None:
        self._layout = layout
        self.names: tuple[str, ...] = (
            (f"{name}.vertices", f"{name}.edges") if layout.elements == "columns" else (name,)
        )

    def build(self, values: Sequence[Any]) -> tuple[Any, ...]:
        import pyarrow

        vertices = _element_lists(
            [None if v is None else v.vertices for v in values], Vertex, self._layout
        )
        edges = _element_lists(
            [None if v is None else v.edges for v in values], Edge, self._layout
        )
        if self._layout.elements == "columns":
            return (vertices, edges)
        return (
            pyarrow.StructArray.from_arrays(
                [vertices, edges], ["vertices", "edges"], mask=_mask(values)
            ),
        )


# -- deciding what a column is --------------------------------------------------------------


def _sql_types(pyarrow: Any) -> dict[int, Any]:
    """The Arrow type a PostgreSQL type reads as, for the types whose oid settles it.

    Only the ones a value cannot settle as well or better. A jsonb column is not here: every Cypher
    expression is jsonb, so the oid says nothing and the values say everything. Numeric is not here
    either, because a Decimal infers to the decimal type that fits it.
    """
    return {
        16: pyarrow.bool_(),
        17: pyarrow.binary(),
        18: pyarrow.utf8(),
        19: pyarrow.utf8(),
        20: pyarrow.int64(),
        21: pyarrow.int16(),
        23: pyarrow.int32(),
        25: pyarrow.utf8(),
        26: pyarrow.uint32(),
        700: pyarrow.float32(),
        701: pyarrow.float64(),
        1042: pyarrow.utf8(),
        1043: pyarrow.utf8(),
        1082: pyarrow.date32(),
        1083: pyarrow.time64("us"),
        1114: pyarrow.timestamp("us"),
        1184: pyarrow.timestamp("us", "UTC"),
    }


_TEXT_OIDS = frozenset({2950})
"""uuid, which arrives as a UUID and has no Arrow type this reads into, so it is written out."""


def _column_for(
    name: str, oid: int, values: Sequence[Any], layout: Layout, declared: Any
) -> Any:
    """How one input column is built.

    The graph types are settled by their oid. Everything else is settled by the first value that is
    not null, because a Cypher projection is jsonb whatever it holds. A column of nothing but nulls
    and no declared type has no type to find, and becomes a null column -- which is what Arrow does
    with one too.

    A map, or a list of maps, becomes JSON text rather than a struct Arrow works out for itself: a
    struct inferred from one chunk is a struct the next chunk has to match, and jsonb promises no
    such thing. A list of anything else is left to Arrow, which reads it as a list column.
    """
    if oid == GRAPHID_OID:
        return _Identities(name, layout)
    if oid == VERTEX_OID:
        return _Elements(name, Vertex, layout)
    if oid == EDGE_OID:
        return _Elements(name, Edge, layout)
    if oid == GRAPHPATH_OID:
        return _Paths(name, layout)
    if oid == _VERTEX_ARRAY_OID:
        return _ElementLists(name, Vertex, layout)
    if oid == _EDGE_ARRAY_OID:
        return _ElementLists(name, Edge, layout)
    if oid == _GRAPHID_ARRAY_OID:
        return _IdentityLists(name, layout)
    if oid in _TEXT_OIDS:
        return _Text(name)

    sample = _first(values)
    if isinstance(sample, Vertex):
        return _Elements(name, Vertex, layout)
    if isinstance(sample, Edge):
        return _Elements(name, Edge, layout)
    if isinstance(sample, Path):
        return _Paths(name, layout)
    if isinstance(sample, GraphId):
        return _Identities(name, layout)
    if isinstance(sample, Vector):
        return _Vectors(name, _declared_width(declared, len(sample)), layout)
    if isinstance(sample, SparseVector):
        return _SparseVectors(name, sample.dimensions, layout)
    inner = _first(sample) if isinstance(sample, list | tuple) else None
    if isinstance(inner, Vertex):
        return _ElementLists(name, Vertex, layout)
    if isinstance(inner, Edge):
        return _ElementLists(name, Edge, layout)
    if isinstance(inner, GraphId):
        return _IdentityLists(name, layout)
    if declared is not None:
        return _Fixed(name, declared)
    if isinstance(sample, dict) or isinstance(inner, dict):
        return _Json(name)
    if oid not in _JSON_OIDS:
        settled = _sql_types(_pyarrow()).get(oid)
        if settled is not None:
            return _Fixed(name, settled)
    return _Inferred(name)


def _declared_width(declared: Any, found: int) -> int:
    """The dimension a vector column is built at: the declared one where there is one."""
    if declared is not None and getattr(declared, "list_size", -1) > 0:
        return int(declared.list_size)
    return found


def _pyarrow() -> Any:
    import pyarrow

    return pyarrow


class Plan:
    """The columns a result becomes, and how each one is built.

    Settled by the first chunk it is given and held thereafter, so every chunk carries the same
    schema, chunks concatenate, and a reader can declare its schema before anything reads from it.
    A chunk that does not fit the settled schema is reported rather than converted quietly.
    """

    __slots__ = ("_columns", "_declared", "_keys", "_layout", "_oids", "_schema")

    def __init__(
        self,
        keys: Sequence[str],
        *,
        oids: Sequence[int] = (),
        layout: Layout = DEFAULT,
        schema: Any = None,
    ) -> None:
        self._keys = list(keys)
        self._oids = list(oids)
        self._layout = layout
        self._declared: dict[str, Any] = (
            {} if schema is None else {field.name: field.type for field in schema}
        )
        self._columns: list[Any] | None = None
        self._schema: Any = None

    @property
    def keys(self) -> list[str]:
        """The column names of the result being built."""
        return list(self._keys)

    @property
    def schema(self) -> Any:
        """The Arrow schema, once a chunk has settled it."""
        if self._schema is None:
            raise ValueError("the schema is settled by the first chunk; build one first")
        return self._schema

    def batch(self, rows: Sequence[Sequence[Any]]) -> Any:
        """One record batch from a chunk of rows."""
        import pyarrow

        values = _transpose(rows, len(self._keys))
        if self._columns is None:
            oids = self._oids if len(self._oids) == len(self._keys) else [0] * len(self._keys)
            self._columns = [
                _column_for(key, oid, held, self._layout, self._declared.get(key))
                for key, oid, held in zip(self._keys, oids, values, strict=True)
            ]
        names: list[str] = []
        arrays: list[Any] = []
        for column, held in zip(self._columns, values, strict=True):
            names.extend(column.names)
            arrays.extend(column.build(held))
        if self._schema is None:
            self._schema = self._settle(names, arrays)
        if not arrays:
            return pyarrow.RecordBatch.from_pydict({}, schema=self._schema)
        return pyarrow.RecordBatch.from_arrays(arrays, schema=self._schema)

    def _settle(self, names: list[str], arrays: list[Any]) -> Any:
        """The schema the first chunk produced, checked against the one that was asked for."""
        import pyarrow

        if self._declared and set(self._declared) != set(names):
            missing = sorted(set(names) - set(self._declared))
            extra = sorted(set(self._declared) - set(names))
            raise ValueError(
                f"the schema names {extra or 'nothing'} that this result has no column for, and "
                f"leaves out {missing or 'nothing'}"
            )
        for name, built in zip(names, arrays, strict=True):
            wanted = self._declared.get(name)
            if wanted is not None and wanted != built.type:
                raise ValueError(
                    f"column {name!r} is built as {built.type} and the schema asks for {wanted}"
                )
        return pyarrow.schema(
            [pyarrow.field(name, built.type) for name, built in zip(names, arrays, strict=True)]
        )


# -- where the rows come from ---------------------------------------------------------------


def _source(
    source: Any, keys: Sequence[str] | None, oids: Sequence[int]
) -> tuple[Any, list[str], list[int]]:
    """Rows, column names and type oids, from a result, a cursor or plain rows.

    A cursor is worth taking because it carries the column types, which a list of rows does not,
    and because it is what a server-side cursor hands out a chunk at a time.
    """
    if isinstance(source, Result):
        return source.records, list(source.keys), list(source.oids)
    described = getattr(source, "description", None)
    if described is not None and hasattr(source, "fetchmany"):
        return (
            source,
            [column.name for column in described],
            [int(column.type_code) for column in described],
        )
    if keys is None:
        raise TypeError(
            "rows on their own do not say what the columns are called: pass keys, or pass the "
            "result or the cursor they came from"
        )
    return source, list(keys), list(oids)


def _chunks(rows: Any, size: int | None) -> Iterator[Sequence[Sequence[Any]]]:
    """The rows in chunks, always at least one chunk so that a schema is settled.

    A cursor is drained with ``fetchmany``, which is what keeps a server-side cursor's rows on the
    server. A sequence is sliced, which copies nothing that was not already held.
    """
    if hasattr(rows, "fetchmany"):
        empty = True
        while True:
            fetched = rows.fetchall() if size is None else rows.fetchmany(size)
            if not fetched:
                break
            empty = False
            yield fetched
            if size is None:
                break
        if empty:
            yield []
        return
    if isinstance(rows, Sequence):
        if size is None or len(rows) <= size:
            yield rows
            return
        for start in range(0, len(rows), size):
            yield rows[start : start + size]
        return
    iterator = iter(rows)
    empty = True
    while True:
        fetched = list(islice(iterator, size)) if size else list(iterator)
        if not fetched:
            break
        empty = False
        yield fetched
        if not size:
            break
    if empty:
        yield []


def batches(
    source: Any,
    keys: Sequence[str] | None = None,
    *,
    size: int | None = CHUNK,
    oids: Sequence[int] = (),
    layout: Layout = DEFAULT,
    schema: Any = None,
) -> Iterator[Any]:
    """Record batches from a result, a cursor or any iterable of rows.

    The first batch settles the schema and every later one is held to it. ``size`` is how many rows
    a batch holds, or ``None`` for all of them in one. Nothing is materialised beyond one chunk when
    *source* is a server-side cursor or an iterator, which is what makes a result larger than memory
    exportable.

    Measured on twenty thousand embeddings of 384 dimensions, peak resident memory: 88 MB reading
    them a thousand at a time from a server-side cursor, 168 MB reading the whole result and building
    one table from it, and 605 MB building that table from lists of Python floats.
    """
    rows, names, types = _source(source, keys, oids)
    plan = Plan(names, oids=types, layout=layout, schema=schema)
    for chunk in _chunks(rows, size):
        yield plan.batch(chunk)


def reader(
    source: Any,
    keys: Sequence[str] | None = None,
    *,
    size: int | None = CHUNK,
    oids: Sequence[int] = (),
    layout: Layout = DEFAULT,
    schema: Any = None,
) -> Any:
    """A ``pyarrow.RecordBatchReader`` a consumer pulls batches from.

    What Arrow's own interfaces take: a dataset writer, ``polars.scan_pyarrow_dataset``, an IPC
    stream. The first batch is built here so that the reader can be handed its schema; the rest are
    built as they are asked for.
    """
    import pyarrow

    stream = batches(source, keys, size=size, oids=oids, layout=layout, schema=schema)
    first = next(stream)
    return pyarrow.RecordBatchReader.from_batches(first.schema, chain([first], stream))


def to_arrow(
    source: Any,
    keys: Sequence[str] | None = None,
    *,
    size: int | None = None,
    oids: Sequence[int] = (),
    layout: Layout = DEFAULT,
    schema: Any = None,
) -> Any:
    """An Arrow table. Needs ``pyarrow``.

    ``size`` is left at ``None``, which builds one batch per chunk of rows the source hands over --
    one batch in all for a result already in memory, so that a column whose type nothing declared is
    settled by all of its values rather than by the first eight thousand of them. Giving a size
    chunks it, and then the first chunk settles the type.
    """
    import pyarrow

    built = list(batches(source, keys, size=size, oids=oids, layout=layout, schema=schema))
    return pyarrow.Table.from_batches(built, schema=built[0].schema)


def to_pandas(
    source: Any,
    keys: Sequence[str] | None = None,
    *,
    dtypes: Literal["arrow", "numpy"] = "arrow",
    size: int | None = None,
    oids: Sequence[int] = (),
    layout: Layout = DEFAULT,
    schema: Any = None,
) -> Any:
    """A pandas frame. Needs ``pandas``.

    Arrow-backed by default, which is what keeps a struct, a list or a vector column in Arrow's own
    layout. ``dtypes="numpy"`` gives pandas' older dtypes instead, and turns a column Arrow holds as
    a struct or a list into a column of Python objects.
    """
    import pandas

    table = to_arrow(source, keys, size=size, oids=oids, layout=layout, schema=schema)
    if dtypes == "arrow":
        return table.to_pandas(types_mapper=pandas.ArrowDtype)
    return table.to_pandas()


def to_polars(
    source: Any,
    keys: Sequence[str] | None = None,
    *,
    size: int | None = None,
    oids: Sequence[int] = (),
    layout: Layout = DEFAULT,
    schema: Any = None,
) -> Any:
    """A polars frame. Needs ``polars``.

    Built from the Arrow table, whose number and vector columns polars takes over rather than
    copying -- checked by buffer address. A string column it holds its own way and copies.
    """
    import polars

    return polars.from_arrow(
        to_arrow(source, keys, size=size, oids=oids, layout=layout, schema=schema)
    )


# -- the other direction --------------------------------------------------------------------


def _arrow_batches(source: Any, size: int) -> Iterator[Any]:
    """Anything columnar as Arrow record batches of at most *size* rows.

    An Arrow table, a polars frame, a pandas frame, a mapping of columns, or anything else offering
    Arrow's C stream, which is how a polars frame is read without being turned into a table first.
    Slicing a batch to *size* copies nothing.
    """
    import pyarrow

    stream: Any
    if isinstance(source, pyarrow.RecordBatchReader):
        stream = source
    elif isinstance(source, pyarrow.RecordBatch):
        stream = (source,)
    elif isinstance(source, pyarrow.Table):
        stream = source.to_batches(max_chunksize=size)
    elif hasattr(source, "__arrow_c_stream__"):
        stream = pyarrow.RecordBatchReader.from_stream(source)
    else:
        stream = pyarrow.table(source).to_batches(max_chunksize=size)
    for batch in stream:
        for start in range(0, batch.num_rows, size):
            yield batch.slice(start, size)


def _payloads(batch: Any) -> list[bytes]:
    """The JSON each row of a batch becomes.

    A single column named ``properties`` holding text or bytes is taken as the map itself, which is
    what :func:`to_arrow` writes and so what a round trip reads back. Otherwise every column becomes
    a property of that name.

    polars writes the JSON if it is installed, which it does in Rust: 98 ms against 1159 for two
    hundred thousand rows built as mappings and encoded here. Both write the same bytes.
    """
    import pyarrow

    if batch.num_columns == 0:
        return [b"{}"] * batch.num_rows
    if batch.num_columns == 1 and batch.schema.field(0).name == "properties":
        kind = batch.schema.field(0).type
        if pyarrow.types.is_binary(kind) or pyarrow.types.is_large_binary(kind):
            return list(batch.column(0).to_pylist())
        if pyarrow.types.is_string(kind) or pyarrow.types.is_large_string(kind):
            return [value.encode() for value in batch.column(0).to_pylist()]
    try:
        import polars
    except ImportError:
        names = batch.schema.names
        held = [column.to_pylist() for column in batch.columns]
        return [
            encode_json(dict(zip(names, row, strict=True))) for row in zip(*held, strict=True)
        ]
    frame: Any = polars.from_arrow(batch)
    written: bytes = frame.write_ndjson().encode()
    return written.split(b"\n")[:-1]


def _endpoints(batch: Any, name: str) -> list[int]:
    """One endpoint column as packed identities.

    An integer column is already the packed value. A text column is the ``labid.locid`` form and is
    parsed. Nothing else can be an identity, and a null cannot be one either -- an edge joins two
    elements or it is not an edge.
    """
    import pyarrow

    column = batch.column(name)
    if column.null_count:
        raise ValueError(f"column {name!r} holds a null; an edge needs both of its endpoints")
    kind = column.type
    if pyarrow.types.is_integer(kind):
        return list(column.to_pylist())
    if pyarrow.types.is_string(kind) or pyarrow.types.is_large_string(kind):
        return [parse_text(value).packed for value in column.to_pylist()]
    raise TypeError(
        f"column {name!r} is {kind}, which is not an identity: pass the packed 64-bit value or "
        f"the 'labid.locid' text"
    )


def vertex_payloads(source: Any, *, size: int = CHUNK) -> Iterator[list[bytes]]:
    """The property maps of a columnar source, written as JSON, a chunk at a time.

    What :func:`~agensgraph.bulk.vertex_blocks` turns into a copy stream.
    """
    for batch in _arrow_batches(source, size):
        yield _payloads(batch)


def edge_payloads(
    source: Any, *, start: str = "start", end: str = "end", size: int = CHUNK
) -> Iterator[list[tuple[int, int, bytes]]]:
    """The endpoints and property maps of a columnar source, a chunk at a time.

    Two of the columns are the identities the edge joins, named by *start* and *end*; the rest are
    its properties. An identity is either the packed 64-bit value or the ``labid.locid`` text.
    """
    for batch in _arrow_batches(source, size):
        starts = _endpoints(batch, start)
        ends = _endpoints(batch, end)
        rest = batch.drop_columns([start, end])
        yield list(zip(starts, ends, _payloads(rest), strict=True))
