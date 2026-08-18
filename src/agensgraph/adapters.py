"""Teaching psycopg to read and write the graph types.

The type oids are fixed. Every graph type is built into the server rather than created by
an extension, so its oid is the same in every database on every supported version and can
be written down here -- which matters because the usual way to find an oid is to look it
up per database, and a map built that way is quietly wrong the moment a connection is
reused against another database. :func:`assert_oids` checks the written-down numbers
against a live server, so the assumption is testable rather than merely stated.

Reading the text rendering needs nothing from the server. Reading the binary rendering
needs a label name, which no wire value carries in either format -- the server resolves it
while rendering, from the session's graph path -- so a binary reader has to resolve it from
the label id itself, and that means a cache belonging to one connection. Text loaders are
therefore registered on a map shared by the whole process, and binary loaders are built per
connection around the cache they will ask.

Three kinds of value are sent, and the interesting one is a string.

**A string says that it is text.** psycopg leaves a ``str``'s type unspecified, so that one
string can stand for whatever type the server infers from where it sits. For Cypher that
inference is the wrong one, and quietly so. The parser has two arms for coercing an expression
to jsonb and which it takes depends on whether the parameter arrived typed: an *unspecified*
one is relabelled jsonb and its bytes handed to jsonb's input function, which **parses** them,
so ``'123'`` arrives as the number 123, ``'null'`` as a JSON null, and ``'Arthur'`` as a parse
error. A *typed* one goes through ``cypher_to_jsonb``, whose purpose is converting a value
whose type is known, and which keeps a string a string -- a text value holding ``[9,9]`` holds
those six characters, not a list. Saying the type reaches the second arm, so looking a property
up by name works with the string the caller already has.

That is a correctness fix and not a convenience. Left unspecified, a search for a value that
happens to read as JSON -- an order number, a postcode, a version -- matched nothing and raised
nothing.

The inference is given up in the one place it was doing useful work: a string passed where the
type is not text-like no longer resolves, and says so while the statement is being parsed,
naming both types. ``where d = %s`` with a string wants ``%s::date``, which is psycopg's own
idiom. Passing the value's real type -- a ``date``, a ``UUID``, an ``int`` -- is untouched, as
is every text, varchar and name position, and :class:`Unspecified` gives the old behaviour back
for a single value. Nothing about the plan changes: ``cypher_to_jsonb`` is immutable, so it
folds into the same constant the jsonb form would have been, and the index condition is
identical.

**A mapping is sent as jsonb**, because psycopg cannot adapt a bare ``dict`` at all and nearly
every parameter a Cypher statement takes is read as jsonb.

**A graph id is sent as itself**, for asking about identity.

A ``list`` is deliberately left as a PostgreSQL array, which is what it should be in the plain
SQL a graph connection is still expected to run. A list meant as a JSON array is wrapped, and
says so.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping
from math import isfinite as _isfinite
from typing import TYPE_CHECKING, Any

from psycopg import postgres, pq
from psycopg.adapt import AdaptersMap, Dumper, Loader
from psycopg.types import TypeInfo
from psycopg.types import json as _json
from psycopg.types.string import StrBinaryDumper, StrDumper

from ._protocol import decode
from ._protocol.graphid import GraphId, pack, parse_text, unpack
from .errors import StaleLabelCache
from .numbers import decode_json, encode_json

if TYPE_CHECKING:
    from collections.abc import Sequence

    from psycopg import AsyncConnection, Connection
    from psycopg.abc import AdaptContext, Buffer

    from ._protocol.labels import LabelCache

__all__ = [
    "OIDS",
    "OID_QUERY",
    "Unspecified",
    "assert_oids",
    "async_assert_oids",
    "check_oids",
    "dump_jsonb",
    "graph_adapters",
    "oid_names",
    "register_binary",
    "register_text",
]

OIDS: dict[str, int] = {
    "graphid": 7002,
    "_graphid": 7001,
    "vertex": 7012,
    "_vertex": 7011,
    "edge": 7022,
    "_edge": 7021,
    "graphpath": 7032,
    "_graphpath": 7031,
    "rowid": 7062,
}
"""Every graph type oid, including the ones nothing here reads.

``rowid`` is listed because it exists and must not be mistaken for something missing: it
has neither a binary nor a text conversion of its own, both of which raise on the server,
so no client can read it and none can be written here either.

``_graphpath`` is listed and not registered. An array of paths renders as arrays nested
inside arrays, which the element separator cannot tell apart from an array of elements, and
no Cypher expression produces one.
"""


def _decode_bytes(data: Buffer) -> bytes:
    """Take a payload as ``bytes``, which means copying it.

    Slicing a ``memoryview`` is cheap and searching one is not -- the generic implementation
    allocates an object per byte compared -- and everything downstream searches. The payload
    also has to outlive the result, since a property map is held as the bytes it arrived as
    and read later.

    It is not an exceptional case. This driver depends on psycopg's C extension, and that hands
    over a ``memoryview`` for every value in both renderings -- counted, a hundred out of a
    hundred each way. So the exact type is asked for first and ``tobytes`` is called on it, which
    knows it is copying a view: 108 nanoseconds against 188 for reaching ``bytes()`` through an
    ``isinstance``.
    """
    return data.tobytes() if type(data) is memoryview else bytes(data)


class GraphIdLoader(Loader):
    """Read ``labid.locid``."""

    format = pq.Format.TEXT

    def load(self, data: Buffer) -> GraphId:
        return parse_text(_decode_bytes(data))


class GraphIdBinaryLoader(Loader):
    """Read the eight wire bytes of a graph id.

    Through the module's own reader, which refuses a payload that is not eight bytes. Read as
    a plain integer instead, a truncated one is not short -- it is a different, valid identity,
    and four zero bytes and a one read as ``0.1``.
    """

    format = pq.Format.BINARY

    def load(self, data: Buffer) -> GraphId:
        return unpack(_decode_bytes(data))


class VertexLoader(Loader):
    """Read ``label[id]{properties}``."""

    format = pq.Format.TEXT

    def load(self, data: Buffer) -> object:
        return decode.vertex_from_text(_decode_bytes(data))


class EdgeLoader(Loader):
    """Read ``label[id][start,end]{properties}``."""

    format = pq.Format.TEXT

    def load(self, data: Buffer) -> object:
        return decode.edge_from_text(_decode_bytes(data))


class PathLoader(Loader):
    """Read ``[vertex,edge,vertex,...]``."""

    format = pq.Format.TEXT

    def load(self, data: Buffer) -> object:
        return decode.path_from_text(_decode_bytes(data))


class VertexArrayLoader(Loader):
    """Read ``[vertex,vertex,...]``.

    An array of graph elements does not render in PostgreSQL's array syntax, so this is
    not the ordinary array loader with an element loader inside it.
    """

    format = pq.Format.TEXT

    def load(self, data: Buffer) -> object:
        return decode.vertices_from_text(_decode_bytes(data))


class EdgeArrayLoader(Loader):
    """Read ``[edge,edge,...]``."""

    format = pq.Format.TEXT

    def load(self, data: Buffer) -> object:
        return decode.edges_from_text(_decode_bytes(data))


class Unspecified(str):
    """A string whose type is left for the server to infer, as psycopg does by default.

    Wanted where the inference was doing the work: a string standing for a date, a uuid or a
    number in plain SQL, without a cast written next to it.

    ::

        conn.execute("select 1 where d = %s", (Unspecified("2026-08-05"),))

    Not for a Cypher expression. There the inference resolves to jsonb and the string is
    parsed as JSON, which is the behaviour this type exists to bring back and which a
    property lookup does not want.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return f"Unspecified({str.__repr__(self)})"


_encode_json = encode_json
"""The same encoder the property map is read back through, so that what a decimal is written as
and what it is read as are one decision rather than two."""

_JSONB_OID = 3802
"""Written down like the graph oids: it is a built-in and does not move."""

_SEQUENCES = (list, tuple, set, frozenset)
"""Written down rather than spelled as a union, which builds a type object on every call: 52
nanoseconds against 316."""


def _has_non_finite(obj: Any) -> bool:
    """Whether a value holds a float that jsonb has no way to store."""
    if isinstance(obj, float):
        return not _isfinite(obj)
    if isinstance(obj, Mapping):
        return any(_has_non_finite(value) for value in obj.values())
    if isinstance(obj, _SEQUENCES):
        return any(_has_non_finite(value) for value in obj)
    return False


def dump_jsonb(obj: Any) -> bytes:
    """Render a property map, refusing a float jsonb cannot hold.

    ``NaN`` and the infinities encode as ``null``, which would store the wrong value rather than
    report anything, so they are refused.

    The walk that finds one runs only when the encoded map holds the four letters of ``null``
    anywhere, which is a substring search and not a token test -- so a real ``None``, and a
    string reading ``annulled``, both pay for it. A regex that could tell a token from those
    letters inside a string measured slower than the walk in every case. Measured: a map with
    neither, 0.44 microseconds; one with either, 1.7; a map of 1536 floats beside a ``None``,
    293.
    """
    out = _encode_json(obj)
    if b"null" in out and _has_non_finite(obj):
        raise ValueError(
            "a property map cannot hold NaN or an infinity: jsonb has no way to store one, and "
            "encoding it would silently write null instead"
        )
    return out


class JsonbLoader(_json.JsonbLoader):
    """Read a jsonb value through the same decoder a property map is read through.

    psycopg reads one with the standard library, which has two consequences a graph driver does
    not want. A non-integer comes back as a float whatever :func:`read_numbers_exactly` was
    asked for, so ``RETURN n.p`` and ``RETURN n`` disagreed about the same stored value -- the
    map kept every digit the server holds and the bare column did not. And the standard
    decoder is the slower one.
    """

    def load(self, data: Buffer) -> Any:
        return decode_json(bytes(data))


class JsonbBinaryLoader(_json.JsonbBinaryLoader):
    """The same, for the rendering that puts a version byte in front of the text."""

    def load(self, data: Buffer) -> Any:
        return decode_json(bytes(data)[1:])


class JsonbDumper(_json.JsonbDumper):
    """Send a property map as jsonb."""

    _dumps = staticmethod(dump_jsonb)


class JsonbBinaryDumper(_json.JsonbBinaryDumper):
    """The same in the binary rendering, which is the version byte and then the same text."""

    _dumps = staticmethod(dump_jsonb)


class UnspecifiedDumper(Dumper):
    """Write a string with no type, leaving the server to decide what it is."""

    format = pq.Format.TEXT
    oid = 0

    def dump(self, obj: str) -> bytes:
        return obj.encode()


_pack_float8 = struct.Struct("!d").pack

_NON_FINITE = (
    "a value here cannot be NaN or an infinity: it reaches the server as a float8, and "
    "converting one to jsonb stores the text {0!r} rather than a number. Store the string "
    "yourself if that is what you meant, or leave the property out"
)


def _refuse_non_finite(obj: float) -> None:
    """Refuse a float jsonb has no way to hold.

    Every Cypher expression is jsonb, and jsonb has no NaN and no infinity, so one sent as a
    float8 arrives as a *string* of its name. Refused for the same reason a property map
    holding one is refused: what comes back is not what was sent.
    """
    if not _isfinite(obj):
        raise ValueError(_NON_FINITE.format(str(obj)))


class FloatDumper(Dumper):
    """Write a float, refusing what jsonb cannot hold.

    Written out rather than derived from psycopg's, which exists to render the three values
    refused here. What is left is what ``repr`` produces, which is what psycopg writes for
    every finite float.
    """

    format = pq.Format.TEXT
    oid = postgres.types["float8"].oid

    def dump(self, obj: float) -> bytes:
        _refuse_non_finite(obj)
        return repr(obj).encode()


class FloatBinaryDumper(Dumper):
    """The same, as the eight bytes ``float8send`` writes."""

    format = pq.Format.BINARY
    oid = postgres.types["float8"].oid

    def dump(self, obj: float) -> bytes:
        _refuse_non_finite(obj)
        return _pack_float8(obj)


class GraphIdDumper(Dumper):
    """Write ``labid.locid``."""

    format = pq.Format.TEXT
    oid = OIDS["graphid"]

    def dump(self, obj: GraphId) -> bytes:
        return str(obj).encode()


class GraphIdBinaryDumper(Dumper):
    """Write the eight wire bytes of a graph id."""

    format = pq.Format.BINARY
    oid = OIDS["graphid"]

    def dump(self, obj: GraphId) -> bytes:
        return pack(obj)


def register_text(context: AdaptContext | AdaptersMap) -> None:
    """Read the graph types in the text rendering, and write a graph id.

    This is everything a connection needs to read an ordinary query, and it needs nothing
    from the server to do it.
    """
    adapters = context if isinstance(context, AdaptersMap) else context.adapters
    # The types are named first, and the loaders registered over the top of them.
    #
    # Naming them is what lets binary copying be told what a column is: it applies no
    # conversions of its own, so it looks a column's type up by name and a type it cannot find
    # cannot be copied into. But naming a type also gives it psycopg's own array support, and
    # psycopg's array reader expects PostgreSQL's array syntax -- which an array of graph
    # elements is not written in. So the loaders go on afterwards, and the last one registered
    # is the one used.
    for name in ("graphid", "vertex", "edge", "graphpath", "rowid"):
        type_info(name).register(adapters)
    adapters.register_loader(OIDS["graphid"], GraphIdLoader)
    adapters.register_loader(OIDS["graphid"], GraphIdBinaryLoader)
    adapters.register_loader(OIDS["vertex"], VertexLoader)
    adapters.register_loader(OIDS["edge"], EdgeLoader)
    adapters.register_loader(OIDS["graphpath"], PathLoader)
    adapters.register_loader(OIDS["_vertex"], VertexArrayLoader)
    adapters.register_loader(OIDS["_edge"], EdgeArrayLoader)
    adapters.register_loader(_JSONB_OID, JsonbLoader)
    adapters.register_loader(_JSONB_OID, JsonbBinaryLoader)
    adapters.register_dumper(GraphId, GraphIdDumper)
    adapters.register_dumper(GraphId, GraphIdBinaryDumper)
    adapters.register_dumper(dict, JsonbDumper)
    adapters.register_dumper(dict, JsonbBinaryDumper)
    # Saying a string is text is what reaches the server's own text-to-jsonb conversion
    # rather than jsonb's parser. psycopg ships these two for exactly this: its own note on
    # StrDumper is that it is for "where the unknown oid is ambiguous and the text oid is
    # required". The last dumper registered is the one an ordinary placeholder uses, so this
    # displaces the unspecified-oid default without touching psycopg's global map.
    adapters.register_dumper(float, FloatDumper)
    adapters.register_dumper(float, FloatBinaryDumper)
    adapters.register_dumper(str, StrDumper)
    adapters.register_dumper(str, StrBinaryDumper)
    adapters.register_dumper(Unspecified, UnspecifiedDumper)


def register_binary(context: AdaptContext, labels: LabelCache) -> None:
    """Also read the composite renderings, resolving label names through *labels*.

    The loaders are built here rather than written above because each one closes over the
    cache it asks, and a cache belongs to a single connection. A label the cache has not
    seen is one created since it was filled, which it says so rather than guessing at.

    Asking for the binary rendering is a decision about the whole result and not about one
    column: a single column of a type with no binary conversion fails the entire result, so
    a caller opts in per statement and only for statements whose columns are known.
    """

    def resolve(labid: int) -> str:
        name = labels.get(labid)
        if name is None:
            raise StaleLabelCache.for_label(labid, graph=labels.graph)
        return name

    class _VertexBinaryLoader(Loader):
        format = pq.Format.BINARY

        def load(self, data: Buffer) -> object:
            return decode.vertex_from_binary(_decode_bytes(data), resolve)

    class _EdgeBinaryLoader(Loader):
        format = pq.Format.BINARY

        def load(self, data: Buffer) -> object:
            return decode.edge_from_binary(_decode_bytes(data), resolve)

    class _PathBinaryLoader(Loader):
        format = pq.Format.BINARY

        def load(self, data: Buffer) -> object:
            return decode.path_from_binary(_decode_bytes(data), resolve)

    class _VertexArrayBinaryLoader(Loader):
        format = pq.Format.BINARY

        def load(self, data: Buffer) -> object:
            return decode.vertices_from_binary(_decode_bytes(data), resolve)

    class _EdgeArrayBinaryLoader(Loader):
        format = pq.Format.BINARY

        def load(self, data: Buffer) -> object:
            return decode.edges_from_binary(_decode_bytes(data), resolve)

    adapters = context.adapters
    adapters.register_loader(OIDS["vertex"], _VertexBinaryLoader)
    adapters.register_loader(OIDS["edge"], _EdgeBinaryLoader)
    adapters.register_loader(OIDS["graphpath"], _PathBinaryLoader)
    adapters.register_loader(OIDS["_vertex"], _VertexArrayBinaryLoader)
    adapters.register_loader(OIDS["_edge"], _EdgeArrayBinaryLoader)


def graph_adapters() -> AdaptersMap:
    """A copy of psycopg's own adapters with the graph types added.

    Taking a copy leaves psycopg's global map alone, so a plain PostgreSQL connection made
    by the same process is unaffected by anything registered here.
    """
    adapters = AdaptersMap(postgres.adapters)
    register_text(adapters)
    return adapters


OID_QUERY = "SELECT typname, oid FROM pg_type WHERE typname = ANY(%s)"
"""The statement that reads the graph types' oids, taking the names to look for."""


def oid_names() -> list[str]:
    """The type names to ask the server about."""
    return sorted(OIDS)


def assert_oids(conn: Connection[Any]) -> None:
    """Check the written-down oids against the connected server.

    One query, so a caller runs it once for a process rather than once for a connection.
    A mismatch is worth failing on rather than working around: an unregistered type read in
    the binary rendering comes back as raw bytes with no error at all, so a wrong oid is a
    silently wrong result rather than a loud one.
    """
    check_oids(conn.execute(OID_QUERY, (oid_names(),)).fetchall())


async def async_assert_oids(conn: AsyncConnection[Any]) -> None:
    """Check the written-down oids against the connected server, waiting on it.

    The same check as :func:`assert_oids`, for a caller whose connection has to be waited on.
    Both hand the rows to the same reading of them, so the two cannot drift apart.
    """
    cursor = await conn.execute(OID_QUERY, (oid_names(),))
    check_oids(await cursor.fetchall())


def check_oids(rows: Sequence[Sequence[Any]]) -> None:
    """Fail if the rows the server sent do not name every graph type at the oid written down."""
    names = oid_names()
    found = {str(name): int(oid) for name, oid in rows}
    missing = [name for name in names if name not in found]
    if missing:
        raise AssertionError(f"the server has no graph types named {missing}")
    wrong = {
        name: (expected, found[name])
        for name, expected in OIDS.items()
        if found[name] != expected
    }
    if wrong:
        detail = ", ".join(
            f"{name} is {actual} and not {expected}"
            for name, (expected, actual) in wrong.items()
        )
        raise AssertionError(
            f"the server's graph type oids are not the expected ones: {detail}"
        )


def type_info(name: str) -> TypeInfo:
    """The oid pair for a graph type, for the psycopg interfaces that ask for one.

    Binary copying needs a type's oid stated rather than inferred, because it applies no
    conversions of its own.
    """
    if name not in OIDS:
        raise KeyError(f"no graph type named {name!r}")
    return TypeInfo(name, OIDS[name], OIDS.get(f"_{name}", 0))
