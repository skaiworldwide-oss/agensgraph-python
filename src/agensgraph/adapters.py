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

from typing import TYPE_CHECKING, Any

from psycopg import postgres, pq
from psycopg.adapt import AdaptersMap, Dumper, Loader
from psycopg.types import TypeInfo
from psycopg.types.json import JsonbBinaryDumper, JsonbDumper
from psycopg.types.string import StrBinaryDumper, StrDumper

from ._protocol import decode
from ._protocol.graphid import GraphId, pack, parse_text
from .errors import StaleLabelCache

if TYPE_CHECKING:
    from psycopg import Connection
    from psycopg.abc import AdaptContext, Buffer

    from ._protocol.labels import LabelCache

__all__ = [
    "OIDS",
    "Unspecified",
    "assert_oids",
    "graph_adapters",
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
    """Take a payload as ``bytes``, copying only when psycopg hands over a view.

    Slicing a ``memoryview`` is cheap and searching one is not -- the generic
    implementation allocates an object per byte compared -- and everything downstream
    searches. psycopg gives ``bytes`` in the ordinary case, so this usually costs a type
    check.
    """
    return data if isinstance(data, bytes) else bytes(data)


class GraphIdLoader(Loader):
    """Read ``labid.locid``."""

    format = pq.Format.TEXT

    def load(self, data: Buffer) -> GraphId:
        return parse_text(_decode_bytes(data))


class GraphIdBinaryLoader(Loader):
    """Read the eight wire bytes of a graph id."""

    format = pq.Format.BINARY

    def load(self, data: Buffer) -> GraphId:
        return GraphId.from_packed(int.from_bytes(_decode_bytes(data), "big"))


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


class UnspecifiedDumper(Dumper):
    """Write a string with no type, leaving the server to decide what it is."""

    format = pq.Format.TEXT
    oid = 0

    def dump(self, obj: str) -> bytes:
        return obj.encode()


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
    adapters.register_dumper(GraphId, GraphIdDumper)
    adapters.register_dumper(GraphId, GraphIdBinaryDumper)
    adapters.register_dumper(dict, JsonbDumper)
    adapters.register_dumper(dict, JsonbBinaryDumper)
    # Saying a string is text is what reaches the server's own text-to-jsonb conversion
    # rather than jsonb's parser. psycopg ships these two for exactly this: its own note on
    # StrDumper is that it is for "where the unknown oid is ambiguous and the text oid is
    # required". The last dumper registered is the one an ordinary placeholder uses, so this
    # displaces the unspecified-oid default without touching psycopg's global map.
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


def assert_oids(conn: Connection[Any]) -> None:
    """Check the written-down oids against the connected server.

    One query, so a caller runs it once for a process rather than once for a connection.
    A mismatch is worth failing on rather than working around: an unregistered type read in
    the binary rendering comes back as raw bytes with no error at all, so a wrong oid is a
    silently wrong result rather than a loud one.
    """
    names = sorted(OIDS)
    rows = conn.execute(
        "SELECT typname, oid FROM pg_type WHERE typname = ANY(%s)", (names,)
    ).fetchall()
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
