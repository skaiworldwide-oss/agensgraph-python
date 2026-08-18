"""Loading a lot of elements at once.

Copying is the fast way in, and the reason is that it is one stream rather than a statement per
row. Measured on this driver against twenty thousand vertices: **225,000 rows a second** by
copying, against 140,000 for a single ``UNWIND ... CREATE`` and 5,400 for one statement per row --
so 1.6 times the best Cypher can do and **thirty-two** times the obvious approach.

An identity does not have to be supplied. A label table's ``id`` column has a default that builds
the graph id from the label's own id and the label's sequence, so copying only the property map
produces exactly the identities a ``CREATE`` would have. That removes the whole business of
generating identities client-side and keeping them unique.

An edge is different, because an edge has to say which two elements it joins, and those are
identities rather than anything in the source data. So loading edges is two steps: read the map
from whatever the source data calls an element to the identity the server gave it, then copy the
edges with the identities filled in. The map is read in one statement rather than one per edge.

Binary copying applies **no** conversions, so every column's type has to be stated. That is why
this module exists rather than a line of documentation: a caller who states them wrongly gets
either a refusal or, worse, bytes read as the wrong type.
"""

from __future__ import annotations

import gc
import struct
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, NamedTuple

from psycopg.types.json import Jsonb

from .cypher import quote_identifier

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Iterator, Mapping, Sequence

    from ._protocol.graphid import GraphId

__all__ = [
    "BLOCK_SIZE",
    "EDGE_LABEL_FACTS_QUERY",
    "PROMOTED_KEY_TYPES",
    "UpsertCounts",
    "build_identity_map",
    "edge_blocks",
    "edge_copy_statement",
    "edge_overlap_update_statement",
    "edge_pairs_all_query",
    "edge_pairs_present_query",
    "freeze_after_import",
    "identity_map_statement",
    "key_spellings",
    "keyed_identity_query",
    "overlap_update_statement",
    "paused_collection",
    "promoted_identity_map_statement",
    "split_by_what_exists",
    "split_edges_by_what_exists",
    "vertex_blocks",
    "vertex_copy_statement",
]


@contextmanager
def paused_collection() -> Generator[None]:
    """Stop the cyclic collector for the duration of a large read or load.

    Worth about 1.03 times on a read of two hundred thousand vertices. It is no more than that
    because a row is a struct the collector does not track, so a result is almost entirely invisible
    to it -- the same read against a row built as an ordinary object was 1.69 times.

    Reference counting still frees whatever stops being referenced. What waits is the collection of
    cycles, so a read that builds cyclic structures holds them until this returns. Nothing is
    collected on the way out beyond what the collector would do next of its own accord.
    """
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def freeze_after_import() -> None:
    """Move everything alive now into a generation the collector will not walk again.

    For calling once at startup, after the imports a process is going to do. Every later collection
    then skips the module-level objects that were never going to be collected.
    """
    gc.freeze()


def vertex_copy_statement(graph: str, label: str) -> str:
    """How to copy vertices into a label.

    Only the property map is copied. The identity column has a default that produces the same
    identities a ``CREATE`` would, so supplying one would mean reproducing the server's own
    numbering and keeping it unique from outside.
    """
    return (
        f"copy {quote_identifier(graph)}.{quote_identifier(label)} (properties) "
        f"from stdin (format binary)"
    )


def edge_copy_statement(graph: str, label: str) -> str:
    """How to copy edges into a label.

    The two endpoints are given and the identity is not, for the same reason as a vertex. ``end``
    is a reserved word, so it is quoted -- which the quoting does without being asked.
    """
    return (
        f"copy {quote_identifier(graph)}.{quote_identifier(label)} "
        f'(start, "end", properties) from stdin (format binary)'
    )


VERTEX_COLUMN_TYPES = ["jsonb"]
"""What a vertex copy sends. Stated because binary copying converts nothing."""

EDGE_COLUMN_TYPES = ["graphid", "graphid", "jsonb"]
"""What an edge copy sends: the two endpoints, then the property map."""


BLOCK_SIZE = 1 << 16
"""How much of the copy stream is handed to the socket at a time."""

_COPY_SIGNATURE = b"PGCOPY\n\xff\r\n\0" + b"\0" * 8
"""What a binary copy stream opens with: the signature, then a word of flags and a word of length,
both zero."""

_COPY_TRAILER = b"\xff\xff"
"""A field count of -1, which is how a binary copy stream ends."""

_JSONB_VERSION = b"\x01"
"""The version byte a binary jsonb value carries ahead of its text."""

_ONE_FIELD = struct.Struct("!hi").pack
"""A field count of one, and that field's length."""

_THREE_FIELDS = struct.Struct("!h").pack(3)

_LENGTH = struct.Struct("!i").pack
_ENDPOINT = struct.Struct("!iQ").pack
"""A graphid field: its length, always eight, and the packed value."""


def vertex_blocks(payloads: Iterable[bytes], *, block: int = BLOCK_SIZE) -> Iterator[bytes]:
    """The whole binary copy stream for a vertex copy, from property maps already written as JSON.

    For ``copy.write()`` rather than ``copy.write_row()``: psycopg's binary formatter treats a
    written block as the complete format, so the signature and the trailer are here and it adds
    neither. Nothing is converted on the way through -- a payload is the JSON text of one property
    map, and the version byte a binary jsonb value needs is prepended.
    """
    out = bytearray(_COPY_SIGNATURE)
    for payload in payloads:
        out += _ONE_FIELD(1, len(payload) + 1)
        out += _JSONB_VERSION
        out += payload
        if len(out) >= block:
            yield bytes(out)
            out.clear()
    out += _COPY_TRAILER
    yield bytes(out)


def edge_blocks(
    rows: Iterable[tuple[int, int, bytes]], *, block: int = BLOCK_SIZE
) -> Iterator[bytes]:
    """The same for an edge copy, from packed endpoint identities and JSON property maps.

    An endpoint is the identity's single 64-bit value, as :attr:`~agensgraph.GraphId.packed` gives
    it, because that is what the eight wire bytes of a graphid hold.
    """
    out = bytearray(_COPY_SIGNATURE)
    for start, end, payload in rows:
        out += _THREE_FIELDS
        out += _ENDPOINT(8, start)
        out += _ENDPOINT(8, end)
        out += _LENGTH(len(payload) + 1)
        out += _JSONB_VERSION
        out += payload
        if len(out) >= block:
            yield bytes(out)
            out.clear()
    out += _COPY_TRAILER
    yield bytes(out)


def promoted_identity_map_statement(graph: str, label: str, key: str) -> str:
    """Read the whole label's map off a key that has a column of its own.

    A promoted key is a column beside the property map rather than inside it, so reading it touches
    no map at all: on 20,000 elements each carrying a 1536-dimension embedding, 3 milliseconds and
    163 buffers against 731 and 60,183 for the same keys out of the map.

    Only for a type whose text reading matches what the map would have given. A boolean does not --
    the column reads back as Python's ``True`` where the map gives ``true`` -- so the caller checks
    the type before choosing this.
    """
    table = f"{quote_identifier(graph)}.{quote_identifier(label)}"
    return f"select {quote_identifier(key)}, id from {table}"


PROMOTED_KEY_TYPES = frozenset(
    {
        "text",
        "character varying",
        "bigint",
        "integer",
        "smallint",
        "numeric",
        "double precision",
        "real",
    }
)
"""Column types whose text reading is the one the property map would have given.

Established by storing a value of each and comparing the two readings. ``boolean`` is the one that
differs and is absent: a column gives Python's ``True`` where the map gives ``true``.
"""


def identity_map_statement(graph: str, label: str) -> str:
    """Read what the server called each element, keyed by a property of the caller's choosing.

    One statement for the whole label rather than one per edge. The key is read as text, because
    a key that is a number in the source data and a string in the property map would otherwise
    match nothing -- and reading both sides as text is the one comparison that always holds.
    """
    return (
        f"select properties ->> %s, id from {quote_identifier(graph)}.{quote_identifier(label)}"
    )


def vertex_rows(properties: Iterable[Mapping[str, Any]]) -> Iterable[list[Any]]:
    """Turn property maps into the rows a vertex copy sends."""
    return ([Jsonb(dict(each))] for each in properties)


def edge_rows(
    edges: Iterable[tuple[GraphId, GraphId, Mapping[str, Any] | None]],
) -> Iterable[list[Any]]:
    """Turn endpoint pairs and property maps into the rows an edge copy sends."""
    return (
        [start, end, Jsonb(dict(props) if props is not None else {})]
        for start, end, props in edges
    )


def build_identity_map(
    rows: Sequence[tuple[Any, Any]], *, label: str, key: str
) -> dict[str, GraphId]:
    """The identities of a label, keyed by a property, refusing a key that does not identify.

    Two elements sharing a key, or one holding none, would each cost an entry. What is lost is
    not the entry: an edge resolved through this map lands on whichever element survived, or on
    nothing, and neither says so.
    """
    found: dict[str, GraphId] = {}
    repeated: set[str] = set()
    missing = 0
    for value, identity in rows:
        if value is None:
            missing += 1
            continue
        text = str(value)
        if text in found:
            repeated.add(text)
        found[text] = identity
    if repeated or missing:
        raise ValueError(
            f"{key!r} does not identify an element of {label!r}: "
            + ", ".join(
                part
                for part in (
                    f"{len(repeated)} value(s) shared by more than one element "
                    f"({sorted(repeated)[:5]})"
                    if repeated
                    else "",
                    f"{missing} element(s) hold no {key!r}" if missing else "",
                )
                if part
            )
            + ". An edge resolved through this map would land on the wrong element or on none"
        )
    return found


class UpsertCounts(NamedTuple):
    """What an upsert did, split by which half of it did the work."""

    inserted: int
    updated: int


def overlap_update_statement(label: str) -> str:
    """Update elements already present, addressed by the identity they already have.

    One statement for the whole overlap. The identity is what makes it cheap: an ``id()`` filter
    over an unwound list plans as an index scan on the label's primary key, one probe per row --
    measured, 20,000 elements in a third of a second against nine times that for a statement each.

    ``+=`` rather than ``=``, so a property the caller did not mention keeps the value it had.
    """
    return (
        f"unwind %s::jsonb as r "
        f"match (n:{quote_identifier(label)}) where id(n) = (r->>'id')::graphid "
        f"set n += r->'props'"
    )


def split_by_what_exists(
    rows: Sequence[Mapping[str, Any]], key: str, known: Mapping[str, GraphId]
) -> tuple[list[Mapping[str, Any]], list[dict[str, Any]]]:
    """The rows that are new, and the updates for the rows that are not.

    The key is read the way :func:`build_identity_map` wrote it, as text, so a key that is a number
    in one and a string in the other still finds its element rather than quietly making a second.
    """
    fresh: list[Mapping[str, Any]] = []
    updates: list[dict[str, Any]] = []
    for row in rows:
        if key not in row:
            raise ValueError(
                f"a row has no {key!r} to identify it by, so it can be neither matched to an "
                f"element nor safely written as a new one"
            )
        value = row[key]
        if value is None:
            raise ValueError(f"a row has {key!r} set to null, which identifies nothing")
        found = known.get(str(value))
        if found is None:
            fresh.append(row)
        else:
            updates.append({"id": str(found), "props": dict(row)})
    return fresh, updates


def key_spellings(value: Any) -> list[Any]:
    """Every JSON form of a key that the text reading of a property would match.

    The full read compares text on both sides: the server extracts the property with ``->>``,
    which renders a number and a string alike, and the caller's value goes through ``str``. So a
    key stored as the number ``1`` and one given as ``"1"`` are the same key.

    A lookup by property compares JSON to JSON, where they are not. Asking for both forms is what
    keeps the two routes agreeing, and it costs one more index probe per key rather than a pass
    over the label.
    """
    text = str(value)
    forms: list[Any] = [text]
    number = _as_number(text)
    if number is not None:
        forms.append(number)
    return forms


def _as_number(text: str) -> int | float | None:
    """The number this text stands for, if a number renders back to exactly this text."""
    try:
        whole = int(text)
    except ValueError:
        pass
    else:
        return whole if str(whole) == text else None
    try:
        fractional = float(text)
    except ValueError:
        return None
    return fractional if str(fractional) == text else None


def keyed_identity_query(label: str, key: str) -> str:
    """Read the identity of named elements, without reading a property of any of them.

    ``k`` comes back rather than the stored property, and that is the point: a property map is one
    column, so reading any key out of it reassembles the whole map, TOAST and all. Handing back the
    key that was asked for needs the index and the identity, and touches no map at all. On a label
    of 30,000 elements each carrying a 1536-dimension embedding, that is the difference between
    90,000 buffers and a few hundred.
    """
    return (
        f"unwind %s::jsonb as k "
        f"match (n:{quote_identifier(label)} {{{quote_identifier(key)}: k}}) "
        f"return k, id(n)"
    )


EDGE_LABEL_FACTS_QUERY = """
select (select count(*) > 0
          from pg_catalog.pg_index x
         where x.indrelid = l.relid
           and x.indisunique
           and x.indexprs is null
           and (select array_agg(a.attname::text order by k.ord)
                  from unnest(x.indkey::smallint[]) with ordinality as k(attnum, ord)
                  join pg_catalog.pg_attribute a
                    on a.attrelid = l.relid and a.attnum = k.attnum) = array['start', 'end']),
       c.reltuples
from pg_catalog.ag_graph g
join pg_catalog.ag_label l on l.graphid = g.oid
join pg_catalog.pg_class c on c.oid = l.relid
where g.graphname = %s
  and l.labname = %s::name
"""
"""Whether one edge per pair of endpoints is kept, and roughly how many edges there are.

Two answers in one read because both are wanted at the same moment and neither is worth a round
trip of its own: the first decides whether the upsert is safe to do at all, the second which shape
of read is cheaper. Measured at 288 microseconds for the pair.

The uniqueness half is read from ``pg_index`` and not through the driver's own index reader, which
cannot answer it: that one admits an expression index or one over a promoted column, being the
conditions the server's own view uses, and the endpoints are ordinary columns. ``indexprs is null``
is what separates a real index over the columns from a property index over properties that happen to
be called ``start`` and ``end``, which the catalogs otherwise print identically.

The size half is the planner's estimate rather than a count, because a count of a large label costs
more than the read it is choosing between. It is ``-1`` on a label nobody has analysed, which is read
as unknown.
"""


def edge_pairs_present_query(graph: str, label: str) -> str:
    """Which of the endpoint pairs asked about already have an edge, and what it is called.

    Written as ``in (select ...)`` rather than a join against the unwound arrays, because the two
    plan differently and only this one reaches an index: measured over 100,000 edges asking about
    1,000 pairs, the join hashed the whole label for 26.6 ms and this took 6.1 ms on an index only
    scan, which is within a sixth of what forcing the scan off achieves.

    The scan is *not* forced off here, unlike the read that looks a vertex up by property. That one
    has to be, because its cost is a property expression over a map the planner cannot see the TOAST
    of. These are two fixed-width columns with real statistics, and the planner was right at every
    size measured: an index up to a thousand pairs, and a sequential scan past that, where a
    sequential scan is genuinely the cheaper of the two.
    """
    return (
        f'select e.start, e."end", e.id '
        f"from {quote_identifier(graph)}.{quote_identifier(label)} e "
        f'where (e.start, e."end") in '
        f"(select s, t from unnest(%s::graphid[], %s::graphid[]) as w(s, t))"
    )


def edge_overlap_update_statement(label: str) -> str:
    """Update edges already present, addressed by the identity the read found.

    A relationship pattern and not ``(e:label)``: an edge label named as a vertex is refused,
    ``label "links" is edge label``.

    ``+=`` rather than ``=``, so a property the caller did not mention keeps the value it had.
    """
    return (
        f"unwind %s::jsonb as r "
        f"match ()-[e:{quote_identifier(label)}]->() "
        f"where id(e) = (r->>'id')::graphid "
        f"set e += r->'props'"
    )


def split_edges_by_what_exists(
    edges: Sequence[tuple[GraphId, GraphId, Mapping[str, Any] | None]],
    present: Mapping[tuple[GraphId, GraphId], GraphId],
) -> tuple[
    list[tuple[GraphId, GraphId, Mapping[str, Any] | None]],
    list[dict[str, Any]],
]:
    """The edges that are not there, and the updates for the ones that are.

    An endpoint pair given twice in one call is written once, and the repeat is dropped. It cannot
    be an update either: the edge it would update is in the same copy and has no identity yet. The
    copy reads nothing back, so a repeat left in would be written as the second edge this exists to
    prevent.
    """
    fresh: list[tuple[GraphId, GraphId, Mapping[str, Any] | None]] = []
    updates: list[dict[str, Any]] = []
    seen: set[tuple[GraphId, GraphId]] = set()
    for start, end, properties in edges:
        if start is None or end is None:
            raise ValueError("an edge joins two elements, and one of these is null")
        pair = (start, end)
        found = present.get(pair)
        if found is not None:
            if properties:
                updates.append({"id": str(found), "props": dict(properties)})
        elif pair not in seen:
            seen.add(pair)
            fresh.append((start, end, properties))
    return fresh, updates


def edge_pairs_all_query(graph: str, label: str) -> str:
    """Every edge of a label as its endpoints and its identity.

    Cheaper than asking about the pairs once the batch is as large as the label, because asking
    sends two identities per pair and this sends none. Measured against a label of 20,000 edges:
    asking took 3.3 ms for 100 pairs, 8.0 for 1,000 and 33.1 for 5,000, all beating the 53.7 this
    takes whatever is asked -- and 143.7 for 20,000, which does not.
    """
    return (
        f'select e.start, e."end", e.id '
        f"from {quote_identifier(graph)}.{quote_identifier(label)} e"
    )
