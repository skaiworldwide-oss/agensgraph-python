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
    "UpsertCounts",
    "build_identity_map",
    "edge_blocks",
    "edge_copy_statement",
    "freeze_after_import",
    "identity_map_statement",
    "overlap_update_statement",
    "paused_collection",
    "split_by_what_exists",
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
