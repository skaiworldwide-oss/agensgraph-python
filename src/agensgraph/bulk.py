"""Loading a lot of elements at once.

Copying is the fast way in, and the reason is that it is one stream rather than a statement per
row. Measured on this driver against ten thousand vertices: **223,000 rows a second** by copying,
against 140,000 for a single ``UNWIND ... CREATE`` and 47,000 for one statement per row -- so
1.6 times the best Cypher can do and 4.7 times the obvious approach.

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
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from psycopg.types.json import Jsonb

from .cypher import quote_identifier

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable, Mapping

    from ._protocol.graphid import GraphId

__all__ = [
    "edge_copy_statement",
    "freeze_after_import",
    "identity_map_statement",
    "paused_collection",
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
