"""What a write changed.

The server keeps five counters for the last graph write and offers them through a function,
so reading them is a second statement rather than something that arrives with the first.

They are reset unevenly, and that is the whole difficulty. A write with no ``RETURN`` is a
graph write all the way to the top of the plan, and starting one zeroes all five. A write
*with* a ``RETURN`` is a select at the top, so only the counters belonging to the clauses it
actually has are zeroed -- the inserts for a ``CREATE`` or ``MERGE`` pattern, the deletes for
a ``DELETE``, the updates for a ``SET``. Whatever the rest held from an earlier statement is
still sitting there.

Measured against a live server, in one session: ``CREATE`` of two vertices reports
``(2, 0, 0, 0, 0)``; a following ``SET ... RETURN`` of two properties reports
``(2, 0, 0, 0, 2)``, and that leading 2 is the earlier statement's. So a summary that
reported all five after the second statement would say two vertices had been created by a
statement that created none.

Nothing here reports a number it cannot account for. A counter it cannot vouch for is
``None``, which is not the same as zero and does not read like it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["COUNTER_COLUMNS", "COUNTER_QUERY", "GraphWriteCounts"]

COUNTER_QUERY = "select * from get_last_graph_write_stats()"
"""The statement that reads the counters. One row of five ``bigint`` columns."""

COUNTER_COLUMNS = (
    "insertedvertices",
    "insertededges",
    "deletedvertices",
    "deletededges",
    "updatedproperties",
)
"""The columns, in the order the server returns them."""


class GraphWriteCounts(NamedTuple):
    """Five counters, each either a number this statement is answerable for or ``None``."""

    inserted_vertices: int | None
    inserted_edges: int | None
    deleted_vertices: int | None
    deleted_edges: int | None
    updated_properties: int | None

    @classmethod
    def unknown(cls) -> GraphWriteCounts:
        """Nothing to report, because nothing was read."""
        return cls(None, None, None, None, None)

    @classmethod
    def exact(cls, after: Sequence[int]) -> GraphWriteCounts:
        """All five counters, for a statement that zeroed all five before it ran.

        That is a write with no ``RETURN``, which the server treats as a graph write to the
        top of the plan.
        """
        if len(after) != 5:
            raise ValueError(f"expected five counters, got {len(after)}")
        return cls(*(int(value) for value in after))

    @classmethod
    def between(cls, before: Sequence[int], after: Sequence[int]) -> GraphWriteCounts:
        """Only the counters a statement can be held to, for a write that returned rows.

        A counter is reported when it changed, since only the statement could have changed
        it, and when it was already zero, since a counter the statement zeroed and a counter
        that was zero already are both zero. It is left unreported when it is unchanged and
        was not zero, which is the one case where a stale number and a real one look alike.
        """
        if len(before) != 5 or len(after) != 5:
            raise ValueError(f"expected five counters each, got {len(before)} and {len(after)}")
        return cls(
            *(
                int(now) if now != then or then == 0 else None
                for then, now in zip(before, after, strict=True)
            )
        )

    @property
    def complete(self) -> bool:
        """Whether every counter is answered for."""
        return all(value is not None for value in self)

    @property
    def total(self) -> int | None:
        """Everything the statement changed, or ``None`` if any counter is unanswered.

        The command tag of a write with no ``RETURN`` carries this same sum, whatever it was
        that changed, which is why the tag is never read as a count of anything in
        particular.
        """
        if not self.complete:
            return None
        return sum(value for value in self if value is not None)
