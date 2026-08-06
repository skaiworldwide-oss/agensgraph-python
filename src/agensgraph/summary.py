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

import enum
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

__all__ = [
    "ASSIGNED_TRANSACTION_QUERY",
    "COUNTER_COLUMNS",
    "COUNTER_QUERY",
    "TRANSACTION_ID_QUERY",
    "TRANSACTION_STATUS_QUERY",
    "CommitOutcome",
    "GraphWriteCounts",
    "read_outcome",
]

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
        """Only the counters a statement can be held to, from two readings alone.

        A counter is reported when it changed, since only the statement could have changed
        it, and when it was already zero, since a counter the statement zeroed and a counter
        that was zero already are both zero. It is left unreported when it is unchanged and
        was not zero, which is the one case where a stale number and a real one look alike.

        :meth:`for_statement` answers more of them, by reading which counters the statement's
        own clauses could move.
        """
        if len(before) != 5 or len(after) != 5:
            raise ValueError(f"expected five counters each, got {len(before)} and {len(after)}")
        return cls(
            *(
                int(now) if now != then or then == 0 else None
                for then, now in zip(before, after, strict=True)
            )
        )

    @classmethod
    def for_statement(
        cls, before: Sequence[int], after: Sequence[int], movable: Collection[int]
    ) -> GraphWriteCounts:
        """The counters, with the statement's own clauses settling what two readings cannot.

        The two readings come first and are never overruled: a counter that moved was moved by
        this statement, whatever its clauses appear to say, because nothing else ran in
        between. A statement can write without naming a clause -- a function it calls may run
        one -- so a counter that changed is reported even when no clause of the text explains
        it.

        *movable* settles the one case two readings cannot: a counter that did not move and
        was not already nought. If no clause of this statement can write that counter, the
        server never zeroed it and the statement never wrote it, so it is nought for this
        statement. If a clause can, a stale number and a real one look alike and it goes
        unreported.
        """
        if len(before) != 5 or len(after) != 5:
            raise ValueError(f"expected five counters each, got {len(before)} and {len(after)}")
        held = cls.between(before, after)
        return cls(
            *(
                value if value is not None else (None if i in movable else 0)
                for i, value in enumerate(held)
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


TRANSACTION_ID_QUERY = "select pg_current_xact_id()::text"
"""Assign this transaction an id, and read it, so its fate can be asked about later.

Asked as text because the id is what has to be sent back, and sending an integer back is
refused: there is no cast from any integer type to the one the server wants.
"""

ASSIGNED_TRANSACTION_QUERY = "select pg_current_xact_id_if_assigned()::text"
"""The id this transaction already has, or nothing if it has not needed one.

A read is given no id at all -- verified -- so this distinguishes a transaction that wrote
from one that only looked, without assigning an id to the one that only looked.
"""

TRANSACTION_STATUS_QUERY = "select pg_xact_status(%s::xid8)"
"""What became of a transaction, asked from a connection that outlived it."""


class CommitOutcome(enum.Enum):
    """What became of a transaction whose commit was interrupted.

    The one failure that cannot be retried is a commit whose outcome nobody knows: retrying
    might apply a write twice, and not retrying might lose it. The server can be asked, from
    another connection, which turns the unanswerable question into an ordinary one.
    """

    COMMITTED = "committed"
    """It landed. There is nothing to retry and nothing was lost."""

    ABORTED = "aborted"
    """It did not land, and nothing of it remains. Safe to run again."""

    IN_PROGRESS = "in progress"
    """Still running somewhere, so it is too early to say. Wait and ask again."""

    UNKNOWN = "unknown"
    """The server cannot say, because the record has been truncated away."""

    @property
    def is_settled(self) -> bool:
        """Whether the question has an answer yet."""
        return self in (CommitOutcome.COMMITTED, CommitOutcome.ABORTED)

    @property
    def safe_to_retry(self) -> bool:
        """Whether the transaction can be run again without applying anything twice.

        True for one outcome only. A transaction known to have aborted left nothing behind, so
        running it again applies it once. Every other answer -- committed, still running, or no
        longer on record -- either applied it already or has not said.
        """
        return self is CommitOutcome.ABORTED


def read_outcome(reported: str | None) -> CommitOutcome:
    """Read what the server said about a transaction.

    A transaction old enough that its record has been truncated reports nothing at all, and
    that is not the same as aborted: it is the server saying it can no longer tell, which has
    to be reported as such rather than guessed either way.
    """
    if reported is None:
        return CommitOutcome.UNKNOWN
    try:
        return CommitOutcome(reported)
    except ValueError:
        return CommitOutcome.UNKNOWN
