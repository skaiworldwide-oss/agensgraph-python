# This file is generated from connection_async.py by tools/async_to_sync.py.
# Edit that file and run the tool; edits made here are lost on the next run.

"""A connection to a graph.

This module is written once, in the awaiting form, and the blocking one is produced from it
by ``tools/async_to_sync.py``. Whichever of the two you are reading, it says the same thing;
the difference is whether the four methods below are waited on.

Everything that does not wait for the server is in :mod:`agensgraph._core` and is shared, in
one copy, by both. What is here is the four things that do wait: making a connection,
selecting a graph, running a statement, and reading the label table again.

The transaction model is psycopg's, unchanged. A statement outside a transaction opens one,
commit and rollback are on the connection, closing rolls back, and ``autocommit`` is how a
statement that cannot run inside a transaction gets to run. There is nothing about a graph
that calls for a different model, and a driver that invents one costs its users everything
they already know.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import psycopg
from psycopg.rows import Row, tuple_row

from ._core import GraphMixin, Result
from .summary import (
    ASSIGNED_TRANSACTION_QUERY,
    COUNTER_QUERY,
    TRANSACTION_ID_QUERY,
    TRANSACTION_STATUS_QUERY,
    CommitOutcome,
    read_outcome,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from psycopg.abc import Params
    from psycopg.rows import RowFactory

    from ._core import Statement
__all__ = ["Connection"]


class Connection(GraphMixin, psycopg.Connection[Row]):
    """A connection that reads the graph types.

    Built on psycopg's, so everything psycopg offers is here unchanged -- cursors, server
    cursors, ``COPY``, pipelines, ``LISTEN``, and plain SQL for the things Cypher has no
    syntax for. What is added is the graph types, a graph to read them from, and a statement
    check for the one shape the server would misread.
    """

    @classmethod
    def connect(cls, conninfo: str = "", **kwargs: Any) -> Connection[Any]:
        """Open a connection and refuse a server this driver cannot read.

        The version arrives in the startup packet, so the refusal costs no round trip and
        happens before the first statement rather than at whichever later one first wants a
        catalog the server has never had.
        """
        conn = super().connect(conninfo, **kwargs)
        try:
            _ = conn.capabilities
        except BaseException:
            conn.close()
            raise
        return conn

    def graph(self, name: str) -> None:
        """Read from a graph, and fill the label table for it.

        Two statements, and only when the table is not already this graph's -- a connection
        taken from a pool and pointed at the graph it is already reading costs nothing. The
        table is what the composite rendering needs to name a label, and filling it here
        means asking for that rendering later does not have to stop and ask.
        """
        if self.labels.graph == name:
            self._run(self._select_graph_statement(name))
            return
        self._run(self._select_graph_statement(name))
        rows = self._fetch(self._label_statement(), (name,))
        self._accept_labels(name, rows)

    def refresh_labels(self) -> None:
        """Fill the label table again, after creating or dropping a label.

        Needed only for the composite rendering, and only for a label created since the
        table was filled. Nothing does this by itself: re-running the statement that hit an
        unknown label would repeat whatever else it did, which for a write that returned rows
        is not something a driver may decide on a caller's behalf.
        """
        graph = self.labels.graph
        if graph is None:
            return
        self._accept_labels(graph, self._fetch(self._label_statement(), (graph,)))

    def execute_query(
        self,
        query: Statement,
        params: Params | None = None,
        *,
        binary_: bool = False,
        counts_: bool = False,
        prepare_: bool | None = None,
        row_: RowFactory[Any] | None = None,
    ) -> Result:
        """Run one statement and read all of it.

        The reserved arguments end in an underscore so that a parameter of any name can be
        passed alongside them. Nothing rewrites the statement: it goes as written, and
        ``binary_`` asks only for the rendering the answer comes back in.

        ``counts_`` reads what the statement changed, which is a second statement because
        the server offers the counters through a function rather than sending them with the
        result. It also reads them beforehand when the statement returned rows, because such
        a statement zeroes only the counters belonging to the clauses it has and the rest
        still hold whatever an earlier statement left -- so without the earlier reading there
        is no way to tell a number this statement earned from one it inherited.
        """
        if isinstance(query, str):
            self._check(query)
        with self.cursor(row_factory=row_) if row_ else self.cursor() as cursor:
            if binary_:
                cursor.format = psycopg.pq.Format.BINARY
            before: Sequence[int] | None = None
            if counts_:
                before = self._counters()
            try:
                cursor.execute(query, params, prepare=prepare_)
            except psycopg.Error as exc:
                raise self._translated(exc) from None
            described = cursor.description
            records = cursor.fetchall() if described is not None else []
            keys = [column.name for column in described or ()]
            tag = cursor.statusmessage
        after = self._counters() if counts_ else None
        return Result(records, keys, self._counts_for(tag, before, after))

    def transaction_id(self, *, assign: bool = False) -> int | None:
        """The id of the transaction now open, so its fate can be asked about if it is lost.

        With ``assign`` left alone this reports the id the transaction already has, and nothing
        if it has not needed one -- a transaction that has only read is given none. With
        ``assign`` set an id is taken whether or not one was needed, which is what a caller
        does before a write whose outcome it intends to be able to establish.

        Keep the number. If the connection is lost while committing, whether the commit landed
        is exactly what has been lost, and :meth:`resolve_commit` on *another* connection is
        the only thing that can answer it.
        """
        statement = TRANSACTION_ID_QUERY if assign else ASSIGNED_TRANSACTION_QUERY
        with super().cursor(row_factory=tuple_row) as cursor:
            cursor.execute(statement)
            row = cursor.fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0])

    def resolve_commit(self, transaction_id: int) -> CommitOutcome:
        """What became of a transaction, asked from this connection.

        Meant to be asked on a connection other than the one that was lost -- the whole point
        is that the original cannot answer. A transaction still running is not an answer yet;
        wait and ask again rather than deciding.
        """
        with super().cursor(row_factory=tuple_row) as cursor:
            cursor.execute(TRANSACTION_STATUS_QUERY, (str(transaction_id),))
            row = cursor.fetchone()
        return read_outcome(None if row is None else row[0])

    def _counters(self) -> Sequence[int]:
        with super().cursor(row_factory=tuple_row) as cursor:
            cursor.execute(COUNTER_QUERY)
            row = cursor.fetchone()
        if row is None:
            raise AssertionError("the write counters returned no row")
        return [int(value) for value in row]

    def _run(self, statement: str) -> None:
        with super().cursor() as cursor:
            cursor.execute(statement)

    def _fetch(self, statement: str, params: Params) -> list[Any]:
        with super().cursor(row_factory=tuple_row) as cursor:
            cursor.execute(statement, params)
            return cursor.fetchall()
