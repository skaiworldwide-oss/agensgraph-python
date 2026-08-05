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
from .cypher import wrap_for_cursor
from .introspect import (
    CONSTRAINTS_QUERY,
    DECLARED_PROPERTIES_QUERY,
    GRAPHS_QUERY,
    INDEXES_QUERY,
    LABELS_QUERY,
    Constraint,
    DeclaredProperty,
    Graph,
    Index,
    Label,
    element_count_query,
)
from .observability import Timer, query_span
from .summary import (
    ASSIGNED_TRANSACTION_QUERY,
    COUNTER_QUERY,
    TRANSACTION_ID_QUERY,
    TRANSACTION_STATUS_QUERY,
    CommitOutcome,
    read_outcome,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from psycopg.abc import Params
    from psycopg.rows import AsyncRowFactory

    from ._core import Statement

__all__ = ["AsyncConnection"]


class AsyncConnection(GraphMixin, psycopg.AsyncConnection[Row]):
    """A connection that reads the graph types.

    Built on psycopg's, so everything psycopg offers is here unchanged -- cursors, server
    cursors, ``COPY``, pipelines, ``LISTEN``, and plain SQL for the things Cypher has no
    syntax for. What is added is the graph types, a graph to read them from, and a statement
    check for the one shape the server would misread.
    """

    @classmethod
    async def connect(cls, conninfo: str = "", **kwargs: Any) -> AsyncConnection[Any]:
        """Open a connection and refuse a server this driver cannot read.

        The version arrives in the startup packet, so the refusal costs no round trip and
        happens before the first statement rather than at whichever later one first wants a
        catalog the server has never had.
        """
        conn = await super().connect(conninfo, **kwargs)
        try:
            _ = conn.capabilities
        except BaseException:
            # The connection is open by now, and refusing it here would otherwise leave it
            # open with nobody holding it -- to be closed whenever it is collected, which is
            # a warning at best and a backend sitting idle on the server at worst.
            await conn.close()
            raise
        return conn

    async def graph(self, name: str) -> None:
        """Read from a graph, and fill the label table for it.

        Two statements, and only when the table is not already this graph's -- a connection
        taken from a pool and pointed at the graph it is already reading costs nothing. The
        table is what the composite rendering needs to name a label, and filling it here
        means asking for that rendering later does not have to stop and ask.
        """
        if self.label_table.graph == name:
            await self._run(self._select_graph_statement(name))
            return
        await self._run(self._select_graph_statement(name))
        rows = await self._fetch(self._label_statement(), (name,))
        self._accept_labels(name, rows)

    async def refresh_labels(self) -> None:
        """Fill the label table again, after creating or dropping a label.

        Needed only for the composite rendering, and only for a label created since the
        table was filled. Nothing does this by itself: re-running the statement that hit an
        unknown label would repeat whatever else it did, which for a write that returned rows
        is not something a driver may decide on a caller's behalf.
        """
        graph = self.label_table.graph
        if graph is None:
            return
        self._accept_labels(graph, await self._fetch(self._label_statement(), (graph,)))

    async def execute_query(
        self,
        query: Statement,
        params: Params | None = None,
        *,
        binary_: bool = False,
        counts_: bool = False,
        prepare_: bool | None = None,
        row_: AsyncRowFactory[Any] | None = None,
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
        text = query if isinstance(query, str) else str(query)
        timer = Timer()

        # The span is taken outside, and is not something to wait for: it is a plain block
        # in both interfaces, so it stays one rather than being converted into a wait.
        with query_span(text, graph=self.label_table.graph):
            async with self.cursor(row_factory=row_) if row_ else self.cursor() as cursor:
                if binary_:
                    cursor.format = psycopg.pq.Format.BINARY
                before: Sequence[int] | None = None
                if counts_:
                    before = await self._counters()
                try:
                    await cursor.execute(query, params, prepare=prepare_)
                except psycopg.Error as exc:
                    failure = self._translated(exc)
                    self._report_query(text, timer, rows=0, error=failure)
                    raise failure from None
                # A statement that changed something without returning rows still has a
                # result, and asking that result for rows raises.  What distinguishes the
                # two is whether it describes any columns.
                described = cursor.description
                records = await cursor.fetchall() if described is not None else []
                keys = [column.name for column in described or ()]
                tag = cursor.statusmessage
        after = await self._counters() if counts_ else None
        self._report_query(text, timer, rows=len(records), error=None)
        return Result(records, keys, self._counts_for(tag, before, after))

    async def transaction_id(self, *, assign: bool = False) -> int | None:
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
        async with super().cursor(row_factory=tuple_row) as cursor:
            await cursor.execute(statement)
            row = await cursor.fetchone()
        if row is None or row[0] is None:
            return None
        return int(row[0])

    async def resolve_commit(self, transaction_id: int) -> CommitOutcome:
        """What became of a transaction, asked from this connection.

        Meant to be asked on a connection other than the one that was lost -- the whole point
        is that the original cannot answer. A transaction still running is not an answer yet;
        wait and ask again rather than deciding.
        """
        async with super().cursor(row_factory=tuple_row) as cursor:
            await cursor.execute(TRANSACTION_STATUS_QUERY, (str(transaction_id),))
            row = await cursor.fetchone()
        return read_outcome(None if row is None else row[0])

    async def stream(
        self,
        query: str,
        params: Params | None = None,
        *,
        size: int = 100,
        binary_: bool = False,
        name: str | None = None,
    ) -> AsyncIterator[Any]:
        """Read a result a chunk at a time, without holding all of it.

        A server-side cursor is what keeps the rows on the server, and the grammar has no arm
        for declaring one over Cypher -- so the statement is placed where a subquery goes, which
        is the one thing that works. That wrap takes only the read-only subset, so a statement
        that writes is refused here, by name, rather than producing a syntax error nobody asked
        for. A trailing ``LIMIT`` or ``ORDER BY`` is fine; one in the middle is not, and the
        server says so clearly enough to be left to.

        An enclosing transaction is required, and not as a formality: a server-side cursor lives
        inside one. That requirement is also what makes abandoning the iterator safe, since
        leaving the transaction closes the cursor with it -- rather than leaving a connection
        with a statement still running and a lock still held, which is how the ordinary
        generator form goes wrong.

        ``size`` is how many rows are fetched at a time. A hundred, because the standard's own
        default of one is a round trip per row.
        """
        if size < 1:
            raise ValueError(f"a chunk holds at least one row, got {size}")
        self._check(query)
        statement = wrap_for_cursor(query)
        async with self.cursor(name=name or "agens_stream") as cursor:
            cursor.itersize = size
            if binary_:
                cursor.format = psycopg.pq.Format.BINARY
            try:
                await cursor.execute(statement, params)
            except psycopg.Error as exc:
                raise self._translated(exc) from None
            while True:
                rows = await cursor.fetchmany(size)
                if not rows:
                    return
                for row in rows:
                    yield row

    # -- reading what is in the database -----------------------------------------------

    async def graphs(self) -> list[Graph]:
        """Every graph, since there is no ``\\d`` that knows about them."""
        return [Graph(*row) for row in await self._fetch(GRAPHS_QUERY, ())]

    async def labels(self, *, graph: str | None = None) -> list[Label]:
        """Every label of a graph, in the order the server gave them ids.

        Defaults to the graph this connection is reading. The two a graph is created with are
        included and say so, because a caller counting labels should not have to know that two
        of them were never asked for.
        """
        return [
            Label(*row) for row in await self._fetch(LABELS_QUERY, (self._graph_of(graph),))
        ]

    async def declared_properties(
        self, label: str | None = None, *, graph: str | None = None
    ) -> list[DeclaredProperty]:
        """Every property given a column of its own, with that column's type.

        A property living in the JSON map is not declared anywhere and so cannot be listed;
        before 2.18 nothing can be promoted at all and this is always empty.
        """
        name = self._graph_of(graph)
        rows = await self._fetch(DECLARED_PROPERTIES_QUERY, (name, label, label))
        return [DeclaredProperty(*row) for row in rows]

    async def indexes(
        self, label: str | None = None, *, graph: str | None = None
    ) -> list[Index]:
        """Every property index. A uniqueness constraint is not one; see :meth:`constraints`."""
        name = self._graph_of(graph)
        return [Index(*row) for row in await self._fetch(INDEXES_QUERY, (name, label, label))]

    async def constraints(
        self, label: str | None = None, *, graph: str | None = None
    ) -> list[Constraint]:
        """Every constraint on a label's properties, uniqueness assertions included.

        Read from the constraint catalog rather than from the property-index view, which filters
        exclusion constraints out -- and a uniqueness assertion is kept as an exclusion, so it
        would otherwise be invisible.
        """
        name = self._graph_of(graph)
        rows = await self._fetch(CONSTRAINTS_QUERY, (name, label, label))
        return [Constraint(*row) for row in rows]

    async def element_counts(self, *, graph: str | None = None) -> dict[str, int]:
        """How many vertices and edges each label holds.

        Two statements and no property read: the label id is part of every element's identity,
        so counting per label needs nothing from a row but its id.
        """
        name = self._graph_of(graph)
        names = {label.id: label.name for label in await self.labels(graph=name)}
        counts: dict[str, int] = {}
        for edges in (False, True):
            for labid, count in await self._fetch(element_count_query(name, edges=edges), ()):
                counts[names.get(labid, str(labid))] = int(count)
        return counts

    def _graph_of(self, given: str | None) -> str:
        """The graph to read about: the one named, or the one this connection is reading."""
        if given is not None:
            return given
        graph = self.label_table.graph
        if graph is None:
            raise ValueError(
                "no graph is selected on this connection, so name the one to read about"
            )
        return graph

    async def _counters(self) -> Sequence[int]:
        async with super().cursor(row_factory=tuple_row) as cursor:
            await cursor.execute(COUNTER_QUERY)
            row = await cursor.fetchone()
        if row is None:
            raise AssertionError("the write counters returned no row")
        return [int(value) for value in row]

    async def _run(self, statement: str) -> None:
        async with super().cursor() as cursor:
            await cursor.execute(statement)

    async def _fetch(self, statement: str, params: Params) -> list[Any]:
        async with super().cursor(row_factory=tuple_row) as cursor:
            await cursor.execute(statement, params)
            return await cursor.fetchall()
