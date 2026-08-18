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

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

import psycopg
from psycopg.adapt import AdaptersMap
from psycopg.pq import TransactionStatus
from psycopg.rows import Row, tuple_row
from psycopg.types.json import Jsonb

from ._core import (
    GRAPH_ADAPTERS,
    GraphMixin,
    Result,
    savepoint_name,
    statement_text,
    stream_name,
    with_keepalives,
)
from ._protocol.labels import CURRENT_GRAPH_QUERY, LabelCache
from .bulk import (
    EDGE_COLUMN_TYPES,
    EDGE_LABEL_FACTS_QUERY,
    PROMOTED_KEY_TYPES,
    VERTEX_COLUMN_TYPES,
    UpsertCounts,
    build_identity_map,
    edge_copy_statement,
    edge_overlap_update_statement,
    edge_pairs_all_query,
    edge_pairs_present_query,
    edge_rows,
    identity_map_statement,
    key_spellings,
    keyed_identity_query,
    overlap_update_statement,
    promoted_identity_map_statement,
    split_by_what_exists,
    split_edges_by_what_exists,
    vertex_copy_statement,
    vertex_rows,
)
from .capabilities import VECTOR_AVAILABLE_QUERY, VECTOR_VERSION_QUERY
from .columnar import CHUNK
from .cypher import (
    changes_graph_path,
    check_bindable_positions,
    needs_a_reading_first,
    quote_identifier,
    wrap_for_cursor,
)
from .deadline import Deadline
from .errors import BatchFailed, ConfigurationError, NoEnclosingTransaction
from .introspect import (
    CONSTRAINTS_FOR_LABEL,
    CONSTRAINTS_QUERY,
    DECLARED_PROPERTIES_FOR_LABEL,
    DECLARED_PROPERTIES_QUERY,
    GATHER_META,
    GRAPHS_QUERY,
    INDEXES_FOR_LABEL,
    INDEXES_QUERY,
    LABELS_QUERY,
    META_FLAG_QUERY,
    META_VALID_QUERY,
    PROMOTION_CATALOG_QUERY,
    SERVER_PROGRAM_QUERY,
    TRIPLES_QUERY,
    Check,
    Constraint,
    DeclaredProperty,
    DesiredIndex,
    DesiredLabel,
    Graph,
    GraphDescription,
    Index,
    Label,
    PropertyShape,
    Triple,
    Unique,
    describe_kind,
    element_count_query,
    for_labels,
    index_properties,
    property_sample_query,
    reconcile_constraints,
    reconcile_indexes,
    reconcile_labels,
)
from .notify import (
    LISTENING_QUERY,
    NOTIFY_QUERY,
    listen_statement,
    unlisten_statement,
)
from .observability import Timer, query_span, report_notice
from .summary import (
    ASSIGNED_TRANSACTION_QUERY,
    COUNTER_QUERY,
    TRANSACTION_ID_QUERY,
    TRANSACTION_STATUS_QUERY,
    CommitOutcome,
    GraphWriteCounts,
    read_outcome,
)
from .vector import search_option_statements

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Iterable, Mapping, Sequence
    from typing import Self

    from psycopg.abc import Params, Query, QueryNoTemplate
    from psycopg.rows import AsyncRowFactory

    from ._core import Statement
    from ._protocol.graphid import GraphId

__all__ = ["AsyncConnection", "AsyncCursor"]


class AsyncCursor(psycopg.AsyncCursor[Row]):
    """A cursor that watches the statements going past it.

    Every way of running a statement arrives here: ``execute_query`` builds a cursor,
    ``connection.execute`` builds one, and a caller may take one and use it directly. So one
    check here holds for all of them.

    Three things are watched for. A statement whose parameter the server would read as
    something else is refused before it is sent. A statement that moves the session to another
    graph leaves the label table describing a graph the session is no longer reading, so the
    table is dropped once such a statement has run. And every statement is reported to whatever
    is listening, which is why it is reported from here: a caller asking what the driver sends
    wants the driver's own catalog reads too, and those do not come through ``execute_query``.

    ``elapsed`` on a record is the round trip, from sending the statement to the server having
    answered -- not the reading of the rows afterwards, which happens outside this call.
    """

    async def execute(
        self,
        query: Query,
        params: Params | None = None,
        *,
        prepare: bool | None = None,
        binary: bool | None = None,
    ) -> Self:
        text = statement_text(query)
        conn = self._guard()
        check_bindable_positions(text)
        timer = Timer.start()
        # psycopg offers this as two overloads, one of which takes a template and no
        # parameters. Its own body takes either and sorts them out, which is what is wanted
        # here: one check, then whatever was written goes on unchanged.
        try:
            await super().execute(
                cast("QueryNoTemplate", query), params, prepare=prepare, binary=binary
            )
        except psycopg.Error as exc:
            # Every statement passes through here, so a refusal the server words badly reads
            # the same however it was sent. A cursor with a name is the one that does not: the
            # server makes those a class of psycopg's own.
            failure = conn._translated(exc)
            conn._report_query(text, timer, rows=0, error=failure)
            raise failure from None
        except BaseException as exc:
            # An interruption, and only an interruption: cancellation, a keyboard interrupt,
            # an interpreter shutting down. The statement did not finish and what the
            # connection holds is whatever psycopg managed to read back, so it is marked here
            # rather than guessed at afterwards, when it looks like any other idle connection.
            # An ordinary exception is left alone -- a value this driver refuses to send never
            # reached the socket, and the connection is untouched by it.
            if not isinstance(exc, Exception):
                conn._agens_cancelled = True
                conn._report_query(text, timer, rows=0, error=exc)
            raise
        conn._report_query(text, timer, rows=self.rowcount, error=None)
        self._watch_graph_path(text)
        return self

    async def executemany(
        self, query: Query, params_seq: Iterable[Params], *, returning: bool = False
    ) -> None:
        text = statement_text(query)
        conn = self._guard()
        check_bindable_positions(text)
        timer = Timer.start()
        try:
            await super().executemany(query, params_seq, returning=returning)
        except psycopg.Error as exc:
            # Every statement passes through here, so a refusal the server words badly reads
            # the same however it was sent. A cursor with a name is the one that does not: the
            # server makes those a class of psycopg's own.
            failure = conn._translated(exc)
            conn._report_query(text, timer, rows=0, error=failure)
            raise failure from None
        except BaseException as exc:
            if not isinstance(exc, Exception):
                conn._agens_cancelled = True
                conn._report_query(text, timer, rows=0, error=exc)
            raise
        conn._report_query(text, timer, rows=self.rowcount, error=None)
        self._watch_graph_path(text)

    def _guard(self) -> AsyncConnection[Row]:
        """The connection, refused if its holder has given it back."""
        conn = cast("AsyncConnection[Row]", self.connection)
        conn._check_lent()
        return conn

    def _watch_graph_path(self, text: str) -> None:
        """Drop the label table if the statement that just ran moved the session elsewhere.

        After the statement rather than before it, because a statement that failed changed
        nothing. Setting the graph path is undone by rolling back, so a change made inside a
        transaction is remembered until that transaction ends.
        """
        if not changes_graph_path(text):
            return
        conn = cast("AsyncConnection[Row]", self.connection)
        conn.label_table.invalidate()
        if not conn.autocommit:
            conn._agens_graph_path_in_transaction = True


class AsyncConnection(GraphMixin, psycopg.AsyncConnection[Row]):
    """A connection that reads the graph types.

    Built on psycopg's, so everything psycopg offers is here unchanged -- cursors, server
    cursors, ``COPY``, pipelines, ``LISTEN``, and plain SQL for the things Cypher has no
    syntax for. What is added is the graph types, a graph to read them from, and a statement
    check for the one shape the server would misread.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Take a connection's own copy of what the rest of it reads.

        Filled here rather than on first use. Two callers reaching a field that fills itself
        both find it empty and both fill it, and one of the two copies is then dropped --
        with whatever was registered on it, or loaded into it, going with it. Building both
        outright costs a few microseconds against a connection that is about to cost a round
        trip, and leaves nothing to race for.
        """
        super().__init__(*args, **kwargs)
        self.cursor_factory = AsyncCursor
        self._agens_adapters = AdaptersMap(GRAPH_ADAPTERS)
        self._agens_labels = LabelCache()

    # -- refusing a handle its holder gave back ------------------------------------------
    #
    # Only what can reach another caller's work: running a statement, taking a cursor to run
    # one with, and ending a transaction that is now somebody else's.

    async def commit(self) -> None:
        self._check_lent()
        await super().commit()
        self._agens_graph_path_in_transaction = False

    @classmethod
    async def connect(cls, conninfo: str = "", **kwargs: Any) -> AsyncConnection[Any]:
        """Open a connection and refuse a server this driver cannot read.

        The version arrives in the startup packet, so the refusal costs no round trip and
        happens before the first statement rather than at whichever later one first wants a
        catalog the server has never had.

        Keepalive is asked for unless the caller decided otherwise; see
        :data:`~agensgraph._core.KEEPALIVE_DEFAULTS` for what that is worth.
        """
        conn = await super().connect(conninfo, **with_keepalives(kwargs, conninfo))
        # Registered once here rather than per statement. psycopg calls this on every notice the
        # server sends, and what it hands over is read into a Notice only when somebody is
        # listening, so a caller who never asked pays one call that returns immediately.
        conn.add_notice_handler(report_notice)
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

        Two statements: one to move the session, one to read the labels of where it now is.
        The table is what the composite rendering needs to name a label, and filling it here
        means asking for that rendering later does not have to stop and ask.
        """
        await self._run(self._select_graph_statement(name))
        rows = await self._fetch(self._label_statement(), (name,))
        self._accept_labels(name, rows)

    async def refresh_labels(self) -> None:
        """Fill the label table again, for whichever graph the session is reading.

        Wanted after creating or dropping a label, and after anything that moved the session
        to another graph. Nothing does this by itself: re-running the statement that hit an
        unknown label would repeat whatever else it did, which for a write that returned rows
        is not something a driver may decide on a caller's behalf.

        A graph the table does not name is asked of the server, which is one statement more
        and is what makes this the way back from any change the driver only saw go past.
        """
        graph = self.label_table.graph
        if graph is None:
            graph = await self._current_graph()
            if graph is None:
                return
        self._accept_labels(graph, await self._fetch(self._label_statement(), (graph,)))

    async def _current_graph(self) -> str | None:
        """The graph the session is reading, or ``None`` if it is reading none."""
        rows = await self._fetch(CURRENT_GRAPH_QUERY, ())
        name = rows[0][0] if rows else ""
        return name or None

    async def rollback(self) -> None:
        """Roll back, and drop the label table if the graph path is going back with it.

        Setting the graph path is part of a transaction, so rolling one back returns the
        session to the graph it was reading before, and a table filled inside that transaction
        describes somewhere the session no longer is.
        """
        self._check_lent()
        await super().rollback()
        if self._agens_graph_path_in_transaction:
            self.label_table.invalidate()
            self._agens_graph_path_in_transaction = False

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

        Parameters are a sequence, as psycopg takes them, rather than arguments of their own.
        A form taking them one by one cannot tell a single sequence parameter from several
        parameters, and this driver binds a list where the grammar wants one. The reserved
        arguments still end in an underscore, so that the shape stays open to a mapping of
        named parameters without any of them colliding with a name a caller chose.

        Nothing rewrites the statement: it goes as written, and ``binary_`` asks only for the
        rendering the answer comes back in.

        Two arguments a caller might look for are deliberately absent, both because they would
        cost round trips that belong somewhere cheaper. Selecting a graph is a statement and
        undone by a rollback, and it drops the label table, so it is :meth:`graph` on the
        connection or ``graph=`` on a pool -- once, not once per statement. Bounding a
        statement's time is two more statements, one to set the limit and one to put it back:
        measured, ``return 1`` costs 161 microseconds and the same read between them costs 373,
        and against a server a network away it would be two more round trips rather than two
        more local ones. So it belongs where it is paid once for many statements -- a pool's
        ``deadline``, or ``options=-c statement_timeout=...`` on the connection.

        ``counts_`` reads what the statement changed, which is a second statement because
        the server offers the counters through a function rather than sending them with the
        result. It also reads them beforehand when the statement returned rows, because such
        a statement zeroes only the counters belonging to the clauses it has and the rest
        still hold whatever an earlier statement left -- so without the earlier reading there
        is no way to tell a number this statement earned from one it inherited.
        """
        text = statement_text(query)

        # The span is taken outside, and is not something to wait for: it is a plain block
        # in both interfaces, so it stays one rather than being converted into a wait.
        with query_span(text, graph=self.label_table.graph):
            async with self.cursor(row_factory=row_) if row_ else self.cursor() as cursor:
                if binary_:
                    self._check_binary()
                    cursor.format = psycopg.pq.Format.BINARY
                before: Sequence[int] | None = None
                if counts_ and needs_a_reading_first(text):
                    before = await self._counters()
                await cursor.execute(query, params, prepare=prepare_)
                # A statement that changed something without returning rows still has a
                # result, and asking that result for rows raises.  What distinguishes the
                # two is whether it describes any columns.
                described = cursor.description
                records = await cursor.fetchall() if described is not None else []
                keys = [column.name for column in described or ()]
                oids = tuple(int(column.type_code) for column in described or ())
        after = await self._counters() if counts_ else None
        return Result(records, keys, self._counts_for(text, before, after), oids)

    async def can_run_server_programs(self) -> bool:
        """Whether this role could run a command on the server's host through ``COPY``.

        Asked once per connection and kept. See :meth:`read_only_transaction`, which is the
        reason it is worth knowing.
        """
        held = self._agens_can_run_programs
        if held is None:
            rows = await self._fetch(SERVER_PROGRAM_QUERY, ())
            held = bool(rows[0][0])
            self._agens_can_run_programs = held
        return held

    @asynccontextmanager
    async def deadline(
        self, budget: Deadline | float | None, *, gap: float = 0.5
    ) -> AsyncGenerator[Deadline]:
        """Bound how long the statements in this block may take, and hand back the budget.

        A pool sets this per borrow, which a connection nobody pooled never gets. For a server the
        unit is a request: one budget, however many statements the request turns out to need.

        ::

            with conn.deadline(5.0) as budget:
                conn.execute_query(one)
                conn.execute_query(two)
                remaining = budget.remaining()

        The limit is the server's, set below what the caller is waiting for by *gap*, so that the
        server gives up and reports a cancelled statement rather than the caller giving up first
        and leaving one running on a connection it has stopped reading. It is put back on the way
        out, including when a statement raised, and put back *by name*: a connection carrying
        ``options=-c statement_timeout=...`` returns to that rather than to none.

        **It costs two statements, so put it where many statements are inside it.** Over a loopback
        connection ``select 1`` alone is 95 microseconds and alone inside a block 303, and ten
        statements in one block are 1.26 times. That is why this is a block and not an argument to
        :meth:`execute_query`.

        The two are sent as they read rather than in one pipeline. Over such a connection a pipeline
        of three costs 681 microseconds against 432, and one statement inside a pipeline costs 307
        against 123 outside it: a loopback round trip is libpq and psycopg at both ends rather than
        latency, so a pipeline has little waiting to remove and its own cost per statement to add.
        Thirty statements come about even. What it costs where a round trip is latency is untested
        here.

        A budget with no limit sets nothing and costs nothing.
        """
        held = budget if isinstance(budget, Deadline) else Deadline(budget)
        limit = held.statement_timeout_ms(gap=gap)
        if limit is None:
            yield held
            return
        await self.execute(f"set statement_timeout = {limit}")
        self._agens_statement_timeout = True
        try:
            yield held
        finally:
            await self.execute("set statement_timeout = default")
            self._agens_statement_timeout = False

    @asynccontextmanager
    async def read_only_transaction(
        self, *, allow_server_programs: bool = False
    ) -> AsyncGenerator[None]:
        """A transaction the server will not let write, for a statement you did not write.

        Model output, most often. The refusal is the server's, so a write is refused however the
        statement is spelled and the driver holds no opinion about what writing looks like. That
        is the whole argument for it over reading the text: a reading has to recognise every way
        of writing and misses, and it is PostgreSQL underneath, so ``INSERT``, ``TRUNCATE`` and
        ``DROP`` are all available and none of them is Cypher. Measured, each of those is refused
        with ``25006``, as is every Cypher write, and a plain read runs.

        The transaction characteristic is psycopg's ``connection.read_only``, which is what to
        set for a whole session. What this adds is the check below.

        **What a read-only transaction does not stop.** ``COPY ... TO PROGRAM`` runs a command on
        the server's host, and it is allowed: it takes rows out of the database rather than
        putting any in, so it is not a write to refuse. Reading the text does not stop it either
        -- a second statement after a semicolon, and a leading comment, both get one past, and
        both were demonstrated. What stops it is not holding the privilege, so that is what is
        asked about, once, and a role holding it is refused here rather than left to find out.
        Pass ``allow_server_programs`` to accept it anyway.

        **It ends by rolling back**, because a transaction that could not write has nothing to
        commit, and one thing a read-only transaction does allow is ``SET``. Committing keeps a
        setting for the rest of the session, which on a pooled connection is somebody else's
        statement: measured, ``set search_path`` inside this block was read back by the next
        borrower of the same connection. Rolling back undoes it.

        **What that costs, since it is not nothing.** psycopg gives up its prepared statements on
        any rollback, so a connection that uses this block keeps none: measured, one prepared name
        before a block and none after, and a ``DEALLOCATE ALL`` round trip whenever any statement
        was prepared. A statement that differs every call -- which is what arrives from somewhere
        else, and what this block is for -- loses nothing by that, and the cost stayed inside the
        noise across two runs. A statement repeated on the same connection pays about a quarter
        more, because it is parsed again every time. Nested inside another transaction the block
        also costs one round trip more than committing, since releasing a savepoint is one
        statement and rolling back to it is two.

        A caller that wants both should keep its own repeated statements on a different connection
        from the ones it did not write.
        """
        if not allow_server_programs and await self.can_run_server_programs():
            raise ConfigurationError(
                "this role may run a command on the server's host, through "
                "COPY ... TO PROGRAM, which a read-only transaction does not stop -- so a "
                "read-only transaction is not a boundary for a statement this role did not "
                "write. Connect as a role that is not a superuser and does not hold "
                "pg_execute_server_program, or pass allow_server_programs=True to accept it."
            )
        async with self.transaction() as block:
            await self._run("set transaction read only")
            yield
            # Named, so that a block nested in an outer transaction rolls back itself and
            # leaves the outer one to its own decision.
            raise psycopg.Rollback(block)

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
        generator form goes wrong. A connection in autocommit is refused here, saying so, rather
        than at the server with a message about ``DECLARE``.

        Each stream names its cursor for itself, so two open at once do not collide -- a
        collision aborts the transaction, and would take the first stream down with the second.
        ``name`` overrides that, for a cursor meant to be found by name.

        ``size`` is how many rows are fetched at a time. A hundred, because the standard's own
        default of one is a round trip per row.
        """
        if size < 1:
            raise ValueError(f"a chunk holds at least one row, got {size}")
        # What is wrong with the statement is said before what is wrong with the connection,
        # so a caller with both is not sent round twice.
        self._check(query)
        statement = wrap_for_cursor(query)
        # What the cursor needs is a transaction, which is not the same question as whether the
        # connection is in autocommit: `with conn.transaction():` opens a real one there, and a
        # cursor inside it lives as long as any other. So the status is read rather than the flag.
        if (
            name is None
            and self.autocommit
            and self.pgconn.transaction_status == TransactionStatus.IDLE
        ):
            raise NoEnclosingTransaction.for_stream()
        # Reported here rather than at the cursor: a server-side cursor is psycopg's own class
        # and not the one that reports for itself, and a stream is a statement like any other.
        timer = Timer.start()
        async with self.cursor(name=name or stream_name()) as cursor:
            cursor.itersize = size
            if binary_:
                self._check_binary()
                cursor.format = psycopg.pq.Format.BINARY
            try:
                await cursor.execute(statement, params)
            except psycopg.Error as exc:
                failure = self._translated(exc)
                self._report_query(statement, timer, rows=0, error=failure)
                raise failure from None
            self._report_query(statement, timer, rows=cursor.rowcount, error=None)
            while True:
                rows = await cursor.fetchmany(size)
                if not rows:
                    return
                for row in rows:
                    yield row

    async def register_vectors(self) -> tuple[str, ...]:
        """Read vectors on this connection, reporting which vector types were found.

        Separate from everything else the connection sets up, because a vector type's oid belongs
        to an extension rather than to the server: it is assigned when the extension is created and
        differs between databases, so it has to be asked for here rather than written down. An
        empty result means the extension is not created in this database.

        Worth doing when a property has been given a column of its own. Such a property arrives as
        that column's type, and with no loader for it a vector arrives as the *string* it prints
        as -- where the same property left in the property map arrives as a list of numbers.
        """
        from psycopg.types import TypeInfo

        from .vector import TYPES, accept

        found: list[str] = []
        for name in TYPES:
            info = await TypeInfo.fetch(self, name)
            if info is None:
                continue
            accept(self, info)
            found.append(name)
        return tuple(found)

    async def has_vectors(self) -> bool:
        """Whether vectors can be read here at all, without registering anything."""
        rows = await self._fetch(VECTOR_AVAILABLE_QUERY, ())
        return bool(rows and rows[0][0])

    async def vector_version(self) -> tuple[int, ...] | None:
        """What version of pgvector is created here, or ``None`` if none is.

        A version rather than a yes or no, because pgvector gates its own features on it: sparse
        vectors and half precision arrived in 0.7.0 and iterative index scans in 0.8.0, so a caller
        deciding whether to use one has a number to compare rather than a boolean to guess from.
        """
        rows = await self._fetch(VECTOR_VERSION_QUERY, ())
        if not rows or rows[0][0] is None:
            return None
        return tuple(int(part) for part in str(rows[0][0]).split(".") if part.isdigit())

    # -- loading a lot at once -----------------------------------------------------------

    async def load_vertices(
        self,
        label: str,
        properties: Iterable[Mapping[str, Any]],
        *,
        graph: str | None = None,
    ) -> int:
        """Copy vertices into a label, and report how many.

        Faster than any statement because it is one stream rather than a statement per row:
        measured at 223,000 rows a second against 140,000 for a single ``UNWIND ... CREATE`` and
        47,000 for one statement each.

        No identity is supplied. The column's default produces the same identities a ``CREATE``
        would, so nothing here has to reproduce the server's numbering.
        """
        name = await self._graph_of(graph)
        loaded = 0
        async with (
            self.cursor() as cursor,
            cursor.copy(vertex_copy_statement(name, label)) as copy,
        ):
            copy.set_types(VERTEX_COLUMN_TYPES)
            for row in vertex_rows(properties):
                await copy.write_row(row)
                loaded += 1
        return loaded

    async def upsert_vertices(
        self,
        label: str,
        key: str,
        rows: Iterable[Mapping[str, Any]],
        *,
        on_existing: str = "skip",
        require_unique: bool = True,
        graph: str | None = None,
    ) -> UpsertCounts:
        """Write elements that are not there, by a property that identifies them.

        ``load_vertices`` is a copy, and a copy only creates, so re-reading a source into a graph
        with it makes a second element for everything it already holds. This reads which keys are
        there, copies only the rows whose key is not, and leaves the rest alone.

        ``on_existing="update"`` merges the given properties into the elements already present, as
        one statement addressed by the identities the read already found. A property the caller
        does not mention keeps its value. Left as ``"skip"`` nothing is written for them at all,
        which is a copy and nothing else, and is why that is the default.

        **A key with no uniqueness behind it is refused.** Two writers merging on the same key
        without one produce duplicate elements rather than an error: eight writers over
        twenty-five shared keys were measured making twenty-seven elements. ``require_unique``
        turns the refusal off for a graph that already has such a key and cannot add one.
        """
        if on_existing not in ("skip", "update"):
            raise ValueError(
                f"on_existing is 'skip' or 'update', not {on_existing!r}: there is no third thing "
                f"to do with an element that is already there"
            )
        name = await self._graph_of(graph)
        material = list(rows)
        indexed = await self._key_is_unique(label, key, graph=name)
        if require_unique and not indexed:
            raise ValueError(
                f"nothing makes {key!r} unique on {label!r}, so two writers merging on it would "
                f"each create an element rather than find one. Add a unique property index or a "
                f"uniqueness constraint, or pass require_unique=False to accept the hazard"
            )
        # Named keys, so the read follows the batch rather than the label. Without something to
        # look a key up by this falls back to the whole label, since a lookup would read it once
        # per key.
        known = await self.identity_map(
            label, key, keys=[row.get(key) for row in material], graph=name
        )
        fresh, updates = split_by_what_exists(material, key, known)
        inserted = await self.load_vertices(label, fresh, graph=name) if fresh else 0
        updated = 0
        if on_existing == "update" and updates:
            await self._run_with(overlap_update_statement(label), (Jsonb(updates),))
            updated = len(updates)
        return UpsertCounts(inserted, updated)

    async def upsert_edges(
        self,
        label: str,
        edges: Iterable[tuple[GraphId, GraphId, Mapping[str, Any] | None]],
        *,
        on_existing: str = "skip",
        require_unique: bool = True,
        graph: str | None = None,
    ) -> UpsertCounts:
        """Write the edges that are not there, by the pair of elements each one joins.

        What :meth:`upsert_vertices` is for a vertex, keyed on the endpoints rather than a property,
        because that pair is what an edge is. :meth:`load_edges` is a copy and a copy only creates,
        so reading a source in twice makes a second edge for every one already there. This reads
        which pairs are there, copies only the pairs that are not, and leaves the rest alone.

        ``on_existing="update"`` merges the given properties into the edges already present, as one
        statement addressed by the identities the read found. A property the caller does not mention
        keeps its value. Left as ``"skip"`` nothing is written for them, which is a copy and nothing
        else, and is why that is the default.

        **A label with nothing keeping one edge per pair is refused.** Two writers merging the same
        pair without it each create an edge rather than finding one: eight writers over twenty-five
        pairs were measured making twenty-seven edges and reporting no failure at all. The index that
        prevents it is plain SQL over the columns, because a property index cannot express it --
        see :class:`~agensgraph.DesiredIndex`::

            create unique index links_pair on "social".links (start, "end")

        With that in place the same eight writers left twenty-five edges, the losers reporting
        ``23505``, which is what :meth:`RetryPolicy.decide` reads as worth trying again when told the
        statement was merging. ``require_unique=False`` accepts the hazard for a graph that cannot
        add the index.

        A pair given twice in one call is written once.
        """
        if on_existing not in ("skip", "update"):
            raise ValueError(
                f"on_existing is 'skip' or 'update', not {on_existing!r}: there is no third thing "
                f"to do with an edge that is already there"
            )
        name = await self._graph_of(graph)
        material = list(edges)
        keyed, estimate = await self._edge_label_facts(label, graph=name)
        if require_unique and not keyed:
            raise ValueError(
                f"nothing keeps one {label!r} edge per pair of endpoints, so two writers merging "
                f"the same pair would each create one rather than find it. Run "
                f'`create unique index on "{name}".{quote_identifier(label)} (start, "end")`, '
                f"which is plain SQL because a property index cannot key on the endpoint columns, "
                f"or pass require_unique=False to accept the hazard"
            )
        present = await self._edges_present(label, material, graph=name, estimate=estimate)
        fresh, updates = split_edges_by_what_exists(material, present)
        inserted = await self.load_edges(label, fresh, graph=name) if fresh else 0
        updated = 0
        if on_existing == "update" and updates:
            await self._run_with(edge_overlap_update_statement(label), (Jsonb(updates),))
            updated = len(updates)
        return UpsertCounts(inserted, updated)

    async def _edge_label_facts(self, label: str, *, graph: str) -> tuple[bool, float]:
        """Whether one edge per pair of endpoints is kept, and roughly how many edges there are.

        Both in one read, since both are wanted at once and the second is only a threshold. The
        size is ``-1`` where nobody has analysed the label, which is read as not knowing.
        """
        rows = await self._fetch(EDGE_LABEL_FACTS_QUERY, (graph, label))
        if not rows:
            raise ValueError(f"{label!r} is not a label of {graph!r}")
        keyed, estimate = rows[0]
        return bool(keyed), float(estimate)

    async def _edges_present(
        self,
        label: str,
        edges: Sequence[tuple[GraphId, GraphId, Mapping[str, Any] | None]],
        *,
        graph: str,
        estimate: float,
    ) -> dict[tuple[GraphId, GraphId], GraphId]:
        """The identity of every edge already joining one of these pairs.

        Asked about the pairs given, so the cost follows the batch rather than the label -- until
        the batch is as large as the label, at which point asking costs more than reading the label
        does, because asking sends two identities per pair and reading sends none. Measured against
        20,000 edges: asking beat reading at 100, 1,000 and 5,000 pairs and lost 143.7 ms to 53.7 at
        20,000. A label nobody has analysed reports no size, and is asked about.
        """
        if not edges:
            return {}
        if 0 <= estimate <= len(edges):
            rows = await self._fetch(edge_pairs_all_query(graph, label), ())
            wanted = {(start, end) for start, end, _ in edges}
            return {(start, end): found for start, end, found in rows if (start, end) in wanted}
        starts = [start for start, _, _ in edges]
        ends = [end for _, end, _ in edges]
        rows = await self._fetch(edge_pairs_present_query(graph, label), (starts, ends))
        return {(start, end): found for start, end, found in rows}

    async def _key_is_unique(self, label: str, key: str, *, graph: str) -> bool:
        """Whether anything on the server keeps one element per value of this property.

        Both catalogs are asked. A uniqueness assertion is kept as an exclusion constraint and is
        filtered out of the index view, so reading indexes alone would report a graph as having no
        uniqueness when it has some.
        """
        for index in await self.indexes(label, graph=graph):
            if index.unique and index_properties(index.definition) == (key,):
                return True
        return any(
            constraint.unique and f"({key})" in constraint.definition.replace('"', "")
            for constraint in await self.constraints(label, graph=graph)
        )

    async def _run_with(self, statement: str, params: Params) -> None:
        """A statement whose rows are not wanted, sent through the reporting cursor."""
        async with self.cursor() as cursor:
            await cursor.execute(statement, params)

    async def load_edges(
        self,
        label: str,
        edges: Iterable[tuple[GraphId, GraphId, Mapping[str, Any] | None]],
        *,
        graph: str | None = None,
    ) -> int:
        """Copy edges into a label, given the identities of the elements each one joins.

        The identities are the caller's to supply, because an edge is about which two elements it
        joins and nothing in a source file says that -- :meth:`identity_map` is how the mapping
        from whatever the source calls an element to the identity the server gave it is read.
        """
        name = await self._graph_of(graph)
        loaded = 0
        async with (
            self.cursor() as cursor,
            cursor.copy(edge_copy_statement(name, label)) as copy,
        ):
            copy.set_types(EDGE_COLUMN_TYPES)
            for row in edge_rows(edges):
                await copy.write_row(row)
                loaded += 1
        return loaded

    async def load_vertex_frame(
        self, label: str, source: Any, *, graph: str | None = None, size: int = CHUNK
    ) -> int:
        """Copy vertices into a label from anything columnar, and report how many.

        An Arrow table, a polars frame, a pandas frame, a mapping of columns -- anything offering
        Arrow's C stream is read without being copied to be read. Each column becomes a property of
        that name, except that a single column named ``properties`` holding text is taken as the
        JSON of the whole map, which is what a round trip through
        :func:`~agensgraph.columnar.to_arrow` reads back.

        A chunk's property maps are written as JSON and the copy stream built from them, so no row
        becomes a Python mapping and no value goes through a dumper.
        """
        from .bulk import vertex_blocks
        from .columnar import vertex_payloads

        name = await self._graph_of(graph)
        chunks = vertex_payloads(source, size=size)
        # The count comes from the copy's command tag, which is set once the copy block has ended.
        async with self.cursor() as cursor:
            async with cursor.copy(vertex_copy_statement(name, label)) as copy:
                for block in vertex_blocks(payload for chunk in chunks for payload in chunk):
                    await copy.write(block)
            return max(cursor.rowcount, 0)

    async def load_edge_frame(
        self,
        label: str,
        source: Any,
        *,
        start: str = "start",
        end: str = "end",
        graph: str | None = None,
        size: int = CHUNK,
    ) -> int:
        """Copy edges into a label from anything columnar, and report how many.

        Two of the columns are the identities each edge joins, named by *start* and *end*, and the
        rest are its properties. An identity is either the packed 64-bit value -- which is what
        :func:`~agensgraph.columnar.to_arrow` writes for an identity column -- or the
        ``labid.locid`` text.
        """
        from .bulk import edge_blocks
        from .columnar import edge_payloads

        name = await self._graph_of(graph)
        chunks = edge_payloads(source, start=start, end=end, size=size)
        async with self.cursor() as cursor:
            async with cursor.copy(edge_copy_statement(name, label)) as copy:
                for block in edge_blocks(row for chunk in chunks for row in chunk):
                    await copy.write(block)
            return max(cursor.rowcount, 0)

    async def identity_map(
        self,
        label: str,
        key: str,
        *,
        keys: Iterable[Any] | None = None,
        graph: str | None = None,
    ) -> dict[str, GraphId]:
        """What the server called each element of a label, keyed by one of its properties.

        The key is read as text on both sides, because a key that is a number in one place and a
        string in the other would otherwise match nothing.

        The key has to identify an element, and it is refused if it does not. A map is what
        :meth:`load_edges` resolves an endpoint through, so an element sharing its key with
        another, or holding no key at all, is not a smaller map -- it is edges attached to the
        wrong vertex, or to none.

        **Name the keys you need, if you know them.** With ``keys`` given, only those are looked
        up, and the cost follows how many you asked for. Without it the whole label is read, and
        that is not a thing an index can make cheaper: every property of an element lives in one
        column, so reading one key reassembles all of them, and PostgreSQL will not answer a
        projection from an index over an expression -- verified, a purpose-built index on the same
        expression is not used even with sequential scans turned off. On 20,000 elements each
        carrying a 1536-dimension embedding the whole label costs 631 milliseconds and 60,000
        buffers, whatever is indexed.

        Two things make it cheap anyway. A key with a column of its own is read from the column,
        which touches no property map: 3 milliseconds for the same 20,000. And ``keys`` turns the
        read into one index lookup per key.
        """
        name = await self._graph_of(graph)
        if keys is not None and await self._key_is_unique(label, key, graph=name):
            return await self._identity_of_keys(label, key, keys, graph=name)
        column = await self._promoted_key_column(label, key, graph=name)
        if column is not None:
            rows = await self._fetch(promoted_identity_map_statement(name, label, key), ())
        else:
            rows = await self._fetch(identity_map_statement(name, label), (key,))
        return build_identity_map(rows, label=label, key=key)

    async def _promoted_key_column(self, label: str, key: str, *, graph: str) -> str | None:
        """The type of this key's own column, where it has one whose text reading matches the map.

        A promoted key sits beside the property map rather than inside it, so reading it detoasts
        nothing. Only a type that reads back as the map would is used: a boolean gives Python's
        ``True`` where the map gives ``true``, and a key that changed its spelling would find a
        different element.
        """
        if not await self.can_promote_properties():
            return None
        for declared in await self.declared_properties(label, graph=graph):
            if declared.name == key:
                return declared.type if declared.type in PROMOTED_KEY_TYPES else None
        return None

    async def _identity_of_keys(
        self, label: str, key: str, values: Iterable[Any], *, graph: str
    ) -> dict[str, GraphId]:
        """The identities of the keys named, asked for one index lookup at a time.

        Every key is asked for in both of its spellings, so the answer is the one the whole label
        would have given -- see :func:`~agensgraph.bulk.key_spellings`.
        """
        asked = [form for value in values if value is not None for form in key_spellings(value)]
        if not asked:
            return {}
        # The planner is not able to choose this. It costs a sequential scan from the heap's page
        # count, and a label whose maps are large has few heap pages and an enormous TOAST table
        # that the cost model does not see. So it takes the scan at every batch size, from ten keys
        # to twice the label, and is between 2 and 1800 times slower for it. Turning the scan off is
        # a preference rather than a prohibition, so a label with nothing to look a key up by still
        # answers.
        #
        # Reading the setting, turning it off and asking all go in one pipeline. They have to happen
        # in that order and do, since a pipeline keeps the order it was filled in; what is not
        # needed is a round trip each, and the reading of the old value is only wanted afterwards.
        # Restoring it is the one that has to wait, because it must not be undone by a failure.
        previous = "on"
        async with self.pipeline():
            before = self.cursor(row_factory=tuple_row)
            await before.execute("show enable_seqscan")
            await self._run("set enable_seqscan = off")
            asking = self.cursor(row_factory=tuple_row)
            await asking.execute(keyed_identity_query(label, key), (Jsonb(asked),))
        try:
            read = await before.fetchall()
            previous = read[0][0] if read else previous
            rows = await asking.fetchall()
        finally:
            await before.close()
            await asking.close()
            # Back to what it was and not to the default, so a caller who had turned the scan off
            # for its own reasons still has it off afterwards.
            await self._run(f"set enable_seqscan = {'on' if previous == 'on' else 'off'}")
        return build_identity_map(rows, label=label, key=key)

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
            Label(*row)
            for row in await self._fetch(LABELS_QUERY, (await self._graph_of(graph),))
        ]

    async def can_promote_properties(self) -> bool:
        """Whether this server can store a property in a column of its own.

        Asked of the catalog once per connection and kept. :meth:`Capabilities.has_property_promotion`
        answers the same question from the version at no round trip, and is right about a release;
        this is right about the server actually connected, which is not the same thing on a
        development build -- two reporting ``2.18-devel`` were found to differ.
        """
        held = self._agens_can_promote
        if held is None:
            rows = await self._fetch(PROMOTION_CATALOG_QUERY, ())
            held = bool(rows[0][0])
            self._agens_can_promote = held
        return held

    async def describe(
        self, *, sample: int = 100, refresh: bool = False, graph: str | None = None
    ) -> GraphDescription:
        """What is in a graph: its labels, what they hold, and what joins what.

        For a prompt, or a schema browser, or anything else that has to say what a graph looks
        like without reading it. Nothing here scans the graph. The labels and their counts come
        from the catalogs, the triples from the one the server keeps for its own planner, and the
        properties from a bounded sample of each label -- ``sample`` rows, not all of them, because
        the shape of a label does not need every row to establish and reading every row of a label
        full of embeddings to learn that one key holds an array is minutes rather than milliseconds.

        Where a property has a column of its own its type is read from the catalog instead, which
        is exact rather than sampled. On a server that cannot promote a property there is nothing
        to read, and everything is sampled.

        **Nothing is installed.** Three of the packages this replaces create a plpgsql function in
        the caller's database to name a JSON type; ``jsonb_typeof`` is built in, and the one thing
        it does not do -- telling a whole number from a fractional one -- is done here.

        The triple catalog is filled by a gather, and ``auto_gather_graphmeta`` is off by default,
        so on a server where nobody has gathered there are no triples and ``meta_gathered`` is
        false. ``refresh`` gathers first. It is not the default because it is a write, and a
        description is not a thing that should write.
        """
        name = await self._graph_of(graph)
        if refresh and not await self.meta_is_current(graph=name):
            await self._run(GATHER_META)
        labels = await self.labels(graph=name)
        counts = await self.element_counts(graph=name)
        triples = tuple(
            Triple(start, edge, end, seen)
            for start, edge, end, seen in await self._fetch(TRIPLES_QUERY, (name,))
        )
        declared: dict[str, dict[str, str]] = {}
        if await self.can_promote_properties():
            for each in await self.declared_properties(graph=name):
                declared.setdefault(each.label, {})[each.name] = each.type
        wanted = [label.name for label in labels if not label.is_builtin]
        sampled = await self.pipeline_query(
            [(property_sample_query(name, label), (sample,)) for label in wanted]
        )
        properties: dict[str, tuple[PropertyShape, ...]] = {}
        for label, result in zip(wanted, sampled, strict=True):
            found = {
                key: PropertyShape(key, describe_kind(kind, fractional), False)
                for key, kind, fractional, _ in result.records
            }
            for key, kind in declared.get(label, {}).items():
                found[key] = PropertyShape(key, kind, True)
            properties[label] = tuple(found[key] for key in sorted(found))
        return GraphDescription(
            graph=name,
            labels=tuple(labels),
            properties=properties,
            triples=triples,
            counts=counts,
            meta_gathered=await self.meta_is_current(graph=name),
        )

    async def meta_is_current(self, *, graph: str | None = None) -> bool:
        """Whether the triple catalog still describes this graph.

        ``regather_graphmeta()`` sets the flag, a transaction that wrote to the graph clears it
        at commit, and a read leaves it alone. So this separates a catalog nobody has gathered
        from one that is current and from one that was gathered and has since been written to.

        The flag arrived with 2.18. Before it, those three cannot be told apart, and the answer
        comes from what the catalog holds: triples for a graph that has edges is as much as can be
        established, and a graph with no edges has nothing to gather.
        """
        name = await self._graph_of(graph)
        if await self._has_meta_flag():
            rows = await self._fetch(META_VALID_QUERY, (name,))
            return bool(rows) and bool(rows[0][0])
        triples = await self._fetch(TRIPLES_QUERY, (name,))
        if triples:
            return True
        counts = await self.element_counts(graph=name)
        labels = await self.labels(graph=name)
        return sum(counts.get(label.name, 0) for label in labels if label.is_edge) == 0

    async def _has_meta_flag(self) -> bool:
        """Whether this server records whether the triple catalog is current. Asked once."""
        held = self._agens_has_meta_flag
        if held is None:
            rows = await self._fetch(META_FLAG_QUERY, ())
            held = bool(rows[0][0])
            self._agens_has_meta_flag = held
        return held

    async def declared_properties(
        self, label: str | None = None, *, graph: str | None = None
    ) -> list[DeclaredProperty]:
        """Every property given a column of its own, with that column's type.

        A property living in the JSON map is not declared anywhere and so cannot be listed.

        A server that cannot promote a property has nowhere to record one, and reading the catalog
        that is not there would raise instead of saying none -- so the catalog is asked for first,
        once per connection. Asked rather than worked out from the version, because the version
        does not answer it: the 2.18 release branch and main both report ``2.18-devel`` and only
        one of them has the catalog.
        """
        if not await self.can_promote_properties():
            return []
        name = await self._graph_of(graph)
        query = DECLARED_PROPERTIES_QUERY if label is None else DECLARED_PROPERTIES_FOR_LABEL
        rows = await self._fetch(query, (name,) if label is None else (name, label))
        return [DeclaredProperty(*row) for row in rows]

    async def indexes(
        self, label: str | None = None, *, graph: str | None = None
    ) -> list[Index]:
        """Every property index. A uniqueness constraint is not one; see :meth:`constraints`."""
        name = await self._graph_of(graph)
        query = INDEXES_QUERY if label is None else INDEXES_FOR_LABEL
        params = (name,) if label is None else (name, label)
        return [Index(*row) for row in await self._fetch(query, params)]

    async def constraints(
        self, label: str | None = None, *, graph: str | None = None
    ) -> list[Constraint]:
        """Every constraint on a label's properties, uniqueness assertions included.

        Read from the constraint catalog rather than from the property-index view, which filters
        exclusion constraints out -- and a uniqueness assertion is kept as an exclusion, so it
        would otherwise be invisible.
        """
        name = await self._graph_of(graph)
        query = CONSTRAINTS_QUERY if label is None else CONSTRAINTS_FOR_LABEL
        rows = await self._fetch(query, (name,) if label is None else (name, label))
        return [Constraint(*row) for row in rows]

    async def pipeline_batch(
        self, statements: Sequence[tuple[str, Params | None]] | Sequence[str]
    ) -> None:
        """Send many statements without waiting for each in turn.

        For a burst whose cost is round trips rather than work. Nothing is read back, so this is for
        statements whose results are not wanted -- a batch of writes, most often.

        **A failure names the batch, not the statement.** A pipeline attributes an error to the wrong
        statement: with four statements of which only the second was bad, the first raised the error
        and the rest raised with no SQLSTATE. So any failure here is raised as
        :class:`~agensgraph.errors.BatchFailed` carrying every statement sent, with the server's own
        error as its cause. Running them one at a time is how to find the one at fault, and is left
        to the caller because replaying a write would apply it twice.
        """
        sent = [(item, None) if isinstance(item, str) else item for item in statements]
        try:
            async with self.pipeline():
                for statement, params in sent:
                    await self.execute(statement, params)
        except psycopg.Error as exc:
            failure = BatchFailed(
                f"one of {len(sent)} pipelined statements failed, and a pipeline does not report "
                f"which: the error below may belong to any of them. Run them one at a time to "
                f"find it."
            )
            failure.statements = tuple(statement for statement, _ in sent)
            raise failure from exc

    async def pipeline_query(
        self,
        statements: Sequence[tuple[str, Params | None]] | Sequence[str],
        *,
        chunk: int = 100,
    ) -> list[Result]:
        """Send many reads without waiting for each in turn, and read every answer back.

        A result per statement, in the order they were given. For a burst of reads whose cost is
        round trips rather than work: an ancestor walk, a degree per node, a batch of lookups by
        id. Measured on 300 indexed reads, three times the same statements run one after another.

        **For reads.** :meth:`pipeline_batch` reads nothing back because a pipeline attributes a
        failure to the wrong statement, and replaying a write to find the real one would apply it
        twice. A read has neither problem, which is what makes this possible: if the batch fails,
        the statements are run again one at a time, so the failure is raised by the statement that
        actually caused it and the results are still correct. A write among them would be applied
        twice by that second pass, so do not put one here.

        Each statement gets a cursor, and cursors are made ``chunk`` at a time so that a long list
        does not allocate one for every statement at once.
        """
        sent = [(item, None) if isinstance(item, str) else item for item in statements]
        results: list[Result] = []
        for start in range(0, len(sent), chunk):
            batch = sent[start : start + chunk]
            # Behind a savepoint, because the asking again below cannot happen in a transaction the
            # failure has aborted: every statement would answer 25P02 and the real error would be
            # lost. The savepoint is taken *inside* the pipeline, so it rides in the round trip the
            # batch was already taking and the ordinary path pays nothing for it.
            mark = savepoint_name() if self._can_hold_a_savepoint() else None
            try:
                results.extend(await self._pipelined(batch, savepoint=mark))
            except psycopg.Error:
                # The pipeline blamed one of them and a pipeline names the wrong one, so the
                # answer comes from asking again singly. Reading twice costs a round trip per
                # statement and buys the right statement in the traceback.
                if mark is not None:
                    await self._run(f"rollback to savepoint {mark}")
                    await self._run(f"release savepoint {mark}")
                for statement, params in batch:
                    results.append(await self.execute_query(statement, params))
        return results

    def _can_hold_a_savepoint(self) -> bool:
        """Whether a savepoint can be taken here at all.

        A transaction the server has already aborted takes no savepoint -- asking for one is
        itself refused -- and a connection in autocommit with nothing open needs none, since a
        failure there aborts nothing to undo. Both are left to fail and be asked again, which for
        the aborted one reports the abort, correctly, as the reason.
        """
        status = self.pgconn.transaction_status
        if status == TransactionStatus.IDLE:
            return not self.autocommit
        return status == TransactionStatus.INTRANS

    async def _pipelined(
        self, batch: Sequence[tuple[str, Params | None]], *, savepoint: str | None = None
    ) -> list[Result]:
        """One pipeline's worth, every cursor read back in the order it was filled.

        The reading follows the pipeline rather than sitting inside it. While it is open the
        statements have been sent and not answered, so a cursor describes no columns and reading
        one gives an empty result rather than the rows.

        A *savepoint* is taken and released here rather than around this, so that both go in the
        pipeline with the batch and neither costs a round trip of its own. A failure leaves the
        release unrun, which is what leaves the savepoint there to roll back to.
        """
        cursors = []
        async with self.pipeline():
            if savepoint is not None:
                await self._run(f"savepoint {savepoint}")
            for statement, params in batch:
                cursor = self.cursor()
                await cursor.execute(statement, params)
                cursors.append(cursor)
            if savepoint is not None:
                await self._run(f"release savepoint {savepoint}")
        gathered = []
        for cursor in cursors:
            described = cursor.description
            records = await cursor.fetchall() if described is not None else []
            keys = [column.name for column in described or ()]
            oids = tuple(int(column.type_code) for column in described or ())
            gathered.append(Result(records, keys, GraphWriteCounts.unknown(), oids))
            await cursor.close()
        return gathered

    async def vector_search_options(
        self, options: Mapping[str, object], *, local: bool = True
    ) -> None:
        """Tune a vector search, by default for the current transaction only.

        Setting ``hnsw.ef_search`` higher looks at more candidates, and so recalls more of the true
        nearest neighbours. :data:`agensgraph.vector.SEARCH_OPTIONS` lists
        what can be set; a name that is not one of them is refused rather than sent, since the server
        accepts an unknown one silently.
        """
        for statement in search_option_statements(options, local=local):
            await self._run(statement)

    async def listen(self, *channels: str) -> None:
        """Subscribe to channels the server announces on.

        The channel is quoted into the statement, since neither ``LISTEN`` nor ``UNLISTEN`` takes a
        parameter for it.
        """
        for channel in channels:
            await self._run(listen_statement(channel))

    async def unlisten(self, *channels: str) -> None:
        """Stop listening. Given no channel, stop listening to all of them."""
        for statement in [unlisten_statement(name) for name in channels] or [
            unlisten_statement()
        ]:
            await self._run(statement)

    async def listening(self) -> list[str]:
        """The channels this connection is subscribed to."""
        rows = await self._fetch(LISTENING_QUERY, ())
        return [row[0] for row in rows]

    async def notify(self, channel: str, payload: str = "") -> None:
        """Announce something on a channel.

        Sent through ``pg_notify``, which takes the channel as a parameter, so a channel name held in a
        variable needs no quoting and cannot carry a statement of its own.
        """
        await self._fetch(NOTIFY_QUERY, (channel, payload))

    async def notifications(
        self, *, timeout: float | None = None, stop_after: int | None = None
    ) -> AsyncIterator[psycopg.Notify]:
        """Read announcements as they arrive.

        Refuses to run while a handler is registered. psycopg reports that pairing as unreliable
        and warns; a warning is easy to miss and what follows is announcements going to one route
        or the other unpredictably, so this raises instead.

        A handler is the other way to read these, and the one to prefer: this holds the
        connection's lock for as long as it is being read, so a caller who stops reading part way
        leaves the connection unusable until the iterator is collected.
        """
        if self._notify_handlers:
            raise RuntimeError(
                "this connection already has a notify handler, and reading announcements both "
                "ways at once delivers each one to whichever route happens to be looking. "
                "Remove the handler, or read them through it alone."
            )
        async for notice in super().notifies(timeout=timeout, stop_after=stop_after):
            yield notice

    async def ensure_indexes(
        self,
        desired: Sequence[DesiredIndex],
        *,
        graph: str | None = None,
        drop_extra: bool = False,
        dry_run: bool = False,
    ) -> list[str]:
        """Make the indexes that exist the ones asked for, and return the statements that took.

        An empty list means nothing had to change, which is what a second run gives. ``dry_run``
        returns the statements without running them.

        ``drop_extra`` also removes indexes nobody asked for, considering only indexes a
        :class:`~agensgraph.DesiredIndex` could describe. One over an expression is left alone.

        It reaches only the labels ``desired`` names. A caller saying what one label should have is
        saying nothing about any other, and what is read here is the whole graph's -- so without
        this, a list naming one label would drop the indexes of every label it did not name,
        including uniqueness something else depends on.

        The state is read again afterwards and anything still outstanding is raised. Naming an
        operator class that is already the default does this, since the server omits a default when
        printing a definition.
        """
        name = await self._graph_of(graph)
        statements = reconcile_indexes(
            desired,
            for_labels(await self.indexes(graph=name), desired, drop_extra),
            drop_extra=drop_extra,
        )
        if dry_run:
            return statements
        for statement in statements:
            await self._run(statement)
        if statements:
            await self._settled(
                reconcile_indexes(desired, await self.indexes(graph=name)), "indexes"
            )
        return statements

    async def ensure_labels(
        self,
        desired: Sequence[DesiredLabel],
        *,
        graph: str | None = None,
        dry_run: bool = False,
    ) -> list[str]:
        """Make the labels asked for that are not there, and return the statements that took.

        An empty list means they all existed, which is what a second run gives. ``dry_run``
        returns the statements without running them.

        Nothing is dropped, whatever is left out: a label holds the elements written to it, so
        removing one is a decision about data. There is no ``drop_extra`` here for that reason.

        Declaring labels is worth doing at startup because a write to a label that does not exist
        makes one, which puts DDL inside the write's transaction -- and two writers arriving
        together report ``42P07`` from each other's label. The label table is reloaded afterwards,
        since what it holds is exactly what these statements changed.
        """
        name = await self._graph_of(graph)
        statements = reconcile_labels(desired, await self.labels(graph=name))
        if dry_run:
            return statements
        for statement in statements:
            await self._run(statement)
        if statements:
            await self.refresh_labels()
        return statements

    async def ensure_constraints(
        self,
        desired: Sequence[Unique | Check],
        *,
        graph: str | None = None,
        drop_extra: bool = False,
        dry_run: bool = False,
    ) -> list[str]:
        """Make the constraints that exist the ones asked for, and return the statements that took.

        Matched by name. A :class:`~agensgraph.Unique` given no name is named after its property; a
        :class:`~agensgraph.Check` has to be given one.

        ``drop_extra`` reaches only the labels ``desired`` names, for the reason given in
        :meth:`ensure_indexes`.
        """
        name = await self._graph_of(graph)
        statements = reconcile_constraints(
            desired,
            for_labels(await self.constraints(graph=name), desired, drop_extra),
            drop_extra=drop_extra,
        )
        if dry_run:
            return statements
        for statement in statements:
            await self._run(statement)
        if statements:
            await self._settled(
                reconcile_constraints(desired, await self.constraints(graph=name)),
                "constraints",
            )
        return statements

    async def _settled(self, outstanding: list[str], what: str) -> None:
        """Raise if the state still differs from what was asked for."""
        if outstanding:
            raise RuntimeError(
                f"the {what} asked for were applied but still do not match what the catalogs "
                f"report, so running this again would repeat the same work. Still outstanding: "
                f"{outstanding}"
            )

    async def element_counts(self, *, graph: str | None = None) -> dict[str, int]:
        """How many vertices and edges each label holds.

        Three statements -- the labels, then the vertices and the edges -- and no property read:
        the label id is part of every element's identity, so counting per label needs nothing
        from a row but its id.
        """
        name = await self._graph_of(graph)
        names = {label.id: label.name for label in await self.labels(graph=name)}
        counts: dict[str, int] = {}
        for edges in (False, True):
            for labid, count in await self._fetch(element_count_query(name, edges=edges), ()):
                counts[names.get(labid, str(labid))] = int(count)
        return counts

    async def _graph_of(self, given: str | None) -> str:
        """The graph to read about: the one named, or the one this connection is reading.

        A table naming no graph is asked about rather than refused. Two states arrive here as
        the same missing name: a session reading no graph, which is the caller's to fix, and a
        table dropped because a statement went past that might have moved the session, which is
        one statement away from being known. Only the server can tell them apart, and asking it
        also answers correctly when the session really did move, where the name held before it
        would name the graph it has left.

        The reading is reached only by a caller that named no graph and a table that names none
        either, so an ordinary call still costs nothing.
        """
        if given is not None:
            return given
        graph = self.label_table.graph or await self._current_graph()
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
