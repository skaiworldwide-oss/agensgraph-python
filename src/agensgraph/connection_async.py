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

import warnings
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, TypeVar, cast

import psycopg
from psycopg.adapt import AdaptersMap
from psycopg.pq import PipelineStatus, TransactionStatus
from psycopg.rows import Row, tuple_row
from psycopg.types.json import Jsonb

from ._core import (
    GRAPH_ADAPTERS,
    GraphMixin,
    ReadOnlyBlock,
    Result,
    savepoint_name,
    statement_text,
    stream_name,
    with_idle_limit,
    with_keepalives,
)
from ._protocol.labels import CURRENT_GRAPH_QUERY, LabelCache
from .bulk import (
    EDGE_COLUMN_TYPES,
    EDGE_LABEL_FACTS_QUERY,
    PROMOTED_KEY_TYPES,
    RESERVE_IDS_QUERY,
    VERTEX_COLUMN_TYPES,
    VERTEX_WITH_ID_COLUMN_TYPES,
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
    vertex_copy_with_ids_statement,
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
from .errors import (
    BatchFailed,
    CapabilityError,
    ConfigurationError,
    NoEnclosingTransaction,
    Retryability,
    StaleLabelCache,
    retryability,
)
from .introspect import (
    AUTO_GATHER_QUERY,
    CONSTRAINTS_FOR_LABEL,
    CONSTRAINTS_QUERY,
    DECLARED_PROPERTIES_FOR_LABEL,
    DECLARED_PROPERTIES_QUERY,
    GATHER_META,
    GRAPHS_QUERY,
    INDEXES_FOR_LABEL,
    INDEXES_QUERY,
    LABEL_POSITIONS_QUERY,
    LABELS_QUERY,
    META_FLAG_QUERY,
    META_VALID_QUERY,
    PROMOTION_CATALOG_QUERY,
    SCHEMA_LOCK_QUERY,
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
    meta_count_check_query,
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
from .observability import (
    Timer,
    detached_span,
    end_span,
    operation_span,
    query_span,
    report_notice,
)
from .settings import check_settings, settings_statement
from .summary import (
    ASSIGNED_TRANSACTION_QUERY,
    COUNTER_QUERY,
    TRANSACTION_ID_QUERY,
    TRANSACTION_STATUS_QUERY,
    CommitOutcome,
    GraphWriteCounts,
    read_outcome,
)
from .untrusted import (
    UNTRUSTED_ROLE_QUERY,
    CheckedText,
    UntrustedResult,
    check_columns,
    check_untrusted,
    role_problems,
)
from .vector import VectorIndex, check_search_option, describe_vector_index

if TYPE_CHECKING:
    from collections.abc import (
        AsyncGenerator,
        AsyncIterator,
        Awaitable,
        Callable,
        Iterable,
        Mapping,
        Sequence,
    )
    from typing import Self

    from psycopg._queries import PostgresQuery
    from psycopg.abc import Params, PQGen, Query, QueryNoTemplate
    from psycopg.copy import AsyncCopy, AsyncWriter
    from psycopg.rows import AsyncRowFactory

    from ._core import Statement
    from ._protocol.graphid import GraphId

__all__ = ["AsyncConnection", "AsyncCursor"]

_RV = TypeVar("_RV")

# The two halves of read_only_is_enforced(): a read the role must be allowed, so that a refusal
# of the write is about writing, and a write that matches nothing.
_READ_ONLY_PROBE_READ = "match (n) where false return n"
_READ_ONLY_PROBE_WRITE = "match (n) where false set n.agensgraph_probe = 1"

# The cursor run_untrusted() reads a wrapped statement through, and the command tags its one
# round trip must answer with, in order. Any other sequence means the statement did something the
# reading did not expect, and the connection is not used again.
_UNTRUSTED_CURSOR = "agensgraph_untrusted"
_UNTRUSTED_TAGS = ("BEGIN", "SELECT 1", "DECLARE CURSOR", "FETCH", "ROLLBACK", "SELECT 1")

# Words that keep a statement out of the cursor wrap on a server with the GQL surface: the grammar
# takes them only at the top of a statement.
_TOP_LEVEL_ONLY = frozenset({"filter", "next"})


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
        conn._check_read_only_statement(text)
        return await self._sent(conn, text, query, params, prepare=prepare, binary=binary)

    async def _execute_own(
        self, statement: str, params: Params | None = None, *, prepare: bool | None = False
    ) -> Self:
        """One of the driver's own statements, reported like any other.

        It skips the read-only block's checks, which are for text from elsewhere. The driver's
        statements are its savepoints, settings and catalog reads: a savepoint it takes inside a
        block is one it is done with before the block closes, so rolling back to it leaves the
        block read-only -- which is also how psycopg's ``transaction()`` sends its own, past every
        cursor. Not prepared by default, since most are sent once or with a different name each
        time, and a string holding two of them -- as opening a model's streamed read does --
        cannot be prepared at all, which ``prepare_threshold=0`` would otherwise try.
        """
        return await self._sent(
            self._guard(), statement, statement, params, prepare=prepare, binary=None
        )

    async def _sent(
        self,
        conn: AsyncConnection[Row],
        text: str,
        query: Query,
        params: Params | None,
        *,
        prepare: bool | None,
        binary: bool | None,
    ) -> Self:
        """The statement sent, its failure translated, and what happened reported."""
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
        conn._check_read_only_kept()
        self._watch_graph_path(text)
        return self

    async def executemany(
        self, query: Query, params_seq: Iterable[Params], *, returning: bool = False
    ) -> None:
        text = statement_text(query)
        conn = self._guard()
        check_bindable_positions(text)
        conn._check_read_only_statement(text)
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
        conn._check_read_only_kept()
        self._watch_graph_path(text)

    @asynccontextmanager
    async def copy(
        self,
        statement: Query,
        params: Params | None = None,
        *,
        writer: AsyncWriter | None = None,
    ) -> AsyncGenerator[AsyncCopy]:
        """psycopg's ``COPY``, held to the same read-only check as every other statement."""
        conn = self._guard()
        conn._check_read_only_statement(statement_text(statement))
        try:
            async with super().copy(statement, params, writer=writer) as copy:
                yield copy
        except psycopg.Error as exc:
            raise conn._translated(exc) from None
        conn._check_read_only_kept()

    async def stream(
        self,
        query: Query,
        params: Params | None = None,
        *,
        binary: bool | None = None,
        size: int = 1,
    ) -> AsyncIterator[Row]:
        """psycopg's row-by-row read, held to the same read-only check as every other statement.

        A statement that ends the transaction is no less one for being read a row at a time.
        """
        conn = self._guard()
        conn._check_read_only_statement(statement_text(query))
        try:
            async for row in super().stream(query, params, binary=binary, size=size):
                yield row
        except psycopg.Error as exc:
            raise conn._translated(exc) from None
        conn._check_read_only_kept()

    def _execute_send(
        self,
        query: PostgresQuery,
        *,
        force_extended: bool = False,
        binary: bool | None = None,
    ) -> None:
        """Send a statement, never over the simple protocol inside a read-only block.

        That protocol runs every statement in a string, so a smuggled ``; commit; ...`` would
        end the block and run the rest read-write. The extended protocol takes one statement and
        refuses a second with ``42601``. psycopg chooses the protocol here, for ``execute``,
        ``executemany`` and ``copy`` alike; a server-side cursor already uses the extended one.
        """
        conn = cast("AsyncConnection[Row]", self.connection)
        super()._execute_send(
            query,
            force_extended=force_extended or conn._agens_read_only is not None,
            binary=binary,
        )

    def _guard(self) -> AsyncConnection[Row]:
        """The connection, refused if its holder has given it back."""
        conn = cast("AsyncConnection[Row]", self.connection)
        conn._check_lent()
        return conn

    def _watch_graph_path(self, text: str) -> None:
        """Drop the label table if the statement that just ran moved the session elsewhere.

        After the statement rather than before it, because a statement that failed changed
        nothing.

        Both signals are read where both are available. A reported path catches what the text
        cannot: a move made inside a function body, one made by a database default the session
        never named, and the move back a rollback performs. The text catches what the report
        cannot: inside a pipeline the statement is queued and its answer has not arrived, so
        the reported value is still the old one.
        """
        conn = cast("AsyncConnection[Row]", self.connection)
        if conn._follow_reported_graph_path() or not changes_graph_path(text):
            return
        conn.label_table.invalidate()
        if not conn.autocommit and not conn._agens_reports_graph_path:
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
        """Commit, and drop the label table if the graph path went back with it.

        A path set with ``SET LOCAL`` ends with its transaction. Seen where the server reports
        the path; the statement text says a path was set, not for how long.
        """
        self._check_lent()
        await super().commit()
        self._transaction_ended(rolled_back=False)

    @classmethod
    async def connect(cls, conninfo: str = "", **kwargs: Any) -> AsyncConnection[Any]:
        """Open a connection and refuse a server this driver cannot read.

        The version arrives in the startup packet, so the refusal costs no round trip and
        happens before the first statement rather than at whichever later one first wants a
        catalog the server has never had.

        Keepalive is asked for unless the caller decided otherwise; see
        :data:`~agensgraph._core.KEEPALIVE_DEFAULTS` for what that is worth.

        ``idle_in_transaction_timeout=`` seconds has the server end a session that leaves a
        transaction open and idle that long, so a caller that stopped cannot hold its locks for
        ever. It is a startup setting, so it costs no statement; it is left unset unless asked for.
        """
        arguments = with_idle_limit(with_keepalives(kwargs, conninfo), conninfo)
        conn = await super().connect(conninfo, **arguments)
        # Registered once here rather than per statement. psycopg calls this on every notice the
        # server sends, and what it hands over is read into a Notice only when somebody is
        # listening, so a caller who never asked pays one call that returns immediately.
        conn.add_notice_handler(report_notice)
        conn._note_graph_path()
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

    async def recover(self, failure: BaseException) -> Retryability:
        """Do on this connection what a failure's category asks before another attempt.

        :func:`~agensgraph.errors.retryability` names the category; this carries out the part of
        it that belongs to the connection, so a caller running its own retry loop does the right
        thing for each without knowing which is which. A prepared statement the server lost
        (``RESET_STATE``, ``26000``) has the statement cache emptied, here and on the server, by
        ``DEALLOCATE ALL``; a label table that no longer describes the graph (``REFRESH_LABELS``)
        is read again. Any other category asks nothing of this connection, and nothing is sent.
        The category is returned, so the loop can go on to the rest of what it says.
        """
        recovery = retryability(failure)
        if recovery.clears_prepared_statements:
            # psycopg empties its own cache when it sees the statement go past.
            await self.execute("deallocate all")
        if recovery.refreshes_labels:
            await self.refresh_labels()
        return recovery

    async def refresh_labels(self) -> None:
        """Fill the label table again, for whichever graph the session is reading.

        Wanted after creating or dropping a label, and after anything that moved the session
        to another graph. Nothing does this by itself: re-running the statement that hit an
        unknown label would repeat whatever else it did, which for a write that returned rows
        is not something a driver may decide on a caller's behalf.

        A graph the table does not name is asked of the server, which is one statement more
        and is what makes this the way back from any change the driver only saw go past.
        """
        self._follow_reported_graph_path()
        graph = self.label_table.graph
        if graph is None:
            graph = await self._current_graph()
            if graph is None:
                return
        self._accept_labels(graph, await self._fetch(self._label_statement(), (graph,)))

    async def _current_graph(self) -> str | None:
        """The graph the session is reading, or ``None`` if it is reading none.

        Free where the server reports the path, which is what every describing method here
        needs before it can name a table -- so on such a server none of them pays a round
        trip to find out where it is.
        """
        if self._agens_reports_graph_path:
            return self.info.parameter_status("graph_path") or None
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
        self._transaction_ended(rolled_back=True)

    async def wait(self, gen: PQGen[_RV], *args: Any, **kwargs: Any) -> _RV:
        """psycopg's wait on the server, timed while a watch is armed (see :meth:`deadline`).

        Every exchange with the server comes through here -- a statement, its rows, a ``COPY``, a
        pipeline's answers, a commit -- so this is where the watchdog learns the server is still
        talking, and where a connection it shut down is reported as :class:`~agensgraph.Stalled`
        to whichever statement was waiting. A connection with no watch armed waits exactly as
        psycopg does.
        """
        if not self._agens_watches:
            return await super().wait(gen, *args, **kwargs)
        try:
            return await super().wait(self._hearing(gen), *args, **kwargs)
        except psycopg.Error as exc:
            if self._given_up():
                raise self._stalled() from exc
            raise

    async def _set_statement_timeout(self, limit: int | None) -> None:
        """Set the server's statement timeout, or put back the default, and keep a note of it.

        Inside a transaction the setting is part of it, so what was held before is noted too, for
        :meth:`_transaction_ended` to put back if the transaction rolls back.
        """
        held = self._agens_statement_timeout_ms
        # Not prepared: each limit is different text, and would take a slot of its own.
        await self._run(f"set statement_timeout = {'default' if limit is None else limit}")
        if (
            self.pgconn.transaction_status != TransactionStatus.IDLE
            and not self._agens_timeout_set_in_transaction
        ):
            self._agens_timeout_set_in_transaction = True
            self._agens_timeout_before_transaction = held
        self._agens_statement_timeout_ms = limit

    def _transaction_ended(self, *, rolled_back: bool) -> None:
        """Bring the label table and the statement timeout's note in step with a transaction.

        Where the server reports the path, what it reports decides. Elsewhere a rollback over
        a path set inside the transaction drops the table. The note of that setting ends with
        the outermost transaction; a block rolled back to a savepoint keeps it. A timeout set inside
        the transaction goes back on a rollback to what it was before, and so does its note.
        """
        if self.closed:
            return
        if (
            self._agens_timeout_set_in_transaction
            and self.pgconn.transaction_status == TransactionStatus.IDLE
        ):
            if rolled_back:
                self._agens_statement_timeout_ms = self._agens_timeout_before_transaction
            self._agens_timeout_set_in_transaction = False
        if self._agens_reports_graph_path:
            self._follow_reported_graph_path()
            return
        if rolled_back and self._agens_graph_path_in_transaction:
            self.label_table.invalidate()
        if self.pgconn.transaction_status == TransactionStatus.IDLE:
            self._agens_graph_path_in_transaction = False

    @asynccontextmanager
    async def transaction(
        self, savepoint_name: str | None = None, force_rollback: bool = False
    ) -> AsyncGenerator[psycopg.AsyncTransaction]:
        """psycopg's transaction block, on a handle its holder still has.

        The block ends with a statement no cursor of this connection runs, so the label table is
        brought in step here as the block closes.
        """
        self._check_lent()
        rolled_back = True
        try:
            async with super().transaction(savepoint_name, force_rollback) as block:
                yield block
                rolled_back = force_rollback
        finally:
            self._transaction_ended(rolled_back=rolled_back)

    @asynccontextmanager
    async def pipeline(self) -> AsyncGenerator[psycopg.AsyncPipeline]:
        """psycopg's pipeline mode, on a handle its holder still has."""
        self._check_lent()
        async with super().pipeline() as pipeline:
            yield pipeline

    async def execute_query(
        self,
        query: Statement,
        params: Params | None = None,
        *,
        binary_: bool = False,
        counts_: bool = False,
        prepare_: bool | None = None,
        row_: AsyncRowFactory[Any] | None = None,
        settings_: Mapping[str, object] | None = None,
    ) -> Result:
        """Run one statement and read all of it.

        Parameters are a sequence, as psycopg takes them, rather than arguments of their own.
        A form taking them one by one cannot tell a single sequence parameter from several
        parameters, and this driver binds a list where the grammar wants one. The reserved
        arguments still end in an underscore, so that the shape stays open to a mapping of
        named parameters without any of them colliding with a name a caller chose.

        Nothing rewrites the statement: it goes as written, and ``binary_`` asks only for the
        rendering the answer comes back in.

        Selecting a graph is deliberately absent: it is a statement and undone by a rollback, and
        it drops the label table, so it is :meth:`graph` on the connection or ``graph=`` on a
        pool -- once, not once per statement.

        ``settings_`` makes settings for this statement alone -- ``hnsw.ef_search`` for a
        search, ``statement_timeout`` for a slow read -- without a round trip of their own. They
        are one ``select set_config(..., true)`` sent in a pipeline with the statement, so in
        autocommit the two share the pipeline's implicit transaction and the settings end with
        it. Inside a transaction they last until it ends, as ``SET LOCAL`` would. Checked by
        :func:`agensgraph.settings.check_settings`, which refuses ``graph_path``.

        ``counts_`` reads what the statement changed, which is a second statement because
        the server offers the counters through a function rather than sending them with the
        result. It also reads them beforehand when the statement returned rows, because such
        a statement zeroes only the counters belonging to the clauses it has and the rest
        still hold whatever an earlier statement left -- so without the earlier reading there
        is no way to tell a number this statement earned from one it inherited.
        """
        text = statement_text(query)
        # Checked before anything is sent, so a misspelt setting costs no round trip.
        settings = settings_statement(check_settings(settings_)) if settings_ else None

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
                if settings is None:
                    await cursor.execute(query, params, prepare=prepare_)
                else:
                    await self._execute_with_settings(
                        cursor, query, params, settings, prepare=prepare_
                    )
                # A statement that changed something without returning rows still has a
                # result, and asking that result for rows raises.  What distinguishes the
                # two is whether it describes any columns.
                described = cursor.description
                records = await self._read_all(cursor) if described is not None else []
                keys = [column.name for column in described or ()]
                oids = tuple(int(column.type_code) for column in described or ())
        after = await self._counters() if counts_ else None
        return Result(records, keys, self._counts_for(text, before, after), oids)

    async def _execute_with_settings(
        self,
        cursor: psycopg.AsyncCursor[Any],
        query: Statement,
        params: Params | None,
        settings: tuple[str, tuple[str, ...]],
        *,
        prepare: bool | None,
    ) -> None:
        """The settings and the statement in one pipeline: one round trip, one transaction.

        A pipeline entered inside another first finishes what the outer one had sent, so the two
        statements here always share their own implicit transaction. A failure in either arrives
        when the pipeline ends rather than from ``execute``, so it is translated here, as the
        cursor would have.
        """
        statement, values = settings
        setting = self.cursor()
        try:
            async with self.pipeline():
                await setting.execute(statement, values)
                await cursor.execute(query, params, prepare=prepare)
        except psycopg.Error as exc:
            raise self._translated(exc) from None
        finally:
            await setting.close()

    async def _read_all(self, cursor: psycopg.AsyncCursor[Any]) -> list[Any]:
        """Every row a cursor holds, decoded again once if it names a label the table lacks.

        A label made by another session after the table was read, or a graph the session moved to
        without the driver, leaves the composite rendering naming an id the table does not have.
        The rows are already here, so the table is read again for the graph the session is on --
        the one the statement read from -- and they are decoded again. The statement is not run a
        second time. Rows that still name an unknown id raise
        :class:`~agensgraph.errors.StaleLabelCache`.
        """
        try:
            return await cursor.fetchall()
        except StaleLabelCache:
            await self.refresh_labels()
            await cursor.scroll(0, mode="absolute")
            return await cursor.fetchall()

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
        self, budget: Deadline | float | None, *, gap: float | None = None
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

        **The client holds the limit too.** A server can only stop a statement it can still hear
        from, so a network that stops answering would leave the block waiting for ever. If a
        statement is running and the server has sent nothing for half a second longer than its own
        limit allows, the socket is shut down: the caller gets :class:`~agensgraph.Stalled`, and the
        connection, which is broken by then, is closed. See :mod:`agensgraph._watchdog`; arming it
        costs a few microseconds, and only a stalled network ever sets it off -- not a block that
        runs past its budget doing work of its own between statements.
        """
        held = budget if isinstance(budget, Deadline) else Deadline(budget)
        limit = held.statement_timeout_ms(gap=gap)
        if limit is None:
            yield held
            return
        await self._set_statement_timeout(limit)
        watch = self._watch()
        try:
            yield held
        finally:
            # Still watched while the limit is put back: that is a round trip too, and a network
            # that stalls during it is reported by wait() rather than holding the caller.
            try:
                if (
                    not self._given_up()
                    and not self.closed
                    and self.pgconn.transaction_status != TransactionStatus.INERROR
                ):
                    # Not in a failed transaction, where the statement would fail too and
                    # replace the error the caller has to see; the rollback owed puts the
                    # setting back, and its note with it.
                    await self._set_statement_timeout(None)
            finally:
                if self._unwatch(watch):
                    # Shut down by the watchdog: whatever the block did, the socket is gone.
                    await self.close()

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

        **What is held to it, on every way of running a statement.** Inside the block every
        statement goes over the extended protocol, so a smuggled ``; commit; ...`` is refused
        with ``42601`` rather than ending the block; text holding a second statement, or one
        that would end or split the transaction, is refused before it is sent. On a server
        that does not refuse a Cypher write in a read-only transaction -- 2.14 and 2.15 -- a
        statement holding a Cypher write clause is refused before it is sent as well, and a
        write the reading misses still runs, shows its effect to the caller, and is undone
        only by the rollback. Any refusal to write raises :class:`ReadOnlyGraphWrite`.

        **The one thing the rollback does not undo** is a session-level advisory lock, which a
        read may take: ``pg_advisory_lock`` is not a write. A pool releases every such lock on a
        connection that ran this block before lending it again; a connection kept by its caller
        keeps them until it closes or calls ``pg_advisory_unlock_all()``. :meth:`run_untrusted`
        releases them itself.
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
            # A transaction can still be made read-write until it takes its first snapshot, so
            # one is taken at once: this is SET TRANSACTION READ ONLY as a query, which a query's
            # snapshot comes with. After it, SET TRANSACTION READ WRITE and setting
            # transaction_read_only from inside the block are both refused by the server. One
            # statement, not two in a string: a connection that prepares every statement
            # (prepare_threshold=0) cannot prepare two, and nor can a block nested in another,
            # whose statements already go over the extended protocol.
            await self._run("select set_config('transaction_read_only', 'on', true)")
            outer = self._agens_read_only
            self._agens_read_only = ReadOnlyBlock(refuse_writes=not self._read_only_gate())
            try:
                yield
            finally:
                self._agens_read_only = outer
            # Named, so that a block nested in an outer transaction rolls back itself and
            # leaves the outer one to its own decision.
            raise psycopg.Rollback(block)

    async def read_only_is_enforced(self) -> bool:
        """Whether this server refuses a Cypher write in a read-only transaction, by asking it.

        The version answers this for free: 2.16 is the first release that refuses. Asking costs
        one round trip and is worth it where the answer matters most, on a 2.14 or 2.15 build
        that carries the fix. A zero-row read and a zero-row ``SET`` run inside a read-only
        transaction that is rolled back, so nothing is written either way.

        The answer is kept for the connection. It is true only when the server refused the
        ``SET`` with ``25006`` or ``42501`` after letting the read run, and false otherwise --
        including when the question could not be put, inside a transaction already open or
        without a graph selected. That errs on the side of the driver refusing writes itself.
        """
        if self._agens_read_only_enforced is not None:
            return self._agens_read_only_enforced
        if self.pgconn.transaction_status != TransactionStatus.IDLE:
            return self.capabilities.has("cypher_write_enforced")
        # None until the server has answered one way or the other.
        enforced: bool | None = None
        try:
            async with self.transaction(force_rollback=True):
                await self._run("set transaction read only")
                await self._run(_READ_ONLY_PROBE_READ)
                try:
                    async with self.transaction():
                        await self._run(_READ_ONLY_PROBE_WRITE)
                except psycopg.Error as exc:
                    if exc.sqlstate in ("25006", "42501"):
                        enforced = True
                else:
                    enforced = False
        except psycopg.Error:
            return False
        if enforced is not None:
            self._agens_read_only_enforced = enforced
        return bool(enforced)

    async def untrusted_role_problems(
        self, *, app_role: str | None = None, schemas: Sequence[str] = ()
    ) -> tuple[str, ...]:
        """What makes this connection's role unfit to run model-written Cypher, if anything.

        Membership is what is asked about, not what is held at this moment: a role can take on
        any role it belongs to with ``SET ROLE``, inside a read-only transaction. ``app_role``
        names the application's role, which the model's role must neither be nor belong to.
        ``schemas`` are the graph's schema and any other the caller keeps data in; where the
        server lets Cypher writes skip privileges, 2.14 and 2.15, the role must hold no write
        privilege on them.

        Asked once per connection for a given role and schemas, and kept.
        """
        key = (app_role, tuple(schemas))
        cached = self._agens_untrusted_role
        if cached is not None and cached[0] == key:
            return cached[1]
        rows = await self._fetch(
            UNTRUSTED_ROLE_QUERY, {"app": app_role, "schemas": list(schemas)}
        )
        problems = role_problems(
            tuple(bool(value) for value in rows[0]), writes_enforced=self._read_only_gate()
        )
        self._agens_untrusted_role = (key, problems)
        return problems

    async def run_untrusted(
        self,
        text: str,
        *,
        max_rows: int = 100,
        timeout: float = 10.0,
        allow_writes: bool = False,
        app_role: str | None = None,
        schemas: Sequence[str] = (),
        unsafe_shared_role: bool = False,
    ) -> UntrustedResult:
        """Run one statement a model wrote, and nothing it could not have meant to.

        The text is read first and refused before it is sent if it is not plainly a Cypher read:
        see :func:`~agensgraph.untrusted.check_untrusted`. Then it runs in a transaction of its
        own that is read-only and always rolled back, under ``timeout`` seconds, and at most
        ``max_rows`` rows are read.

        One round trip in the usual case: the statement is read through a cursor declared over
        ``SELECT * FROM (<text>) AS t`` in a single pipeline that also opens, limits and rolls
        back the transaction and releases any advisory lock. A statement that wrap cannot hold
        -- one starting with ``WITH``, and on 2.18 one using ``FILTER`` or ``NEXT`` -- is
        streamed as it is instead, in three round trips, inside the same kind of transaction.
        Either way the result must have the columns the statement's ``RETURN`` declares.

        The connection's role must be fit for this: not a superuser, not able to reach the
        server's files, programs or other sessions, and not the application's role or a member
        of it (:meth:`untrusted_role_problems`). On 2.14 and 2.15 that is required; from 2.16
        ``unsafe_shared_role`` accepts the risk.

        ``allow_writes`` runs Cypher write clauses in a read-write transaction that commits. It
        needs a server that checks those writes against privileges itself -- a 2.16 or later
        release, confirmed by :meth:`read_only_is_enforced` -- and a role allowed exactly the
        writes the model may make. It is never retried.
        """
        if max_rows < 1:
            raise ValueError(f"at least one row has to be readable, got {max_rows}")
        checked = check_untrusted(text, allow_writes=allow_writes)
        self._check_lent()
        if self.pgconn.transaction_status != TransactionStatus.IDLE:
            raise psycopg.ProgrammingError(
                "model-written Cypher runs in a transaction of its own, and this connection has "
                "one open; commit or roll it back first"
            )
        problems = await self.untrusted_role_problems(app_role=app_role, schemas=schemas)
        if problems and not (unsafe_shared_role and self._read_only_gate()):
            raise ConfigurationError(
                "this role must not run model-written Cypher: "
                + "; ".join(problems)
                + ". Connect as a role made for it: one that can log in and read the graph, and "
                "holds nothing else"
                + (
                    ", or pass unsafe_shared_role=True to accept the risk"
                    if self._read_only_gate()
                    else ""
                )
            )
        if allow_writes and not (
            self.capabilities.has("cypher_write_enforced")
            and await self.read_only_is_enforced()
        ):
            raise CapabilityError(
                "model-written writes need a server that checks Cypher writes against privileges "
                "and READ ONLY itself: an AgensGraph 2.16 or later release, confirmed by "
                f"read_only_is_enforced(); this server is {self.capabilities.reported}"
            )
        timeout_ms = str(max(1, int(timeout * 1000)))
        was_autocommit = self.autocommit
        if not was_autocommit:
            await self.set_autocommit(True)
        try:
            if allow_writes:
                result = await self._untrusted_write(checked, max_rows, timeout_ms)
            else:
                result = await self._untrusted_read(checked, max_rows, timeout_ms)
        finally:
            if not was_autocommit:
                await self.set_autocommit(False)
        try:
            check_columns(checked.columns, result.columns)
        except ValueError as exc:
            raise self._boundary_lost(
                f"model-written Cypher returned unexpected columns: {exc}"
            ) from None
        return result

    async def _untrusted_read(
        self, checked: CheckedText, max_rows: int, timeout_ms: str
    ) -> UntrustedResult:
        """A read, through the cursor wrap where its grammar takes the text, streamed otherwise."""
        # 2.18's wrap also refuses FILTER, NEXT and a mid-statement ORDER BY, SKIP or LIMIT;
        # those go straight to the stream rather than spend a round trip failing.
        gql = self.capabilities.has("gql_clauses")
        if checked.wrappable and not (
            gql and (checked.limits_mid_statement or checked.words & _TOP_LEVEL_ONLY)
        ):
            wrapped = await self._untrusted_wrapped(checked, max_rows, timeout_ms)
            if wrapped is not None:
                return wrapped
        return await self._untrusted_streamed(checked, max_rows, timeout_ms)

    async def _untrusted_wrapped(
        self, checked: CheckedText, max_rows: int, timeout_ms: str
    ) -> UntrustedResult | None:
        """The one-round-trip read, or ``None`` where the wrap's grammar refuses the statement."""
        statements: tuple[tuple[str, tuple[str, ...] | None], ...] = (
            ("begin read only", None),
            settings_statement([("statement_timeout", timeout_ms)]),
            (
                f"declare {_UNTRUSTED_CURSOR} no scroll cursor for {wrap_for_cursor(checked.text)}",
                None,
            ),
            (f"fetch {max_rows + 1} from {_UNTRUSTED_CURSOR}", None),
            ("rollback", None),
            ("select pg_advisory_unlock_all()", None),
        )
        # Rows as tuples whatever the connection's row factory, as the result promises.
        cursors = [self.cursor(row_factory=tuple_row) for _ in statements]
        try:
            try:
                async with self.pipeline():
                    for cursor, (statement, params) in zip(cursors, statements, strict=True):
                        await cursor.execute(statement, params, prepare=False)
            except psycopg.Error as exc:
                await self._roll_back_and_unlock()
                if exc.sqlstate == "42601":
                    return None
                raise self._translated(exc) from None
            tags = tuple(cursor.statusmessage or "" for cursor in cursors)
            if tuple(tag.split(" ")[0] if tag.startswith("FETCH") else tag for tag in tags) != (
                _UNTRUSTED_TAGS
            ):
                raise self._boundary_lost(
                    f"model-written Cypher answered with {tags}, not the expected sequence"
                )
            fetched = cursors[3]
            rows = await fetched.fetchall()
            names = tuple(column.name for column in fetched.description or ())
        finally:
            for cursor in cursors:
                await cursor.close()
        return UntrustedResult.capped(names, rows, max_rows)

    async def _untrusted_streamed(
        self, checked: CheckedText, max_rows: int, timeout_ms: str
    ) -> UntrustedResult:
        """The read the wrap cannot hold, streamed as written, in three round trips."""
        # More than one row per chunk needs libpq 17's chunked mode.
        size = 100 if psycopg.pq.version() >= 170000 else 1
        rows: list[tuple[Any, ...]] = []
        names: tuple[str, ...] = ()
        await self._run(
            f"begin read only; select set_config('statement_timeout', '{timeout_ms}', true)"
        )
        try:
            async with self.cursor(row_factory=tuple_row) as cursor:
                try:
                    async for row in cursor.stream(checked.text, size=size):
                        rows.append(row)
                        if len(rows) > max_rows:
                            break
                except psycopg.Error as exc:
                    raise self._translated(exc) from None
                names = tuple(column.name for column in cursor.description or ())
            if self.pgconn.transaction_status == TransactionStatus.IDLE:
                raise self._boundary_lost(
                    "the read-only transaction model-written Cypher ran in ended under it"
                )
        finally:
            await self._roll_back_and_unlock()
        return UntrustedResult.capped(names, rows, max_rows)

    async def _untrusted_write(
        self, checked: CheckedText, max_rows: int, timeout_ms: str
    ) -> UntrustedResult:
        """A model-written write, committed once, after checking it stayed in its transaction."""
        rows: list[tuple[Any, ...]] = []
        names: tuple[str, ...] = ()
        async with self.transaction():
            await self._run(*settings_statement([("statement_timeout", timeout_ms)]))
            async with self.cursor(row_factory=tuple_row) as cursor:
                # In a pipeline, so it goes over the extended protocol whatever it holds.
                async with self.pipeline():
                    await cursor.execute(checked.text, prepare=False)
                if cursor.description is not None:
                    rows = list(await cursor.fetchmany(max_rows + 1))
                    names = tuple(column.name for column in cursor.description)
            if self.pgconn.transaction_status != TransactionStatus.INTRANS:
                raise self._boundary_lost(
                    "the transaction model-written Cypher ran in ended before it was committed"
                )
        return UntrustedResult.capped(names, rows, max_rows)

    async def _roll_back_and_unlock(self) -> None:
        """Roll back whatever text from elsewhere left open, and release every session advisory
        lock, which a rollback leaves in place: one round trip either way.

        In a transaction both go in one string. Out of one the release goes in autocommit, so that
        a connection outside it does not open a transaction around the release only to end it.
        """
        if self.pgconn.transaction_status != TransactionStatus.IDLE:
            await self._run("rollback; select pg_advisory_unlock_all()")
            return
        was_autocommit = self.autocommit
        if not was_autocommit:
            await self.set_autocommit(True)
        try:
            await self._run("select pg_advisory_unlock_all()")
        finally:
            if not was_autocommit:
                await self.set_autocommit(False)

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
        self._check_lent()
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
        # Not made current: this yields between rows, and a current span would hold every span
        # the caller takes meanwhile under it.
        span = detached_span(
            "STREAM", graph=self.label_table.graph, **{"db.query.text": statement}
        )
        failure: BaseException | None = None
        try:
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
        except BaseException as exc:
            failure = failure or exc
            raise
        finally:
            end_span(span, failure)

    async def register_vectors(self) -> tuple[str, ...]:
        """Read vectors on this connection, reporting which vector types were found.

        Separate from everything else the connection sets up, because a vector type's oid belongs
        to an extension rather than to the server: it is assigned when the extension is created and
        differs between databases, so it has to be asked for here rather than written down. An
        empty result means the extension is not created in this database.

        Worth doing when a property has been given a column of its own. Such a property arrives as
        that column's type, and with no loader for it a vector arrives as the *string* it prints
        as -- where the same property left in the property map arrives as a list of numbers.

        Every type is looked up in one statement. Where any is found, ``bit`` is read and sent as
        :class:`~agensgraph.vector.Bits` as well, since that is what a quantised vector is and
        what its Hamming and Jaccard distances measure.
        """
        from .vector import TYPES, VECTOR_TYPES_QUERY, accept, accept_bits, vector_type_infos

        found: list[str] = []
        for info in vector_type_infos(await self._fetch(VECTOR_TYPES_QUERY, (list(TYPES),))):
            accept(self, info)
            found.append(info.name)
        if found:
            accept_bits(self)
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

    @asynccontextmanager
    async def _describing_a_copy(self) -> AsyncGenerator[None]:
        """Describe a failure a copy reports the way one from a statement is described.

        A copy's failure arrives when its block ends rather than from a statement, so the cursor
        that describes every statement never sees it. Undescribed, the whole of what the server
        said reaches the caller, and for a copy that is the row it refused: measured,
        ``load_vertices`` of a duplicate key put the conflicting value into ``str(exc)`` while the
        same write through a statement did not.

        The handle is checked here as well, since a copy runs through no cursor of this
        connection's that would check it. And a span is taken here, since no statement's span
        covers the rows a copy sends.
        """
        self._check_lent()
        with operation_span("COPY", graph=self.label_table.graph):
            try:
                yield
            except psycopg.Error as exc:
                raise self._translated(exc) from None

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
        47,000 for the same rows through ``executemany``, which psycopg pipelines. A plain loop of
        one statement a row is slower again; see :mod:`agensgraph.bulk` for the spread.

        No identity is supplied. The column's default produces the same identities a ``CREATE``
        would, so nothing here has to reproduce the server's numbering.
        """
        name = await self._graph_of(graph)
        return await self._copy_rows(
            vertex_copy_statement(name, label), VERTEX_COLUMN_TYPES, vertex_rows(properties)
        )

    async def _copy_rows(
        self, statement: str, types: Sequence[str], rows: Iterable[Sequence[Any]]
    ) -> int:
        """Rows sent through one binary ``COPY``, under its span; how many were sent."""
        sent = 0
        async with (
            self._describing_a_copy(),
            self.cursor() as cursor,
            cursor.copy(statement) as copy,
        ):
            copy.set_types(list(types))
            for row in rows:
                await copy.write_row(row)
                sent += 1
        return sent

    async def reserve_ids(
        self, label: str, count: int, *, graph: str | None = None
    ) -> list[GraphId]:
        """Take ``count`` identities for a label, the way its own id default would, at once.

        One round trip. The label's id and its sequence are read from the catalog rather than
        from the default, which names the label by a string that 2.14.1 and 2.15.0 leave stale
        after a rename. Identities taken and not used leave a gap, as a rolled-back ``CREATE``
        would.
        """
        if count < 0:
            raise ValueError(f"cannot reserve a negative number of identities, got {count}")
        if count == 0:
            return []
        name = await self._graph_of(graph)
        rows = await self._fetch(
            RESERVE_IDS_QUERY, {"graph": name, "label": label, "count": count}
        )
        if len(rows) != count:
            raise ValueError(f"there is no label {label!r} in graph {name!r}")
        return [row[0] for row in rows]

    async def load_vertices_with_ids(
        self,
        label: str,
        properties: Iterable[Mapping[str, Any]],
        *,
        graph: str | None = None,
    ) -> list[GraphId]:
        """Copy vertices into a label and return their identities, in the order given.

        The identities are reserved first (:meth:`reserve_ids`) and copied with the vertices, so a
        caller can write rows that refer to them -- a vector table keyed by the vertex, say -- in
        the same transaction without reading anything back. Two round trips and one copy stream.
        """
        name = await self._graph_of(graph)
        material = [dict(each) for each in properties]
        ids = await self.reserve_ids(label, len(material), graph=name)
        if not ids:
            return ids
        await self._copy_rows(
            vertex_copy_with_ids_statement(name, label),
            VERTEX_WITH_ID_COLUMN_TYPES,
            ([identity, Jsonb(props)] for identity, props in zip(ids, material, strict=True)),
        )
        return ids

    async def merge_vertices(
        self,
        label: str,
        key: str,
        rows: Iterable[Mapping[str, Any]],
        *,
        on_existing: str = "update",
        require_unique: bool = True,
        graph: str | None = None,
    ) -> dict[str, GraphId]:
        """Write elements by a key, as :meth:`upsert_vertices` does, and return every identity.

        The answer maps each row's key, as text, to the identity of its element -- the one already
        there or the one just created -- so a caller can go on to write rows that refer to them.
        Keys are read the way :meth:`identity_map` reads them, so ``1`` and ``"1"`` are one key.
        """
        material = list(rows)
        name, known, fresh, updates = await self._by_what_exists(
            label,
            key,
            material,
            on_existing=on_existing,
            require_unique=require_unique,
            graph=graph,
        )
        identities = dict(known)
        if fresh:
            created = await self.load_vertices_with_ids(label, fresh, graph=name)
            identities.update(
                (str(row[key]), identity) for row, identity in zip(fresh, created, strict=True)
            )
        if on_existing == "update" and updates:
            await self._run_with(overlap_update_statement(label), (Jsonb(updates),))
        return {str(row[key]): identities[str(row[key])] for row in material}

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
        name, _, fresh, updates = await self._by_what_exists(
            label,
            key,
            list(rows),
            on_existing=on_existing,
            require_unique=require_unique,
            graph=graph,
        )
        inserted = await self.load_vertices(label, fresh, graph=name) if fresh else 0
        updated = 0
        if on_existing == "update" and updates:
            await self._run_with(overlap_update_statement(label), (Jsonb(updates),))
            updated = len(updates)
        return UpsertCounts(inserted, updated)

    async def _by_what_exists(
        self,
        label: str,
        key: str,
        material: Sequence[Mapping[str, Any]],
        *,
        on_existing: str,
        require_unique: bool,
        graph: str | None,
    ) -> tuple[str, dict[str, GraphId], list[Mapping[str, Any]], list[dict[str, Any]]]:
        """The graph, the identities already there by *key*, the new rows, and updates for the rest.

        Refuses an *on_existing* other than ``"skip"`` or ``"update"`` before anything is read,
        and a key nothing makes unique unless *require_unique* is off.
        """
        if on_existing not in ("skip", "update"):
            raise ValueError(
                f"on_existing is 'skip' or 'update', not {on_existing!r}: there is no third thing "
                f"to do with an element that is already there"
            )
        name = await self._graph_of(graph)
        if require_unique and not await self._key_is_unique(label, key, graph=name):
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
        return name, known, fresh, updates

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
        return await self._copy_rows(
            edge_copy_statement(name, label), EDGE_COLUMN_TYPES, edge_rows(edges)
        )

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
        async with self._describing_a_copy(), self.cursor() as cursor:
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
        async with self._describing_a_copy(), self.cursor() as cursor:
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
        self,
        *,
        sample: int = 100,
        refresh: bool = False,
        counts: bool = True,
        graph: str | None = None,
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

        ``counts=False`` leaves the element counts out, which a schema for a prompt does not
        need: counting is the one part here that reads every element of the graph. ``counts``
        on the result is then empty.

        ``meta_gathered`` is the flag where the server keeps one. Before 2.18 it says only that
        the catalog holds triples, or that the graph has no edges to gather; whether it is still
        current is :meth:`meta_is_current`, which counts the edges to say so.
        """
        name = await self._graph_of(graph)
        if refresh and not await self.meta_is_current(graph=name):
            await self._run(GATHER_META)
        labels = await self.labels(graph=name)
        element_counts = await self.element_counts(graph=name) if counts else {}
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
        gathered = await self._meta_flag(name)
        if gathered is None:
            edges = sum(element_counts.get(label.name, 0) for label in labels if label.is_edge)
            gathered = bool(triples) or (counts and edges == 0)
        return GraphDescription(
            graph=name,
            labels=tuple(labels),
            properties=properties,
            triples=triples,
            counts=element_counts,
            meta_gathered=gathered,
        )

    async def meta_is_current(self, *, graph: str | None = None) -> bool:
        """Whether the triple catalog still describes this graph.

        ``regather_graphmeta()`` sets the flag, a transaction that wrote to the graph clears it
        at commit, and a read leaves it alone. So this separates a catalog nobody has gathered
        from one that is current and from one that was gathered and has since been written to.

        The flag arrived with 2.18. Before it, a server with ``auto_gather_graphmeta`` on keeps
        the catalog current itself; otherwise the catalog's edge counts are compared with the
        edges the graph holds, label by label. That comparison is exact and reads every edge
        (about 30 ms for 300,000), so a caller that asks often keeps :meth:`label_positions`
        from its last look and asks this only when those have moved.
        """
        name = await self._graph_of(graph)
        flagged = await self._meta_flag(name)
        if flagged is not None:
            return flagged
        auto = await self._fetch(AUTO_GATHER_QUERY, ())
        if auto and auto[0][0] == "on":
            return True
        rows = await self._fetch(meta_count_check_query(name), {"g": name})
        return bool(rows[0][0])

    async def label_positions(self, *, graph: str | None = None) -> dict[str, int]:
        """How far each label's id sequence has gone, by label.

        One catalog read, under a millisecond. Unmoved positions mean nothing was added since
        they were taken, which is the cheap way to decide whether :meth:`meta_is_current` is
        worth asking. A label whose sequence has never been used is missing from the answer.
        """
        name = await self._graph_of(graph)
        rows = await self._fetch(LABEL_POSITIONS_QUERY, (name,))
        return {label: int(position) for label, position in rows if position is not None}

    async def _meta_flag(self, graph: str) -> bool | None:
        """The server's own word on whether *graph*'s triple catalog is current; ``None`` before 2.18.

        Whether the server keeps that flag at all is asked once per connection.
        """
        held = self._agens_has_meta_flag
        if held is None:
            rows = await self._fetch(META_FLAG_QUERY, ())
            held = self._agens_has_meta_flag = bool(rows[0][0])
        if not held:
            return None
        rows = await self._fetch(META_VALID_QUERY, (graph,))
        return bool(rows) and bool(rows[0][0])

    async def declared_properties(
        self, label: str | None = None, *, graph: str | None = None
    ) -> list[DeclaredProperty]:
        """Every property given a column of its own, with that column's type.

        A property living in the JSON map is not declared anywhere and so cannot be listed.

        A server that cannot promote a property has nowhere to record one, and reading the catalog
        that is not there would raise instead of saying none -- so the catalog is asked for first,
        once per connection. Asked rather than worked out from the version, because the version
        does not answer it: a development build reports the line it is on and nothing of what it
        carries, and two servers both reporting ``2.18-devel`` were found to differ, one with the
        catalog and one without.
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

    async def vector_indexes(
        self, label: str | None = None, *, graph: str | None = None
    ) -> list[VectorIndex]:
        """Every vector index in its parts: property, method, operator class, type, width and
        the options it was made with.

        The width of an index over a property in the map is the cast's. The width of one over a
        promoted column is the column's, read from :meth:`declared_properties`.
        """
        name = await self._graph_of(graph)
        found = await self.indexes(label, graph=name)
        types = {(d.label, d.name): d.type for d in await self.declared_properties(graph=name)}
        described = (describe_vector_index(index, types) for index in found)
        return [index for index in described if index is not None]

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
        binary_: bool = False,
        prepare_: bool | None = None,
        row_: AsyncRowFactory[Any] | None = None,
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

        ``binary_``, ``prepare_`` and ``row_`` mean what they do on :meth:`execute_query`, for
        every statement in the list and for the asking again alike: the binary rendering, which
        carries a vector as packed floats rather than text, whether to prepare each statement, and
        the shape each row is read into.
        """
        sent = [(item, None) if isinstance(item, str) else item for item in statements]
        if binary_ and sent:
            self._check_binary()
        results: list[Result] = []
        for start in range(0, len(sent), chunk):
            batch = sent[start : start + chunk]
            # Behind a savepoint, because the asking again below cannot happen in a transaction the
            # failure has aborted: every statement would answer 25P02 and the real error would be
            # lost. The savepoint is taken *inside* the pipeline, so it rides in the round trip the
            # batch was already taking and the ordinary path pays nothing for it.
            mark = savepoint_name() if self._can_hold_a_savepoint() else None
            try:
                with operation_span(
                    "PIPELINE",
                    graph=self.label_table.graph,
                    **{"db.operation.batch.size": len(batch)},
                ):
                    results.extend(
                        await self._pipelined(
                            batch, savepoint=mark, binary=binary_, prepare=prepare_, row=row_
                        )
                    )
            except psycopg.Error:
                # The pipeline blamed one of them and a pipeline names the wrong one, so the
                # answer comes from asking again singly. Reading twice costs a round trip per
                # statement and buys the right statement in the traceback.
                if mark is not None:
                    # Two statements: inside a read-only block everything goes over the extended
                    # protocol, which takes one at a time.
                    await self._run(f"rollback to savepoint {mark}")
                    await self._run(f"release savepoint {mark}")
                for statement, params in batch:
                    results.append(
                        await self.execute_query(
                            statement, params, binary_=binary_, prepare_=prepare_, row_=row_
                        )
                    )
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
        self,
        batch: Sequence[tuple[str, Params | None]],
        *,
        savepoint: str | None = None,
        binary: bool = False,
        prepare: bool | None = None,
        row: AsyncRowFactory[Any] | None = None,
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
                cursor = self.cursor(row_factory=row) if row else self.cursor()
                if binary:
                    cursor.format = psycopg.pq.Format.BINARY
                await cursor.execute(statement, params, prepare=prepare)
                cursors.append(cursor)
            if savepoint is not None:
                await self._run(f"release savepoint {savepoint}")
        gathered = []
        for cursor in cursors:
            described = cursor.description
            records = await self._read_all(cursor) if described is not None else []
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
        accepts an unknown one silently. Every option goes in one statement.

        A setting local to a transaction needs one to be open. In autocommit, outside a pipeline,
        there is none, and the server would only warn and leave the search as it was -- so this
        warns instead, and sends nothing. 3.0 will raise. To tune one search,
        ``execute_query(..., settings_=...)`` sends the settings with it in one round trip.
        """
        checked = [(name, check_search_option(name, value)) for name, value in options.items()]
        if not checked:
            return
        if local and self._outside_any_transaction():
            warnings.warn(
                "vector_search_options(local=True) does nothing in autocommit outside a "
                "transaction or pipeline: the setting would end before any search could see it. "
                "Pass the settings to execute_query(settings_=...) to send them with the search, "
                "open a transaction, or pass local=False. This will raise in agensgraph 3.0.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        statement, values = settings_statement(checked, local=local)
        async with super().cursor() as cursor:
            await cursor.execute(statement, values)

    def _outside_any_transaction(self) -> bool:
        """Whether a setting local to the transaction would end before the next statement."""
        return (
            self.autocommit
            and self.pgconn.pipeline_status == PipelineStatus.OFF
            and self.pgconn.transaction_status == TransactionStatus.IDLE
        )

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

        An element cast to a type is refused on a property that has a column of its own. A
        search reads that column uncast, and never uses such an index.
        """
        name = await self._graph_of(graph)
        cast = {
            (want.label, element.property)
            for want in desired
            for element in want.elements
            if element.cast is not None
        }
        if cast:
            declared = {(d.label, d.name) for d in await self.declared_properties(graph=name)}
            promoted = sorted(cast & declared)
            if promoted:
                raise ValueError(
                    f"{promoted} have columns of their own, and an index over a cast of the "
                    f"property is never used by a search on the column. Leave the cast out."
                )

        async def planned() -> list[str]:
            found = for_labels(await self.indexes(graph=name), desired, drop_extra)
            return reconcile_indexes(desired, found, drop_extra=drop_extra)

        async def settled() -> None:
            await self._settled(
                reconcile_indexes(desired, await self.indexes(graph=name)), "indexes"
            )

        return await self._reconciled(name, planned, settled, dry_run=dry_run)

    async def _reconciled(
        self,
        graph: str,
        planned: Callable[[], Awaitable[list[str]]],
        settled: Callable[[], Awaitable[None]],
        *,
        dry_run: bool,
    ) -> list[str]:
        """What reconciling asks for, run unless this is a dry run; the statements either way.

        Worked out first without the graph's schema lock, so a graph that already matches costs
        the reading alone and takes no lock. Otherwise worked out again under the lock -- another
        process may have done some of it meanwhile -- run, and checked by *settled*.
        """
        statements = await planned()
        if dry_run or not statements:
            return statements
        async with self.schema_lock(graph=graph):
            statements = await planned()
            for statement in statements:
                await self._run(statement)
            if statements:
                await settled()
        return statements

    @asynccontextmanager
    async def schema_lock(self, *, graph: str | None = None) -> AsyncGenerator[None]:
        """Hold this graph's schema lock in a transaction, for changing what the graph declares.

        Every ``ensure_*`` method runs under it, so two processes reconciling one graph at once
        run one after the other: the second reads what the first left and has nothing to do,
        rather than both creating the same label and one failing with ``42P07``. A caller making
        DDL of its own takes it the same way. It lasts until the transaction ends, so a failure or a
        lost connection releases it; inside an open transaction it lasts until that one ends.
        """
        name = await self._graph_of(graph)
        async with self.transaction():
            await self._fetch(SCHEMA_LOCK_QUERY, (f"agensgraph:schema:{name}",))
            yield

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

        A label declaring :class:`~agensgraph.PromotedProperty` columns is refused where the
        server cannot promote one. On a label already there the columns it lacks are added;
        one it has, its own or inherited, is left as it is.
        """
        name = await self._graph_of(graph)
        declared: list[DeclaredProperty] = []
        if any(want.properties for want in desired):
            if not await self.can_promote_properties():
                raise CapabilityError.for_feature(
                    "a property with a column of its own",
                    required="2.18",
                    found=self.capabilities.reported,
                )
            declared = await self.declared_properties(graph=name)

        async def planned() -> list[str]:
            return reconcile_labels(desired, await self.labels(graph=name), declared)

        return await self._reconciled(name, planned, self.refresh_labels, dry_run=dry_run)

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

        async def planned() -> list[str]:
            found = for_labels(await self.constraints(graph=name), desired, drop_extra)
            return reconcile_constraints(desired, found, drop_extra=drop_extra)

        async def settled() -> None:
            await self._settled(
                reconcile_constraints(desired, await self.constraints(graph=name)),
                "constraints",
            )

        return await self._reconciled(name, planned, settled, dry_run=dry_run)

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

        Where the server reports the path, the table is checked against it first. psycopg ends
        a ``transaction()`` block without a cursor of this connection, so no statement watch
        saw the path go back.
        """
        if given is not None:
            return given
        self._follow_reported_graph_path()
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

    async def _run(self, statement: str, params: Params | None = None) -> None:
        """One of the driver's own statements: a savepoint, a setting, a lock released.

        Never prepared, and past the read-only block's checks; see
        :meth:`AsyncCursor._execute_own`.
        """
        async with AsyncCursor(self) as cursor:
            await cursor._execute_own(statement, params)

    async def _fetch(self, statement: str, params: Params) -> list[Any]:
        """The rows of one of the driver's own reads, as tuples, past the read-only block's checks.

        Prepared as psycopg would, since the catalog reads it serves come back again and again.
        """
        async with AsyncCursor(self, row_factory=tuple_row) as cursor:
            await cursor._execute_own(statement, params, prepare=None)
            return await cursor.fetchall()
