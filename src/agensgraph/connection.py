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

from typing import TYPE_CHECKING, Any, cast

import psycopg
from psycopg.adapt import AdaptersMap
from psycopg.rows import Row, tuple_row

from ._core import (
    GRAPH_ADAPTERS,
    GraphMixin,
    Result,
    statement_text,
    stream_name,
    with_keepalives,
)
from ._protocol.labels import CURRENT_GRAPH_QUERY, LabelCache
from .bulk import (
    EDGE_COLUMN_TYPES,
    VERTEX_COLUMN_TYPES,
    build_identity_map,
    edge_copy_statement,
    edge_rows,
    identity_map_statement,
    vertex_copy_statement,
    vertex_rows,
)
from .capabilities import VECTOR_AVAILABLE_QUERY, VECTOR_VERSION_QUERY
from .columnar import CHUNK
from .cypher import changes_graph_path, check_bindable_positions, wrap_for_cursor
from .errors import BatchFailed, NoEnclosingTransaction
from .introspect import (
    CONSTRAINTS_QUERY,
    DECLARED_PROPERTIES_QUERY,
    GRAPHS_QUERY,
    INDEXES_QUERY,
    LABELS_QUERY,
    Check,
    Constraint,
    DeclaredProperty,
    DesiredIndex,
    Graph,
    Index,
    Label,
    Unique,
    element_count_query,
    reconcile_constraints,
    reconcile_indexes,
)
from .notify import LISTENING_QUERY, NOTIFY_QUERY, listen_statement, unlisten_statement
from .observability import Timer, query_span, report_notice
from .summary import (
    ASSIGNED_TRANSACTION_QUERY,
    COUNTER_QUERY,
    TRANSACTION_ID_QUERY,
    TRANSACTION_STATUS_QUERY,
    CommitOutcome,
    read_outcome,
)
from .vector import search_option_statements

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from typing import Self

    from psycopg.abc import Params, Query, QueryNoTemplate
    from psycopg.rows import RowFactory

    from ._core import Statement
    from ._protocol.graphid import GraphId
__all__ = ["Connection", "Cursor"]


class Cursor(psycopg.Cursor[Row]):
    """A cursor that watches the statements going past it.

    Every way of running a statement arrives here: ``execute_query`` builds a cursor,
    ``connection.execute`` builds one, and a caller may take one and use it directly. So one
    check here holds for all of them.

    Two things are watched for. A statement whose parameter the server would read as something
    else is refused before it is sent. A statement that moves the session to another graph
    leaves the label table describing a graph the session is no longer reading, so the table
    is dropped once such a statement has run.
    """

    def execute(
        self,
        query: Query,
        params: Params | None = None,
        *,
        prepare: bool | None = None,
        binary: bool | None = None,
    ) -> Self:
        text = statement_text(query)
        self._guard()
        check_bindable_positions(text)
        super().execute(cast("QueryNoTemplate", query), params, prepare=prepare, binary=binary)
        self._watch_graph_path(text)
        return self

    def executemany(
        self, query: Query, params_seq: Iterable[Params], *, returning: bool = False
    ) -> None:
        text = statement_text(query)
        self._guard()
        check_bindable_positions(text)
        super().executemany(query, params_seq, returning=returning)
        self._watch_graph_path(text)

    def _guard(self) -> Connection[Row]:
        """The connection, refused if its holder has given it back."""
        conn = cast("Connection[Row]", self.connection)
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
        conn = cast("Connection[Row]", self.connection)
        conn.label_table.invalidate()
        if not conn.autocommit:
            conn._agens_graph_path_in_transaction = True


class Connection(GraphMixin, psycopg.Connection[Row]):
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
        self.cursor_factory = Cursor
        self._agens_adapters = AdaptersMap(GRAPH_ADAPTERS)
        self._agens_labels = LabelCache()

    def commit(self) -> None:
        self._check_lent()
        super().commit()
        self._agens_graph_path_in_transaction = False

    @classmethod
    def connect(cls, conninfo: str = "", **kwargs: Any) -> Connection[Any]:
        """Open a connection and refuse a server this driver cannot read.

        The version arrives in the startup packet, so the refusal costs no round trip and
        happens before the first statement rather than at whichever later one first wants a
        catalog the server has never had.

        Keepalive is asked for unless the caller decided otherwise; see
        :data:`~agensgraph._core.KEEPALIVE_DEFAULTS` for what that is worth.
        """
        conn = super().connect(conninfo, **with_keepalives(kwargs, conninfo))
        conn.add_notice_handler(report_notice)
        try:
            _ = conn.capabilities
        except BaseException:
            conn.close()
            raise
        return conn

    def graph(self, name: str) -> None:
        """Read from a graph, and fill the label table for it.

        Two statements: one to move the session, one to read the labels of where it now is.
        The table is what the composite rendering needs to name a label, and filling it here
        means asking for that rendering later does not have to stop and ask.
        """
        self._run(self._select_graph_statement(name))
        rows = self._fetch(self._label_statement(), (name,))
        self._accept_labels(name, rows)

    def refresh_labels(self) -> None:
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
            graph = self._current_graph()
            if graph is None:
                return
        self._accept_labels(graph, self._fetch(self._label_statement(), (graph,)))

    def _current_graph(self) -> str | None:
        """The graph the session is reading, or ``None`` if it is reading none."""
        rows = self._fetch(CURRENT_GRAPH_QUERY, ())
        name = rows[0][0] if rows else ""
        return name or None

    def rollback(self) -> None:
        """Roll back, and drop the label table if the graph path is going back with it.

        Setting the graph path is part of a transaction, so rolling one back returns the
        session to the graph it was reading before, and a table filled inside that transaction
        describes somewhere the session no longer is.
        """
        self._check_lent()
        super().rollback()
        if self._agens_graph_path_in_transaction:
            self.label_table.invalidate()
            self._agens_graph_path_in_transaction = False

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
        text = statement_text(query)
        timer = Timer()
        with query_span(text, graph=self.label_table.graph):
            with self.cursor(row_factory=row_) if row_ else self.cursor() as cursor:
                if binary_:
                    self._check_binary()
                    cursor.format = psycopg.pq.Format.BINARY
                before: Sequence[int] | None = None
                if counts_:
                    before = self._counters()
                try:
                    cursor.execute(query, params, prepare=prepare_)
                except psycopg.Error as exc:
                    failure = self._translated(exc)
                    self._report_query(text, timer, rows=0, error=failure)
                    raise failure from None
                described = cursor.description
                records = cursor.fetchall() if described is not None else []
                keys = [column.name for column in described or ()]
                oids = tuple(int(column.type_code) for column in described or ())
        after = self._counters() if counts_ else None
        self._report_query(text, timer, rows=len(records), error=None)
        return Result(records, keys, self._counts_for(text, before, after), oids)

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

    def stream(
        self,
        query: str,
        params: Params | None = None,
        *,
        size: int = 100,
        binary_: bool = False,
        name: str | None = None,
    ) -> Iterator[Any]:
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
        self._check(query)
        statement = wrap_for_cursor(query)
        if self.autocommit and name is None:
            raise NoEnclosingTransaction.for_stream()
        with self.cursor(name=name or stream_name()) as cursor:
            cursor.itersize = size
            if binary_:
                self._check_binary()
                cursor.format = psycopg.pq.Format.BINARY
            try:
                cursor.execute(statement, params)
            except psycopg.Error as exc:
                raise self._translated(exc) from None
            while True:
                rows = cursor.fetchmany(size)
                if not rows:
                    return
                for row in rows:
                    yield row

    def register_vectors(self) -> tuple[str, ...]:
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
            info = TypeInfo.fetch(self, name)
            if info is None:
                continue
            accept(self, info)
            found.append(name)
        return tuple(found)

    def has_vectors(self) -> bool:
        """Whether vectors can be read here at all, without registering anything."""
        rows = self._fetch(VECTOR_AVAILABLE_QUERY, ())
        return bool(rows and rows[0][0])

    def vector_version(self) -> tuple[int, ...] | None:
        """What version of pgvector is created here, or ``None`` if none is.

        A version rather than a yes or no, because pgvector gates its own features on it: sparse
        vectors and half precision arrived in 0.7.0 and iterative index scans in 0.8.0, so a caller
        deciding whether to use one has a number to compare rather than a boolean to guess from.
        """
        rows = self._fetch(VECTOR_VERSION_QUERY, ())
        if not rows or rows[0][0] is None:
            return None
        return tuple(int(part) for part in str(rows[0][0]).split(".") if part.isdigit())

    def load_vertices(
        self, label: str, properties: Iterable[Mapping[str, Any]], *, graph: str | None = None
    ) -> int:
        """Copy vertices into a label, and report how many.

        Faster than any statement because it is one stream rather than a statement per row:
        measured at 223,000 rows a second against 140,000 for a single ``UNWIND ... CREATE`` and
        47,000 for one statement each.

        No identity is supplied. The column's default produces the same identities a ``CREATE``
        would, so nothing here has to reproduce the server's numbering.
        """
        name = self._graph_of(graph)
        loaded = 0
        with self.cursor() as cursor, cursor.copy(vertex_copy_statement(name, label)) as copy:
            copy.set_types(VERTEX_COLUMN_TYPES)
            for row in vertex_rows(properties):
                copy.write_row(row)
                loaded += 1
        return loaded

    def load_edges(
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
        name = self._graph_of(graph)
        loaded = 0
        with self.cursor() as cursor, cursor.copy(edge_copy_statement(name, label)) as copy:
            copy.set_types(EDGE_COLUMN_TYPES)
            for row in edge_rows(edges):
                copy.write_row(row)
                loaded += 1
        return loaded

    def load_vertex_frame(
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

        name = self._graph_of(graph)
        chunks = vertex_payloads(source, size=size)
        with self.cursor() as cursor:
            with cursor.copy(vertex_copy_statement(name, label)) as copy:
                for block in vertex_blocks(payload for chunk in chunks for payload in chunk):
                    copy.write(block)
            return max(cursor.rowcount, 0)

    def load_edge_frame(
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

        name = self._graph_of(graph)
        chunks = edge_payloads(source, start=start, end=end, size=size)
        with self.cursor() as cursor:
            with cursor.copy(edge_copy_statement(name, label)) as copy:
                for block in edge_blocks(row for chunk in chunks for row in chunk):
                    copy.write(block)
            return max(cursor.rowcount, 0)

    def identity_map(
        self, label: str, key: str, *, graph: str | None = None
    ) -> dict[str, GraphId]:
        """What the server called each element of a label, keyed by one of its properties.

        One statement for the whole label. The key is read as text on both sides, because a key
        that is a number in one place and a string in the other would otherwise match nothing.

        The key has to identify an element, and it is refused if it does not. A map is what
        :meth:`load_edges` resolves an endpoint through, so an element sharing its key with
        another, or holding no key at all, is not a smaller map -- it is edges attached to the
        wrong vertex, or to none.
        """
        name = self._graph_of(graph)
        rows = self._fetch(identity_map_statement(name, label), (key,))
        return build_identity_map(rows, label=label, key=key)

    def graphs(self) -> list[Graph]:
        """Every graph, since there is no ``\\d`` that knows about them."""
        return [Graph(*row) for row in self._fetch(GRAPHS_QUERY, ())]

    def labels(self, *, graph: str | None = None) -> list[Label]:
        """Every label of a graph, in the order the server gave them ids.

        Defaults to the graph this connection is reading. The two a graph is created with are
        included and say so, because a caller counting labels should not have to know that two
        of them were never asked for.
        """
        return [Label(*row) for row in self._fetch(LABELS_QUERY, (self._graph_of(graph),))]

    def declared_properties(
        self, label: str | None = None, *, graph: str | None = None
    ) -> list[DeclaredProperty]:
        """Every property given a column of its own, with that column's type.

        A property living in the JSON map is not declared anywhere and so cannot be listed;
        before 2.18 nothing can be promoted at all and this is always empty.
        """
        name = self._graph_of(graph)
        rows = self._fetch(DECLARED_PROPERTIES_QUERY, (name, label, label))
        return [DeclaredProperty(*row) for row in rows]

    def indexes(self, label: str | None = None, *, graph: str | None = None) -> list[Index]:
        """Every property index. A uniqueness constraint is not one; see :meth:`constraints`."""
        name = self._graph_of(graph)
        return [Index(*row) for row in self._fetch(INDEXES_QUERY, (name, label, label))]

    def constraints(
        self, label: str | None = None, *, graph: str | None = None
    ) -> list[Constraint]:
        """Every constraint on a label's properties, uniqueness assertions included.

        Read from the constraint catalog rather than from the property-index view, which filters
        exclusion constraints out -- and a uniqueness assertion is kept as an exclusion, so it
        would otherwise be invisible.
        """
        name = self._graph_of(graph)
        rows = self._fetch(CONSTRAINTS_QUERY, (name, label, label))
        return [Constraint(*row) for row in rows]

    def pipeline_batch(
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
            with self.pipeline():
                for statement, params in sent:
                    self.execute(statement, params)
        except psycopg.Error as exc:
            failure = BatchFailed(
                f"one of {len(sent)} pipelined statements failed, and a pipeline does not report which: the error below may belong to any of them. Run them one at a time to find it."
            )
            failure.statements = tuple((statement for statement, _ in sent))
            raise failure from exc

    def vector_search_options(
        self, options: Mapping[str, object], *, local: bool = True
    ) -> None:
        """Tune a vector search, by default for the current transaction only.

        Setting ``hnsw.ef_search`` higher looks at more candidates, and so recalls more of the true
        nearest neighbours. :data:`agensgraph.vector.SEARCH_OPTIONS` lists
        what can be set; a name that is not one of them is refused rather than sent, since the server
        accepts an unknown one silently.
        """
        for statement in search_option_statements(options, local=local):
            self._run(statement)

    def listen(self, *channels: str) -> None:
        """Subscribe to channels the server announces on.

        The channel is quoted into the statement, since neither ``LISTEN`` nor ``UNLISTEN`` takes a
        parameter for it.
        """
        for channel in channels:
            self._run(listen_statement(channel))

    def unlisten(self, *channels: str) -> None:
        """Stop listening. Given no channel, stop listening to all of them."""
        for statement in [unlisten_statement(name) for name in channels] or [
            unlisten_statement()
        ]:
            self._run(statement)

    def listening(self) -> list[str]:
        """The channels this connection is subscribed to."""
        rows = self._fetch(LISTENING_QUERY, ())
        return [row[0] for row in rows]

    def notify(self, channel: str, payload: str = "") -> None:
        """Announce something on a channel.

        Sent through ``pg_notify``, which takes the channel as a parameter, so a channel name held in a
        variable needs no quoting and cannot carry a statement of its own.
        """
        self._fetch(NOTIFY_QUERY, (channel, payload))

    def notifications(
        self, *, timeout: float | None = None, stop_after: int | None = None
    ) -> Iterator[psycopg.Notify]:
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
                "this connection already has a notify handler, and reading announcements both ways at once delivers each one to whichever route happens to be looking. Remove the handler, or read them through it alone."
            )
        for notice in super().notifies(timeout=timeout, stop_after=stop_after):
            yield notice

    def ensure_indexes(
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

        The state is read again afterwards and anything still outstanding is raised. Naming an
        operator class that is already the default does this, since the server omits a default when
        printing a definition.
        """
        name = self._graph_of(graph)
        statements = reconcile_indexes(desired, self.indexes(graph=name), drop_extra=drop_extra)
        if dry_run:
            return statements
        for statement in statements:
            self._run(statement)
        if statements:
            self._settled(reconcile_indexes(desired, self.indexes(graph=name)), "indexes")
        return statements

    def ensure_constraints(
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
        """
        name = self._graph_of(graph)
        statements = reconcile_constraints(
            desired, self.constraints(graph=name), drop_extra=drop_extra
        )
        if dry_run:
            return statements
        for statement in statements:
            self._run(statement)
        if statements:
            self._settled(
                reconcile_constraints(desired, self.constraints(graph=name)), "constraints"
            )
        return statements

    def _settled(self, outstanding: list[str], what: str) -> None:
        """Raise if the state still differs from what was asked for."""
        if outstanding:
            raise RuntimeError(
                f"the {what} asked for were applied but still do not match what the catalogs report, so running this again would repeat the same work. Still outstanding: {outstanding}"
            )

    def element_counts(self, *, graph: str | None = None) -> dict[str, int]:
        """How many vertices and edges each label holds.

        Two statements and no property read: the label id is part of every element's identity,
        so counting per label needs nothing from a row but its id.
        """
        name = self._graph_of(graph)
        names = {label.id: label.name for label in self.labels(graph=name)}
        counts: dict[str, int] = {}
        for edges in (False, True):
            for labid, count in self._fetch(element_count_query(name, edges=edges), ()):
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
