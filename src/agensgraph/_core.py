"""The part of a connection that does no input or output.

A driver that offers both a blocking and an awaiting interface has to decide what it keeps
in one copy. Everything that does not touch the socket lives here, in one copy, shared by
both: the adapters map, the capability gate, the label table, the statement checks, and the
assembly of a result. What is left over is the handful of methods that send something and
wait for an answer, and those exist twice because there is no way to write them once.

The split is worth keeping honest, because it is the whole argument for not generating one
interface from the other: the more that lives here, the less there is to keep in step.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING, Any, NamedTuple, TypeVar, Union, cast

from psycopg import errors as _pg
from psycopg.conninfo import conninfo_to_dict as _conninfo_to_dict
from psycopg.pq import TransactionStatus

from ._protocol.labels import LabelCache
from ._watchdog import GRACE, WATCHDOG
from .adapters import graph_adapters, register_binary, register_text
from .capabilities import Capabilities
from .cypher import (
    check_bindable_positions,
    check_single_statement,
    check_transaction_kept,
    first_write_clause,
    is_session_command,
    quote_identifier,
    writable_counters,
)
from .deadline import Stalled
from .errors import (
    ReadOnlyBoundaryLost,
    ReadOnlyGraphWrite,
    ReleasedConnection,
    StaleLabelCache,
    explain_string_type,
    redact_details,
    translate,
)
from .observability import QueryRecord, logging_wanted, report
from .summary import GraphWriteCounts

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from psycopg import ConnectionInfo
    from psycopg.abc import PQGen
    from psycopg.adapt import AdaptersMap
    from psycopg.pq.abc import PGconn
    from psycopg.sql import SQL, Composed

    from ._watchdog import Watch
_RV = TypeVar("_RV")

__all__ = [
    "GRAPH_ADAPTERS",
    "AsyncConninfoSource",
    "ConninfoSource",
    "GraphMixin",
    "Result",
    "Statement",
    "savepoint_name",
    "stream_name",
]


@dataclass(frozen=True)
class ReadOnlyBlock:
    """What a read-only block open on a connection holds every statement to.

    ``refuse_writes`` is set where the server would let a Cypher write through a read-only
    transaction, so the driver refuses one before it is sent rather than relying on the
    rollback at the end of the block to undo it.
    """

    refuse_writes: bool


GRAPH_ADAPTERS: AdaptersMap = graph_adapters()
"""The graph types, registered once for the process.

Every connection derives its own map from this one, so registering something on a connection
does not reach the others and nothing here reaches a plain PostgreSQL connection made by the
same process.
"""


class Result(NamedTuple):
    """What one statement produced.

    ``records`` are the rows as the row factory built them, ``keys`` the column names, and
    ``counts`` what the statement changed -- which for a statement that changed nothing is
    five ``None`` rather than five zeros, because it was never asked.

    ``oids`` are the type of each column as the server described it, which is what lets a
    columnar export give an empty result its schema. A Cypher expression is jsonb whatever it
    evaluates to, so for most of a graph result these say only that.
    """

    records: list[Any]
    keys: list[str]
    counts: GraphWriteCounts
    oids: tuple[int, ...] = ()


KEEPALIVE_DEFAULTS = {
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 3,
}
"""What a connection asks for unless the caller says otherwise.

Without these a connection whose network stops carrying packets waits for the kernel, which is two
hours and a quarter on Linux by default. Measured against a server whose traffic was dropped: with
nothing set the wait had not ended after twenty-two seconds; with these it ends in about a minute.

Each is filled in on its own, so naming one of them does not silently leave the others at the
system's values. ``keepalives=0`` turns them all off.

``tcp_user_timeout`` is **not** among them, and it is worth saying why, because it is the setting
usually recommended for this. It bounds how long *transmitted* data may go unacknowledged, and a
connection waiting for a reply has transmitted nothing -- measured, ``tcp_user_timeout`` alone left
the same wait unbounded. What it does do is bound the keepalive probing once that is on, so it is
useful alongside these and useless instead of them. It is left unset because it also applies to a
connection that is busy sending, where too small a value would end a healthy one.
"""


_stream_numbers = count(1)
_savepoint_numbers = count(1)


def stream_name() -> str:
    """A name no other stream on this connection is using.

    Two server-side cursors of one name collide, and the collision aborts the transaction --
    which takes down the stream that was already reading, not only the one that failed.
    """
    return f"agens_stream_{next(_stream_numbers)}"


def savepoint_name() -> str:
    """A name no other savepoint this driver takes is using.

    Generated rather than fixed so that one taken inside another, which a caller nesting these
    would produce, releases the one it took and not the one around it.
    """
    return f"agens_batch_{next(_savepoint_numbers)}"


def with_keepalives(kwargs: dict[str, Any], conninfo: str = "") -> dict[str, Any]:
    """The connection arguments, with any keepalive setting the caller left out filled in.

    Filled in one key at a time rather than all or nothing, because the settings do nothing useful
    apart: a caller who names ``keepalives_interval`` and not ``keepalives_idle`` would otherwise get
    the system's idle time, which is two hours and a quarter, and their interval would never be
    reached.

    ``keepalives=0`` is the one thing that turns the rest off, since it says so.

    The connection string is read as well as the arguments, because a setting is as likely to be
    written there and psycopg lays the arguments over it. A default filled in as an argument would
    otherwise displace what the string asked for, which is the opposite of leaving a caller's
    decision alone.

    ``tcp_user_timeout`` does **not** count as having decided. It bounds how long transmitted data may
    go unacknowledged, and a connection waiting for a reply has transmitted nothing -- so a caller who
    sets only that has asked for something that does not bound a hung read, and keepalive is still
    filled in for them.
    """
    named = dict(_conninfo_to_dict(conninfo)) if conninfo else {}
    named.update(kwargs)
    if str(named.get("keepalives", 1)) == "0":
        return kwargs
    return {**{k: v for k, v in KEEPALIVE_DEFAULTS.items() if k not in named}, **kwargs}


def with_idle_limit(kwargs: dict[str, Any], conninfo: str = "") -> dict[str, Any]:
    """The connection arguments, with ``idle_in_transaction_timeout`` made a startup setting.

    A connection that opens a transaction and then stops -- a crashed worker, a caller holding a
    transaction across a slow call of its own -- holds its locks until something ends it. The
    server's ``idle_in_transaction_session_timeout`` ends it, and as a startup setting it costs no
    statement. It goes into ``options`` after whatever the caller already put there, from the
    arguments or the connection string, rather than replacing it.
    """
    seconds = kwargs.pop("idle_in_transaction_timeout", None)
    if seconds is None:
        return kwargs
    if isinstance(seconds, bool) or not isinstance(seconds, int | float) or seconds <= 0:
        raise ValueError(f"idle_in_transaction_timeout is a number of seconds, got {seconds!r}")
    given = kwargs.get("options")
    if given is None and conninfo:
        given = _conninfo_to_dict(conninfo).get("options")
    setting = f"-c idle_in_transaction_session_timeout={max(1, round(seconds * 1000))}"
    return {**kwargs, "options": f"{given} {setting}" if given else setting}


def _look_again_at(conn: GraphMixin, now: float) -> float | None:
    """When the watchdog should look at a connection again, or ``None`` to shut it down now.

    Quiet too long is a statement running and nothing heard for ``GRACE`` longer than the limit
    the server holds now -- read each time, since a rollback can take the limit back. With no limit
    there is nothing to measure silence against, so it only looks again. On the watchdog's thread:
    it reads two attributes and does nothing else.
    """
    limit = conn._agens_statement_timeout_ms
    if limit is None:
        return now + GRACE
    quiet = limit / 1000 + GRACE
    heard = conn._agens_heard_at
    if heard is None:
        return now + quiet
    return None if now - heard >= quiet else heard + quiet


class GraphMixin:
    """Everything a graph connection knows that does not require waiting for the server."""

    _agens_adapters: AdaptersMap
    pgconn: PGconn

    if TYPE_CHECKING:
        # psycopg defines it as a property, not a field.
        @property
        def info(self) -> ConnectionInfo: ...

    _agens_capabilities: Capabilities | None = None
    _agens_has_meta_flag: bool | None = None
    """Whether this server records that a graph's triple catalog is current.

    Asked once and kept, since a server does not grow a catalog column while a connection is open.
    """

    _agens_can_run_programs: bool | None = None
    """Whether this role could run a command on the server's host through ``COPY``.

    Asked once and kept, since a role does not gain the privilege while a connection is open.
    """

    _agens_can_promote: bool | None = None
    """Whether this server has the catalog a promoted property is recorded in.

    Asked once and kept, since a server does not grow the catalog while a connection is open.
    """
    _agens_labels: LabelCache
    _agens_binary_ready: bool = False
    _agens_generation: int = 0
    """Which generation of a pool this connection belongs to, if it came from one.

    Set when the pool creates it and read when it comes back, so that a pool which has moved
    on closes it instead of handing it to somebody else.
    """

    _agens_graph_path_in_transaction: bool = False
    """Whether the graph path was set inside the transaction now open.

    Setting it is part of the transaction, so a rollback puts it back and takes the label
    table out of step with the session. Kept only where the path is read from the statements
    that go past; a server that reports it says so itself after the rollback.
    """

    _agens_reports_graph_path: bool = False
    """Whether the server sends the graph path as a run-time parameter.

    From 2.18.6. Not a version question, though: 2.18.4 and 2.18.6 report the same two leading
    numbers and differ here, so it is read off the connection rather than off the version.
    """

    _agens_graph_path_seen: str = ""
    """The graph path this connection last read from a reported parameter."""

    _agens_pooled: bool = False
    """Whether a pool owns this connection, and so decides who may use it and when."""

    _agens_lent: bool = False
    """Whether a pool has lent this connection to somebody, and not had it back."""

    _agens_statement_timeout_ms: int | None = None
    """The statement timeout the driver set on this connection, or ``None`` for the default.

    Kept so that a caller wanting the limit the connection already holds is not sent the statement
    that sets it again, and so that the watchdog knows how long the server may go unheard.
    """

    _agens_timeout_before_transaction: int | None = None
    _agens_timeout_set_in_transaction: bool = False
    """Whether the limit was set inside the transaction now open, and what was held before it.

    Such a setting is part of the transaction, so a rollback takes it back, and the note above goes
    back with it when the transaction ends.
    """

    _agens_watches: frozenset[Watch] = frozenset()
    """The watches armed on this connection; while there is any, its waits are timed."""

    _agens_heard_at: float | None = None
    """When the server was last heard from while a statement ran, on the monotonic clock.

    ``None`` while no statement is running, and noted only while a watch is armed. Written by the
    connection's own thread and read by the watchdog's.
    """

    _agens_returned_at: float = 0.0
    """When a pool last had this connection back, on the monotonic clock.

    A connection returned moments ago, with nothing since suggesting the server went away, is lent
    again without the round trip a health check costs.
    """

    _agens_cancelled: bool = False
    """Whether a statement on this connection was interrupted rather than finished.

    A pool closes such a connection instead of lending it again. psycopg does try to leave one
    clean -- it asks the server to cancel and then reads what is still coming -- and when that
    finishes the connection really is idle and really is reusable. What it cannot promise is
    that it finished: the read it re-enters is bounded by nothing
    (``psycopg/connection_async.py:526``), so on a connection whose packets stopped arriving it
    is still there. A connection that has been through that is cheaper to replace than to
    reason about, and a cancellation is rare enough that replacing it costs nothing anybody
    will see.
    """

    _agens_read_only: ReadOnlyBlock | None = None
    """The read-only block open on this connection, if one is."""

    _agens_ran_others_text: bool = False
    """Whether a read-only block has run a statement since a pool last had this connection back.

    Such a statement is text from elsewhere, which may take a session-level advisory lock: the one
    thing it can do that the block's rollback does not undo. A pool releases them before lending
    the connection to anybody else; a block that ran nothing costs nothing.
    """

    _agens_read_only_enforced: bool | None = None
    """Whether this server refused a Cypher write in a read-only transaction when asked.

    Asked once, by :meth:`read_only_is_enforced`, and kept: neither the server nor the role
    changes while a connection is open. Until it is asked, the version answers.
    """

    _agens_untrusted_role: tuple[tuple[str | None, tuple[str, ...]], tuple[str, ...]] | None = (
        None
    )
    """The last answer about this role's fitness for model-written Cypher, and what it was for.

    Kept for one application role and set of schemas: a role's memberships do not change while a
    connection is open, and a connection serves one application.
    """

    def _read_only_gate(self) -> bool:
        """Whether this server refuses a Cypher write in a read-only transaction.

        The probe's answer where it has been asked, and the version's otherwise. The version
        answers no for a development build or a release candidate of any line, which may predate
        the fix, and for a 2.14 or 2.15 that carries it; that only costs those servers a refusal
        the driver did not need to make, and :meth:`read_only_is_enforced` removes it.
        """
        if self._agens_read_only_enforced is not None:
            return self._agens_read_only_enforced
        return self.capabilities.has("cypher_write_enforced")

    def _check_read_only_statement(self, text: str) -> None:
        """Refuse, before it is sent, a statement a read-only block cannot hold.

        More than one statement, a statement that ends the transaction, and on a server that
        lets Cypher writes through a read-only transaction, a Cypher write.
        """
        block = self._agens_read_only
        if block is None:
            return
        check_single_statement(text)
        check_transaction_kept(text)
        # A session command is SQL, which the server holds to the transaction on every release;
        # making the transaction read-write again is refused once it has taken its snapshot.
        if block.refuse_writes and not is_session_command(text):
            clause = first_write_clause(text)
            if clause is not None:
                raise ReadOnlyGraphWrite.before_sending(clause, self.capabilities.reported)
        self._agens_ran_others_text = True

    def _check_read_only_kept(self) -> None:
        """Refuse to go on if the block's transaction ended under it."""
        if (
            self._agens_read_only is not None
            and self.pgconn.transaction_status == TransactionStatus.IDLE
        ):
            raise self._boundary_lost(
                "the read-only transaction ended while the block was open, so what ran after it "
                "was not read-only"
            )

    def _boundary_lost(self, message: str) -> ReadOnlyBoundaryLost:
        """The error for a boundary that did not hold, with the connection marked as not to be
        trusted again: a pool closes it rather than lend it to the next caller."""
        self._agens_cancelled = True
        return ReadOnlyBoundaryLost(message)

    def _check_lent(self) -> None:
        """Refuse a connection its holder has already given back.

        A pool hands out the connection itself, so a caller who keeps it past the block that
        borrowed it holds the same object the pool later lends to somebody else. Using it then
        reaches into another caller's work: a ``rollback`` through such a handle discards a
        transaction that belongs to whoever holds it now.
        """
        if self._agens_pooled and not self._agens_lent:
            raise ReleasedConnection.for_use()

    @property
    def adapters(self) -> AdaptersMap:
        """The map this connection reads and writes through, which starts from the graph one.

        Derived per connection, so registering something on one connection does not reach the
        others and nothing here reaches a plain PostgreSQL connection in the same process.

        Named apart from psycopg's own field rather than filling it, because psycopg reads
        that one in the property this one replaces and nowhere else, and a field of our own
        needs no agreement with a base class about when it may be empty.
        """
        return self._agens_adapters

    # -- the capability gate ------------------------------------------------------------

    @property
    def capabilities(self) -> Capabilities:
        """What this server can do, read from what it reported at startup.

        Built on first use and kept. Nothing is asked of the server: the version arrived
        with the connection.
        """
        caps = self._agens_capabilities
        if caps is None:
            caps = Capabilities.of(self)
            self._agens_capabilities = caps
        return caps

    # -- the label table ---------------------------------------------------------------

    @property
    def label_table(self) -> LabelCache:
        """The label ids and names of the graph this connection is reading.

        Only the composite rendering needs this, because only it leaves the label name out
        of the value. It is filled when a graph is selected and can be filled again after
        a label is created.
        """
        return self._agens_labels

    def _accept_labels(self, graph: str, rows: Sequence[tuple[int, str]]) -> None:
        """Take the label table for a graph, and make the composite loaders available.

        The text loaders for a path and an element array are replaced at the same time with
        ones that check each label against the table.
        """
        self.label_table.load(graph, list(rows))
        if not self._agens_binary_ready:
            register_binary(self, self.label_table)  # type: ignore[arg-type]
            register_text(self, self.label_table)  # type: ignore[arg-type]
            self._agens_binary_ready = True

    # -- the graph the session is reading -------------------------------------------------

    def _note_graph_path(self) -> None:
        """Read whether the server reports the graph path, and record what it reports."""
        reported = self.info.parameter_status("graph_path")
        self._agens_reports_graph_path = reported is not None
        self._agens_graph_path_seen = reported or ""

    def _graph_path_moved(self) -> bool:
        """Whether the reported graph path differs from the one last read here.

        Reading it records it, so one change is reported once. Only meaningful where the
        server reports the path at all.
        """
        now = self.info.parameter_status("graph_path") or ""
        if now == self._agens_graph_path_seen:
            return False
        self._agens_graph_path_seen = now
        return True

    def _follow_reported_graph_path(self) -> bool:
        """Drop the label table if the reported graph path has moved since it was last read.

        Nothing on a server that does not report it. Called after a commit or a rollback, and
        before a describing method reads which graph the table names.
        """
        if not self._agens_reports_graph_path or not self._graph_path_moved():
            return False
        self.label_table.invalidate()
        return True

    # -- statements ---------------------------------------------------------------------

    def _check(self, statement: Any) -> None:
        """Refuse a statement the server would accept and read as something else."""
        check_bindable_positions(statement_text(statement))

    def _check_binary(self) -> None:
        """Refuse the composite rendering while nothing can read it.

        The loaders for it are built around a label table, so until one has been filled the
        connection has none registered, and a graph value in the composite rendering arrives
        as the bytes it was sent as. Bytes where a vertex belongs is a wrong answer that says
        nothing, so it is refused instead.
        """
        if not self._agens_binary_ready:
            raise StaleLabelCache.for_binary()

    # -- the client's own deadline ----------------------------------------------------------

    def _watch(self) -> Watch | None:
        """Watch this connection for a server gone quiet, or not at all without a limit.

        Quiet is a statement running and nothing heard for ``GRACE`` longer than the server's own
        limit allows: a statement that is merely slow the server ends itself, so silence that long
        is the network. A block running past its budget, a caller busy between statements, and a
        long batch whose answers keep coming are none of them quiet.
        """
        limit = self._agens_statement_timeout_ms
        if limit is None:
            return None
        watch = WATCHDOG.arm(
            time.monotonic() + limit / 1000 + GRACE,
            self,  # type: ignore[arg-type]
            _look_again_at,
        )
        self._agens_watches |= {watch}
        return watch

    def _unwatch(self, watch: Watch | None) -> bool:
        """Stop watching; whether the watchdog had already shut the connection down."""
        if watch is None:
            return False
        self._agens_watches -= {watch}
        return WATCHDOG.disarm(watch)

    def _hearing(self, gen: PQGen[_RV]) -> PQGen[_RV]:
        """psycopg's generator, noting each time the server is heard from while a statement runs.

        A step that finds the socket ready is the server, or at least the network, moving. psycopg
        also steps it every tenth of a second with nothing ready, so that an interruption is seen;
        those are not heard. A connection that is not running a statement -- waiting for a
        notification, say -- is not waiting on the server, and notes nothing.
        """
        try:
            asked = next(gen)
            self._note_heard()
            while True:
                ready = yield asked
                if ready:
                    self._note_heard()
                asked = gen.send(ready)
        except StopIteration as done:
            return cast("_RV", done.value)
        finally:
            self._agens_heard_at = None

    def _note_heard(self) -> None:
        running = self.pgconn.transaction_status == TransactionStatus.ACTIVE
        self._agens_heard_at = time.monotonic() if running else None

    def _given_up(self) -> bool:
        """Whether the watchdog has shut this connection down."""
        return any(watch.fired for watch in self._agens_watches)

    def _stalled(self) -> Stalled:
        """What a caller whose connection the watchdog shut down is told."""
        limit = self._agens_statement_timeout_ms
        return Stalled.unheard((limit or 0) / 1000 + GRACE, limit)

    @staticmethod
    def _select_graph_statement(name: str) -> str:
        """The statement that selects a graph.

        The name is quoted rather than bound, because the grammar has no place for a
        parameter here. Setting more than one graph is an error the server reports, and
        supplying one in the connection's own options skips the check that it exists, so
        neither is done.
        """
        return f"set graph_path = {quote_identifier(name)}"

    @staticmethod
    def _label_statement() -> str:
        return LabelCache().query

    # -- results ------------------------------------------------------------------------

    @staticmethod
    def _counts_for(
        statement: str, before: Sequence[int] | None, after: Sequence[int] | None
    ) -> GraphWriteCounts:
        """Read the counters as far as the statement allows.

        The counters live on the session and only a graph write moves them, so reading them
        after a statement says what the session holds and not what the statement did. Two
        things narrow that to the statement.

        Its clauses say which counters it could move at all: the server zeroes a counter only
        for a clause that can write it, so a counter no clause names is nought for this
        statement whatever the session holds.

        For the rest, a reading from before the statement says which of them it moved. A
        counter that changed can only have been changed here. One that did not, and was
        already nought, is nought either way. One that did not and was not nought is the one
        case a stale number and a real one look alike, and it is left unanswered.

        The command tag is not consulted. A graph write reports its command as an update, and
        so does an ordinary SQL update, whose row count has nothing to do with these.
        """
        if after is None:
            return GraphWriteCounts.unknown()
        movable = writable_counters(statement)
        if before is None:
            return GraphWriteCounts(*(int(after[i]) if i in movable else 0 for i in range(5)))
        return GraphWriteCounts.for_statement(before, after, movable)

    def _report_query(
        self, statement: str, timer: Any, *, rows: int, error: BaseException | None
    ) -> None:
        """Tell anything listening what one statement did, and cost nothing when nothing is.

        Called from the cursor rather than from the methods that build one, so that every
        statement is reported once and the driver's own catalog reads are reported too.

        The connection is identified by an opaque number rather than by anything it was
        configured with, because its settings hold the password.
        """
        if not logging_wanted():
            return
        report(
            QueryRecord(
                connection=id(self),
                statement=statement,
                elapsed=timer.elapsed,
                rows=rows,
                failed=error is not None,
                error=error,
            )
        )

    def _translated(self, exc: BaseException) -> BaseException:
        """A failure the server described badly, described properly, or the failure itself.

        A failure caused by a string's declared type also picks up a note saying why the
        parameter was text, which is the one thing the server's own message cannot know. The
        server's version decides what a refused plain SQL write to a label table is told to do.

        Inside a read-only block, any refusal to write reads as the write it was: the server names
        the command rather than what it did, refusing a Cypher ``CREATE`` as ``nextval()`` and a
        write on 2.16 and 2.17 under the tag ``SELECT``.
        """
        better = translate(exc, graph_dml_setting=self.capabilities.has("graph_dml_setting"))
        replacement = redact_details(better or exc)
        if (
            self._agens_read_only is not None
            and isinstance(replacement, _pg.ReadOnlySqlTransaction)
            and not isinstance(replacement, ReadOnlyGraphWrite)
        ):
            return ReadOnlyGraphWrite.by_the_server()
        hint = explain_string_type(replacement)
        if hint is not None:
            replacement.add_note(hint)
        return replacement


ConninfoSource = Union[str, "Callable[[], str]"]
"""How a blocking pool may be told where to connect.

A callable is re-read for every connection attempt, which is how a rotating credential is
supplied without restarting anything.
"""

AsyncConninfoSource = Union[ConninfoSource, "Callable[[], Awaitable[str]]"]
"""The same, for an awaiting pool, which may also be told by something to wait for."""


Statement = Union[str, bytes, "SQL", "Composed"]
"""What a statement may be written as.

Narrower than what psycopg's own type allows, because a template is not something a cursor
will take, and a caller should hear that from a type checker rather than at run time.
"""


def statement_text(statement: Any) -> str:
    """The text of a statement, however it was written.

    A check that reads a statement has to reach every spelling of one, since the shape it
    refuses is as reachable through bytes or a composed statement as through text. A
    template is read by joining its literal parts around the placeholder each hole stands
    for, which is the statement the server is going to be sent.
    """
    if type(statement) is str:
        return statement
    if isinstance(statement, bytes):
        return statement.decode("utf-8", "replace")
    as_string = getattr(statement, "as_string", None)
    if as_string is not None:
        return str(as_string(None))
    strings = getattr(statement, "strings", None)
    if strings is not None:
        return "%s".join(strings)
    return str(statement)
