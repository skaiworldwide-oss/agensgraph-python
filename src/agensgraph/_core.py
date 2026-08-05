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

from typing import TYPE_CHECKING, Any, NamedTuple, Union

from psycopg.adapt import AdaptersMap

from ._protocol.labels import LabelCache
from .adapters import graph_adapters, register_binary
from .capabilities import Capabilities
from .cypher import check_bindable_positions, quote_identifier
from .errors import explain_string_type, translate
from .observability import QueryRecord, logging_wanted, report
from .summary import GraphWriteCounts

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from psycopg.pq.abc import PGconn
    from psycopg.sql import SQL, Composed

__all__ = [
    "GRAPH_ADAPTERS",
    "AsyncConninfoSource",
    "ConninfoSource",
    "GraphMixin",
    "Result",
    "Statement",
]

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


def with_keepalives(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The connection arguments, with any keepalive setting the caller left out filled in.

    Filled in one key at a time rather than all or nothing, because the settings do nothing useful
    apart: a caller who names ``keepalives_interval`` and not ``keepalives_idle`` would otherwise get
    the system's idle time, which is two hours and a quarter, and their interval would never be
    reached.

    ``keepalives=0`` is the one thing that turns the rest off, since it says so.

    ``tcp_user_timeout`` does **not** count as having decided. It bounds how long transmitted data may
    go unacknowledged, and a connection waiting for a reply has transmitted nothing -- so a caller who
    sets only that has asked for something that does not bound a hung read, and keepalive is still
    filled in for them.
    """
    if str(kwargs.get("keepalives", 1)) == "0":
        return kwargs
    return {**KEEPALIVE_DEFAULTS, **kwargs}


class GraphMixin:
    """Everything a graph connection knows that does not require waiting for the server."""

    _adapters: AdaptersMap | None
    pgconn: PGconn

    _agens_capabilities: Capabilities | None = None
    _agens_labels: LabelCache
    _agens_binary_ready: bool = False
    _agens_generation: int = 0
    """Which generation of a pool this connection belongs to, if it came from one.

    Set when the pool creates it and read when it comes back, so that a pool which has moved
    on closes it instead of handing it to somebody else.
    """

    @property
    def adapters(self) -> AdaptersMap:
        """The map this connection reads and writes through, which starts from the graph one.

        Derived per connection, so registering something on one connection does not reach the
        others and nothing here reaches a plain PostgreSQL connection in the same process.
        """
        if not self._adapters:
            self._adapters = AdaptersMap(GRAPH_ADAPTERS)
        return self._adapters

    # -- the capability gate ------------------------------------------------------------

    @property
    def capabilities(self) -> Capabilities:
        """What this server can do, read from what it reported at startup.

        Built on first use and kept. Nothing is asked of the server: the version arrived
        with the connection.
        """
        caps = self._agens_capabilities
        if caps is None:
            caps = Capabilities.of(self)  # type: ignore[arg-type]
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
        try:
            return self._agens_labels
        except AttributeError:
            self._agens_labels = LabelCache()
            return self._agens_labels

    def _accept_labels(self, graph: str, rows: Sequence[tuple[int, str]]) -> None:
        """Take the label table for a graph, and make the composite loaders available."""
        self.label_table.load(graph, list(rows))
        if not self._agens_binary_ready:
            register_binary(self, self.label_table)  # type: ignore[arg-type]
            self._agens_binary_ready = True

    # -- statements ---------------------------------------------------------------------

    def _check(self, statement: Any) -> None:
        """Refuse a statement the server would accept and read as something else."""
        check_bindable_positions(statement_text(statement))

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
        tag: str | None, before: Sequence[int] | None, after: Sequence[int] | None
    ) -> GraphWriteCounts:
        """Read the counters as far as the statement allows.

        A write with no ``RETURN`` reports its command as an update, and starting one zeroes
        all five counters, so all five belong to it. A write that returned rows reports a
        select, and only the counters for the clauses it has were zeroed, so it is held to
        the ones that changed and to the ones that were zero to begin with.
        """
        if after is None:
            return GraphWriteCounts.unknown()
        if tag is not None and tag.split(" ", 1)[0].upper() == "UPDATE":
            return GraphWriteCounts.exact(after)
        if before is None:
            return GraphWriteCounts.unknown()
        return GraphWriteCounts.between(before, after)

    def _report_query(
        self, statement: str, timer: Any, *, rows: int, error: BaseException | None
    ) -> None:
        """Tell anything listening what one statement did, and cost nothing when nothing is.

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

    @staticmethod
    def _translated(exc: BaseException) -> BaseException:
        """A failure the server described badly, described properly, or the failure itself.

        A failure caused by a string's declared type also picks up a note saying why the
        parameter was text, which is the one thing the server's own message cannot know.
        """
        replacement = translate(exc) or exc
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
