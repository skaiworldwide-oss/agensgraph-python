# This file is generated from pool_async.py by tools/async_to_sync.py.
# Edit that file and run the tool; edits made here are lost on the next run.

"""A pool of graph connections.

psycopg's pool is wrapped, not reimplemented and not exposed. A pool is a large piece of
concurrency bookkeeping that psycopg has debugged in public across a dozen releases, and this
driver's contribution is the graph layer rather than socket accounting. But handing psycopg's
pool to a caller as our own type would make its semantics our contract, including the ones we
would like to be free to fix, so it is held rather than inherited from.

Its default reset already does the right thing and is why wrapping is safe: a connection
returned idle is kept, one returned in a transaction is rolled back, and one returned while a
statement is still running is closed outright. That last is the structural defence against the
bug class where a cancelled request's connection is handed to somebody else with a reply still
coming.

Four things are added.

**A version check before anything is pooled.** `_connect` catches every exception, counts it as
a connection error and lets the reconnect logic retry -- so pointing a pool at a server this
driver cannot read would retry for the whole reconnect timeout and only then report, rather
than saying at once what is wrong. So :meth:`open` makes one connection of its own first, lets
the check run on it, and closes it. A server that cannot be read is refused there, by name.

**A generation, so a restart is one event rather than N.** After a server restarts, every
pooled connection is dead at the same moment, and discovering that one caller at a time hands
the same failure to as many callers as there are connections. :meth:`invalidate` moves the
generation on; a connection from an older one is closed when it comes back rather than reused.
The mechanism is psycopg's own: a reset hook that raises makes it close the connection.

**A split between what is done once per connection and once per use.** psycopg has a hook for
a new connection and one for a returning one, but none for a connection being handed out, and
that is exactly where a graph and a statement timeout belong -- they follow the caller, not the
socket. Registering adapters and reading the label table follow the socket, and are done once.

**A budget threaded through waiting for a connection.** Time spent waiting is time spent, so
the wait comes out of the caller's deadline rather than a constant of its own, and a caller
that could not finish in what is left is turned away before a connection is taken from anyone.
"""

from __future__ import annotations

import inspect
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import psycopg_pool
from psycopg.pq import TransactionStatus

from .connection import Connection
from .deadline import Deadline
from .errors import StaleGeneration

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Generator

    from psycopg.abc import Params

    from ._core import ConninfoSource, Result, Statement
__all__ = ["ConnectionPool"]
DEFAULT_MIN_SIZE = 4
DEFAULT_MAX_SIZE = 16


class ConnectionPool:
    """Connections to one server, kept and reused.

    Nothing is opened by constructing one. :meth:`open` connects, and :meth:`wait` blocks until
    the pool is full enough to serve, which is how a misconfigured connection string becomes an
    error at startup rather than a timeout under load much later.
    """

    def __init__(
        self,
        conninfo: ConninfoSource = "",
        *,
        graph: str | None = None,
        min_size: int = DEFAULT_MIN_SIZE,
        max_size: int | None = DEFAULT_MAX_SIZE,
        kwargs: dict[str, Any] | None = None,
        configure: Callable[[Connection[Any]], Awaitable[None]] | None = None,
        setup: Callable[[Connection[Any]], Awaitable[None]] | None = None,
        reset: Callable[[Connection[Any]], Awaitable[None]] | None = None,
        timeout: float = 30.0,
        max_waiting: int = 0,
        max_lifetime: float = 3600.0,
        max_idle: float = 600.0,
        reconnect_timeout: float = 300.0,
        reconnect_failed: Callable[[Any], None] | None = None,
        num_workers: int = 3,
        check_connections: bool = True,
        name: str | None = None,
        statement_timeout_gap: float = 0.5,
    ) -> None:
        """Set a pool up.

        *graph* is selected on every connection handed out, so a caller does not have to. Both
        *min_size* and *max_size* are given values on purpose: psycopg leaves the upper bound
        off, and an unbounded pool answers a server under load by asking it for more
        connections. What the right number is depends on the server, and
        :meth:`get_stats` and :meth:`resize` are how it is found rather than guessed.

        *configure* runs once for each new connection, alongside the driver's own setting up.
        *setup* runs each time one is handed out, which psycopg has no hook for and which is
        where anything belonging to the caller rather than the socket goes.

        *check_connections* is on by default. The check is an empty statement rather than a
        query, deliberately cheap, and without it the first callers after a server restart each
        receive the failure themselves.
        """
        self._conninfo = conninfo
        self._kwargs = kwargs
        self._graph = graph
        self._setup = setup
        self._user_configure = configure
        self._user_reset = reset
        self._generation = 0
        self._counters = threading.Lock()
        self._statement_timeout_gap = statement_timeout_gap
        self._retired = 0
        self._pool: psycopg_pool.ConnectionPool[Connection[Any]] = psycopg_pool.ConnectionPool(
            conninfo,
            connection_class=Connection,
            kwargs=kwargs,
            min_size=min_size,
            max_size=max_size,
            open=False,
            configure=self._configure,
            check=self._check if check_connections else None,
            reset=self._reset,
            timeout=timeout,
            max_waiting=max_waiting,
            max_lifetime=max_lifetime,
            max_idle=max_idle,
            reconnect_timeout=reconnect_timeout,
            reconnect_failed=reconnect_failed,
            num_workers=num_workers,
            name=name,
        )

    def open(self, *, wait: bool = False, timeout: float = 30.0) -> None:
        """Check the server, then start filling.

        The check is a connection of this pool's own, made and closed before any worker starts,
        because a refusal raised inside the pool would be counted as a connection error and
        retried rather than reported.
        """
        self._check_server()
        self._pool.open(wait=wait, timeout=timeout)

    def _check_server(self) -> None:
        """Make one connection of our own, let it be checked, and close it.

        Not one of the pool's: making a connection through the pool runs the hook for a new
        connection, which selects the graph, reads the label table and calls whatever the caller
        gave -- all of it wasted on a connection that is about to be thrown away, and counted as
        though the pool had made it. The check itself happens inside ``connect``, so opening one
        and closing it is the whole of it.
        """
        conn = Connection.connect(self._resolve_conninfo(), **self._kwargs or {})
        conn.close()

    def _resolve_conninfo(self) -> str:
        """Where to connect, asking for it again if it is something that has to be asked.

        A callable is re-read on every attempt, which is how a credential that rotates is
        supplied without restarting anything.
        """
        value = self._conninfo() if callable(self._conninfo) else self._conninfo
        if inspect.isawaitable(value):
            value = value
        return value

    def wait(self, timeout: float = 30.0) -> None:
        """Block until the pool holds *min_size* connections, or fail saying it cannot."""
        self._pool.wait(timeout=timeout)

    def close(self, timeout: float = 5.0) -> None:
        self._pool.close(timeout=timeout)

    @property
    def closed(self) -> bool:
        return self._pool.closed

    def __enter__(self) -> ConnectionPool:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def _working_on(self, conn: Connection[Any]) -> Generator[None]:
        """Mark a connection as the pool's own to use for the moment.

        Every hook here runs on a connection no caller holds, and so does the check, which
        sends an empty statement. Without this the guard against a caller keeping a handle
        would refuse the pool its own connections.
        """
        conn._agens_lent = True
        try:
            yield
        finally:
            conn._agens_lent = False

    def _check(self, conn: Connection[Any]) -> None:
        """psycopg's own check, on a connection the pool is holding rather than lending."""
        with self._working_on(conn):
            psycopg_pool.ConnectionPool.check_connection(conn)

    def _configure(self, conn: Connection[Any]) -> None:
        """Once per new connection: everything that follows the socket rather than the caller.

        psycopg discards a connection this hook does not leave idle, so a statement run here
        outside autocommit -- selecting the graph, or anything a caller's own hook does -- is
        committed before the connection goes into the pool.
        """
        conn._agens_generation = self._generation
        with self._working_on(conn):
            self._set_up(conn)
        conn._agens_pooled = True

    def _set_up(self, conn: Connection[Any]) -> None:
        """What a new connection needs before it can be lent."""
        if self._graph is not None:
            conn.graph(self._graph)
        if self._user_configure is not None:
            self._user_configure(conn)
        if conn.pgconn.transaction_status != TransactionStatus.IDLE:
            conn.commit()

    def _reset(self, conn: Connection[Any]) -> None:
        """Once per returning connection, and where a retired one is refused.

        Raising is how a connection is discarded: psycopg closes one whose reset hook fails.
        That is the whole of the generation mechanism -- nothing has to reach into the pool's
        own bookkeeping to retire a set of connections.
        """
        if conn._agens_generation != self._generation:
            with self._counters:
                self._retired += 1
            raise StaleGeneration.for_connection(
                conn._agens_generation, current=self._generation
            )
        if self._user_reset is not None:
            with self._working_on(conn):
                self._user_reset(conn)

    @contextmanager
    def connection(
        self, *, timeout: float | None = None, deadline: Deadline | None = None
    ) -> Generator[Connection[Any]]:
        """Take a connection for the duration of a block.

        The wait comes out of *deadline* if one is given, so waiting for a connection is spent
        from the same budget as the statement that will run on it. A caller whose budget has
        already gone is turned away here rather than after taking a connection somebody else
        could have used.
        """
        budget = deadline if deadline is not None else Deadline(timeout)
        budget.check("waiting for a connection")
        wait = budget.bounded(timeout)
        lent: Connection[Any] | None = None
        try:
            with self._pool.connection(timeout=wait) as conn:
                lent = conn
                conn._agens_lent = True
                self._prepare(conn, budget)
                yield conn
        finally:
            if lent is not None:
                lent._agens_lent = False

    def _prepare(self, conn: Connection[Any], budget: Deadline) -> None:
        """Once per use: what follows the caller rather than the socket.

        A statement timeout belongs to the caller it was worked out for, so one left by the
        caller before is taken off rather than inherited. Inherited, it cancels a statement
        that was given no budget at all, and the cancellation is reported as a failure that
        another attempt would not fix.

        Only ever one statement: a caller with a budget replaces whatever was there, and a
        caller without one pays nothing unless the connection is carrying a limit.
        """
        limit = budget.statement_timeout_ms(gap=self._statement_timeout_gap)
        if limit is not None:
            conn.execute(f"set statement_timeout = {limit}")
            conn._agens_statement_timeout = True
        elif conn._agens_statement_timeout:
            conn.execute("set statement_timeout = default")
            conn._agens_statement_timeout = False
        if self._setup is not None:
            self._setup(conn)

    def execute_query(
        self,
        query: Statement,
        params: Params | None = None,
        *,
        timeout: float | None = None,
        deadline: Deadline | None = None,
        **kwargs: Any,
    ) -> Result:
        """Run one statement on a connection taken for just that statement."""
        with self.connection(timeout=timeout, deadline=deadline) as conn:
            return conn.execute_query(query, params, **kwargs)

    def invalidate(self) -> int:
        """Retire every connection now held, and return the generation now in force.

        Nothing is closed here and nothing waits. A connection in use stays usable until its
        holder is finished with it, and is closed when it comes back; a connection sitting idle
        is closed the next time it is taken out and returned. So a server restart costs one
        call rather than one failure per connection.
        """
        with self._counters:
            self._generation += 1
            return self._generation

    @property
    def generation(self) -> int:
        """The generation connections must belong to in order to be reused."""
        return self._generation

    @property
    def retired(self) -> int:
        """How many connections have been closed for belonging to an older generation."""
        return self._retired

    def check(self) -> None:
        """Check every idle connection now, and replace the ones that fail."""
        self._pool.check()

    def resize(self, min_size: int, max_size: int | None = None) -> None:
        self._pool.resize(min_size, max_size)

    def get_stats(self) -> dict[str, int]:
        """Everything psycopg counts, and what this pool adds to it."""
        stats = dict(self._pool.get_stats())
        stats["generation"] = self._generation
        stats["connections_retired"] = self._retired
        return stats

    def pop_stats(self) -> dict[str, int]:
        """The same, with the counters that accumulate reset, for reporting an interval."""
        stats = dict(self._pool.pop_stats())
        stats["generation"] = self._generation
        stats["connections_retired"] = self._retired
        return stats

    @property
    def min_size(self) -> int:
        return self._pool.min_size

    @property
    def max_size(self) -> int:
        return self._pool.max_size

    @property
    def name(self) -> str:
        return self._pool.name

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r}, {self.min_size}-{self.max_size}, generation {self._generation})"
