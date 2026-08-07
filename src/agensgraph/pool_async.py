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
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, ClassVar

import psycopg_pool
from psycopg.pq import TransactionStatus
from psycopg_pool import errors as _pool_errors

from .connection_async import AsyncConnection
from .deadline import Deadline
from .errors import InterruptedConnection, StaleGeneration

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable

    from psycopg.abc import Params

    from ._core import AsyncConninfoSource, Result, Statement

__all__ = ["AsyncConnectionPool", "AsyncNullConnectionPool"]

DEFAULT_MIN_SIZE = 4
DEFAULT_MAX_SIZE = 16


class AsyncConnectionPool:
    """Connections to one server, kept and reused.

    Nothing is opened by constructing one. :meth:`open` connects, and :meth:`wait` blocks until
    the pool is full enough to serve, which is how a misconfigured connection string becomes an
    error at startup rather than a timeout under load much later.
    """

    _pool_class: ClassVar[type[psycopg_pool.AsyncConnectionPool[Any]]] = (
        psycopg_pool.AsyncConnectionPool
    )
    """Which of psycopg's pools does the keeping. :class:`AsyncNullConnectionPool` keeps none."""

    def __init__(
        self,
        conninfo: AsyncConninfoSource = "",
        *,
        graph: str | None = None,
        min_size: int = DEFAULT_MIN_SIZE,
        max_size: int | None = DEFAULT_MAX_SIZE,
        kwargs: dict[str, Any] | None = None,
        configure: Callable[[AsyncConnection[Any]], Awaitable[None]] | None = None,
        setup: Callable[[AsyncConnection[Any]], Awaitable[None]] | None = None,
        reset: Callable[[AsyncConnection[Any]], Awaitable[None]] | None = None,
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
        # psycopg returns connections from any of its worker threads, so the two counters
        # a returning connection touches are not incremented from one thread alone.
        self._counters = threading.Lock()
        self._statement_timeout_gap = statement_timeout_gap
        self._retired = 0
        self._interrupted = 0
        self._pool: psycopg_pool.AsyncConnectionPool[AsyncConnection[Any]] = self._pool_class(
            conninfo,
            connection_class=AsyncConnection,
            kwargs=kwargs,
            min_size=min_size,
            max_size=max_size,
            # Never opened by constructing one: psycopg deprecated that and will make it an
            # error, and opening in a constructor gives a caller nothing to await and no
            # way to hear that the server is unreachable.
            open=False,
            configure=self._configure,
            check=self._check if check_connections else None,
            reset=self._reset,
            timeout=timeout,
            max_waiting=max_waiting,
            # The jitter psycopg subtracts from this is kept: it is there so that a pool
            # filled at once does not empty itself all at once an hour later.
            max_lifetime=max_lifetime,
            max_idle=max_idle,
            reconnect_timeout=reconnect_timeout,
            reconnect_failed=reconnect_failed,
            num_workers=num_workers,
            name=name,
        )

    # -- opening and closing ------------------------------------------------------------

    async def open(self, *, wait: bool = False, timeout: float = 30.0) -> None:
        """Check the server, then start filling.

        The check is a connection of this pool's own, made and closed before any worker starts,
        because a refusal raised inside the pool would be counted as a connection error and
        retried rather than reported.
        """
        await self._check_server()
        await self._pool.open(wait=wait, timeout=timeout)

    async def _check_server(self) -> None:
        """Make one connection of our own, let it be checked, and close it.

        Not one of the pool's: making a connection through the pool runs the hook for a new
        connection, which selects the graph, reads the label table and calls whatever the caller
        gave -- all of it wasted on a connection that is about to be thrown away, and counted as
        though the pool had made it. The check itself happens inside ``connect``, so opening one
        and closing it is the whole of it.
        """
        conn = await AsyncConnection.connect(
            await self._resolve_conninfo(), **(self._kwargs or {})
        )
        await conn.close()

    async def _resolve_conninfo(self) -> str:
        """Where to connect, asking for it again if it is something that has to be asked.

        A callable is re-read on every attempt, which is how a credential that rotates is
        supplied without restarting anything.
        """
        value = self._conninfo() if callable(self._conninfo) else self._conninfo
        if inspect.isawaitable(value):
            # Only reachable in the awaiting interface; a blocking pool's callable returns a
            # string, which is what its own type says.
            value = await value
        return value

    async def wait(self, timeout: float = 30.0) -> None:
        """Block until the pool holds *min_size* connections, or fail saying it cannot."""
        await self._pool.wait(timeout=timeout)

    async def close(self, timeout: float = 5.0) -> None:
        await self._pool.close(timeout=timeout)

    @property
    def closed(self) -> bool:
        return self._pool.closed

    async def __aenter__(self) -> AsyncConnectionPool:
        await self.open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    # -- the hooks psycopg has, and the one it does not ---------------------------------

    @asynccontextmanager
    async def _working_on(self, conn: AsyncConnection[Any]) -> AsyncGenerator[None]:
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

    async def _check(self, conn: AsyncConnection[Any]) -> None:
        """psycopg's own check, on a connection the pool is holding rather than lending."""
        async with self._working_on(conn):
            await psycopg_pool.AsyncConnectionPool.check_connection(conn)

    async def _configure(self, conn: AsyncConnection[Any]) -> None:
        """Once per new connection: everything that follows the socket rather than the caller.

        psycopg discards a connection this hook does not leave idle, so a statement run here
        outside autocommit -- selecting the graph, or anything a caller's own hook does -- is
        committed before the connection goes into the pool.
        """
        conn._agens_generation = self._generation
        async with self._working_on(conn):
            await self._set_up(conn)
        conn._agens_pooled = True

    async def _set_up(self, conn: AsyncConnection[Any]) -> None:
        """What a new connection needs before it can be lent."""
        if self._graph is not None:
            await conn.graph(self._graph)
        if self._user_configure is not None:
            await self._user_configure(conn)
        if conn.pgconn.transaction_status != TransactionStatus.IDLE:
            await conn.commit()

    async def _reset(self, conn: AsyncConnection[Any]) -> None:
        """Once per returning connection, and where a retired one is refused.

        Raising is how a connection is discarded: psycopg closes one whose reset hook fails.
        That is the whole of the generation mechanism -- nothing has to reach into the pool's
        own bookkeeping to retire a set of connections.
        """
        if conn._agens_cancelled:
            with self._counters:
                self._interrupted += 1
            raise InterruptedConnection.for_reuse()
        if conn._agens_generation != self._generation:
            with self._counters:
                self._retired += 1
            raise StaleGeneration.for_connection(
                conn._agens_generation, current=self._generation
            )
        if self._user_reset is not None:
            async with self._working_on(conn):
                await self._user_reset(conn)

    # -- taking one out -----------------------------------------------------------------

    @asynccontextmanager
    async def connection(
        self, *, timeout: float | None = None, deadline: Deadline | None = None
    ) -> AsyncGenerator[AsyncConnection[Any]]:
        """Take a connection for the duration of a block.

        The wait comes out of *deadline* if one is given, so waiting for a connection is spent
        from the same budget as the statement that will run on it. A caller whose budget has
        already gone is turned away here rather than after taking a connection somebody else
        could have used.
        """
        budget = deadline if deadline is not None else Deadline(timeout)
        budget.check("waiting for a connection")
        lent: AsyncConnection[Any] | None = None
        try:
            # A connection from a retired generation is handed back rather than lent, and
            # another taken. Retiring one only as it returns would let each of them serve one
            # more caller first, which is the failure per caller the epoch bump replaces.
            for _ in range(self._pool.max_size + 1):
                budget.check("waiting for a connection")
                async with self._pool.connection(timeout=budget.bounded(timeout)) as conn:
                    # Marked before the test and left so through the end of psycopg's own
                    # block, which commits and resets: the pool doing that is not a caller
                    # reaching past its turn. One rejected here is closed on the way out, so
                    # what the flag says about it afterwards reaches nobody.
                    lent = conn
                    conn._agens_lent = True
                    if conn._agens_generation != self._generation:
                        continue
                    await self._prepare(conn, budget)
                    yield conn
                    return
            raise StaleGeneration.for_pool(self._generation)
        except _pool_errors.PoolTimeout:
            # A caller who gave a deadline and ran out of it is not a pool with nothing to
            # give: one is the server being short and the other is not, and they are worth
            # retrying differently. A caller who only said how long to wait, and waited that
            # long, met the pool.
            if deadline is not None:
                budget.check("waiting for a connection")
            raise
        finally:
            if lent is not None:
                lent._agens_lent = False

    async def _prepare(self, conn: AsyncConnection[Any], budget: Deadline) -> None:
        """Once per use: what follows the caller rather than the socket.

        A statement timeout is one statement, and there is no way to make it none. A limit
        worked out for *this* caller cannot travel in the connection's own options, which are
        a startup parameter fixed for the life of the connection; so a per-caller deadline
        costs one round trip on acquire, measured at 84 microseconds. A pool that wants a
        ceiling and no round trip sets one for every connection instead::

            ConnectionPool(dsn, kwargs={"options": "-c statement_timeout=5000"})

        which this narrows per caller only when a deadline asks for something shorter, and
        restores by name rather than by number when it does not.

        A statement timeout belongs to the caller it was worked out for, so one left by the
        caller before is taken off rather than inherited. Inherited, it cancels a statement
        that was given no budget at all, and the cancellation is reported as a failure that
        another attempt would not fix.

        Only ever one statement: a caller with a budget replaces whatever was there, and a
        caller without one pays nothing unless the connection is carrying a limit.
        """
        limit = budget.statement_timeout_ms(gap=self._statement_timeout_gap)
        if limit is not None:
            await conn.execute(f"set statement_timeout = {limit}")
            conn._agens_statement_timeout = True
        elif conn._agens_statement_timeout:
            await conn.execute("set statement_timeout = default")
            conn._agens_statement_timeout = False
        if self._setup is not None:
            await self._setup(conn)

    async def execute_query(
        self,
        query: Statement,
        params: Params | None = None,
        *,
        timeout: float | None = None,
        deadline: Deadline | None = None,
        **kwargs: Any,
    ) -> Result:
        """Run one statement on a connection taken for just that statement."""
        async with self.connection(timeout=timeout, deadline=deadline) as conn:
            return await conn.execute_query(query, params, **kwargs)

    # -- retiring a set of connections --------------------------------------------------

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

    @property
    def interrupted(self) -> int:
        """How many connections have been closed for having had a statement interrupted."""
        return self._interrupted

    # -- what psycopg already offers, passed through ------------------------------------

    async def check(self) -> None:
        """Check every idle connection now, and replace the ones that fail."""
        await self._pool.check()

    async def drain(self) -> None:
        """Close every connection this pool holds and open the same number again.

        For a change that a connection carries from the moment it is made: a registration on the
        adapters map, a session setting asked for in *configure*, a rotated password behind a
        callable connection string. A connection somebody is holding is closed when it comes
        back rather than taken from them.

        This is not :meth:`invalidate` and does not do its job. The generation is untouched,
        because these connections are not wrong to be reused -- they are replaced so that the
        replacements are built the new way. :meth:`invalidate` is for connections that must not
        be used again at all, and it costs nothing until each is next handled.
        """
        await self._pool.drain()

    async def resize(self, min_size: int, max_size: int | None = None) -> None:
        await self._pool.resize(min_size, max_size)

    def get_stats(self) -> dict[str, int]:
        """Everything psycopg counts, and what this pool adds to it."""
        stats = dict(self._pool.get_stats())
        stats["generation"] = self._generation
        stats["connections_retired"] = self._retired
        stats["connections_interrupted"] = self._interrupted
        return stats

    def pop_stats(self) -> dict[str, int]:
        """The same, with the counters that accumulate reset, for reporting an interval."""
        stats = dict(self._pool.pop_stats())
        stats["generation"] = self._generation
        stats["connections_retired"] = self._retired
        stats["connections_interrupted"] = self._interrupted
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
        return (
            f"{type(self).__name__}({self.name!r}, {self.min_size}-{self.max_size}, "
            f"generation {self._generation})"
        )


class AsyncNullConnectionPool(AsyncConnectionPool):
    """A pool that keeps nothing: one connection per caller, closed when they are done.

    For a process that handles one request and exits, or a serverless one that may be frozen
    between requests, where a kept connection is a connection the server holds open for nobody.
    Everything else is the same -- the graph is selected, the version is checked, *configure* and
    *setup* run, the counters are counted -- so moving between the two is a change of class and
    nothing else.

    *min_size* is nought and cannot be anything else. *max_size* bounds how many callers may be
    connected at once, and left out it bounds nothing, which is the point: there is no pool to
    exhaust, only a server to reach.

    What it costs is a connection every time. Measured against a kept pool on one box, a borrow
    and one statement is 4.2 ms against 0.5 ms -- so this is for a process that would not have
    reused the connection anyway, and a loss for one that would.
    """

    _pool_class: ClassVar[type[psycopg_pool.AsyncConnectionPool[Any]]] = (
        psycopg_pool.AsyncNullConnectionPool
    )

    def __init__(
        self,
        conninfo: AsyncConninfoSource = "",
        *,
        graph: str | None = None,
        min_size: int = 0,
        max_size: int | None = None,
        kwargs: dict[str, Any] | None = None,
        configure: Callable[[AsyncConnection[Any]], Awaitable[None]] | None = None,
        setup: Callable[[AsyncConnection[Any]], Awaitable[None]] | None = None,
        reset: Callable[[AsyncConnection[Any]], Awaitable[None]] | None = None,
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
        super().__init__(
            conninfo,
            graph=graph,
            min_size=min_size,
            max_size=max_size,
            kwargs=kwargs,
            configure=configure,
            setup=setup,
            reset=reset,
            timeout=timeout,
            max_waiting=max_waiting,
            max_lifetime=max_lifetime,
            max_idle=max_idle,
            reconnect_timeout=reconnect_timeout,
            reconnect_failed=reconnect_failed,
            num_workers=num_workers,
            check_connections=check_connections,
            name=name,
            statement_timeout_gap=statement_timeout_gap,
        )
