"""What happens when a caller gives up partway through.

The failure this guards against is not an exception being missed. It is a cancellation being
*absorbed* -- a task told to stop, which then carries on as though it had not been, so the thing
that cancelled it waits forever. Every surveyed driver has had a version of that bug, twice with
a CVE attached, and the fix in each case was to decide the connection's fate synchronously and
throw it away rather than trying to clean it up in place.

So these assert on the cancellation itself, with ``cancelling()`` and ``cancelled()``, rather than
on an exception having been raised. Asserting only the latter is what let the bug through.
"""

from __future__ import annotations

import asyncio

import psycopg
import pytest
import pytest_asyncio

import agensgraph

pytestmark = pytest.mark.server


@pytest_asyncio.fixture
async def conn(dsn: str):  # type: ignore[no-untyped-def]
    connection = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
    async with connection:
        graph = "cancel_check"
        await connection.execute(f'drop graph if exists "{graph}" cascade')
        await connection.execute(f'create graph "{graph}"')
        await connection.graph(graph)
        await connection.execute("create vlabel thing")
        await connection.execute("create (:thing {n: 1})")
        try:
            yield connection
        finally:
            await connection.execute("reset graph_path")
            await connection.execute(f'drop graph "{graph}" cascade')


class TestCancellingAStatement:
    @pytest.mark.asyncio
    async def test_the_cancellation_is_delivered_and_not_absorbed(self, conn) -> None:  # type: ignore[no-untyped-def]
        """The assertion that matters. An absorbed cancellation raises nothing and hangs."""
        started = asyncio.Event()

        async def slow() -> None:
            started.set()
            await conn.execute_query("select pg_sleep(30)")

        task = asyncio.create_task(slow())
        await started.wait()
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert task.cancelled(), "the cancellation was absorbed rather than delivered"

    @pytest.mark.asyncio
    async def test_a_timeout_around_a_statement_fires(self, dsn: str) -> None:
        """A timeout works *by* cancelling, so one that does not fire is the same bug."""
        connection = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
        async with connection:
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.2):
                    await connection.execute_query("select pg_sleep(30)")

    @pytest.mark.asyncio
    async def test_a_second_cancellation_is_delivered_too(self, dsn: str) -> None:
        """Almost every surveyed bug here is 'one cancellation gets absorbed'."""
        for _ in range(2):
            connection = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
            async with connection:
                task = asyncio.create_task(connection.execute_query("select pg_sleep(30)"))
                await asyncio.sleep(0.05)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert task.cancelled()


class TestCancellingWhileAPoolIsInvolved:
    @pytest_asyncio.fixture
    async def pool(self, dsn: str):  # type: ignore[no-untyped-def]
        p = agensgraph.AsyncConnectionPool(
            dsn, min_size=1, max_size=1, kwargs={"autocommit": True}
        )
        await p.open(wait=True)
        try:
            yield p
        finally:
            await p.close()

    @pytest.mark.asyncio
    async def test_a_cancelled_waiter_hears_about_it(self, pool) -> None:  # type: ignore[no-untyped-def]
        """A waiter that receives its connection *and* never sees the cancellation hangs whoever
        awaits it -- which is a live issue in the pool underneath, so it is asserted here."""
        held = asyncio.Event()
        release = asyncio.Event()

        async def hold() -> None:
            async with pool.connection():
                held.set()
                await release.wait()

        holder = asyncio.create_task(hold())
        await held.wait()

        async def wait_for_one() -> None:
            async with pool.connection():
                pass

        waiter = asyncio.create_task(wait_for_one())
        await asyncio.sleep(0.05)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert waiter.cancelled(), "the waiter never saw the cancellation"

        release.set()
        await holder

    @pytest.mark.asyncio
    async def test_the_pool_is_usable_afterwards(self, pool) -> None:  # type: ignore[no-untyped-def]
        """A cancellation must not cost the pool a connection it never gets back."""
        task = asyncio.create_task(pool.execute_query("select pg_sleep(30)"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        for _ in range(200):
            if pool.get_stats().get("pool_available", 0) >= 1:
                break
            await asyncio.sleep(0.01)
        result = await pool.execute_query("select 1")
        assert result.records == [(1,)]

    @pytest.mark.asyncio
    async def test_a_cancellation_inside_the_reset_hook_does_not_lose_the_pool(
        self, dsn: str
    ) -> None:
        """Injected rather than raced for, because the case that corrupts state is a cancellation
        landing *inside* cleanup, and waiting for the scheduler to produce that is a coin toss."""
        failures = 0

        async def reset(connection: object) -> None:
            nonlocal failures
            failures += 1
            raise asyncio.CancelledError

        p = agensgraph.AsyncConnectionPool(
            dsn, min_size=1, max_size=1, kwargs={"autocommit": True}, reset=reset
        )
        await p.open(wait=True)
        try:
            async with p.connection() as conn:
                await conn.execute("select 1")
            for _ in range(200):
                if failures:
                    break
                await asyncio.sleep(0.01)
            assert failures >= 1, "the reset hook was never reached"
            # The connection it happened on is gone, and the pool replaces it rather than
            # starving.
            result = await p.execute_query("select 1")
            assert result.records == [(1,)]
        finally:
            await p.close()


class TestWhatIsLeftBehind:
    @pytest.mark.asyncio
    async def test_the_server_is_not_left_running_the_statement(self, conn, dsn: str) -> None:  # type: ignore[no-untyped-def]
        """Asserted from a second connection, because the first one is the one in doubt."""
        onlooker = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
        async with onlooker:
            victim = conn.info.backend_pid
            task = asyncio.create_task(conn.execute_query("select pg_sleep(30)"))
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            for _ in range(200):
                cursor = await onlooker.execute(
                    "select query from pg_stat_activity where pid = %s and state = 'active'",
                    (victim,),
                )
                rows = await cursor.fetchall()
                if not any("pg_sleep" in (row[0] or "") for row in rows):
                    break
                await asyncio.sleep(0.01)
            else:
                pytest.fail("the server was left running the statement")


class TestAConnectionThatSawACancel:
    """It is closed rather than lent to somebody else.

    psycopg does try to leave it clean: it asks the server to cancel and then reads what is
    still coming, and when that finishes the connection really is idle and really is reusable
    -- measured, the next caller got the right answer. What it cannot promise is that it
    finished. The read it re-enters is bounded by nothing, so on a connection whose packets
    stopped arriving it is still in there, and because an asyncio timeout works *by*
    cancelling, no timeout on that connection can fire either. That is psycopg's to bound and
    not this driver's to fix; what this driver can do is not hand such a connection to the next
    caller, which costs one connection on an event that is rare by definition.
    """

    async def backend_pid(self, pool) -> int:  # type: ignore[no-untyped-def]
        async with pool.connection() as conn:
            result = await conn.execute_query("select pg_backend_pid()")
            return int(result.records[0][0])

    @pytest.mark.asyncio
    async def test_the_next_caller_gets_a_different_backend(self, dsn: str) -> None:
        pool = agensgraph.AsyncConnectionPool(dsn, min_size=1, max_size=1)
        await pool.open(wait=True)
        try:
            before = await self.backend_pid(pool)

            async def slow() -> None:
                async with pool.connection() as conn:
                    await conn.execute_query("select pg_sleep(5)")

            task = asyncio.create_task(slow())
            await asyncio.sleep(0.3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0.5)
            after = await self.backend_pid(pool)
            assert before != after
            assert pool.interrupted == 1
            assert pool.get_stats()["connections_interrupted"] == 1
        finally:
            await pool.close()

    @pytest.mark.asyncio
    async def test_the_pool_still_serves(self, dsn: str) -> None:
        pool = agensgraph.AsyncConnectionPool(dsn, min_size=1, max_size=1)
        await pool.open(wait=True)
        try:

            async def slow() -> None:
                async with pool.connection() as conn:
                    await conn.execute_query("select pg_sleep(5)")

            task = asyncio.create_task(slow())
            await asyncio.sleep(0.3)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(0.5)
            result = await pool.execute_query("return 42")
            assert result.records == [(42,)]
        finally:
            await pool.close()

    @pytest.mark.asyncio
    async def test_a_statement_that_merely_failed_keeps_its_connection(self, dsn: str) -> None:
        """The mark is for an interruption and not for any failure: a refused value never
        reached the socket, and a server's own refusal left the connection idle and answered."""
        pool = agensgraph.AsyncConnectionPool(dsn, min_size=1, max_size=1)
        await pool.open(wait=True)
        try:
            before = await self.backend_pid(pool)
            with pytest.raises(ValueError):
                async with pool.connection() as conn:
                    await conn.execute_query("select %s", ({"x": float("nan")},))
            await asyncio.sleep(0.2)
            assert await self.backend_pid(pool) == before
            with pytest.raises(psycopg.errors.UndefinedFunction):
                async with pool.connection() as conn:
                    await conn.execute_query("select nosuchfunction()")
            await asyncio.sleep(0.2)
            assert await self.backend_pid(pool) == before
            assert pool.interrupted == 0
        finally:
            await pool.close()


class TestAskingTheServerToStop:
    """`cancel_safe` is how a running statement is stopped, and it is a different thing from a
    task being cancelled: one ends the statement cleanly and the other leaves it unfinished.
    """

    SLOW = "select pg_sleep(20)"

    async def backend_state(self, watcher, pid: int) -> str:  # type: ignore[no-untyped-def]
        cursor = await watcher.execute(
            "select state from pg_stat_activity where pid = %s", (pid,)
        )
        rows = await cursor.fetchall()
        return str(rows[0][0]) if rows else "gone"

    @pytest.mark.asyncio
    async def test_it_stops_the_backend_and_the_caller_is_told(self, dsn: str) -> None:
        """`57014` on the statement's own caller is the only evidence a cancel landed."""
        watcher = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
        conn = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
        async with watcher, conn:
            cursor = await conn.execute("select pg_backend_pid()")
            (pid,) = await cursor.fetchone()
            running = asyncio.create_task(conn.execute(self.SLOW))
            await asyncio.sleep(0.5)
            assert await self.backend_state(watcher, pid) == "active"
            await conn.cancel_safe(timeout=5.0)
            with pytest.raises(psycopg.Error) as caught:
                await running
            assert caught.value.sqlstate == "57014"
            assert await self.backend_state(watcher, pid) == "idle"

    @pytest.mark.asyncio
    async def test_the_connection_is_left_usable(self, dsn: str) -> None:
        """Which is why it is kept rather than replaced: nothing about it is unknown."""
        conn = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
        async with conn:
            running = asyncio.create_task(conn.execute(self.SLOW))
            await asyncio.sleep(0.5)
            await conn.cancel_safe(timeout=5.0)
            with pytest.raises(psycopg.Error):
                await running
            assert not conn.closed
            assert not conn.broken
            cursor = await conn.execute("select 42")
            assert await cursor.fetchone() == (42,)

    @pytest.mark.asyncio
    async def test_a_pool_hands_that_one_back_and_replaces_the_interrupted_one(
        self, dsn: str
    ) -> None:
        """The two cases differ, so the pool is asserted to treat them differently."""
        pool = agensgraph.AsyncConnectionPool(dsn, min_size=1, max_size=1)
        await pool.open(wait=True)
        try:
            async with pool.connection() as conn:
                cursor = await conn.execute("select pg_backend_pid()")
                (cancelled_on,) = await cursor.fetchone()
                running = asyncio.create_task(conn.execute(self.SLOW))
                await asyncio.sleep(0.5)
                await conn.cancel_safe(timeout=5.0)
                with pytest.raises(psycopg.Error):
                    await running
            async with pool.connection() as conn:
                cursor = await conn.execute("select pg_backend_pid()")
                (after_cancel,) = await cursor.fetchone()
            assert after_cancel == cancelled_on, "cleanly cancelled, so nothing to replace"

            async with pool.connection() as conn:
                cursor = await conn.execute("select pg_backend_pid()")
                (interrupted_on,) = await cursor.fetchone()
                running = asyncio.create_task(conn.execute(self.SLOW))
                await asyncio.sleep(0.5)
                running.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await running
                assert conn._agens_cancelled
            async with pool.connection() as conn:
                cursor = await conn.execute("select pg_backend_pid()")
                (after_interrupt,) = await cursor.fetchone()
            assert after_interrupt != interrupted_on, "interrupted, so its state is unknown"
        finally:
            await pool.close()
