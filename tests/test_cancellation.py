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
