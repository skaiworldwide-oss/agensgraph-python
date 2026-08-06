"""A pool of graph connections, in both interfaces.

What is tested here is what the wrapper adds, not what psycopg's pool already does. So: that a
server this driver cannot read is refused before any worker starts, that retiring a generation
closes connections rather than reusing them, that the two hooks fire at the times they claim
to, and that waiting for a connection is spent from the caller's budget.
"""

from __future__ import annotations

import time

import psycopg_pool
import pytest
import pytest_asyncio

import agensgraph
from agensgraph.deadline import Deadline
from agensgraph.errors import ReleasedConnection, Retryability, StaleGeneration, retryability

pytestmark = pytest.mark.server


def until(condition, timeout: float = 5.0) -> bool:  # type: ignore[no-untyped-def]
    """Wait for something the pool does on a worker thread.

    A connection is not put back by the thread that finishes with it -- psycopg hands it to a
    worker -- so anything checked straight afterwards is checked too early.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return False


@pytest.fixture
def graph_name(dsn: str) -> str:
    """A graph the pooled connections can be pointed at."""
    name = "pool_fixture"
    with agensgraph.connect(dsn, autocommit=True) as conn:
        conn.execute(f'drop graph if exists "{name}" cascade')
        conn.execute(f'create graph "{name}"')
        conn.graph(name)
        conn.execute("create vlabel thing")
        conn.execute("create (:thing {n: 1})")
    yield name
    with agensgraph.connect(dsn, autocommit=True) as conn:
        conn.execute(f'drop graph "{name}" cascade')


@pytest.fixture
def pool(dsn: str, graph_name: str):  # type: ignore[no-untyped-def]
    p = agensgraph.ConnectionPool(
        dsn, graph=graph_name, min_size=2, max_size=4, kwargs={"autocommit": True}
    )
    p.open(wait=True)
    try:
        yield p
    finally:
        p.close()


class TestOpening:
    def test_nothing_connects_until_it_is_opened(self, dsn: str, graph_name: str) -> None:
        """Opening in a constructor leaves a caller nothing to await and no way to hear."""
        p = agensgraph.ConnectionPool(dsn, graph=graph_name, min_size=1)
        try:
            # pool_size is the size intended; pool_available is what has actually been made.
            assert p.get_stats()["pool_available"] == 0
        finally:
            p.close()

    def test_waiting_fills_it(self, pool) -> None:  # type: ignore[no-untyped-def]
        assert pool.get_stats()["pool_size"] >= 2

    def test_an_unreachable_server_is_refused_promptly(self, graph_name: str) -> None:
        """Not after the reconnect timeout, which is five minutes by default."""
        p = agensgraph.ConnectionPool(
            "host=127.0.0.1 port=1 dbname=nothing", min_size=1, reconnect_timeout=300.0
        )
        started = time.monotonic()
        try:
            with pytest.raises(Exception):
                p.open(wait=True)
        finally:
            p.close()
        assert time.monotonic() - started < 10.0

    def test_a_server_the_driver_cannot_read_is_refused_by_the_probe(
        self, dsn: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every exception a connection attempt raises is swallowed and retried by the pool.

        So the check cannot live inside it. This stands in for a server below the supported
        version, which is what the check exists for and which there is no second server here
        to be.
        """
        from agensgraph import capabilities

        refusal = agensgraph.errors.CapabilityError("this server is 2.15")

        def refuse(cls: object, conn: object) -> object:
            raise refusal

        monkeypatch.setattr(capabilities.Capabilities, "of", classmethod(refuse))
        p = agensgraph.ConnectionPool(dsn, min_size=1, reconnect_timeout=300.0)
        started = time.monotonic()
        try:
            with pytest.raises(agensgraph.errors.CapabilityError, match=r"2\.15"):
                p.open(wait=True)
        finally:
            p.close()
        assert time.monotonic() - started < 10.0


class TestUsingIt:
    def test_a_connection_comes_with_its_graph_selected(self, pool, graph_name: str) -> None:  # type: ignore[no-untyped-def]
        with pool.connection() as conn:
            assert conn.label_table.graph == graph_name
            assert conn.execute_query("match (n:thing) return n").records

    def test_the_composite_rendering_works_without_further_setting_up(self, pool) -> None:  # type: ignore[no-untyped-def]
        """Which is the point of reading the label table once per connection."""
        with pool.connection() as conn:
            result = conn.execute_query("match (n:thing) return n", binary_=True)
            assert result.records[0][0].label == "thing"

    def test_one_statement_needs_no_block(self, pool) -> None:  # type: ignore[no-untyped-def]
        assert pool.execute_query("match (n:thing) return n").records

    def test_a_connection_comes_back(self, pool) -> None:  # type: ignore[no-untyped-def]
        before = pool.get_stats()["pool_available"]
        with pool.connection():
            pass
        assert until(lambda: pool.get_stats()["pool_available"] == before)

    def test_the_same_backend_comes_back(self, dsn: str, graph_name: str) -> None:
        """Reclaimed rather than thrown away and replaced, which the counts alone would hide.

        One connection only, so there is no second one to be handed instead.
        """
        p = agensgraph.ConnectionPool(
            dsn, graph=graph_name, min_size=1, max_size=1, kwargs={"autocommit": True}
        )
        p.open(wait=True)
        try:
            with p.connection() as conn:
                first = conn.info.backend_pid
            assert until(lambda: p.get_stats()["pool_available"] == 1)
            with p.connection() as conn:
                assert conn.info.backend_pid == first
        finally:
            p.close()


class TestTheTwoHooks:
    def test_one_fires_per_connection_and_the_other_per_use(
        self, dsn: str, graph_name: str
    ) -> None:
        """psycopg has the first and not the second, which is where a caller's own state goes."""
        configured: list[int] = []
        prepared: list[int] = []

        def configure(conn) -> None:  # type: ignore[no-untyped-def]
            configured.append(conn.info.backend_pid)

        def setup(conn) -> None:  # type: ignore[no-untyped-def]
            prepared.append(conn.info.backend_pid)

        p = agensgraph.ConnectionPool(
            dsn,
            graph=graph_name,
            min_size=1,
            max_size=1,
            kwargs={"autocommit": True},
            configure=configure,
            setup=setup,
        )
        p.open(wait=True)
        try:
            for _ in range(3):
                with p.connection():
                    pass
        finally:
            p.close()
        assert len(configured) == 1, "a new connection is configured once"
        assert len(set(configured)) == 1, "and the probe's connection is not one of them"
        assert len(prepared) == 3, "each use is prepared"


class TestRetiringAGeneration:
    def test_it_reports_the_generation_now_in_force(self, pool) -> None:  # type: ignore[no-untyped-def]
        assert pool.generation == 0
        assert pool.invalidate() == 1
        assert pool.generation == 1

    def test_a_retired_connection_is_closed_rather_than_reused(self, pool) -> None:  # type: ignore[no-untyped-def]
        with pool.connection() as conn:
            old = conn.info.backend_pid
        pool.invalidate()
        with pool.connection():
            pass
        assert until(lambda: pool.retired >= 1)
        with pool.connection() as conn:
            assert conn.info.backend_pid != old

    def test_it_is_counted(self, pool) -> None:  # type: ignore[no-untyped-def]
        pool.invalidate()
        with pool.connection():
            pass
        assert until(lambda: pool.get_stats()["connections_retired"] >= 1)
        assert pool.get_stats()["generation"] == 1

    def test_a_connection_in_use_is_left_alone(self, pool) -> None:  # type: ignore[no-untyped-def]
        """Retiring is not closing: whoever holds one keeps it until they are finished."""
        with pool.connection() as conn:
            pool.invalidate()
            assert conn.execute_query("match (n:thing) return n").records

    def test_the_report_names_both_generations(self) -> None:
        exc = StaleGeneration.for_connection(0, current=2)
        assert exc.generation == 0
        assert exc.current == 2
        assert "0" in str(exc)
        assert "2" in str(exc)


class TestTheBudget:
    def test_it_reaches_the_server(self, pool) -> None:  # type: ignore[no-untyped-def]
        """Set below what the caller waits for, so the server gives up first and says so."""
        with pool.connection(deadline=Deadline(5.0)) as conn:
            (limit,) = conn.execute("show statement_timeout").fetchone()
        assert limit.endswith("ms")
        assert 4000 < int(limit.removesuffix("ms")) < 5000

    def test_no_budget_asks_for_no_limit(self, pool) -> None:  # type: ignore[no-untyped-def]
        with pool.connection() as conn:
            (limit,) = conn.execute("show statement_timeout").fetchone()
        assert limit == "0"

    def test_a_budget_already_gone_is_refused_before_a_connection_is_taken(self, pool) -> None:  # type: ignore[no-untyped-def]
        """Taking one to fail with it is a connection somebody else could have used."""
        budget = Deadline(0.01)
        time.sleep(0.02)
        available = pool.get_stats()["pool_available"]
        with pytest.raises(agensgraph.Expired), pool.connection(deadline=budget):
            pass
        assert pool.get_stats()["pool_available"] == available


class TestHowPoolFailuresClassify:
    """All three are an OperationalError with no SQLSTATE, so a code cannot tell them apart."""

    @pytest.mark.parametrize(
        ("failure", "expected"),
        [
            (psycopg_pool.PoolTimeout("waited"), Retryability.BACKPRESSURE),
            (psycopg_pool.TooManyRequests("queue full"), Retryability.BACKPRESSURE),
            (psycopg_pool.PoolClosed("closed"), Retryability.FATAL),
        ],
    )
    def test_the_classification(self, failure: Exception, expected: Retryability) -> None:
        assert retryability(failure) is expected

    def test_none_of_them_asks_for_a_new_connection(self) -> None:
        """There is no connection to replace, so replacing one would be meaningless."""
        for failure in (
            psycopg_pool.PoolTimeout("x"),
            psycopg_pool.TooManyRequests("x"),
            psycopg_pool.PoolClosed("x"),
        ):
            assert not retryability(failure).needs_new_connection

    def test_a_closed_pool_is_not_retried(self) -> None:
        assert not retryability(psycopg_pool.PoolClosed("x")).is_retryable

    def test_waiting_is_retried_later_rather_than_sooner(self) -> None:
        recovery = retryability(psycopg_pool.PoolTimeout("x"))
        assert recovery.is_retryable
        assert recovery.wants_longer_delay


class TestStats:
    def test_psycopgs_counters_are_kept(self, pool) -> None:  # type: ignore[no-untyped-def]
        """A counter appears once it has counted something, so ask after asking for something."""
        pool.execute_query("match (n:thing) return n")
        stats = pool.get_stats()
        for key in ("pool_size", "pool_available", "requests_num", "connections_num"):
            assert key in stats

    def test_and_ours_are_added(self, pool) -> None:  # type: ignore[no-untyped-def]
        stats = pool.get_stats()
        assert stats["generation"] == pool.generation
        assert stats["connections_retired"] == pool.retired

    def test_an_interval_can_be_reported(self, pool) -> None:  # type: ignore[no-untyped-def]
        pool.execute_query("match (n:thing) return n")
        first = pool.pop_stats()
        assert first["requests_num"] >= 1
        assert pool.pop_stats().get("requests_num", 0) == 0


class TestTheAwaitingInterface:
    """The blocking pool is generated from this one, so both are exercised."""

    @pytest_asyncio.fixture
    async def apool(self, dsn: str, graph_name: str):  # type: ignore[no-untyped-def]
        p = agensgraph.AsyncConnectionPool(
            dsn, graph=graph_name, min_size=2, max_size=4, kwargs={"autocommit": True}
        )
        await p.open(wait=True)
        try:
            yield p
        finally:
            await p.close()

    @pytest.mark.asyncio
    async def test_a_connection_comes_with_its_graph_selected(
        self, apool, graph_name: str
    ) -> None:  # type: ignore[no-untyped-def]
        async with apool.connection() as conn:
            assert conn.label_table.graph == graph_name
            result = await conn.execute_query("match (n:thing) return n")
            assert result.records

    @pytest.mark.asyncio
    async def test_one_statement_needs_no_block(self, apool) -> None:  # type: ignore[no-untyped-def]
        result = await apool.execute_query("match (n:thing) return n")
        assert result.records

    @pytest.mark.asyncio
    async def test_the_composite_rendering_works_straight_away(self, apool) -> None:  # type: ignore[no-untyped-def]
        async with apool.connection() as conn:
            result = await conn.execute_query("match (n:thing) return n", binary_=True)
            assert result.records[0][0].label == "thing"

    @pytest.mark.asyncio
    async def test_a_retired_connection_is_closed_rather_than_reused(self, apool) -> None:  # type: ignore[no-untyped-def]
        import asyncio

        async with apool.connection() as conn:
            old = conn.info.backend_pid
        apool.invalidate()
        async with apool.connection():
            pass
        for _ in range(200):
            if apool.retired >= 1:
                break
            await asyncio.sleep(0.005)
        assert apool.retired >= 1
        async with apool.connection() as conn:
            assert conn.info.backend_pid != old

    @pytest.mark.asyncio
    async def test_the_budget_reaches_the_server(self, apool) -> None:  # type: ignore[no-untyped-def]
        async with apool.connection(deadline=Deadline(5.0)) as conn:
            row = await (await conn.execute("show statement_timeout")).fetchone()
        assert row[0].endswith("ms")

    @pytest.mark.asyncio
    async def test_a_budget_already_gone_is_refused(self, apool) -> None:  # type: ignore[no-untyped-def]
        budget = Deadline(0.01)
        time.sleep(0.02)
        with pytest.raises(agensgraph.Expired):
            async with apool.connection(deadline=budget):
                pass

    @pytest.mark.asyncio
    async def test_it_closes_on_leaving_the_block(self, dsn: str, graph_name: str) -> None:
        async with agensgraph.AsyncConnectionPool(
            dsn, graph=graph_name, min_size=1, kwargs={"autocommit": True}
        ) as p:
            await p.wait()
            assert not p.closed
        assert p.closed


class TestTheDefaultsWithoutAutocommit:
    """Selecting the graph runs a statement, which outside autocommit opens a transaction, and
    psycopg discards a connection its configure hook does not leave idle."""

    def test_a_pool_naming_a_graph_opens(self, dsn: str, graph_name: str) -> None:
        p = agensgraph.ConnectionPool(
            dsn, graph=graph_name, min_size=2, max_size=2, timeout=5.0
        )
        p.open(wait=True, timeout=10.0)
        try:
            with p.connection() as conn:
                assert conn.execute("show graph_path").fetchone()[0] == graph_name
                assert conn.execute_query("return 1").records == [(1,)]
        finally:
            p.close()

    def test_a_connection_comes_back_idle(self, dsn: str, graph_name: str) -> None:
        from psycopg.pq import TransactionStatus

        p = agensgraph.ConnectionPool(
            dsn, graph=graph_name, min_size=1, max_size=1, timeout=5.0
        )
        p.open(wait=True, timeout=10.0)
        try:
            with p.connection() as conn:
                assert conn.pgconn.transaction_status == TransactionStatus.IDLE
        finally:
            p.close()

    def test_a_configure_hook_that_runs_a_statement_keeps_its_work(
        self, dsn: str, graph_name: str
    ) -> None:
        def configure(conn: object) -> None:
            conn.execute("set application_name = 'from the hook'")  # type: ignore[attr-defined]

        p = agensgraph.ConnectionPool(
            dsn, graph=graph_name, min_size=1, max_size=1, configure=configure, timeout=5.0
        )
        p.open(wait=True, timeout=10.0)
        try:
            with p.connection() as conn:
                assert conn.execute("show application_name").fetchone()[0] == "from the hook"
        finally:
            p.close()


@pytest.mark.asyncio
async def test_the_awaiting_pool_opens_without_autocommit(dsn: str) -> None:
    name = "pool_async_defaults"
    async with await agensgraph.AsyncConnection.connect(dsn, autocommit=True) as conn:
        await conn.execute(f'drop graph if exists "{name}" cascade')
        await conn.execute(f'create graph "{name}"')
    p = agensgraph.AsyncConnectionPool(dsn, graph=name, min_size=2, max_size=2, timeout=5.0)
    await p.open(wait=True, timeout=10.0)
    try:
        async with p.connection() as conn:
            result = await conn.execute_query("return 1")
            assert result.records == [(1,)]
    finally:
        await p.close()
        async with await agensgraph.AsyncConnection.connect(dsn, autocommit=True) as conn:
            await conn.execute(f'drop graph "{name}" cascade')


@pytest.mark.server
class TestAHandleKeptPastItsBlock:
    """The pool lends the connection itself, so a handle held past the block is the object
    the pool later lends to somebody else."""

    def test_every_way_of_using_one_is_refused(self, dsn: str) -> None:
        with agensgraph.ConnectionPool(dsn, min_size=1, max_size=1, timeout=5.0) as pool:
            with pool.connection() as conn:
                escaped, escaped_cursor = conn, conn.cursor()
            for run in (
                lambda: escaped.execute("select 1"),
                lambda: escaped.rollback(),
                lambda: escaped.commit(),
                lambda: escaped.execute_query("return 1"),
                lambda: escaped_cursor.execute("select 1"),
            ):
                with pytest.raises(ReleasedConnection):
                    run()

    def test_it_is_refused_while_the_connection_sits_in_the_pool(self, dsn: str) -> None:
        with agensgraph.ConnectionPool(dsn, min_size=1, max_size=1, timeout=5.0) as pool:
            with pool.connection() as conn:
                escaped = conn
            with pytest.raises(ReleasedConnection):
                escaped.rollback()

    @pytest.mark.xfail(
        reason="the pool lends the connection itself, so once it is lent again the stale "
        "handle and the live one are one object and no state on it can tell them apart. "
        "Catching this needs a handle of its own per borrow, which changes what "
        "pool.connection() yields",
        strict=True,
    )
    def test_it_is_refused_once_the_connection_is_lent_to_somebody_else(
        self, dsn: str
    ) -> None:
        with agensgraph.ConnectionPool(dsn, min_size=1, max_size=1, timeout=5.0) as pool:
            with pool.connection() as conn:
                escaped = conn
            with pool.connection(), pytest.raises(ReleasedConnection):
                escaped.rollback()

    def test_a_connection_of_one_s_own_is_never_refused(self, dsn: str) -> None:
        """The guard is the pool's, so a connection nobody pooled is unaffected."""
        with agensgraph.Connection.connect(dsn) as conn:
            conn.execute("select 1")
            conn.rollback()
            conn.commit()


@pytest.mark.server
class TestAStatementTimeoutBelongsToOneBorrow:
    def test_it_is_not_inherited_by_the_next_caller(self, dsn: str) -> None:
        """Inherited, it cancels a statement given no budget, and reports it as a failure
        another attempt would not fix."""
        from agensgraph.deadline import Deadline

        with agensgraph.ConnectionPool(dsn, min_size=1, max_size=1, timeout=5.0) as pool:
            with pool.connection(deadline=Deadline(2.0)) as conn:
                held = conn.execute("show statement_timeout").fetchone()[0]
                assert held not in ("0", "")
            with pool.connection() as conn:
                assert conn.execute("show statement_timeout").fetchone()[0] == "0"

    def test_a_later_budget_replaces_an_earlier_one(self, dsn: str) -> None:
        from agensgraph.deadline import Deadline

        with agensgraph.ConnectionPool(dsn, min_size=1, max_size=1, timeout=5.0) as pool:
            with pool.connection(deadline=Deadline(2.0)) as conn:
                first = conn.execute("show statement_timeout").fetchone()[0]
            with pool.connection(deadline=Deadline(20.0)) as conn:
                second = conn.execute("show statement_timeout").fetchone()[0]
            assert first != second


@pytest.mark.server
class TestARetiredConnectionIsNeverLent:
    """Retiring one only as it comes back lets each of them serve one more caller first, which
    is the failure-per-caller the epoch bump replaces."""

    def test_no_borrow_after_invalidate_gets_an_older_generation(self, dsn: str) -> None:
        with agensgraph.ConnectionPool(dsn, min_size=3, max_size=3, timeout=5.0) as pool:
            pool.wait()
            for _ in range(5):
                with pool.connection() as conn:
                    conn.execute("select 1")
            current = pool.invalidate()
            served = []
            for _ in range(4):
                with pool.connection() as conn:
                    served.append(conn._agens_generation)
            assert served == [current] * 4

    def test_they_are_all_retired_rather_than_one_per_caller(self, dsn: str) -> None:
        with agensgraph.ConnectionPool(dsn, min_size=3, max_size=3, timeout=5.0) as pool:
            pool.wait()
            for _ in range(4):
                with pool.connection() as conn:
                    conn.execute("select 1")
            pool.invalidate()
            with pool.connection() as conn:
                conn.execute("select 1")
            assert pool.retired >= 3

    def test_the_pool_still_serves_afterwards(self, dsn: str) -> None:
        with agensgraph.ConnectionPool(dsn, min_size=2, max_size=2, timeout=5.0) as pool:
            pool.wait()
            pool.invalidate()
            assert pool.execute_query("return 1 as n").records == [(1,)]


@pytest.mark.server
class TestWhatRunningOutOfTimeIsCalled:
    """A caller that ran out of its own budget and a pool with nothing to give want retrying
    differently: one is the server being short and the other is not."""

    def test_a_spent_deadline_is_reported_as_the_budget_expiring(self, dsn: str) -> None:
        from agensgraph.deadline import Deadline, Expired
        from agensgraph.errors import Retryability, retryability

        with agensgraph.ConnectionPool(dsn, min_size=1, max_size=1, timeout=5.0) as pool:
            pool.wait()
            with (
                pool.connection(),
                pytest.raises(Expired) as caught,
                pool.connection(deadline=Deadline(0.3)),
            ):
                pass
            assert retryability(caught.value) is Retryability.SAFE

    def test_a_plain_wait_is_reported_as_the_pool_being_empty(self, dsn: str) -> None:
        import psycopg_pool

        from agensgraph.errors import Retryability, retryability

        with agensgraph.ConnectionPool(dsn, min_size=1, max_size=1, timeout=5.0) as pool:
            pool.wait()
            with (
                pool.connection(),
                pytest.raises(psycopg_pool.PoolTimeout) as caught,
                pool.connection(timeout=0.3),
            ):
                pass
            assert retryability(caught.value) is Retryability.BACKPRESSURE


@pytest.mark.server
class TestAPoolWideStatementTimeout:
    """The per-caller limit costs a round trip and cannot not; a ceiling for every connection
    costs none, because the options travel in the startup packet."""

    def test_it_is_set_without_a_round_trip_and_narrowed_by_a_deadline(self, dsn: str) -> None:
        from agensgraph.deadline import Deadline

        with agensgraph.ConnectionPool(
            dsn, min_size=1, max_size=1, timeout=5.0,
            kwargs={"options": "-c statement_timeout=5000"},
        ) as pool:
            pool.wait()
            with pool.connection() as conn:
                assert conn.execute("show statement_timeout").fetchone()[0] == "5s"
            with pool.connection(deadline=Deadline(2.0)) as conn:
                assert conn.execute("show statement_timeout").fetchone()[0] != "5s"
            with pool.connection() as conn:
                assert conn.execute("show statement_timeout").fetchone()[0] == "5s"
