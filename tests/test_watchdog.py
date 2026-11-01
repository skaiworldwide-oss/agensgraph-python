"""A budget the client holds as well as the server.

A server stops a statement it can still hear from. Through a network that stops answering -- here a
proxy that stops forwarding and closes nothing -- a call under a server-side limit alone waits for
ever, measured past fifteen seconds. These pin that a budget ends the wait from the client side
within half a second of running out, in a blocking connection, an awaiting one and a pool, and
that in every other case the server is the one that stops the statement.
"""

from __future__ import annotations

import gc
import statistics
import time
import weakref

import psycopg
import pytest

import agensgraph
from agensgraph._watchdog import GRACE, Watchdog
from agensgraph.deadline import Deadline, Expired, Stalled
from agensgraph.errors import Retryability, retryability

from .stall_proxy import StallProxy


class _Closed:
    """A connection already closed, which the watchdog must leave alone."""

    closed = True


def _at_once(conn: object, now: float) -> float | None:
    """Give the connection up the first time it is looked at."""
    return None


class TestTheWatchdog:
    def test_arming_and_disarming_cost_microseconds(self) -> None:
        """It is armed for every block with a budget, so it has to be close to free."""
        dog = Watchdog()
        samples = []
        for _ in range(5000):
            started = time.perf_counter()
            dog.disarm(dog.arm(time.monotonic() + 30, _Closed(), _at_once))  # type: ignore[arg-type]
            samples.append(time.perf_counter() - started)
        assert statistics.median(samples) < 20e-6

    def test_watches_fire_in_the_order_they_are_due(self) -> None:
        dog = Watchdog()
        now = time.monotonic()
        late = dog.arm(now + 0.2, _Closed(), _at_once)  # type: ignore[arg-type]
        early = dog.arm(now + 0.02, _Closed(), _at_once)  # type: ignore[arg-type]
        time.sleep(0.08)
        assert early.fired
        assert not late.fired
        assert dog.disarm(late) is False
        assert dog.disarm(early) is True, "disarming one that fired says so"

    def test_a_disarmed_watch_never_fires(self) -> None:
        dog = Watchdog()
        watch = dog.arm(time.monotonic() + 0.02, _Closed(), _at_once)  # type: ignore[arg-type]
        assert dog.disarm(watch) is False
        time.sleep(0.06)
        assert not watch.fired

    def test_disarmed_watches_do_not_pile_up(self) -> None:
        dog = Watchdog()
        for _ in range(5000):
            dog.disarm(dog.arm(time.monotonic() + 3600, _Closed(), _at_once))  # type: ignore[arg-type]
        assert dog.pending() == 0
        assert len(dog._heap) < 3000

    def test_a_disarmed_watch_keeps_nothing_alive(self) -> None:
        """It waits in the heap until its moment, so it must not hold the connection there."""
        dog = Watchdog()
        held = _Closed()
        gone = weakref.ref(held)
        dog.disarm(dog.arm(time.monotonic() + 3600, held, _at_once))  # type: ignore[arg-type]
        del held
        gc.collect()
        assert gone() is None

    def test_a_watch_is_asked_again_until_it_answers_none(self) -> None:
        dog = Watchdog()
        asked: list[object] = []
        held = _Closed()

        def look(conn: object, now: float) -> float | None:
            asked.append(conn)
            return now + 0.01 if len(asked) < 3 else None

        watch = dog.arm(time.monotonic() + 0.01, held, look)  # type: ignore[arg-type]
        time.sleep(0.15)
        assert asked == [held, held, held], "handed the connection each time"
        assert watch.fired


class TestTheGap:
    """How far below the budget the server is asked to stop."""

    def test_a_long_budget_keeps_the_half_second_it_always_had(self) -> None:
        assert Deadline(10.0).statement_timeout_ms() == pytest.approx(9500, abs=2)

    def test_a_short_budget_takes_a_fifth_as_its_gap(self) -> None:
        """Half a second would leave a budget of half a second or less no time at all."""
        assert Deadline(0.3).statement_timeout_ms() == pytest.approx(240, abs=2)
        assert Deadline(1.0).statement_timeout_ms() == pytest.approx(800, abs=2)

    def test_a_gap_given_is_used_as_it_is(self) -> None:
        assert Deadline(10.0).statement_timeout_ms(gap=1.0) == pytest.approx(9000, abs=2)
        with pytest.raises(Expired):
            Deadline(0.3).statement_timeout_ms(gap=0.5)


@pytest.fixture
def proxy(dsn: str):  # type: ignore[no-untyped-def]
    with StallProxy(dsn) as stalling:
        yield stalling


@pytest.mark.server
class TestAStalledNetwork:
    def test_a_blocking_call_is_freed_when_its_budget_runs_out(self, proxy) -> None:  # type: ignore[no-untyped-def]
        with agensgraph.connect(proxy.dsn, autocommit=True) as conn:

            def stalled() -> None:
                with conn.deadline(1.0):
                    proxy.stall()
                    conn.execute("select 1")

            started = time.monotonic()
            with pytest.raises(Stalled, match="stopped answering"):
                stalled()
            waited = time.monotonic() - started
            assert waited < 1.0 + GRACE + 1.0, f"freed after {waited:.2f}s"
            assert conn.closed, "the connection was broken, so it is closed rather than reused"

    @pytest.mark.asyncio
    async def test_an_awaiting_call_is_freed_too(self, proxy) -> None:  # type: ignore[no-untyped-def]
        conn = await agensgraph.AsyncConnection.connect(proxy.dsn, autocommit=True)

        async def stalled() -> None:
            async with conn.deadline(1.0):
                proxy.stall()
                await conn.execute("select 1")

        async with conn:
            started = time.monotonic()
            with pytest.raises(Expired):
                await stalled()
            assert time.monotonic() - started < 1.0 + GRACE + 1.0

    def test_a_pool_borrow_is_freed_and_the_pool_recovers(self, proxy) -> None:  # type: ignore[no-untyped-def]
        pool = agensgraph.ConnectionPool(
            proxy.dsn, min_size=1, max_size=1, kwargs={"autocommit": True}
        )
        pool.open(wait=True)
        try:

            def stalled() -> None:
                with pool.connection(deadline=Deadline(1.0)) as conn:
                    proxy.stall()
                    conn.execute("select 1")

            started = time.monotonic()
            with pytest.raises(Expired):
                stalled()
            assert time.monotonic() - started < 1.0 + GRACE + 1.0
            proxy.resume()
            with pool.connection(timeout=10) as conn:
                assert conn.execute("select 1").fetchone() == (1,)
        finally:
            pool.close()

    def test_without_a_budget_nothing_is_watched(self, dsn) -> None:  # type: ignore[no-untyped-def]
        from agensgraph._watchdog import WATCHDOG

        before = WATCHDOG.pending()
        with agensgraph.connect(dsn, autocommit=True) as conn, conn.deadline(None):
            assert WATCHDOG.pending() == before
            conn.execute("select 1")


def test_a_stall_is_a_lost_connection_and_not_a_budget_declined() -> None:
    """The statement went, so whether it ran is unknown: after a write, an outcome to settle."""
    stalled = Stalled.unheard(1.3, 800)
    assert retryability(stalled) is Retryability.RECONNECT
    assert retryability(stalled, wrote=True) is Retryability.UNKNOWN
    assert retryability(Expired.after(1.0, 1.0, doing="waiting")) is Retryability.SAFE


@pytest.mark.server
class TestABlockThatIsMerelyLong:
    """Only a server gone quiet ends a connection; running past the budget does not."""

    def test_work_of_its_own_past_the_budget_keeps_the_connection(self, dsn) -> None:  # type: ignore[no-untyped-def]
        with agensgraph.connect(dsn, autocommit=True) as conn:
            with conn.deadline(0.4):
                conn.execute("select 1")
                time.sleep(0.4 + GRACE + 0.3)
                assert conn.execute("select 2").fetchone() == (2,)
            assert not conn.closed

    def test_many_quick_statements_past_the_budget_keep_it_too(self, dsn) -> None:  # type: ignore[no-untyped-def]
        with agensgraph.connect(dsn, autocommit=True) as conn, conn.deadline(0.4):
            until = time.monotonic() + 0.4 + GRACE + 0.3
            while time.monotonic() < until:
                conn.execute("select pg_sleep(0.01)")
            assert not conn.closed

    def test_a_pool_borrow_past_its_budget_keeps_it(self, dsn) -> None:  # type: ignore[no-untyped-def]
        """``timeout`` bounds each statement through the server's limit; time the caller spends
        between statements is its own."""
        with (
            agensgraph.ConnectionPool(
                dsn, min_size=1, max_size=1, kwargs={"autocommit": True}
            ) as pool,
            pool.connection(timeout=0.4) as conn,
        ):
            time.sleep(0.4 + GRACE + 0.3)
            assert conn.execute("select 1").fetchone() == (1,)

    def test_a_limit_a_rollback_took_back_is_not_held_against_the_server(self, dsn) -> None:  # type: ignore[no-untyped-def]
        """Set inside a transaction, the limit goes when it rolls back: a statement slower than
        it then runs to its end rather than being taken for a network gone quiet."""
        with agensgraph.connect(dsn) as conn, conn.deadline(0.4):
            conn.execute("select 1")
            conn.rollback()
            assert conn.execute("show statement_timeout").fetchone() == ("0",)
            assert conn.execute("select pg_sleep(0.4 + 0.5 + 0.4), 1").fetchone()[1] == 1
            assert not conn.closed


@pytest.mark.server
class TestAServerThatAnswers:
    def test_the_server_stops_a_slow_statement_and_the_connection_lives(self, dsn) -> None:  # type: ignore[no-untyped-def]
        with agensgraph.connect(dsn, autocommit=True) as conn:
            with pytest.raises(psycopg.errors.QueryCanceled), conn.deadline(1.0):
                conn.execute("select pg_sleep(5)")
            assert not conn.closed
            assert conn.execute("select 1").fetchone() == (1,)

    def test_a_short_budget_runs(self, dsn) -> None:  # type: ignore[no-untyped-def]
        with agensgraph.connect(dsn, autocommit=True) as conn, conn.deadline(0.4):
            assert conn.execute("select 1").fetchone() == (1,)


@pytest.mark.server
class TestAnIdleTransactionLimit:
    def test_it_is_a_startup_setting(self, dsn) -> None:  # type: ignore[no-untyped-def]
        with agensgraph.connect(dsn, idle_in_transaction_timeout=7) as conn:
            assert conn.execute("show idle_in_transaction_session_timeout").fetchone() == (
                "7s",
            )

    def test_options_already_given_are_kept(self, dsn) -> None:  # type: ignore[no-untyped-def]
        with agensgraph.connect(
            dsn, options="-c statement_timeout=4000", idle_in_transaction_timeout=7
        ) as conn:
            assert conn.execute("show statement_timeout").fetchone() == ("4s",)
            assert conn.execute("show idle_in_transaction_session_timeout").fetchone() == (
                "7s",
            )

    def test_a_session_left_idle_in_a_transaction_is_ended(self, dsn) -> None:  # type: ignore[no-untyped-def]
        with agensgraph.connect(dsn, idle_in_transaction_timeout=0.5) as conn:
            conn.execute("select 1")
            time.sleep(1.5)
            with pytest.raises(psycopg.errors.IdleInTransactionSessionTimeout):
                conn.execute("select 1")

    @pytest.mark.parametrize("bad", [0, -1, True, "5"])
    def test_a_limit_that_is_not_a_time_is_refused(self, dsn, bad: object) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="seconds"):
            agensgraph.connect(dsn, idle_in_transaction_timeout=bad)
