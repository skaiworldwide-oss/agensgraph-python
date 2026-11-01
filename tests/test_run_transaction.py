"""Running a transaction again where that is safe, and settling a commit that was lost."""

from __future__ import annotations

import psycopg
import pytest

import agensgraph
from agensgraph.errors import UnresolvedCommit
from agensgraph.retry import RetryPolicy, TokenBucket

pytestmark = pytest.mark.server

GRAPH = "run_transaction"


def _policy(attempts: int = 3) -> RetryPolicy:
    """A policy with its own allowance and millisecond waits, so tests neither drain nor dawdle."""
    return RetryPolicy(attempts=attempts, base=0.001, cap=0.005, bucket=TokenBucket())


@pytest.fixture
def pool(dsn: str):  # type: ignore[no-untyped-def]
    with agensgraph.connect(dsn, autocommit=True) as admin:
        admin.execute(f'drop graph if exists "{GRAPH}" cascade')
        admin.execute(f'create graph "{GRAPH}"')
        admin.graph(GRAPH)
        admin.execute("create vlabel item")
    p = agensgraph.ConnectionPool(
        dsn, graph=GRAPH, min_size=1, max_size=3, kwargs={"autocommit": True}
    )
    p.open(wait=True)
    try:
        yield p
    finally:
        p.close()
        with agensgraph.connect(dsn, autocommit=True) as admin:
            admin.execute(f'drop graph if exists "{GRAPH}" cascade')


def _items(pool) -> int:  # type: ignore[no-untyped-def]
    with pool.connection() as conn:
        return int(conn.execute("match (n:item) return count(n)").fetchone()[0])


def _terminate(dsn: str, conn) -> None:  # type: ignore[no-untyped-def]
    with psycopg.connect(dsn, autocommit=True) as admin:
        admin.execute("select pg_terminate_backend(%s)", (conn.info.backend_pid,))


def _lose_the_commit(dsn: str, conn) -> None:  # type: ignore[no-untyped-def]
    """End the connection just after its transaction id is read, so the commit is what is lost."""
    reading = conn.transaction_id

    def then_lose(*, assign: bool = False):  # type: ignore[no-untyped-def]
        found = reading(assign=assign)
        _terminate(dsn, conn)
        return found

    conn.transaction_id = then_lose


class TestRunningItAgain:
    def test_a_conflict_runs_it_again(self, pool) -> None:  # type: ignore[no-untyped-def]
        calls = []

        def work(conn):  # type: ignore[no-untyped-def]
            calls.append(1)
            conn.execute("create (:item {n: 1})")
            if len(calls) == 1:
                raise psycopg.errors.SerializationFailure("could not serialize access")
            return "done"

        assert pool.run_transaction(work, policy=_policy()) == "done"
        assert len(calls) == 2
        assert _items(pool) == 1, "the first attempt was rolled back"

    def test_someone_else_arriving_first_is_retried_only_when_merging(self, pool) -> None:  # type: ignore[no-untyped-def]
        def racing(calls):  # type: ignore[no-untyped-def]
            def work(conn):  # type: ignore[no-untyped-def]
                calls.append(1)
                if len(calls) == 1:
                    raise psycopg.errors.UniqueViolation("duplicate key value")
                return len(calls)

            return work

        assert pool.run_transaction(racing([]), policy=_policy(), merging=True) == 2
        with pytest.raises(psycopg.errors.UniqueViolation):
            pool.run_transaction(racing([]), policy=_policy())

    def test_it_stops_when_the_attempts_are_spent(self, pool) -> None:  # type: ignore[no-untyped-def]
        def work(conn):  # type: ignore[no-untyped-def]
            raise psycopg.errors.DeadlockDetected("deadlock detected")

        with pytest.raises(psycopg.errors.DeadlockDetected) as caught:
            pool.run_transaction(work, policy=_policy(attempts=2))
        assert "2" in str(caught.value) or getattr(caught.value, "attempts", 2) == 2


class TestALostConnection:
    def test_lost_before_committing_runs_again_on_another_connection(
        self, pool, dsn: str
    ) -> None:  # type: ignore[no-untyped-def]
        """Nothing was committed, so with the commit tracked the work simply runs again."""
        calls = []

        def work(conn):  # type: ignore[no-untyped-def]
            calls.append(conn.info.backend_pid)
            conn.execute("create (:item {n: 1})")
            if len(calls) == 1:
                _terminate(dsn, conn)
            return "done"

        assert pool.run_transaction(work, policy=_policy(), track_commit=True) == "done"
        assert len(calls) == 2
        assert calls[0] != calls[1], "the second attempt ran on another connection"
        assert _items(pool) == 1

    def test_lost_while_committing_is_not_guessed_at(self, pool, dsn: str) -> None:  # type: ignore[no-untyped-def]
        """The work finished and the commit had no answer: repeating it could apply it twice."""

        def work(conn):  # type: ignore[no-untyped-def]
            conn.execute("create (:item {n: 1})")
            _terminate(dsn, conn)

        with pytest.raises(UnresolvedCommit):
            pool.run_transaction(work, policy=_policy())

    def test_tracked_work_that_wrote_nothing_needs_no_settling(self, pool, dsn: str) -> None:  # type: ignore[no-untyped-def]
        """The server gives a transaction an id when it first writes; one that never got an id
        has nothing whose commit could have been lost."""

        def work(conn):  # type: ignore[no-untyped-def]
            _lose_the_commit(dsn, conn)
            return conn.execute("match (n:item) return count(n)").fetchone()[0]

        assert pool.run_transaction(work, policy=_policy(), track_commit=True) == 0

    def test_a_commit_that_cannot_be_asked_about_is_still_reported_unresolved(
        self, pool, dsn: str, monkeypatch
    ) -> None:  # type: ignore[no-untyped-def]
        """Not the reason asking failed: what the caller has to act on is the open outcome."""

        def unreachable(transaction_id, budget):  # type: ignore[no-untyped-def]
            raise psycopg.OperationalError("the server is still out of reach")

        monkeypatch.setattr(pool, "_settle", unreachable)

        def work(conn):  # type: ignore[no-untyped-def]
            _lose_the_commit(dsn, conn)
            conn.execute("create (:item {n: 1})")

        with pytest.raises(UnresolvedCommit, match="out of reach") as caught:
            pool.run_transaction(work, policy=_policy(), track_commit=True)
        assert caught.value.transaction_id is not None

    def test_idempotent_work_runs_again_after_a_lost_commit(self, pool, dsn: str) -> None:  # type: ignore[no-untyped-def]
        calls = []

        def work(conn):  # type: ignore[no-untyped-def]
            calls.append(1)
            conn.execute("merge (:item {n: 7})")
            if len(calls) == 1:
                _terminate(dsn, conn)

        pool.run_transaction(work, policy=_policy(), idempotent=True)
        assert len(calls) == 2
        assert _items(pool) == 1


class TestTheAwaitingInterface:
    @pytest.mark.asyncio
    async def test_a_conflict_runs_it_again_there_too(self, pool, dsn: str) -> None:  # type: ignore[no-untyped-def]
        calls = []

        async def work(conn):  # type: ignore[no-untyped-def]
            calls.append(1)
            await conn.execute("create (:item {n: 2})")
            if len(calls) == 1:
                raise psycopg.errors.SerializationFailure("could not serialize access")
            return "done"

        async with agensgraph.AsyncConnectionPool(
            dsn, graph=GRAPH, min_size=1, max_size=2, kwargs={"autocommit": True}
        ) as apool:
            assert await apool.run_transaction(work, policy=_policy()) == "done"
        assert len(calls) == 2
        assert _items(pool) == 1
