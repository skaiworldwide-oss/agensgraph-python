"""Doing on a connection what a failure's category asks, before another attempt.

A prepared statement the server lost and a label table out of date are both worth another attempt
on the same connection, and each needs something different done first. Done wrong, the attempt
fails the same way: in autocommit, a lost prepared statement fails with ``26000`` on every attempt
until the statement cache is emptied.
"""

from __future__ import annotations

import psycopg
import pytest
import pytest_asyncio

import agensgraph
from agensgraph.errors import Retryability, StaleLabelCache
from agensgraph.retry import RetryPolicy

pytestmark = pytest.mark.server


def lose_prepared_statements(conn) -> None:  # type: ignore[no-untyped-def]
    """Drop them on the server behind psycopg's back, as a pooler resetting a session does."""
    conn.pgconn.exec_(b"deallocate all")


class TestALostPreparedStatement:
    def test_it_fails_on_every_attempt_until_recovered(self, dsn: str) -> None:
        with agensgraph.connect(dsn, autocommit=True) as conn:
            conn.execute("select 1", prepare=True)
            lose_prepared_statements(conn)
            for _ in range(2):
                with pytest.raises(psycopg.errors.InvalidSqlStatementName) as caught:
                    conn.execute("select 1", prepare=True)
            assert conn.recover(caught.value) is Retryability.RESET_STATE
            assert conn.execute("select 1", prepare=True).fetchone() == (1,)

    def test_a_transaction_run_again_gets_past_it(self, dsn: str) -> None:
        pool = agensgraph.ConnectionPool(
            dsn, min_size=1, max_size=1, kwargs={"autocommit": True}
        )
        pool.open(wait=True)
        try:
            with pool.connection() as conn:
                conn.execute("select 2", prepare=True)
                lose_prepared_statements(conn)
            attempts: list[int] = []

            def work(conn) -> int:  # type: ignore[no-untyped-def]
                attempts.append(1)
                return conn.execute("select 2", prepare=True).fetchone()[0]

            assert pool.run_transaction(work, policy=RetryPolicy(attempts=3)) == 2
            assert len(attempts) == 2
        finally:
            pool.close()


class TestALabelTableOutOfDate:
    def test_it_is_read_again(self, agens, dsn: str) -> None:  # type: ignore[no-untyped-def]
        graph = agens.label_table.graph
        with agensgraph.connect(dsn, autocommit=True) as other:
            other.graph(graph)
            other.execute("create vlabel later")
            other.execute("create (:later {k: 1})")
        with pytest.raises(StaleLabelCache) as caught:
            agens.execute("match (n:later) return n", binary=True).fetchall()
        assert agens.recover(caught.value) is Retryability.REFRESH_LABELS
        ((vertex,),) = agens.execute("match (n:later) return n", binary=True).fetchall()
        assert vertex.label == "later"


class TestNothingElseIsTouched:
    def test_a_failure_asking_nothing_of_the_connection_sends_nothing(self, agens) -> None:  # type: ignore[no-untyped-def]
        sent: list[str] = []

        def record(record) -> None:  # type: ignore[no-untyped-def]
            sent.append(record.statement)

        agensgraph.add_query_logger(record)
        try:
            failure = psycopg.errors.SerializationFailure("conflict")
            assert agens.recover(failure) is Retryability.SAFE
        finally:
            agensgraph.remove_query_logger(record)
        assert sent == []


class TestTheAwaitingInterface:
    @pytest_asyncio.fixture
    async def conn(self, dsn: str):  # type: ignore[no-untyped-def]
        connection = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
        async with connection:
            yield connection

    @pytest.mark.asyncio
    async def test_a_lost_prepared_statement(self, conn) -> None:  # type: ignore[no-untyped-def]
        await conn.execute("select 1", prepare=True)
        lose_prepared_statements(conn)
        with pytest.raises(psycopg.errors.InvalidSqlStatementName) as caught:
            await conn.execute("select 1", prepare=True)
        assert await conn.recover(caught.value) is Retryability.RESET_STATE
        cursor = await conn.execute("select 1", prepare=True)
        assert await cursor.fetchone() == (1,)
