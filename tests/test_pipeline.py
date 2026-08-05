"""Sending a burst of statements without waiting for each.

The behaviour worth pinning is the one that makes a pipeline dangerous to read errors from: it
attributes a failure to the wrong statement. These tests assert that it does, and that the batch
helper says so rather than passing the misattribution on.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

import agensgraph

pytestmark = pytest.mark.server


@pytest.fixture
def graph(agens):  # type: ignore[no-untyped-def]
    agens.execute("create vlabel doc")
    return agens


class TestWhatAPipelineDoesWithAnError:
    def test_the_error_is_reported_against_the_wrong_statement(self, dsn) -> None:  # type: ignore[no-untyped-def]
        """Four statements, only the second bad: the first raises it and the rest raise with no
        SQLSTATE. This is why the batch helper reports the batch instead."""
        with agensgraph.Connection.connect(dsn) as conn:
            with conn.pipeline():
                cursors = [conn.cursor() for _ in range(4)]
                for cursor, statement in zip(
                    cursors,
                    ["select 1", "select 1/0", "select 3", "select 4"],
                    strict=True,
                ):
                    cursor.execute(statement)
                raised = []
                for cursor in cursors:
                    try:
                        cursor.fetchall()
                        raised.append(None)
                    except agensgraph.errors.Error as exc:
                        raised.append(exc)
            assert raised[0] is not None
            assert raised[0].sqlstate == "22012"
            assert [exc.sqlstate for exc in raised[1:]] == [None, None, None]
            conn.rollback()


class TestTheBatchHelper:
    def test_a_batch_of_writes_is_applied(self, graph) -> None:  # type: ignore[no-untyped-def]
        graph.pipeline_batch([f"create (:doc {{n: {n}}})" for n in range(10)])
        assert graph.execute_query("match (n:doc) return count(*)").records[0][0] == 10

    def test_parameters_are_carried(self, graph) -> None:  # type: ignore[no-untyped-def]
        graph.pipeline_batch([("create (:doc %s)", ({"n": n},)) for n in range(5)])
        assert graph.execute_query("match (n:doc) return count(*)").records[0][0] == 5

    def test_a_failure_names_the_batch_and_carries_the_cause(self, graph) -> None:  # type: ignore[no-untyped-def]
        statements = ["create (:doc {n: 1})", "create (:doc {n: 1/0})", "create (:doc {n: 3})"]
        with pytest.raises(agensgraph.errors.BatchFailed) as caught:
            graph.pipeline_batch(statements)
        assert caught.value.statements == tuple(statements)
        assert "does not report which" in str(caught.value)
        assert isinstance(caught.value.__cause__, agensgraph.errors.Error)

    def test_the_batch_is_not_applied_in_part(self, graph) -> None:  # type: ignore[no-untyped-def]
        """One transaction, so a failure anywhere leaves nothing behind."""
        with pytest.raises(agensgraph.errors.BatchFailed):
            graph.pipeline_batch(["create (:doc {n: 1})", "create (:doc {n: 1/0})"])
        graph.rollback()
        assert graph.execute_query("match (n:doc) return count(*)").records[0][0] == 0

    def test_an_empty_batch_does_nothing(self, graph) -> None:  # type: ignore[no-untyped-def]
        graph.pipeline_batch([])
        assert graph.execute_query("match (n:doc) return count(*)").records[0][0] == 0

    def test_running_them_one_at_a_time_is_what_finds_the_culprit(self, graph) -> None:  # type: ignore[no-untyped-def]
        """What the helper tells a caller to do, shown working -- reads, so replaying is safe."""
        statements = [
            "match (n:doc) return 1",
            "match (n:doc) return 1/0",
            "match (n:doc) return 3",
        ]
        with pytest.raises(agensgraph.errors.BatchFailed):
            graph.pipeline_batch(statements)
        graph.rollback()
        blamed = []
        for index, statement in enumerate(statements):
            try:
                graph.execute_query(statement)
            except agensgraph.errors.Error:
                blamed.append(index)
                graph.rollback()
        assert blamed == [1]


class TestTheAwaitingInterface:
    @pytest_asyncio.fixture
    async def conn(self, dsn: str):  # type: ignore[no-untyped-def]
        name = "pipeline_async"
        connection = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
        async with connection:
            await connection.execute(f'drop graph if exists "{name}" cascade')
            await connection.execute(f'create graph "{name}"')
            await connection.graph(name)
            await connection.execute("create vlabel doc")
            try:
                yield connection
            finally:
                await connection.execute("reset graph_path")
                await connection.execute(f'drop graph "{name}" cascade')

    @pytest.mark.asyncio
    async def test_a_batch_is_applied(self, conn) -> None:  # type: ignore[no-untyped-def]
        await conn.pipeline_batch([f"create (:doc {{n: {n}}})" for n in range(10)])
        result = await conn.execute_query("match (n:doc) return count(*)")
        assert result.records[0][0] == 10

    @pytest.mark.asyncio
    async def test_a_failure_names_the_batch(self, conn) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(agensgraph.errors.BatchFailed) as caught:
            await conn.pipeline_batch(["create (:doc {n: 1})", "create (:doc {n: 1/0})"])
        assert len(caught.value.statements) == 2
