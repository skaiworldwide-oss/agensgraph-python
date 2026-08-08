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


class TestReadingTheAnswersBack:
    """What the batch helper cannot do, and why a read can.

    A pipeline blames the wrong statement, which is why writes are sent without reading anything
    back. A read has the same problem and a way out of it: run them again one at a time. That is
    only safe because reading twice changes nothing.
    """

    @pytest.fixture
    def loaded(self, agens):  # type: ignore[no-untyped-def]
        agens.execute("create vlabel person")
        agens.load_vertices("person", [{"k": i, "name": f"n{i}"} for i in range(200)])
        agens.execute("create property index on person (k)")
        return agens

    def test_every_answer_comes_back_in_the_order_it_was_asked(self, loaded) -> None:  # type: ignore[no-untyped-def]
        asked = [("match (n:person) where n.k = %s return n.name", (i,)) for i in range(50)]
        results = loaded.pipeline_query(asked)
        assert [result.records for result in results] == [
            loaded.execute_query(statement, params).records for statement, params in asked
        ]

    def test_the_columns_are_described_as_they_are_by_the_other_route(self, loaded) -> None:  # type: ignore[no-untyped-def]
        """Read inside the pipeline these are empty, because nothing has answered yet."""
        (result,) = loaded.pipeline_query(["match (n:person) return n.name limit 1"])
        assert result.keys == ["name"]
        assert result.oids

    def test_a_statement_returning_nothing_is_still_a_result(self, loaded) -> None:  # type: ignore[no-untyped-def]
        (result,) = loaded.pipeline_query(["match (n:person) where n.k = 99999 return n"])
        assert result.records == []

    def test_a_failure_names_the_statement_that_caused_it(self, loaded) -> None:  # type: ignore[no-untyped-def]
        """Which the batch helper cannot do, because it may not run a write a second time."""
        good = "match (n:person) where n.k = 1 return n.name"
        with pytest.raises(agensgraph.errors.Error) as caught:
            loaded.pipeline_query([good, "match (n:person) return n.nope::int", good])
        assert not isinstance(caught.value, agensgraph.errors.BatchFailed)
        assert caught.value.sqlstate == "42601"

    def test_a_list_longer_than_one_chunk_is_still_one_list(self, loaded) -> None:  # type: ignore[no-untyped-def]
        asked = ["match (n:person) where n.k = 1 return n.name"] * 40
        assert len(loaded.pipeline_query(asked, chunk=7)) == 40

    def test_an_empty_list_asks_nothing(self, loaded) -> None:  # type: ignore[no-untyped-def]
        assert loaded.pipeline_query([]) == []

    def test_it_beats_asking_one_at_a_time(self, loaded) -> None:  # type: ignore[no-untyped-def]
        """Not a benchmark -- a floor, so a change that makes it slower than the plain route
        fails rather than merely disappoints. Interleaved, because measuring one and then the
        other measures the order as much as the routes."""
        import time

        asked = [("match (n:person) where n.k = %s return n.name", (i,)) for i in range(200)]
        singly = pipelined = 0.0
        for _ in range(3):
            started = time.monotonic()
            for statement, params in asked:
                loaded.execute_query(statement, params)
            singly += time.monotonic() - started
            started = time.monotonic()
            loaded.pipeline_query(asked)
            pipelined += time.monotonic() - started
        assert pipelined < singly


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

    @pytest.mark.asyncio
    async def test_answers_come_back_here_too(self, conn) -> None:  # type: ignore[no-untyped-def]
        await conn.pipeline_batch([f"create (:doc {{n: {n}}})" for n in range(5)])
        results = await conn.pipeline_query(
            [("match (n:doc) where n.n = %s return n.n", (n,)) for n in range(5)]
        )
        assert [result.records for result in results] == [[(n,)] for n in range(5)]
