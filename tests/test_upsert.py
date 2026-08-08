"""Writing elements that may already be there.

A copy only creates, so reading a source in twice with ``load_vertices`` makes a second element for
everything the graph already holds. Every knowledge-graph package re-ingests, so what they all
reach for is a merge, and every one of them writes it as a statement per row.
"""

from __future__ import annotations

import contextlib
import threading

import psycopg
import pytest
import pytest_asyncio

import agensgraph

pytestmark = pytest.mark.server


@pytest.fixture
def docs(agens):  # type: ignore[no-untyped-def]
    """A label whose key is unique, which is what an upsert asks for."""
    agens.execute("create vlabel doc")
    agens.execute("create unique property index on doc (k)")
    agens.refresh_labels()
    return agens


def count(conn, statement: str = "match (n:doc) return count(*)") -> int:  # type: ignore[no-untyped-def]
    return int(conn.execute_query(statement).records[0][0])


class TestWhatItWrites:
    def test_the_first_pass_writes_everything(self, docs) -> None:  # type: ignore[no-untyped-def]
        assert docs.upsert_vertices("doc", "k", [{"k": i} for i in range(50)]) == (50, 0)
        assert count(docs) == 50

    def test_the_second_pass_writes_nothing(self, docs) -> None:  # type: ignore[no-untyped-def]
        rows = [{"k": i, "v": 0} for i in range(50)]
        docs.upsert_vertices("doc", "k", rows)
        assert docs.upsert_vertices("doc", "k", [{"k": i, "v": 1} for i in range(50)]) == (0, 0)
        assert count(docs, "match (n:doc) where n.v = 0 return count(*)") == 50

    def test_only_the_rows_that_are_new_are_written(self, docs) -> None:  # type: ignore[no-untyped-def]
        docs.upsert_vertices("doc", "k", [{"k": i} for i in range(50)])
        assert docs.upsert_vertices("doc", "k", [{"k": i} for i in range(25, 75)]) == (25, 0)
        assert count(docs) == 75

    def test_asking_for_an_update_writes_the_overlap_too(self, docs) -> None:  # type: ignore[no-untyped-def]
        docs.upsert_vertices("doc", "k", [{"k": i, "v": 0} for i in range(50)])
        written = docs.upsert_vertices(
            "doc", "k", [{"k": i, "v": 1} for i in range(25, 75)], on_existing="update"
        )
        assert written == (25, 25)
        assert count(docs, "match (n:doc) where n.v = 1 return count(*)") == 50
        assert count(docs, "match (n:doc) where n.v = 0 return count(*)") == 25

    def test_an_update_merges_rather_than_replaces(self, docs) -> None:  # type: ignore[no-untyped-def]
        """A property the caller did not mention keeps the value it had."""
        docs.upsert_vertices("doc", "k", [{"k": 1, "kept": "yes", "v": 0}])
        docs.upsert_vertices("doc", "k", [{"k": 1, "v": 9}], on_existing="update")
        assert docs.execute_query("match (n:doc) return n.kept, n.v").records[0] == ("yes", 9)

    def test_an_empty_list_writes_nothing(self, docs) -> None:  # type: ignore[no-untyped-def]
        assert docs.upsert_vertices("doc", "k", []) == (0, 0)


class TestWhatItRefuses:
    def test_a_key_with_no_uniqueness_behind_it(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel doc")
        agens.execute("create property index on doc (k)")
        agens.refresh_labels()
        with pytest.raises(ValueError, match="unique"):
            agens.upsert_vertices("doc", "k", [{"k": 1}])

    def test_unless_the_caller_accepts_the_hazard(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel doc")
        agens.refresh_labels()
        assert agens.upsert_vertices("doc", "k", [{"k": 1}], require_unique=False) == (1, 0)

    def test_a_uniqueness_constraint_counts_as_well_as_an_index(self, agens) -> None:  # type: ignore[no-untyped-def]
        """It is kept as an exclusion constraint, so reading indexes alone would not find it."""
        agens.execute("create vlabel doc")
        agens.execute("create constraint doc_k_unique on doc assert k is unique")
        agens.refresh_labels()
        assert agens.upsert_vertices("doc", "k", [{"k": 1}]) == (1, 0)

    def test_a_row_with_no_key(self, docs) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="identify it by"):
            docs.upsert_vertices("doc", "k", [{"other": 1}])

    def test_a_row_whose_key_is_null(self, docs) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="identifies nothing"):
            docs.upsert_vertices("doc", "k", [{"k": None}])

    def test_a_third_thing_to_do_with_an_existing_element(self, docs) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="skip"):
            docs.upsert_vertices("doc", "k", [{"k": 1}], on_existing="replace")


class TestTheHazardTheUniqueKeyIsFor:
    """Why the key must be unique, shown rather than asserted.

    Two writers merging on the same key both miss and both create. This is the measurement the
    refusal above is based on, so it is worth having on the record.
    """

    def test_concurrent_merges_duplicate_without_it(self, agens, dsn) -> None:  # type: ignore[no-untyped-def]
        keys, writers = 25, 8
        agens.execute("create vlabel t")
        agens.execute("create property index on t (k)")
        graph = agens.execute("select current_setting('graph_path')").fetchone()[0]
        ready = threading.Barrier(writers)

        def merge_them() -> None:
            with agensgraph.connect(dsn, autocommit=True) as conn:
                conn.graph(graph)
                ready.wait()
                for k in range(keys):
                    # A conflict is the expected path here, not a surprise: this is what two
                    # writers merging on one key do.
                    with contextlib.suppress(psycopg.Error):
                        conn.execute("merge (n:t {k: %s}) set n.seen = 1", (k,))

        threads = [threading.Thread(target=merge_them) for _ in range(writers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        made = count(agens, "match (n:t) return count(*)")
        distinct = count(agens, "match (n:t) return count(distinct n.k)")
        assert distinct == keys
        assert made >= distinct, "which is the point: more elements than keys"


class TestItIsWorthDoing:
    def test_it_beats_a_merge_per_row(self, docs, agens) -> None:  # type: ignore[no-untyped-def]
        """Not a benchmark -- a floor, so a regression that makes it slower than the alternative
        fails rather than merely disappoints."""
        import time

        rows = [{"k": i, "v": 1} for i in range(2000)]
        docs.upsert_vertices("doc", "k", rows[:1000])
        started = time.monotonic()
        docs.upsert_vertices("doc", "k", rows, on_existing="update")
        upserting = time.monotonic() - started

        docs.execute("match (n:doc) delete n")
        docs.upsert_vertices("doc", "k", rows[:1000])
        started = time.monotonic()
        for row in rows:
            docs.execute("merge (n:doc {k: %s}) set n.v = %s", (row["k"], row["v"]))
        one_at_a_time = time.monotonic() - started
        assert upserting < one_at_a_time


class TestTheAwaitingInterface:
    @pytest_asyncio.fixture
    async def conn(self, dsn: str):  # type: ignore[no-untyped-def]
        name = "upsert_async"
        connection = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
        async with connection:
            await connection.execute(f'drop graph if exists "{name}" cascade')
            await connection.execute(f'create graph "{name}"')
            await connection.graph(name)
            await connection.execute("create vlabel doc")
            await connection.execute("create unique property index on doc (k)")
            await connection.refresh_labels()
            try:
                yield connection
            finally:
                await connection.execute("reset graph_path")
                await connection.execute(f'drop graph "{name}" cascade')

    @pytest.mark.asyncio
    async def test_it_writes_and_then_does_not(self, conn) -> None:  # type: ignore[no-untyped-def]
        rows = [{"k": i, "v": 0} for i in range(20)]
        assert await conn.upsert_vertices("doc", "k", rows) == (20, 0)
        assert await conn.upsert_vertices("doc", "k", rows) == (0, 0)
        written = await conn.upsert_vertices(
            "doc", "k", [{"k": i, "v": 1} for i in range(20)], on_existing="update"
        )
        assert written == (0, 20)
        result = await conn.execute_query("match (n:doc) where n.v = 1 return count(*)")
        assert result.records[0][0] == 20
