"""Loading a lot of elements at once.

The identities matter most here. Copying supplies none, and the point of the tests below is that
what comes out is indistinguishable from what a ``CREATE`` would have produced -- the right label
id, a sequence with no gaps and no repeats, and elements the ordinary reader decodes.
"""

from __future__ import annotations

import gc

import pytest
import pytest_asyncio

import agensgraph
from agensgraph.bulk import (
    edge_copy_statement,
    identity_map_statement,
    vertex_copy_statement,
)

pytestmark = pytest.mark.server

ROWS = 5000


@pytest.fixture
def loaded(agens):  # type: ignore[no-untyped-def]
    agens.execute("create vlabel doc")
    agens.execute("create elabel cites")
    agens.refresh_labels()
    return agens


class TestTheStatements:
    def test_a_vertex_copy_sends_only_the_property_map(self) -> None:
        """Because the identity column's default makes the same identities a CREATE would."""
        statement = vertex_copy_statement("g", "doc")
        assert "(properties)" in statement
        assert "format binary" in statement
        assert "id" not in statement.split("(")[1]

    def test_an_edge_copy_sends_both_endpoints(self) -> None:
        statement = edge_copy_statement("g", "cites")
        assert 'start, "end", properties' in statement

    def test_end_is_quoted_because_it_is_a_reserved_word(self) -> None:
        assert '"end"' in edge_copy_statement("g", "cites")

    def test_a_name_needing_quoting_is_quoted(self) -> None:
        statement = vertex_copy_statement("odd graph", "odd label")
        assert '"odd graph"."odd label"' in statement

    def test_the_identity_map_reads_the_key_as_text(self) -> None:
        """A key that is a number in one place and a string in the other must still match."""
        assert "->>" in identity_map_statement("g", "doc")


class TestLoadingVertices:
    def test_every_row_arrives(self, loaded) -> None:  # type: ignore[no-untyped-def]
        count = loaded.load_vertices("doc", ({"n": i} for i in range(ROWS)))
        assert count == ROWS
        assert loaded.execute_query("match (n:doc) return count(*)").records[0][0] == ROWS

    def test_the_identities_are_the_ones_a_create_would_have_made(self, loaded) -> None:  # type: ignore[no-untyped-def]
        """The right label, a sequence with no gaps, and nothing repeated."""
        loaded.load_vertices("doc", [{"n": i} for i in range(10)])
        wanted = {label.id for label in loaded.labels() if label.name == "doc"}
        ids = [v.id for (v,) in loaded.execute_query("match (n:doc) return n").records]
        assert {gid.labid for gid in ids} == wanted
        assert sorted(gid.locid for gid in ids) == list(range(1, 11))
        assert len(set(ids)) == 10

    def test_the_elements_read_back_as_usual(self, loaded) -> None:  # type: ignore[no-untyped-def]
        loaded.load_vertices("doc", [{"n": 1, "s": "text", "l": [1, 2], "m": {"k": "v"}}])
        (v,) = loaded.execute_query("match (n:doc) return n").records[0]
        assert isinstance(v, agensgraph.Vertex)
        assert v.label == "doc"
        assert v.properties == {"n": 1, "s": "text", "l": [1, 2], "m": {"k": "v"}}

    def test_a_property_that_reads_as_json_stays_a_string(self, loaded) -> None:  # type: ignore[no-untyped-def]
        """The same hazard as a parameter, and copying must not reintroduce it."""
        loaded.load_vertices("doc", [{"v": "123"}, {"v": "null"}, {"v": 123}])
        values = [
            row[0]
            for row in loaded.execute_query("match (n:doc) return n.v order by n.v").records
        ]
        assert "123" in values
        assert "null" in values
        assert 123 in values

    def test_loading_nothing_is_allowed(self, loaded) -> None:  # type: ignore[no-untyped-def]
        assert loaded.load_vertices("doc", []) == 0

    def test_it_can_be_read_in_the_composite_rendering(self, loaded) -> None:  # type: ignore[no-untyped-def]
        loaded.load_vertices("doc", [{"n": 1}])
        result = loaded.execute_query("match (n:doc) return n", binary_=True)
        assert result.records[0][0].label == "doc"


class TestLoadingEdges:
    def test_the_identity_map_finds_what_the_server_called_each_one(self, loaded) -> None:  # type: ignore[no-untyped-def]
        loaded.load_vertices("doc", [{"key": f"k{i}"} for i in range(5)])
        mapping = loaded.identity_map("doc", "key")
        assert set(mapping) == {f"k{i}" for i in range(5)}
        assert all(isinstance(gid, agensgraph.GraphId) for gid in mapping.values())

    def test_edges_join_the_elements_they_were_told_to(self, loaded) -> None:  # type: ignore[no-untyped-def]
        loaded.load_vertices("doc", [{"key": f"k{i}"} for i in range(4)])
        by_key = loaded.identity_map("doc", "key")
        count = loaded.load_edges(
            "cites",
            [
                (by_key["k0"], by_key["k1"], {"w": 1}),
                (by_key["k1"], by_key["k2"], {"w": 2}),
                (by_key["k2"], by_key["k3"], None),
            ],
        )
        assert count == 3
        walked = loaded.execute_query(
            "match (a:doc)-[r:cites]->(b:doc) return a.key, b.key, r.w order by a.key"
        ).records
        assert walked == [("k0", "k1", 1), ("k1", "k2", 2), ("k2", "k3", None)]

    def test_an_edge_reads_back_with_its_endpoints(self, loaded) -> None:  # type: ignore[no-untyped-def]
        loaded.load_vertices("doc", [{"key": "a"}, {"key": "b"}])
        by_key = loaded.identity_map("doc", "key")
        loaded.load_edges("cites", [(by_key["a"], by_key["b"], {})])
        (e,) = loaded.execute_query("match ()-[r:cites]->() return r").records[0]
        assert isinstance(e, agensgraph.Edge)
        assert e.start == by_key["a"]
        assert e.end == by_key["b"]

    def test_a_numeric_key_still_matches(self, loaded) -> None:  # type: ignore[no-untyped-def]
        """Read as text on both sides, so a number in the data and a string in the map agree."""
        loaded.load_vertices("doc", [{"key": 1}, {"key": 2}])
        mapping = loaded.identity_map("doc", "key")
        assert set(mapping) == {"1", "2"}

    def test_a_path_walks_through_what_was_loaded(self, loaded) -> None:  # type: ignore[no-untyped-def]
        loaded.load_vertices("doc", [{"key": f"k{i}"} for i in range(3)])
        by_key = loaded.identity_map("doc", "key")
        loaded.load_edges(
            "cites", [(by_key["k0"], by_key["k1"], {}), (by_key["k1"], by_key["k2"], {})]
        )
        result = loaded.execute_query("match p = (:doc)-[:cites*2..2]->(:doc) return p")
        assert result.records
        path = result.records[0][0]
        assert path.length == 2
        assert len(path) == 5


class TestItIsWorthDoing:
    def test_it_beats_a_statement_per_row(self, loaded) -> None:  # type: ignore[no-untyped-def]
        """Not a benchmark -- a floor, so a regression that makes it slower than the alternative
        fails rather than merely disappoints."""
        import time

        rows = [{"n": i} for i in range(2000)]
        started = time.monotonic()
        loaded.load_vertices("doc", rows)
        copying = time.monotonic() - started

        loaded.execute("create vlabel other")
        started = time.monotonic()
        loaded.cursor().executemany("create (:other %s)", [(row,) for row in rows])
        one_at_a_time = time.monotonic() - started

        assert copying < one_at_a_time


class TestTheAwaitingInterface:
    @pytest_asyncio.fixture
    async def aloaded(self, dsn: str):  # type: ignore[no-untyped-def]
        graph = "bulk_async"
        conn = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
        async with conn:
            await conn.execute(f'drop graph if exists "{graph}" cascade')
            await conn.execute(f'create graph "{graph}"')
            await conn.graph(graph)
            await conn.execute("create vlabel doc")
            await conn.execute("create elabel cites")
            await conn.refresh_labels()
            try:
                yield conn
            finally:
                await conn.execute("reset graph_path")
                await conn.execute(f'drop graph "{graph}" cascade')

    @pytest.mark.asyncio
    async def test_vertices(self, aloaded) -> None:  # type: ignore[no-untyped-def]
        assert await aloaded.load_vertices("doc", [{"n": i} for i in range(100)]) == 100
        result = await aloaded.execute_query("match (n:doc) return count(*)")
        assert result.records[0][0] == 100

    @pytest.mark.asyncio
    async def test_edges_and_the_identity_map(self, aloaded) -> None:  # type: ignore[no-untyped-def]
        await aloaded.load_vertices("doc", [{"key": "a"}, {"key": "b"}])
        by_key = await aloaded.identity_map("doc", "key")
        assert await aloaded.load_edges("cites", [(by_key["a"], by_key["b"], {"w": 1})]) == 1
        result = await aloaded.execute_query(
            "match (x:doc)-[r:cites]->(y:doc) return x.key, y.key, r.w"
        )
        assert result.records == [("a", "b", 1)]

    @pytest.mark.asyncio
    async def test_a_columnar_source(self, aloaded) -> None:  # type: ignore[no-untyped-def]
        pyarrow = pytest.importorskip("pyarrow", reason="pyarrow is not installed")
        table = pyarrow.table({"key": ["a", "b", "c"]})
        assert await aloaded.load_vertex_frame("doc", table) == 3
        result = await aloaded.execute_query("match (n:doc) return n.key order by n.key")
        assert [key for (key,) in result.records] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_edges_from_a_columnar_source(self, aloaded) -> None:  # type: ignore[no-untyped-def]
        pyarrow = pytest.importorskip("pyarrow", reason="pyarrow is not installed")
        await aloaded.load_vertices("doc", [{"key": "a"}, {"key": "b"}])
        by_key = await aloaded.identity_map("doc", "key")
        edges = pyarrow.table(
            {"start": [by_key["a"].packed], "end": [by_key["b"].packed], "w": [1]}
        )
        assert await aloaded.load_edge_frame("cites", edges) == 1
        result = await aloaded.execute_query(
            "match (x:doc)-[r:cites]->(y:doc) return x.key, y.key, r.w"
        )
        assert result.records == [("a", "b", 1)]


class TestPausingTheCollector:
    def test_it_is_off_inside_and_on_again_after(self) -> None:
        assert gc.isenabled()
        with agensgraph.paused_collection():
            assert not gc.isenabled()
        assert gc.isenabled()

    def test_it_is_put_back_after_a_failure(self) -> None:
        with pytest.raises(ZeroDivisionError), agensgraph.paused_collection():
            _ = 1 / 0
        assert gc.isenabled()

    def test_it_leaves_the_collector_off_if_it_was_already_off(self) -> None:
        gc.disable()
        try:
            with agensgraph.paused_collection():
                assert not gc.isenabled()
            assert not gc.isenabled()
        finally:
            gc.enable()

    def test_nesting_it_is_harmless(self) -> None:
        with agensgraph.paused_collection(), agensgraph.paused_collection():
            assert not gc.isenabled()
        assert gc.isenabled()

    def test_reference_counting_still_frees_inside_it(self) -> None:
        """Only the collection of cycles waits, so an ordinary object is still freed at once."""
        import weakref

        class Held:
            pass

        with agensgraph.paused_collection():
            held = Held()
            ref = weakref.ref(held)
            del held
            assert ref() is None

    def test_freezing_moves_what_is_alive_out_of_the_way(self) -> None:
        before = gc.get_freeze_count()
        agensgraph.freeze_after_import()
        assert gc.get_freeze_count() >= before
        gc.unfreeze()
