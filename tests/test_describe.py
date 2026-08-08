"""Saying what is in a graph without reading it.

The question a prompt asks, and the one every integration answers with three full scans and a
plpgsql function installed in the caller's database. The catalogs already hold the labels, the
counts and which label joins which; only the property names need looking at any rows at all, and
those come from a bounded sample.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

import agensgraph

pytestmark = pytest.mark.server


@pytest.fixture
def described(agens):  # type: ignore[no-untyped-def]
    """A graph with the things every framework's version drops."""
    agens.execute('create vlabel "Spaced Label"')
    agens.execute("create vlabel person")
    agens.execute("create vlabel employee inherits (person)")
    agens.execute("create elabel knows")
    agens.execute(
        "create (:person {name: 'a', age: 30, score: 1.5, tags: ['x'], ok: true, nothing: null})"
    )
    agens.execute("create (:person {name: 'b', age: 31})")
    agens.execute('create (:"Spaced Label" {k: 1})')
    agens.execute(
        "match (a:person {name: 'a'}), (b:person {name: 'b'}) "
        "create (a)-[:knows {since: 2020}]->(b)"
    )
    agens.refresh_labels()
    return agens


class TestWhatItFinds:
    def test_every_label_with_its_kind_and_what_it_inherits(self, described) -> None:  # type: ignore[no-untyped-def]
        found = {
            label.name: (label.kind, label.parent)
            for label in described.describe().labels
            if not label.is_builtin
        }
        assert found["person"] == ("v", "ag_vertex")
        assert found["employee"] == ("v", "person"), "the inheritance, which a scan cannot see"
        assert found["knows"] == ("e", "ag_edge")

    def test_a_label_whose_name_has_a_space(self, described) -> None:  # type: ignore[no-untyped-def]
        """The server writes a label raw and unquoted, so a reading of the text loses this one."""
        described_graph = described.describe()
        assert "Spaced Label" in described_graph.properties
        assert described_graph.counts["Spaced Label"] == 1

    def test_how_many_elements_each_label_has(self, described) -> None:  # type: ignore[no-untyped-def]
        counts = described.describe().counts
        assert counts["person"] == 2
        assert counts["knows"] == 1

    def test_the_properties_of_a_vertex_label_with_their_kinds(self, described) -> None:  # type: ignore[no-untyped-def]
        found = {p.name: p.kind for p in described.describe().properties["person"]}
        assert found["name"] == "string"
        assert found["age"] == "integer"
        assert found["score"] == "float", "a whole number is told from a fractional one"
        assert found["tags"] == "array"
        assert found["ok"] == "boolean"

    def test_the_properties_of_an_edge_label(self, described) -> None:  # type: ignore[no-untyped-def]
        """Which every framework's decoder throws away, so their schemas do not have them."""
        assert [p.name for p in described.describe().properties["knows"]] == ["since"]

    def test_a_label_with_no_elements_has_no_properties(self, described) -> None:  # type: ignore[no-untyped-def]
        assert described.describe().properties["employee"] == ()


class TestTheTriples:
    def test_they_are_empty_and_say_so_before_anything_gathers_them(self, described) -> None:  # type: ignore[no-untyped-def]
        """``auto_gather_graphmeta`` is off, so a new graph has none. Empty is not the same as
        gathered-and-empty, and the difference is reported rather than guessed at."""
        found = described.describe()
        if found.triples:
            pytest.skip("something on this server gathers automatically")
        assert found.meta_gathered is False

    def test_asking_for_a_refresh_fills_them(self, described) -> None:  # type: ignore[no-untyped-def]
        found = described.describe(refresh=True)
        assert found.meta_gathered is True
        assert found.triples == (agensgraph.Triple("person", "knows", "person", 1),)

    def test_a_graph_with_no_edges_is_gathered_by_having_none(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel lonely")
        agens.execute("create (:lonely {a: 1})")
        agens.refresh_labels()
        found = agens.describe()
        assert found.triples == ()
        assert found.meta_gathered is True, "no edges and no triples agree with each other"


class TestItInstallsNothing:
    def test_no_function_is_created_in_the_caller_s_database(self, described) -> None:  # type: ignore[no-untyped-def]
        """Three of the packages this replaces install a plpgsql function to name a JSON type.

        ``jsonb_typeof`` is built in, and the one thing it does not do is done in Python.
        """
        before = described.execute("select count(*) from pg_proc").fetchone()[0]
        described.describe()
        after = described.execute("select count(*) from pg_proc").fetchone()[0]
        assert after == before

    def test_no_table_is_created_either(self, described) -> None:  # type: ignore[no-untyped-def]
        before = described.execute("select count(*) from pg_class").fetchone()[0]
        described.describe()
        after = described.execute("select count(*) from pg_class").fetchone()[0]
        assert after == before


class TestWhereTheTypesComeFrom:
    def test_a_promoted_property_is_declared_rather_than_sampled(self, agens) -> None:  # type: ignore[no-untyped-def]
        """Its type is on the column, so it is read rather than guessed from what a sample held."""
        if not agens.can_promote_properties():
            pytest.skip("this server cannot store a property in a column of its own")
        agens.execute("create vlabel doc (title text generated)")
        agens.execute("create (:doc {title: 'a'})")
        agens.refresh_labels()
        found = {p.name: p for p in agens.describe().properties["doc"]}
        assert found["title"].declared is True
        assert found["title"].kind == "text"

    def test_a_property_in_the_map_is_sampled(self, described) -> None:  # type: ignore[no-untyped-def]
        assert all(not p.declared for p in described.describe().properties["person"])

    def test_the_sample_is_bounded(self, agens) -> None:  # type: ignore[no-untyped-def]
        """A key beyond the sample is not reported, which is the bargain a sample makes."""
        agens.execute("create vlabel many")
        agens.load_vertices("many", [{"common": i} for i in range(200)])
        agens.load_vertices("many", [{"rare": 1}])
        agens.refresh_labels()
        keys = {p.name for p in agens.describe(sample=10).properties["many"]}
        assert keys == {"common"}
        keys = {p.name for p in agens.describe(sample=1000).properties["many"]}
        assert keys == {"common", "rare"}


class TestItIsWorthDoing:
    def test_it_beats_scanning_the_graph_for_the_same_answer(self, agens) -> None:  # type: ignore[no-untyped-def]
        """Not a benchmark -- a floor. The scans grow with the graph and this does not, so the gap
        widens with size; the assertion only needs it to be the right way round."""
        import time

        agens.execute("create vlabel person")
        agens.execute("create elabel knows")
        agens.load_vertices("person", [{"k": i, "bio": "x" * 100} for i in range(4000)])
        agens.refresh_labels()
        ids = agens.identity_map("person", "k")
        agens.load_edges(
            "knows", [(ids[str(i)], ids[str((i + 1) % 4000)], None) for i in range(4000)]
        )
        agens.execute("select regather_graphmeta()")

        started = time.monotonic()
        agens.describe()
        describing = time.monotonic() - started

        started = time.monotonic()
        agens.execute_query(
            "match (a) unwind keys(properties(a)) as prop "
            "with distinct label(a) as label, prop as property return label, collect(property)"
        )
        agens.execute_query("match (s)-[r]->(e) return distinct label(s), type(r), label(e)")
        scanning = time.monotonic() - started
        assert describing < scanning


class TestTheAwaitingInterface:
    @pytest_asyncio.fixture
    async def conn(self, dsn: str):  # type: ignore[no-untyped-def]
        name = "describe_async"
        connection = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
        async with connection:
            await connection.execute(f'drop graph if exists "{name}" cascade')
            await connection.execute(f'create graph "{name}"')
            await connection.graph(name)
            await connection.execute("create vlabel person")
            await connection.execute("create (:person {name: 'a'})")
            await connection.refresh_labels()
            try:
                yield connection
            finally:
                await connection.execute("reset graph_path")
                await connection.execute(f'drop graph "{name}" cascade')

    @pytest.mark.asyncio
    async def test_it_describes_there_too(self, conn) -> None:  # type: ignore[no-untyped-def]
        found = await conn.describe()
        assert found.counts["person"] == 1
        assert [p.name for p in found.properties["person"]] == ["name"]
