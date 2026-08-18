"""Loading a lot of elements at once.

The identities matter most here. Copying supplies none, and the point of the tests below is that
what comes out is indistinguishable from what a ``CREATE`` would have produced -- the right label
id, a sequence with no gaps and no repeats, and elements the ordinary reader decodes.
"""

from __future__ import annotations

import gc

import psycopg
import pytest
import pytest_asyncio

import agensgraph
from agensgraph import GraphId
from agensgraph.bulk import (
    build_identity_map,
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


class TestNamingTheKeysWanted:
    """The whole label is not a thing an index can make cheaper.

    Every property of an element lives in one column, so reading one key reassembles all of them,
    and PostgreSQL will not answer a projection from an index over an expression -- a purpose-built
    index on the same expression is not used even with sequential scans turned off. So the way to
    make it cheap is to ask for fewer keys, or to give the key a column of its own.
    """

    @pytest.fixture
    def keyed(self, agens):  # type: ignore[no-untyped-def]
        agens.execute("create vlabel doc")
        agens.execute("create unique property index on doc (k)")
        agens.refresh_labels()
        agens.load_vertices("doc", [{"k": f"k{i}", "pad": "x" * 200} for i in range(500)])
        return agens

    def test_naming_them_gives_the_same_identities(self, keyed) -> None:  # type: ignore[no-untyped-def]
        whole = keyed.identity_map("doc", "k")
        named = keyed.identity_map("doc", "k", keys=[f"k{i}" for i in range(50)])
        assert len(named) == 50
        assert all(whole[key] == identity for key, identity in named.items())

    def test_a_key_that_is_not_there_is_simply_absent(self, keyed) -> None:  # type: ignore[no-untyped-def]
        named = keyed.identity_map("doc", "k", keys=["k1", "nope"])
        assert set(named) == {"k1"}

    def test_naming_none_of_them_reads_nothing(self, keyed) -> None:  # type: ignore[no-untyped-def]
        assert keyed.identity_map("doc", "k", keys=[]) == {}

    def test_either_spelling_of_a_key_finds_it(self, keyed) -> None:  # type: ignore[no-untyped-def]
        """The whole-label read compares text on both sides, so the named read has to as well."""
        keyed.execute("create vlabel numbered")
        keyed.execute("create unique property index on numbered (k)")
        keyed.refresh_labels()
        keyed.execute("create (:numbered {k: 5})")
        assert set(keyed.identity_map("numbered", "k", keys=[5])) == {"5"}
        assert set(keyed.identity_map("numbered", "k", keys=["5"])) == {"5"}

    def test_with_nothing_to_look_a_key_up_by_the_label_is_read(self, agens) -> None:  # type: ignore[no-untyped-def]
        """A lookup would read the label once per key, so the whole map is the cheaper of the two."""
        agens.execute("create vlabel plain")
        agens.refresh_labels()
        agens.load_vertices("plain", [{"k": f"k{i}"} for i in range(10)])
        named = agens.identity_map("plain", "k", keys=["k1"])
        assert set(named) == {f"k{i}" for i in range(10)}, "the whole label came back"

    def test_naming_them_beats_reading_the_label(self, keyed) -> None:  # type: ignore[no-untyped-def]
        """Not a benchmark -- a floor. The whole read costs the same whatever is asked for."""
        import time

        keyed.load_vertices("doc", [{"k": f"m{i}", "pad": "x" * 400} for i in range(4000)])
        wanted = [f"k{i}" for i in range(20)]
        started = time.monotonic()
        keyed.identity_map("doc", "k", keys=wanted)
        named = time.monotonic() - started
        started = time.monotonic()
        keyed.identity_map("doc", "k")
        whole = time.monotonic() - started
        assert named < whole


class TestAKeyWithAColumnOfItsOwn:
    """A promoted key sits beside the property map rather than inside it, so reading it detoasts
    nothing. The map it produces has to be the same one the map route gives."""

    def test_the_column_is_read_rather_than_the_map(self, agens) -> None:  # type: ignore[no-untyped-def]
        if not agens.can_promote_properties():
            pytest.skip("this server cannot store a property in a column of its own")
        agens.execute("create vlabel doc (k text generated)")
        agens.refresh_labels()
        agens.load_vertices("doc", [{"k": f"k{i}"} for i in range(20)])
        sent: list[str] = []

        def record(record_of) -> None:  # type: ignore[no-untyped-def]
            sent.append(record_of.statement)

        agensgraph.add_query_logger(record)
        try:
            found = agens.identity_map("doc", "k")
        finally:
            agensgraph.remove_query_logger(record)
        assert len(found) == 20
        assert not [text for text in sent if "properties ->>" in text], "the map was read"

    def test_it_agrees_with_the_map_route(self, agens) -> None:  # type: ignore[no-untyped-def]
        if not agens.can_promote_properties():
            pytest.skip("this server cannot store a property in a column of its own")
        agens.execute("create vlabel promoted (k text generated)")
        agens.execute("create vlabel plain")
        agens.refresh_labels()
        rows = [{"k": f"k{i}"} for i in range(20)]
        agens.load_vertices("promoted", rows)
        agens.load_vertices("plain", rows)
        assert set(agens.identity_map("promoted", "k")) == set(agens.identity_map("plain", "k"))

    def test_a_boolean_column_is_not_read_that_way(self, agens) -> None:  # type: ignore[no-untyped-def]
        """It reads back as Python's ``True`` where the map gives ``true``, so the key would
        change spelling and find a different element."""
        if not agens.can_promote_properties():
            pytest.skip("this server cannot store a property in a column of its own")
        agens.execute("create vlabel flagged (k boolean generated)")
        agens.refresh_labels()
        agens.execute("create (:flagged {k: true})")
        sent: list[str] = []

        def record(record_of) -> None:  # type: ignore[no-untyped-def]
            sent.append(record_of.statement)

        agensgraph.add_query_logger(record)
        try:
            found = agens.identity_map("flagged", "k")
        finally:
            agensgraph.remove_query_logger(record)
        assert set(found) == {"true"}, "the spelling the map gives"
        assert any("properties ->>" in text for text in sent)


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


class TestAnIdentityMapNeedsAKeyThatIdentifies:
    """What is lost is not an entry: an edge resolved through the map lands on whichever
    element survived, or on nothing, and neither says so."""

    def test_a_key_that_identifies_builds_a_map(self) -> None:
        rows = [("a", GraphId(3, 1)), ("b", GraphId(3, 2))]
        assert build_identity_map(rows, label="t", key="k") == {
            "a": GraphId(3, 1),
            "b": GraphId(3, 2),
        }

    def test_a_key_shared_by_two_elements_is_refused(self) -> None:
        rows = [("a", GraphId(3, 1)), ("a", GraphId(3, 2))]
        with pytest.raises(ValueError, match="shared by more than one"):
            build_identity_map(rows, label="t", key="k")

    def test_an_element_holding_no_key_is_refused(self) -> None:
        rows = [("a", GraphId(3, 1)), (None, GraphId(3, 2))]
        with pytest.raises(ValueError, match="hold no"):
            build_identity_map(rows, label="t", key="k")

    def test_the_message_names_the_label_and_the_key(self) -> None:
        with pytest.raises(ValueError, match=r"'k' does not identify an element of 't'"):
            build_identity_map([(None, GraphId(3, 1))], label="t", key="k")

    def test_a_number_and_its_text_are_one_key(self) -> None:
        """Both sides read the key as text, so 1 and '1' are the same element's key."""
        with pytest.raises(ValueError, match="shared by more than one"):
            build_identity_map([(1, GraphId(3, 1)), ("1", GraphId(3, 2))], label="t", key="k")


class TestTheGraphIdBinaryLoader:
    def test_a_payload_that_is_not_eight_bytes_is_refused(self) -> None:
        """Read as a plain integer a truncated one is not short, it is another valid identity."""
        from agensgraph.adapters import GraphIdBinaryLoader

        for payload in (b"", b"\x00\x00\x00\x01", b"\x00" * 9):
            with pytest.raises(ValueError, match="8 bytes"):
                GraphIdBinaryLoader(7002).load(payload)

    def test_eight_bytes_read_as_the_identity_they_hold(self) -> None:
        from agensgraph.adapters import GraphIdBinaryLoader

        assert GraphIdBinaryLoader(7002).load(b"\x00\x03\x00\x00\x00\x00\x00\x01") == GraphId(
            3, 1
        )


@pytest.mark.server
class TestACopysFailureIsDescribedToo:
    """A copy reports its failure when the block ends rather than from a statement, so the cursor
    that describes every statement never saw it, and the whole of what the server said reached the
    caller. For a copy that is the row it refused.
    """

    def test_the_row_it_refused_is_not_in_the_message(self, agens) -> None:  # type: ignore[no-untyped-def]
        secret = "alice@example.com"
        agens.execute("create vlabel doc")
        agens.refresh_labels()
        agens.execute("create unique property index on doc (email)")
        agens.execute("create (:doc {email: %s})", (secret,))
        with pytest.raises(psycopg.Error) as caught:
            agens.load_vertices("doc", [{"email": secret}])
        assert caught.value.sqlstate == "23505"
        assert secret not in str(caught.value)
        assert secret in (caught.value.diag.message_detail or ""), (
            "only the one line is redacted; a post-mortem still needs the row"
        )

    def test_a_statement_and_a_copy_read_the_same_way(self, agens) -> None:  # type: ignore[no-untyped-def]
        """Which is the point: how a failure reads should not depend on how it was sent."""
        secret = "bob@example.com"
        agens.execute("create vlabel doc")
        agens.refresh_labels()
        agens.execute("create unique property index on doc (email)")
        agens.execute("create (:doc {email: %s})", (secret,))
        with pytest.raises(psycopg.Error) as by_statement:
            agens.execute("create (:doc {email: %s})", (secret,))
        with pytest.raises(psycopg.Error) as by_copy:
            agens.load_vertices("doc", [{"email": secret}])
        assert str(by_statement.value) == str(by_copy.value)
