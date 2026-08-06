"""What a live server actually does.

These tests exist to keep the driver's assumptions honest. Several of them assert
behaviour that looks like a defect and is not the driver's to fix -- a refusal reported
without a code of its own, a message with ``???`` in it, a query shape the server accepts
and misreads. Each is written down here so that the workaround it justifies cannot outlive
the behaviour it works around: if a later release reports these properly, one of these
tests fails and says which workaround to drop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import psycopg
import pytest

from agensgraph import Edge, GraphId, Path, Vertex
from agensgraph._protocol.labels import LabelCache
from agensgraph.adapters import OIDS, assert_oids, register_binary
from agensgraph.capabilities import Capabilities
from agensgraph.errors import ConfigurationError, ReadOnlyGraphWrite, StaleLabelCache, translate

if TYPE_CHECKING:
    from psycopg import Connection

pytestmark = pytest.mark.server

TEXT_OID = 25
"""What a promoted read of a ``text`` property comes back as, where jsonb is 3802."""


@pytest.fixture
def graph(conn: Connection[object]) -> Connection[object]:
    """A small graph: two people, one edge, one path between them."""
    conn.execute("create vlabel person")
    conn.execute("create elabel knows")
    conn.execute(
        "create (:person {name: 'a', n: 1})-[:knows {since: 2020}]->(:person {name: 'b'})"
    )
    return conn


@pytest.fixture
def labels(graph: Connection[object]) -> LabelCache:
    cache = LabelCache()
    name = graph.execute("select current_setting('graph_path')").fetchone()[0]
    cache.load(name, graph.execute(cache.query, (name,)).fetchall())
    return cache


class TestTheWrittenDownOids:
    def test_they_match_the_server(self, conn: Connection[object]) -> None:
        assert_oids(conn)

    def test_a_graph_id_is_exactly_eight_bytes(self, conn: Connection[object]) -> None:
        (length,) = conn.execute("select octet_length(graphid_send('3.1'::graphid))").fetchone()
        assert length == 8

    def test_the_first_label_a_user_creates_has_id_three(
        self, graph: Connection[object]
    ) -> None:
        """Two labels come with the graph, so a user's first is the third."""
        (v,) = graph.execute("match (n:person) return n limit 1").fetchone()
        assert v.id.labid == 3


class TestCapabilities:
    def test_the_version_arrives_without_being_asked_for(
        self, conn: Connection[object]
    ) -> None:
        caps = Capabilities.of(conn)
        assert caps.version >= (2, 16)
        assert caps.reported == conn.info.parameter_status("agversion")

    def test_the_graph_path_does_not_arrive_that_way(self, conn: Connection[object]) -> None:
        """So a label table cannot be invalidated by watching for a parameter change."""
        assert conn.info.parameter_status("graph_path") is None


class TestBothRenderingsAgree:
    """The differential check, against values the server itself produced."""

    @pytest.fixture(autouse=True)
    def _binary(self, graph: Connection[object], labels: LabelCache) -> None:
        register_binary(graph, labels)

    def both(self, conn: Connection[object], query: str) -> tuple[object, object]:
        return (
            conn.execute(query).fetchall(),
            conn.execute(query, binary=True).fetchall(),
        )

    @pytest.mark.parametrize(
        "query",
        [
            "match (n:person) return n order by n.name",
            "match ()-[r:knows]->() return r",
            "match p = (:person)-[:knows]->(:person) return p",
            "match p = (:person)-[:knows]->(:person) return nodes(p)",
            "match p = (:person)-[:knows]->(:person) return relationships(p)",
        ],
    )
    def test_the_same_query_read_twice(self, graph: Connection[object], query: str) -> None:
        from_text, from_binary = self.both(graph, query)
        assert from_text == from_binary

    def test_and_the_parts_agree_too(self, graph: Connection[object]) -> None:
        """Equality is on identity alone, so it would not notice a wrong label or property."""
        query = "match (n:person) return n order by n.name"
        (from_text, from_binary) = self.both(graph, query)
        for (t,), (b,) in zip(from_text, from_binary, strict=True):
            assert (t.id, t.label, t.properties) == (b.id, b.label, b.properties)

    def test_an_edges_endpoints_agree(self, graph: Connection[object]) -> None:
        (from_text, from_binary) = self.both(graph, "match ()-[r:knows]->() return r")
        (t,), (b,) = from_text[0], from_binary[0]
        assert (t.start, t.end, t.label, t.properties) == (
            b.start,
            b.end,
            b.label,
            b.properties,
        )

    def test_a_label_created_after_the_table_was_filled(
        self, graph: Connection[object], labels: LabelCache
    ) -> None:
        """The text rendering carries the name; only the composite one has to resolve it."""
        graph.execute("create vlabel later")
        graph.execute("create (:later)")
        assert graph.execute("match (n:later) return n").fetchone()[0].label == "later"
        with pytest.raises(StaleLabelCache):
            graph.execute("match (n:later) return n", binary=True).fetchall()


class TestValuesTheServerRarelyProduces:
    def test_an_empty_path(self, graph: Connection[object]) -> None:
        (p,) = graph.execute(
            "match (n:person) return shortestpath((n)-[*0..0]->(n)) limit 1"
        ).fetchone()
        assert isinstance(p, Path)

    def test_a_label_holding_the_renderings_own_characters(
        self, conn: Connection[object]
    ) -> None:
        """The server accepts it and writes it back with no escaping."""
        conn.execute('create vlabel "a,b}c{"')
        conn.execute("""create (:"a,b}c{" {k: '},{'})""")
        (v,) = conn.execute('match (n:"a,b}c{") return n').fetchone()
        assert v.label == "a,b}c{"
        assert v.properties == {"k": "},{"}

    def test_a_graph_id_at_its_maximum(self, conn: Connection[object]) -> None:
        (gid,) = conn.execute("select '65535.281474976710655'::graphid").fetchone()
        assert gid == GraphId(65535, 281474976710655)

    def test_a_graph_id_above_the_signed_range_in_both_renderings(
        self, conn: Connection[object]
    ) -> None:
        query = "select '40000.7'::graphid"
        assert conn.execute(query).fetchone()[0] == GraphId(40000, 7)
        assert conn.execute(query, binary=True).fetchone()[0] == GraphId(40000, 7)

    def test_a_property_map_holding_what_looks_like_an_edge(
        self, conn: Connection[object]
    ) -> None:
        """A value that made the earlier driver invent endpoints out of property text."""
        conn.execute("create vlabel v")
        conn.execute("""create (:v {a: '][1.1,2.2]'})""")
        (v,) = conn.execute("match (n:v) return n").fetchone()
        assert isinstance(v, Vertex)
        assert v.properties == {"a": "][1.1,2.2]"}


class TestRefusalsTheServerReportsBadly:
    def test_plain_sql_writing_to_a_label_table(self, graph: Connection[object]) -> None:
        """Reported as an internal fault, with the message as the only thing naming it."""
        name = graph.execute("select current_setting('graph_path')").fetchone()[0]
        with pytest.raises(psycopg.errors.InternalError_) as caught:
            graph.execute(f"insert into \"{name}\".person values (default, '{{}}'::jsonb)")
        assert caught.value.sqlstate == "XX000"
        replacement = translate(caught.value)
        assert isinstance(replacement, ConfigurationError)
        assert replacement.setting == "enable_graph_dml"

    def test_a_write_of_more_than_one_clause_with_eagerness_off(
        self, graph: Connection[object]
    ) -> None:
        graph.execute("set enable_eager = off")
        try:
            with pytest.raises(psycopg.errors.InternalError_) as caught:
                graph.execute("match (n:person) set n.b = 1 create (:person {c: 2})")
        finally:
            graph.execute("reset enable_eager")
        assert caught.value.sqlstate == "XX000"
        replacement = translate(caught.value)
        assert isinstance(replacement, ConfigurationError)
        assert replacement.setting == "enable_eager"

    def test_a_graph_write_in_a_read_only_transaction(self, graph: Connection[object]) -> None:
        """Classified correctly and described with a literal '???', because it has no name."""
        graph.autocommit = False
        graph.execute("set transaction read only")
        try:
            with pytest.raises(psycopg.errors.ReadOnlySqlTransaction) as caught:
                graph.execute("create (:person)")
        finally:
            graph.rollback()
            graph.autocommit = True
        assert caught.value.sqlstate == "25006"
        assert "???" in str(caught.value)
        assert isinstance(translate(caught.value), ReadOnlyGraphWrite)


class TestShapesTheServerAcceptsAndMisreads:
    def test_a_variable_length_bound_binds_without_complaint(
        self, graph: Connection[object]
    ) -> None:
        """It is read as an unbounded walk plus a property map, and nothing says so.

        Every other position that cannot take a parameter fails with a syntax error, so this
        one shape has to be refused by the driver before it is sent.
        """
        graph.execute("prepare vle as match (a)-[r*1..$1]->(b) return a")
        try:
            (types,) = graph.execute(
                "select parameter_types::text[] from pg_prepared_statements where name = 'vle'"
            ).fetchone()
        finally:
            graph.execute("deallocate vle")
        assert types == ["jsonb"]

    def test_a_composite_field_reference_is_read_as_a_property(
        self, graph: Connection[object]
    ) -> None:
        """``(n).id`` asks for a property named id, which no vertex has, so it is null.

        Reading the parts of an element uses the functions, not field access.
        """
        row = graph.execute("match (n:person) return (n).id, (n).properties limit 1").fetchone()
        assert row == (None, None)

    def test_the_functions_are_what_read_the_parts(self, graph: Connection[object]) -> None:
        row = graph.execute(
            "match (n:person) return id(n), properties(n), label(n) limit 1"
        ).fetchone()
        gid, props, label = row
        assert isinstance(gid, GraphId)
        assert isinstance(props, dict)
        assert label == "person"

    def test_a_property_set_to_null_is_not_stored_at_all(
        self, graph: Connection[object]
    ) -> None:
        """So a write is not a round trip, and no client-side check may assume it is."""
        graph.execute("create (:person {name: 'c', nickname: null})")
        (props,) = graph.execute("match (n:person {name: 'c'}) return properties(n)").fetchone()
        assert props == {"name": "c"}


class TestBinaryIsAllOrNothing:
    def test_one_unreadable_column_fails_the_whole_result(
        self, conn: Connection[object]
    ) -> None:
        """Which is why the binary rendering is asked for per statement and never by default."""
        conn.execute("select 1::int4").fetchall()
        with pytest.raises(psycopg.errors.ProgrammingError):
            conn.execute("select 1::int4, '{}'::aclitem[]", binary=True).fetchall()

    def test_the_type_with_no_conversion_at_all(self, conn: Connection[object]) -> None:
        """``rowid`` cannot be read in either rendering, so nothing pretends otherwise."""
        assert "rowid" in OIDS
        with pytest.raises(psycopg.Error):
            conn.execute("select '(1,1)'::rowid").fetchall()


class TestWhatBindsAsWhat:
    """The parameter type the server chooses, which decides what a caller may pass."""

    @pytest.mark.parametrize(
        ("statement", "expected"),
        [
            ("match (n:person) where n.name = $1 return n", ["jsonb"]),
            ("match (n:person) return n limit $1", ["bigint"]),
            ("create (:person $1)", ["jsonb"]),
            ("match (n:person) set n = $1", ["jsonb"]),
            ("match (n:person) where id(n) = $1 return n", ["graphid"]),
            ("match (n:person) return size($1)", ["text"]),
        ],
    )
    def test_the_binding(
        self, graph: Connection[object], statement: str, expected: list[str]
    ) -> None:
        graph.execute(f"prepare p as {statement}")
        try:
            (types,) = graph.execute(
                "select parameter_types::text[] from pg_prepared_statements where name = 'p'"
            ).fetchone()
        finally:
            graph.execute("deallocate p")
        assert types == expected

    @pytest.mark.parametrize(
        "statement",
        [
            "match (n:$1) return n",
            "match (n:person) return n.$1",
            "match (n:person) return {$1: 1}",
            "match (a)-[r*$1..2]->(b) return a",
        ],
    )
    def test_what_cannot_be_bound_at_all(
        self, graph: Connection[object], statement: str
    ) -> None:
        with pytest.raises(psycopg.errors.SyntaxError):
            graph.execute(f"prepare p as {statement}")


class TestAGraphIdBindsBothWays:
    def test_finding_a_vertex_by_its_identity(self, graph: Connection[object]) -> None:
        (v,) = graph.execute("match (n:person {name: 'a'}) return n").fetchone()
        (again,) = graph.execute(
            "match (n:person) where id(n) = %s return n", (v.id,)
        ).fetchone()
        assert again.id == v.id
        assert again == v

    def test_and_in_the_binary_rendering(self, graph: Connection[object]) -> None:
        (v,) = graph.execute("match (n:person {name: 'a'}) return n").fetchone()
        (gid,) = graph.execute(
            "select id(n) from (match (n:person) where id(n) = %b return n) t",
            (v.id,),
        ).fetchone()
        assert gid == v.id


def test_an_edge_and_a_vertex_are_usable_as_keys(graph: Connection[object]) -> None:
    """Which the earlier driver made impossible by defining equality without hashing."""
    rows = graph.execute("match (a)-[r]->(b) return a, r, b").fetchall()
    seen: dict[object, int] = {}
    for a, r, b in rows:
        for element in (a, r, b):
            seen[element] = seen.get(element, 0) + 1
    assert len(seen) == 3
    assert all(isinstance(k, Vertex | Edge) for k in seen)


class TestThePromotedSentinelColumn:
    """A hidden column the server projects to carry a promoted read across a scope boundary.

    It is named ``_agens_default_prop:<var>:<key>`` and it is real: with promotion on, a
    ``WITH`` or ``LET`` that carries a graph element forward appends one target entry per
    promoted property of that element's label, so a later ``WHERE`` or ``ORDER BY`` still
    reaches the typed column and its index rather than the property map.

    It cannot reach a client, and the driver therefore does not filter it. Two things in the
    server say so, and both are asserted below rather than taken on trust. Sentinels are
    appended for ``WITH`` and ``LET`` only -- ``RETURN`` is terminal and excluded, so that its
    projection stays exactly what was written -- and a Cypher statement's columns are its
    terminal ``RETURN``'s. And star expansion drops every target entry whose name is a
    sentinel, so ``RETURN *`` after such a ``WITH`` does not carry them either.

    Filtering them here would be worse than not: a record is a tuple, so dropping a name
    without dropping the value beside it would leave the names and the values misaligned.
    """

    SHAPES = (
        "match (d:doc) return d.title",
        "match (d:doc) with d as e where e.title = 'a' return *",
        "match (d:doc) with d where d.n > 0 return d",
        "match (d:doc) with d, d.title as t return *",
        "match (d:doc) with d order by d.title return d.title",
        "match (d:doc) with d limit 2 with d return *",
        "match (d:doc) call { with d match (d)-[:link]->(x:doc) return x } return *",
        "match (d:doc) let t = d.title return *",
        "match (d:doc) with d match (d)-[r:link]->(y) return *",
        "match (d:doc) return *",
    )

    @pytest.fixture
    def promoted(self, agens):  # type: ignore[no-untyped-def]
        if not agens.capabilities.has_property_promotion():
            pytest.skip("this server has no property promotion")
        on = agens.execute_query("show enable_property_promotion").records[0][0]
        assert on == "on", "promotion is off on this server, so the sentinel is never projected"
        agens.execute("create vlabel doc (title text generated, n int generated)")
        agens.execute("create elabel link")
        agens.execute("create (:doc {title: 'a', n: 1})-[:link]->(:doc {title: 'b', n: 2})")
        agens.refresh_labels()
        return agens

    @pytest.mark.parametrize("statement", SHAPES)
    def test_no_shape_surfaces_it_as_a_column(self, promoted, statement) -> None:  # type: ignore[no-untyped-def]
        keys = promoted.execute_query(statement).keys
        assert not [key for key in keys if key.startswith("_agens_default_")]

    def test_the_promoted_read_is_the_column_s_own_type(self, promoted) -> None:  # type: ignore[no-untyped-def]
        """Which is what the sentinel is projected for, and what a filter here would risk.

        Without promotion the same read is jsonb; the column's own type is what reaches the
        typed column's index, and it is how this fixture is known to be promoted at all.
        """
        result = promoted.execute_query("match (d:doc) where d.title = 'a' return d.title")
        assert result.records == [("a",)]
        assert result.oids == (TEXT_OID,)
