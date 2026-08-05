"""Reading what is in a database.

There is no ``\\d`` for a graph, so this is the only way to find out. Two of these tests exist
because the obvious way to write the driver would get them wrong: a uniqueness constraint is
absent from the property-index view, and asking the constraint-definition function about a
not-null constraint raises rather than returning nothing.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

import agensgraph
from agensgraph.introspect import element_count_query

pytestmark = pytest.mark.server


@pytest.fixture
def described(agens):  # type: ignore[no-untyped-def]
    """A graph with everything worth reading about: inheritance, an index, two constraints."""
    agens.execute("create vlabel person")
    agens.execute("create vlabel employee inherits (person)")
    agens.execute("create elabel knows")
    agens.execute("create constraint on person assert name is unique")
    agens.execute("create constraint on person assert age > 0")
    agens.execute("create property index on person (name)")
    agens.execute(
        "create (:person {name: 'a', age: 3})-[:knows]->(:employee {name: 'b', age: 4})"
    )
    agens.execute("create (:person {name: 'c', age: 5})")
    return agens


class TestGraphs:
    def test_the_one_being_read_is_among_them(self, described) -> None:  # type: ignore[no-untyped-def]
        names = [g.name for g in described.graphs()]
        assert described.label_table.graph in names

    def test_each_carries_its_schema_and_a_label_count(self, described) -> None:  # type: ignore[no-untyped-def]
        mine = next(g for g in described.graphs() if g.name == described.label_table.graph)
        assert mine.schema == mine.name
        assert mine.labels == 5, "two the graph came with, plus person, employee and knows"


class TestLabels:
    def test_every_label_including_the_two_that_came_with_it(self, described) -> None:  # type: ignore[no-untyped-def]
        labels = described.labels()
        assert [label.name for label in labels] == [
            "ag_vertex",
            "ag_edge",
            "person",
            "employee",
            "knows",
        ]

    def test_the_ids_are_the_ones_in_every_element(self, described) -> None:  # type: ignore[no-untyped-def]
        by_name = {label.name: label.id for label in described.labels()}
        assert by_name["ag_vertex"] == 1
        assert by_name["ag_edge"] == 2
        (v,) = described.execute_query("match (n:person) return n limit 1").records[0]
        assert v.id.labid == by_name["person"]

    def test_a_kind_says_which_it_is(self, described) -> None:  # type: ignore[no-untyped-def]
        by_name = {label.name: label for label in described.labels()}
        assert by_name["person"].is_vertex
        assert not by_name["person"].is_edge
        assert by_name["knows"].is_edge
        assert not by_name["knows"].is_vertex

    def test_inheritance_is_reported(self, described) -> None:  # type: ignore[no-untyped-def]
        by_name = {label.name: label for label in described.labels()}
        assert by_name["employee"].parent == "person"
        assert by_name["person"].parent == "ag_vertex"
        assert by_name["knows"].parent == "ag_edge"

    def test_the_two_a_graph_comes_with_say_so(self, described) -> None:  # type: ignore[no-untyped-def]
        """A caller counting its own labels should not have to know which two it never made."""
        builtin = [label.name for label in described.labels() if label.is_builtin]
        assert builtin == ["ag_vertex", "ag_edge"]

    def test_another_graph_can_be_asked_about(self, described, dsn: str) -> None:  # type: ignore[no-untyped-def]
        other = "introspect_other"
        described.execute(f'drop graph if exists "{other}" cascade')
        described.execute(f'create graph "{other}"')
        try:
            assert len(described.labels(graph=other)) == 2
        finally:
            described.execute(f'drop graph "{other}" cascade')

    def test_with_no_graph_selected_it_says_to_name_one(self, dsn: str) -> None:
        with (
            agensgraph.connect(dsn, autocommit=True) as conn,
            pytest.raises(ValueError, match="name the one"),
        ):
            conn.labels()


class TestIndexes:
    def test_a_property_index_is_listed(self, described) -> None:  # type: ignore[no-untyped-def]
        indexes = described.indexes()
        assert [i.label for i in indexes] == ["person"]
        assert "btree (name)" in indexes[0].definition
        assert not indexes[0].unique

    def test_one_label_can_be_asked_about(self, described) -> None:  # type: ignore[no-untyped-def]
        assert described.indexes("person")
        assert described.indexes("knows") == []

    def test_a_uniqueness_constraint_is_not_among_them(self, described) -> None:  # type: ignore[no-untyped-def]
        """The view filters exclusion constraints out, and uniqueness is kept as one."""
        assert not any(i.unique for i in described.indexes())


class TestConstraints:
    def test_a_uniqueness_assertion_is_found(self, described) -> None:  # type: ignore[no-untyped-def]
        """Which the property-index view cannot show, so it is read from the constraints."""
        unique = [c for c in described.constraints() if c.unique]
        assert len(unique) == 1
        assert unique[0].label == "person"
        assert "IS UNIQUE" in unique[0].definition

    def test_a_check_is_found_too(self, described) -> None:  # type: ignore[no-untyped-def]
        checks = [c for c in described.constraints() if not c.unique]
        assert checks
        assert any("age" in c.definition for c in checks)

    def test_the_ones_every_label_carries_are_not_asked_about(self, described) -> None:  # type: ignore[no-untyped-def]
        """Asking the definition function about a not-null constraint raises, not returns."""
        names = [c.name for c in described.constraints()]
        assert not any("not_null" in name for name in names)
        assert not any(name.endswith("_pkey") for name in names)

    def test_one_label_can_be_asked_about(self, described) -> None:  # type: ignore[no-untyped-def]
        assert described.constraints("person")
        assert described.constraints("knows") == []


class TestDeclaredProperties:
    def test_a_property_in_the_json_map_is_not_declared(self, described) -> None:  # type: ignore[no-untyped-def]
        """Nothing declares it, so nothing can be read about it -- which is not 'no properties'."""
        assert described.declared_properties("person") == []

    def test_the_catalog_is_readable_even_when_empty(self, described) -> None:  # type: ignore[no-untyped-def]
        assert described.declared_properties() == []


class TestElementCounts:
    def test_vertices_and_edges_are_counted_per_label(self, described) -> None:  # type: ignore[no-untyped-def]
        counts = described.element_counts()
        assert counts["person"] == 2
        assert counts["employee"] == 1
        assert counts["knows"] == 1

    def test_a_label_holding_nothing_is_absent_rather_than_zero(self, described) -> None:  # type: ignore[no-untyped-def]
        """Counting reads only the rows that exist, so a label with none produces no row."""
        described.execute("create vlabel empty")
        assert "empty" not in described.element_counts()

    def test_the_count_reads_no_property(self, described) -> None:  # type: ignore[no-untyped-def]
        """The label id is part of the identity, so grouping needs nothing but the id column."""
        statement = element_count_query(described.label_table.graph)
        plan = "\n".join(
            row[0] for row in described.execute("explain (verbose, costs off) " + statement)
        )
        assert "properties" not in plan

    def test_a_graph_name_needing_quoting(self, agens, dsn: str) -> None:  # type: ignore[no-untyped-def]
        """The schema cannot be bound, so it is quoted, and this is what proves it."""
        odd = "count me, please"
        agens.execute(f'drop graph if exists "{odd}" cascade')
        agens.execute(f'create graph "{odd}"')
        try:
            assert agens.element_counts(graph=odd) == {}
        finally:
            agens.execute(f'drop graph "{odd}" cascade')


class TestTheAwaitingInterface:
    @pytest_asyncio.fixture
    async def adescribed(self, dsn: str):  # type: ignore[no-untyped-def]
        graph = "introspect_async"
        conn = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
        async with conn:
            await conn.execute(f'drop graph if exists "{graph}" cascade')
            await conn.execute(f'create graph "{graph}"')
            await conn.graph(graph)
            await conn.execute("create vlabel person")
            await conn.execute("create elabel knows")
            await conn.execute("create constraint on person assert name is unique")
            await conn.execute("create property index on person (name)")
            await conn.execute("create (:person {name: 'a'})-[:knows]->(:person {name: 'b'})")
            try:
                yield conn
            finally:
                await conn.execute("reset graph_path")
                await conn.execute(f'drop graph "{graph}" cascade')

    @pytest.mark.asyncio
    async def test_labels(self, adescribed) -> None:  # type: ignore[no-untyped-def]
        names = [label.name for label in await adescribed.labels()]
        assert "person" in names
        assert "knows" in names

    @pytest.mark.asyncio
    async def test_indexes_and_constraints(self, adescribed) -> None:  # type: ignore[no-untyped-def]
        assert [i.label for i in await adescribed.indexes()] == ["person"]
        assert any(c.unique for c in await adescribed.constraints())

    @pytest.mark.asyncio
    async def test_element_counts(self, adescribed) -> None:  # type: ignore[no-untyped-def]
        counts = await adescribed.element_counts()
        assert counts["person"] == 2
        assert counts["knows"] == 1

    @pytest.mark.asyncio
    async def test_graphs(self, adescribed) -> None:  # type: ignore[no-untyped-def]
        assert "introspect_async" in [g.name for g in await adescribed.graphs()]
