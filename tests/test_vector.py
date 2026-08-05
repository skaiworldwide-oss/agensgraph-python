"""Reading and indexing embedding vectors.

Skipped where the extension is not created, since it is created per database rather than per
server. The two things worth pinning are that promotion changes what a vector property reads as --
which is the whole reason this module exists -- and that the dimension in a cast is the difference
between an index that exists and one the server refuses.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

import agensgraph
from agensgraph.vector import (
    TYPES,
    expression_index,
    generated_column,
    nearest,
    parse_vector_text,
)

pytestmark = pytest.mark.server

EMBEDDING = [1.0, 2.0, 3.0, 4.0]


@pytest.fixture
def vectors(agens):  # type: ignore[no-untyped-def]
    """A graph with both routes set up, or a skip if the extension is not here."""
    if not agens.has_vectors():
        pytest.skip("the vector extension is not created in this database")
    found = agens.register_vectors()
    assert "vector" in found
    agens.execute(f"create vlabel emb ({generated_column('v', 4)})")
    agens.execute("create vlabel loose")
    agens.refresh_labels()
    return agens


class TestReadingTheText:
    """No server needed: the text form is a bracketed list and nothing else."""

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("[1,2,3]", [1.0, 2.0, 3.0]),
            ("[1.5,-2.25]", [1.5, -2.25]),
            ("[]", []),
            ("  [1, 2]  ", [1.0, 2.0]),
            ("[1e3]", [1000.0]),
        ],
    )
    def test_it_is_read_as_numbers(self, text: str, expected: list[float]) -> None:
        assert parse_vector_text(text) == expected

    @pytest.mark.parametrize("text", ["1,2,3", "[1,2", "1,2]", '"[1,2]"', "", "null"])
    def test_something_that_is_not_one_is_refused(self, text: str) -> None:
        """Rather than half-read, which is what a value that went through a quoter looks like."""
        with pytest.raises(ValueError, match="not a vector"):
            parse_vector_text(text)


class TestTheStatementsItBuilds:
    def test_a_generated_column_carries_its_dimension(self) -> None:
        assert generated_column("v", 1024) == "v vector(1024) generated"

    def test_generated_is_not_optional(self) -> None:
        """A promoted column that is not generated is refused by the server outright."""
        assert "generated" in generated_column("v", 4)

    def test_half_precision_can_be_asked_for(self) -> None:
        assert generated_column("v", 4, type="halfvec") == "v halfvec(4) generated"

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_dimensionless_vector_is_refused(self, bad: int) -> None:
        with pytest.raises(ValueError, match="at least one dimension"):
            generated_column("v", bad)

    def test_an_unknown_type_is_refused(self) -> None:
        with pytest.raises(ValueError, match="expected one of"):
            generated_column("v", 4, type="sparsevec")

    def test_an_expression_index_carries_the_dimension_in_the_cast(self) -> None:
        """Which is the difference between an index and 'column does not have dimensions'."""
        statement = expression_index("g", "doc", "v", 4)
        assert "::vector(4)" in statement
        assert "using hnsw" in statement

    def test_a_name_needing_quoting_is_quoted(self) -> None:
        assert '"odd graph"."odd label"' in expression_index("odd graph", "odd label", "v", 4)

    def test_the_nearest_statement_is_sql_because_cypher_has_no_operator(self) -> None:
        statement = nearest("g", "emb", "v", limit=5)
        assert "<->" in statement
        assert "limit 5" in statement

    def test_and_it_casts_its_parameter(self) -> None:
        """Because this driver says a string is text, and there is no vector-to-text operator."""
        assert "%s::vector" in nearest("g", "emb", "v")


class TestPromotionChangesWhatAPropertyReadsAs:
    """The reason this module exists, asserted rather than described."""

    def test_without_a_column_of_its_own_it_is_a_list(self, vectors) -> None:  # type: ignore[no-untyped-def]
        vectors.execute("create (:loose {v: [1,2,3,4]})")
        (value,) = vectors.execute_query("match (n:loose) return n.v").records[0]
        assert value == EMBEDDING
        assert isinstance(value, list)

    def test_with_one_it_is_the_columns_type_and_is_read_as_numbers(self, vectors) -> None:  # type: ignore[no-untyped-def]
        """Registering is what makes this a list of numbers rather than the string it prints as."""
        vectors.execute("create (:emb {v: [1,2,3,4]})")
        (value,) = vectors.execute_query("match (n:emb) return n.v").records[0]
        assert value == EMBEDDING
        assert isinstance(value, list)
        assert all(isinstance(each, float) for each in value)

    def test_without_registering_it_would_be_a_string(self, agens) -> None:  # type: ignore[no-untyped-def]
        """The failure this guards against: correct without promotion, wrong with it."""
        if not agens.has_vectors():
            pytest.skip("the vector extension is not created in this database")
        agens.execute(f"create vlabel emb ({generated_column('v', 4)})")
        agens.execute("create (:emb {v: [1,2,3,4]})")
        (value,) = agens.execute_query("match (n:emb) return n.v").records[0]
        assert isinstance(value, str), (
            "psycopg has no loader for the type, so it arrives as text"
        )
        assert value.startswith("[")

    def test_both_renderings_agree(self, vectors) -> None:  # type: ignore[no-untyped-def]
        vectors.execute("create (:emb {v: [1,2,3,4]})")
        graph = vectors.label_table.graph
        text = vectors.execute(f'select v from "{graph}".emb').fetchone()[0]
        binary = vectors.execute(f'select v from "{graph}".emb', binary=True).fetchone()[0]
        assert text == binary == EMBEDDING


class TestTheServerEnforcesTheDimension:
    def test_a_wrong_length_value_is_refused_at_the_moment_it_is_written(self, vectors) -> None:  # type: ignore[no-untyped-def]
        """Which is what putting the dimension on the column buys."""
        with pytest.raises(agensgraph.errors.Error, match="dimensions"):
            vectors.execute("create (:emb {v: [1,2,3]})")

    def test_the_declared_type_carries_it(self, vectors) -> None:  # type: ignore[no-untyped-def]
        declared = vectors.declared_properties("emb")
        assert [(each.name, each.type) for each in declared] == [("v", "vector(4)")]

    def test_a_loose_property_is_not_declared_at_all(self, vectors) -> None:  # type: ignore[no-untyped-def]
        vectors.execute("create (:loose {v: [1,2,3,4]})")
        assert vectors.declared_properties("loose") == []


class TestIndexing:
    def test_a_column_can_be_indexed(self, vectors) -> None:  # type: ignore[no-untyped-def]
        graph = vectors.label_table.graph
        vectors.execute("create (:emb {v: [1,2,3,4]})")
        vectors.execute(f'create index on "{graph}".emb using hnsw (v vector_l2_ops)')
        found = vectors.execute(
            "select count(*) from pg_indexes where schemaname = %s and indexdef like %s",
            (graph, "%hnsw%"),
        ).fetchone()[0]
        assert found == 1

    def test_a_bare_cast_cannot_be_indexed(self, vectors) -> None:  # type: ignore[no-untyped-def]
        """The trap. Without a dimension the server refuses outright."""
        graph = vectors.label_table.graph
        with pytest.raises(agensgraph.errors.Error, match="does not have dimensions"):
            vectors.execute(
                f'create index on "{graph}".loose using hnsw '
                f"(((properties ->> 'v')::vector) vector_l2_ops)"
            )

    def test_a_cast_carrying_the_dimension_can_be(self, vectors) -> None:  # type: ignore[no-untyped-def]
        graph = vectors.label_table.graph
        vectors.execute(expression_index(graph, "loose", "v", 4))

    def test_a_nearest_neighbour_search_runs(self, vectors) -> None:  # type: ignore[no-untyped-def]
        graph = vectors.label_table.graph
        vectors.execute("create (:emb {v: [1,2,3,4]})")
        vectors.execute("create (:emb {v: [9,9,9,9]})")
        rows = vectors.execute(nearest(graph, "emb", "v", limit=1), ("[1,2,3,4]",)).fetchall()
        assert len(rows) == 1
        assert rows[0][1] == 0.0, "the closest thing to a vector is itself"


class TestRegistering:
    def test_it_reports_what_it_found(self, agens) -> None:  # type: ignore[no-untyped-def]
        if not agens.has_vectors():
            pytest.skip("the vector extension is not created in this database")
        found = agens.register_vectors()
        assert "vector" in found
        assert set(found) <= set(TYPES)

    def test_asking_costs_nothing_and_raises_nothing(self, agens) -> None:  # type: ignore[no-untyped-def]
        """Asking about a database one did not set up is reasonable, so it answers rather than fails."""
        assert isinstance(agens.has_vectors(), bool)


class TestTheAwaitingInterface:
    @pytest_asyncio.fixture
    async def avectors(self, dsn: str):  # type: ignore[no-untyped-def]
        graph = "vector_async"
        conn = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
        async with conn:
            if not await conn.has_vectors():
                pytest.skip("the vector extension is not created in this database")
            await conn.execute(f'drop graph if exists "{graph}" cascade')
            await conn.execute(f'create graph "{graph}"')
            await conn.graph(graph)
            await conn.register_vectors()
            await conn.execute(f"create vlabel emb ({generated_column('v', 4)})")
            await conn.refresh_labels()
            try:
                yield conn
            finally:
                await conn.execute("reset graph_path")
                await conn.execute(f'drop graph "{graph}" cascade')

    @pytest.mark.asyncio
    async def test_a_vector_reads_as_numbers(self, avectors) -> None:  # type: ignore[no-untyped-def]
        await avectors.execute("create (:emb {v: [1,2,3,4]})")
        result = await avectors.execute_query("match (n:emb) return n.v")
        assert result.records[0][0] == EMBEDDING

    @pytest.mark.asyncio
    async def test_the_dimension_is_enforced(self, avectors) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(agensgraph.errors.Error, match="dimensions"):
            await avectors.execute("create (:emb {v: [1,2]})")

    @pytest.mark.asyncio
    async def test_it_is_declared(self, avectors) -> None:  # type: ignore[no-untyped-def]
        declared = await avectors.declared_properties("emb")
        assert [(each.name, each.type) for each in declared] == [("v", "vector(4)")]
