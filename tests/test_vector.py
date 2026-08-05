"""Reading and indexing embedding vectors.

Skipped where the extension is not created, since it is created per database rather than per
server. The two things worth pinning are that promotion changes what a vector property reads as --
which is the whole reason this module exists -- and that the dimension in a cast is the difference
between an index that exists and one the server refuses.
"""

from __future__ import annotations

import random
import struct
from array import array

import pytest
import pytest_asyncio

import agensgraph
from agensgraph.vector import (
    TYPES,
    Distance,
    SparseVector,
    Vector,
    generated_column,
    nearest,
    parse_vector_text,
    search_option_statements,
    vector_index,
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

    def test_a_sparse_column_can_be_asked_for(self) -> None:
        assert generated_column("v", 6, type="sparsevec") == "v sparsevec(6) generated"

    @pytest.mark.parametrize("unknown", ["bit", "float8", "vector4", ""])
    def test_a_type_this_driver_cannot_read_is_refused(self, unknown: str) -> None:
        with pytest.raises(ValueError, match="expected one of"):
            generated_column("v", 4, type=unknown)

    def test_an_index_over_a_property_in_the_map_is_a_property_index(self) -> None:
        """Not a plain SQL index. The route is what makes the planner match it."""
        statement = vector_index("movie", "embedding", dimensions=4)
        assert statement.startswith("create property index on movie")
        assert "using hnsw" in statement
        assert "((embedding::vector(4)) vector_cosine_ops)" in statement
        assert "->>" not in statement, "the SQL spelling is never matched by a Cypher query"

    def test_an_index_over_a_promoted_column_needs_no_cast(self) -> None:
        statement = vector_index("emb", "v", operator_class="vector_l2_ops")
        assert "((v) vector_l2_ops)" in statement
        assert "::" not in statement

    def test_the_method_and_its_options(self) -> None:
        statement = vector_index(
            "movie", "embedding", dimensions=4, method="ivfflat", options={"lists": 10}
        )
        assert "using ivfflat" in statement
        assert statement.endswith("with (lists=10)")

    def test_a_name_can_be_given(self) -> None:
        assert vector_index("movie", "e", dimensions=4, name="my_idx").startswith(
            "create property index my_idx on movie"
        )

    def test_the_operator_class_comes_from_the_operator_being_searched(self) -> None:
        """A cosine index answers ``<=>`` alone, so this pairing is not decoration."""
        statement = vector_index(
            "movie", "e", dimensions=4, operator_class=Distance.L2.operator_class
        )
        assert "vector_l2_ops" in statement

    def test_a_name_needing_quoting_is_quoted(self) -> None:
        statement = vector_index("odd label", "odd prop", dimensions=4)
        assert '"odd label"' in statement
        assert '"odd prop"::vector(4)' in statement

    def test_half_precision_goes_through_the_same_route(self) -> None:
        statement = vector_index(
            "movie", "e", dimensions=4, type="halfvec", operator_class="halfvec_l2_ops"
        )
        assert "((e::halfvec(4)) halfvec_l2_ops)" in statement

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_dimension_below_one_is_refused(self, bad: int) -> None:
        with pytest.raises(ValueError, match="at least one dimension"):
            vector_index("movie", "e", dimensions=bad)

    def test_a_type_this_driver_cannot_read_is_refused_here_too(self) -> None:
        with pytest.raises(ValueError, match="expected one of"):
            vector_index("movie", "e", dimensions=4, type="bit")

    def test_the_nearest_statement_is_cypher(self) -> None:
        """Cypher has words for a distance operator, so the search does not drop to SQL."""
        statement = nearest("movie", "embedding", dimensions=4, limit=5)
        assert statement.startswith("match (n:movie)")
        assert "<=>" in statement
        assert "limit 5" in statement

    def test_and_it_casts_its_parameter_carrying_the_dimension(self) -> None:
        """Because this driver says a string is text, and because a bare cast loses the index."""
        assert "%s::vector(4)" in nearest("movie", "embedding", dimensions=4)

    def test_a_promoted_column_is_searched_without_a_cast_on_the_column(self) -> None:
        statement = nearest("emb", "v")
        assert "n.v <=>" in statement
        assert "%s::vector" in statement

    def test_the_index_and_the_search_spell_the_cast_identically(self) -> None:
        """The trap this pairing exists to prevent: a dimension in one and not the other, or two
        different dimensions, costs the index silently -- the planner sorts a scan and says
        nothing. Asserted by taking the spelling out of one and finding it in the other."""
        for dimensions in (None, 1, 4, 1536):
            index = vector_index("movie", "embedding", dimensions=dimensions)
            search = nearest("movie", "embedding", dimensions=dimensions)
            expression = index.split("((")[1].split(") vector_cosine_ops")[0]
            assert f"n.{expression} <=>" in search


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
        assert value == EMBEDDING, "and it compares equal to a plain list, which is the point"
        assert isinstance(value, Vector), "not a list -- a value that only unpacks if asked"
        assert value.tolist() == EMBEDDING
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


def plan_of(conn, statement: str, params: tuple[object, ...] = ()) -> str:  # type: ignore[no-untyped-def]
    """The plan for a statement, as one string.

    Read with sequential scans switched off. That is deliberate: what is being asserted is whether
    the planner *matches* an index expression at all, and with a sequential scan available a plan
    showing one proves only that it looked cheaper on a small table. Penalised, a sequential scan
    means the index genuinely could not be used.
    """
    if "%s" in statement and not params:
        raise AssertionError("this statement takes a parameter, so the plan needs one too")
    conn.execute("set enable_seqscan = off")
    try:
        rows = conn.execute(f"explain (costs off) {statement}", params or None).fetchall()
    finally:
        conn.execute("reset enable_seqscan")
    return "\n".join(row[0] for row in rows)


class TestIndexing:
    """Which indexes a vector search actually uses, asserted on the plan rather than on the DDL
    being accepted. An index that is created and never matched is the failure mode here, and it is
    silent -- so every one of these reads a plan."""

    def test_a_property_in_the_map_is_indexed_and_the_index_is_used(self, vectors) -> None:  # type: ignore[no-untyped-def]
        """The documented route, and the one a caller reaches for first: no promotion at all."""
        vectors.execute(vector_index("loose", "v", dimensions=4))
        for value in ([1, 2, 3, 4], [9, 9, 9, 9], [4, 3, 2, 1]):
            vectors.execute(f"create (:loose {{v: {value}}})")
        plan = plan_of(vectors, nearest("loose", "v", dimensions=4, limit=1), ("[1,2,3,4]",))
        assert "Index Scan" in plan, f"the index was not used:\n{plan}"

    def test_a_promoted_column_is_indexed_and_the_index_is_used(self, vectors) -> None:  # type: ignore[no-untyped-def]
        """The other route. Same statement builder, no cast."""
        vectors.execute(vector_index("emb", "v", operator_class="vector_l2_ops"))
        for value in ([1, 2, 3, 4], [9, 9, 9, 9]):
            vectors.execute(f"create (:emb {{v: {value}}})")
        plan = plan_of(
            vectors, nearest("emb", "v", operator=Distance.L2, limit=1), ("[1,2,3,4]",)
        )
        assert "Index Scan" in plan, f"the index was not used:\n{plan}"

    def test_the_sql_spelling_of_the_same_index_is_never_matched(self, vectors) -> None:  # type: ignore[no-untyped-def]
        """Why :func:`vector_index` builds a property index and not a plain one.

        The index is accepted and is dead weight: Cypher compiles ``n.v`` to its own access
        operator, so an index over the jsonb arrow operator holds a different expression and is
        never a candidate. This is the mistake that motivated the whole helper.
        """
        graph = vectors.label_table.graph
        vectors.execute(
            f'create index loose_arrow on "{graph}".loose using hnsw '
            f"(((properties ->> 'v')::vector(4)) vector_l2_ops)"
        )
        vectors.execute("create (:loose {v: [1,2,3,4]})")
        plan = plan_of(
            vectors, nearest("loose", "v", dimensions=4, operator=Distance.L2), ("[1,2,3,4]",)
        )
        assert "loose_arrow" not in plan, "an arrow-operator index was matched after all"
        assert "Seq Scan" in plan

    def test_a_bare_cast_cannot_be_indexed_at_all(self, vectors) -> None:  # type: ignore[no-untyped-def]
        """Without a dimension the server refuses outright, which is the loud half of the trap."""
        with pytest.raises(agensgraph.errors.Error, match="does not have dimensions"):
            vectors.execute(
                "create property index on loose using hnsw ((v::vector) vector_l2_ops)"
            )

    def test_a_search_casting_to_a_bare_vector_loses_the_index(self, vectors) -> None:  # type: ignore[no-untyped-def]
        """And the quiet half. The dimension is part of what the expression is, so a search that
        drops it sorts a scan and says nothing about having done so."""
        vectors.execute(vector_index("loose", "v", dimensions=4))
        vectors.execute("create (:loose {v: [1,2,3,4]})")
        plan = plan_of(
            vectors,
            "match (n:loose) return n order by n.v::vector <=> '[1,2,3,4]'::vector limit 1",
        )
        assert "Seq Scan" in plan

    def test_a_search_casting_to_another_dimension_loses_it_too(self, vectors) -> None:  # type: ignore[no-untyped-def]
        vectors.execute(vector_index("loose", "v", dimensions=4))
        vectors.execute("create (:loose {v: [1,2,3,4]})")
        plan = plan_of(
            vectors,
            "match (n:loose) return n order by n.v::vector(3) <=> '[1,2,1]'::vector(3) limit 1",
        )
        assert "Seq Scan" in plan

    def test_an_operator_the_index_does_not_serve_falls_back(self, vectors) -> None:  # type: ignore[no-untyped-def]
        """A cosine index answers ``<=>``. Ordering by L2 against it sorts a scan, which is why the
        operator class has to match the operator rather than merely be present."""
        vectors.execute(vector_index("loose", "v", dimensions=4))  # cosine, by default
        vectors.execute("create (:loose {v: [1,2,3,4]})")
        plan = plan_of(
            vectors, nearest("loose", "v", dimensions=4, operator=Distance.L2), ("[1,2,3,4]",)
        )
        assert "Sort" in plan

    def test_an_ivfflat_index_with_its_list_count(self, vectors) -> None:  # type: ignore[no-untyped-def]
        vectors.execute(
            vector_index(
                "loose",
                "v",
                dimensions=4,
                method="ivfflat",
                operator_class="vector_l2_ops",
                options={"lists": 4},
            )
        )
        vectors.execute("create (:loose {v: [1,2,3,4]})")
        plan = plan_of(
            vectors,
            nearest("loose", "v", dimensions=4, operator=Distance.L2, limit=1),
            ("[1,2,3,4]",),
        )
        assert "Index Scan" in plan, f"the index was not used:\n{plan}"

    def test_a_half_precision_index_over_a_property_in_the_map(self, vectors) -> None:  # type: ignore[no-untyped-def]
        vectors.execute(
            vector_index(
                "loose", "v", dimensions=4, type="halfvec", operator_class="halfvec_l2_ops"
            )
        )
        vectors.execute("create (:loose {v: [1,2,3,4]})")
        statement = nearest(
            "loose", "v", dimensions=4, type="halfvec", operator=Distance.L2, limit=1
        )
        assert "Index Scan" in plan_of(vectors, statement, ("[1,2,3,4]",))

    def test_the_index_is_reported_by_the_catalogs(self, vectors) -> None:  # type: ignore[no-untyped-def]
        vectors.execute(vector_index("loose", "v", dimensions=4, name="loose_v_cos"))
        found = {index.name: index for index in vectors.indexes("loose")}
        assert "loose_v_cos" in found
        assert "hnsw" in found["loose_v_cos"].definition

    def test_a_search_finds_the_nearest_thing_first(self, vectors) -> None:  # type: ignore[no-untyped-def]
        vectors.execute(vector_index("loose", "v", dimensions=4))
        vectors.execute("create (:loose {v: [1,2,3,4]})")
        vectors.execute("create (:loose {v: [9,9,9,9]})")
        result = vectors.execute_query(
            nearest("loose", "v", dimensions=4, limit=1), ("[1,2,3,4]",)
        )
        assert len(result.records) == 1
        assert result.records[0][0].properties["v"] == EMBEDDING, (
            "the closest thing to a vector is itself"
        )


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


class TestSparseVectorsWithoutAServer:
    """A sparse vector is a value rather than a list, and this is why and how."""

    def test_it_keeps_only_what_is_not_zero(self) -> None:
        """Measured: three entries in a million dimensions is 36 wire bytes against 8 MiB dense."""
        v = SparseVector({0: 1.0, 499_999: 2.0}, 1_000_000)
        assert len(v) == 2, "len counts what is stored, not the dimensions"
        assert v.dimensions == 1_000_000
        assert v.indices == (0, 499_999)

    def test_the_entries_are_ordered(self) -> None:
        v = SparseVector([(3, 1.0), (0, 2.0)], 4)
        assert v.indices == (0, 3)
        assert v.values == (2.0, 1.0)

    def test_it_can_be_built_from_a_mapping_or_from_pairs(self) -> None:
        assert SparseVector({0: 1.0}, 4) == SparseVector([(0, 1.0)], 4)

    def test_a_dense_list_can_be_narrowed_and_widened_again(self) -> None:
        dense = [1.0, 0.0, 0.0, 2.0]
        v = SparseVector.from_dense(dense)
        assert v.dimensions == 4
        assert len(v) == 2
        assert v.to_dense() == dense

    def test_the_mapping_view_agrees_with_the_dense_one(self) -> None:
        v = SparseVector({0: 1.0, 3: 2.0}, 4)
        assert v.to_dict() == {0: 1.0, 3: 2.0}
        assert [v.to_dense()[i] for i in v.indices] == list(v.values)

    def test_it_is_a_value(self) -> None:
        v = SparseVector({0: 1.0}, 4)
        assert v == SparseVector({0: 1.0}, 4)
        assert {v: "seen"}[SparseVector({0: 1.0}, 4)] == "seen"
        with pytest.raises(AttributeError):
            v.dimensions = 9  # type: ignore[misc]

    def test_the_dimension_is_part_of_the_value(self) -> None:
        """An empty vector of four is not an empty vector of a thousand."""
        assert SparseVector({}, 4) != SparseVector({}, 1000)

    def test_an_explicit_zero_is_dropped_because_the_server_drops_it(self) -> None:
        """Keeping it would make a round trip lossy in one direction only."""
        assert SparseVector({0: 0.0, 1: 5.0}, 4).to_text() == "{2:5}/4"

    @pytest.mark.parametrize(
        ("entries", "dimensions", "why"),
        [
            ([(0, 1.0), (0, 2.0)], 4, "more than once"),
            ({4: 1.0}, 4, "outside"),
            ({-1: 1.0}, 4, "outside"),
            ({}, 0, "at least one dimension"),
            ({}, -1, "at least one dimension"),
            ({0: float("nan")}, 4, "no infinity and no nan"),
            ({0: float("inf")}, 4, "no infinity and no nan"),
            ({0: float("-inf")}, 4, "no infinity and no nan"),
        ],
    )
    def test_what_the_server_would_refuse_is_refused_here_first(
        self, entries, dimensions: int, why: str
    ) -> None:  # type: ignore[no-untyped-def]
        """Before the round trip is the only place a caller can still do something about it."""
        with pytest.raises(ValueError, match=why):
            SparseVector(entries, dimensions)


class TestTheTwoIndexBases:
    """The trap in the type: text counts from one, the wire counts from zero."""

    def test_the_text_form_counts_from_one(self) -> None:
        assert SparseVector({0: 9.0}, 3).to_text() == "{1:9}/3"

    def test_and_reading_it_converts_back(self) -> None:
        v = SparseVector.from_text("{1:9}/3")
        assert v.indices == (0,), "the first text entry is index zero here"
        assert v.to_dense() == [9.0, 0.0, 0.0]

    @pytest.mark.parametrize(
        "text",
        ["{1:1,3:2}/4", "{}/4", "{1:1.5}/1", "{1:1000}/4", "{ 1 : 1 }/4", "{1:-2.25,4:1}/8"],
    )
    def test_a_round_trip_through_the_text_form_is_stable(self, text: str) -> None:
        once = SparseVector.from_text(text)
        assert SparseVector.from_text(once.to_text()) == once

    @pytest.mark.parametrize(
        "text", ["1:1/4", "{1}/4", "{1:1}", "{1:1}/", "", "[1,2]", "{1:1}4", "{0:1}/4"]
    )
    def test_something_that_is_not_one_is_refused(self, text: str) -> None:
        with pytest.raises(ValueError):
            SparseVector.from_text(text)

    def test_an_index_of_zero_in_the_text_form_is_out_of_bounds(self) -> None:
        """As the server says: its text indices start at one, so zero is below the first."""
        with pytest.raises(ValueError, match="outside"):
            SparseVector.from_text("{0:1}/4")


class TestSparseVectorsAgainstAServer:
    @pytest.fixture
    def sparse(self, agens):  # type: ignore[no-untyped-def]
        if not agens.has_vectors():
            pytest.skip("the vector extension is not created in this database")
        found = agens.register_vectors()
        if "sparsevec" not in found:
            pytest.skip("this pgvector has no sparsevec")
        agens.execute("create vlabel s (v sparsevec(6) generated)")
        agens.refresh_labels()
        return agens

    def test_both_renderings_agree(self, sparse) -> None:  # type: ignore[no-untyped-def]
        text = sparse.execute("select %s::sparsevec", ("{1:1,4:2}/6",)).fetchone()[0]
        binary = sparse.execute(
            "select %s::sparsevec", ("{1:1,4:2}/6",), binary=True
        ).fetchone()[0]
        assert text == binary == SparseVector({0: 1.0, 3: 2.0}, 6)

    def test_the_server_reads_back_what_we_sent(self, sparse) -> None:  # type: ignore[no-untyped-def]
        mine = SparseVector({0: 1.0, 3: 2.0}, 6)
        assert sparse.execute("select %s::sparsevec", (mine,)).fetchone()[0] == mine

    def test_a_large_one_stays_small(self, sparse) -> None:  # type: ignore[no-untyped-def]
        """The reason it is not read densely: this would be 8 MiB of Python floats."""
        big = SparseVector({0: 1.0, 499_999: 2.0, 999_999: 3.0}, 1_000_000)
        back = sparse.execute("select %s::sparsevec", (big,)).fetchone()[0]
        assert back == big
        assert len(back) == 3

    def test_it_can_be_written_through_cypher(self, sparse) -> None:  # type: ignore[no-untyped-def]
        """A list cannot -- the server refuses it -- so sending the value is the way in."""
        mine = SparseVector({1: 5.0}, 6)
        sparse.execute("create (:s {v: %s})", (mine,))
        (stored,) = sparse.execute_query("match (n:s) return n.v").records[0]
        assert stored == mine
        assert isinstance(stored, SparseVector)

    def test_a_list_is_refused_by_the_server(self, sparse) -> None:  # type: ignore[no-untyped-def]
        """Unlike a dense column, which takes one -- which is why the dumper exists."""
        with pytest.raises(agensgraph.errors.Error, match="must start with"):
            sparse.execute("create (:s {v: [1,0,0,2,0,0]})")

    def test_the_dimension_is_enforced_at_write(self, sparse) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(agensgraph.errors.Error):
            sparse.execute("create (:s {v: %s})", (SparseVector({0: 1.0}, 99),))

    def test_it_indexes_and_answers_a_distance_query(self, sparse) -> None:  # type: ignore[no-untyped-def]
        graph = sparse.label_table.graph
        sparse.execute("create (:s {v: %s})", (SparseVector({0: 1.0, 3: 2.0}, 6),))
        sparse.execute("create (:s {v: %s})", (SparseVector({5: 9.0}, 6),))
        sparse.execute(f'create index on "{graph}".s using hnsw (v sparsevec_l2_ops)')
        rows = sparse.execute(
            f'select id, v <-> %s::sparsevec as d from "{graph}".s order by d limit 1',
            (SparseVector({0: 1.0, 3: 2.0}, 6),),
        ).fetchall()
        assert rows[0][1] == 0.0, "the closest thing to a vector is itself"

    def test_it_is_declared_with_its_dimension(self, sparse) -> None:  # type: ignore[no-untyped-def]
        assert [(p.name, p.type) for p in sparse.declared_properties("s")] == [
            ("v", "sparsevec(6)")
        ]

    def test_registering_reports_it(self, sparse) -> None:  # type: ignore[no-untyped-def]
        assert "sparsevec" in TYPES


def random_single(rng: random.Random) -> float:
    """A value drawn from the whole of single precision, not from the easy part of it."""
    while True:
        value = struct.unpack(">f", struct.pack(">I", rng.getrandbits(32)))[0]
        if value == value and abs(value) != float("inf"):
            return value


def random_half(rng: random.Random) -> float:
    """The same for half precision."""
    while True:
        value = struct.unpack(">e", struct.pack(">H", rng.getrandbits(16)))[0]
        if value == value and abs(value) != float("inf"):
            return value


class TestPrecision:
    """The two renderings have to agree on values that are hard, not only on whole numbers.

    A version of the agreement test using ``[1,2,3,4]`` passed while the two readings of an
    ordinary embedding differed on 390 values out of 400. Small whole numbers survive every
    conversion, so they cannot tell anyone anything -- these draw from the whole of the type.
    """

    def test_nine_digits_is_what_a_single_precision_value_needs(self) -> None:
        """Six is the default and loses up to five parts in a million. Measured, not assumed."""
        rng = random.Random(1)
        worst_six = worst_nine = 0.0
        for _ in range(20_000):
            value = random_single(rng)
            for text, which in ((f"{value:g}", "six"), (f"{value:.9g}", "nine")):
                back = struct.unpack(">f", struct.pack(">f", float(text)))[0]
                error = 0.0 if back == value else abs(back - value) / abs(value or 1.0)
                if which == "six":
                    worst_six = max(worst_six, error)
                else:
                    worst_nine = max(worst_nine, error)
        assert worst_nine == 0.0, "nine digits must reproduce every single-precision value"
        assert worst_six > 1e-7, "six digits loses precision, which is why nine are written"

    def test_a_sparse_vector_writes_enough_digits(self) -> None:
        rng = random.Random(2)
        for _ in range(2_000):
            value = random_single(rng)
            once = SparseVector({0: value}, 1)
            assert SparseVector.from_text(once.to_text()).values == once.values

    def test_reading_text_narrows_to_what_the_server_keeps(self) -> None:
        """Otherwise a decimal reads back as a double the stored value never equals."""
        from agensgraph.vector import parse_vector_text

        rng = random.Random(3)
        for _ in range(2_000):
            value = random_single(rng)
            (read,) = parse_vector_text(f"[{value:.9g}]")
            assert read == value

    def test_and_at_half_precision_it_narrows_to_half(self) -> None:
        from agensgraph.vector import parse_vector_text

        rng = random.Random(4)
        for _ in range(2_000):
            value = random_half(rng)
            (read,) = parse_vector_text(f"[{value:.9g}]", half=True)
            assert read == value

    def test_half_precision_narrows_and_says_so(self) -> None:
        """What survives is worth knowing, since it is chosen for memory rather than accuracy."""
        from agensgraph.vector import parse_vector_text

        assert parse_vector_text("[1]", half=True) == [1.0]
        assert parse_vector_text("[65504]", half=True) == [65504.0]
        assert parse_vector_text("[65505]", half=True) == [65504.0], "the largest it holds"
        assert parse_vector_text("[1e-8]", half=True) == [0.0], "below the smallest it holds"


@pytest.mark.parametrize("dimensions", [1, 4, 384, 1536])
def test_the_two_renderings_agree_on_hard_values(vectors, dimensions: int) -> None:  # type: ignore[no-untyped-def]
    """The test that should have caught the disagreement, drawn from the whole type."""
    rng = random.Random(dimensions)
    for _ in range(5):
        values = [random_single(rng) for _ in range(dimensions)]
        literal = "[" + ",".join(f"{v:.9g}" for v in values) + "]"
        text = vectors.execute(f"select %s::vector({dimensions})", (literal,)).fetchone()[0]
        binary = vectors.execute(
            f"select %s::vector({dimensions})", (literal,), binary=True
        ).fetchone()[0]
        assert text == binary
        assert text == values, "and both agree with what was sent"


def test_a_sparse_vector_survives_the_server_exactly(vectors) -> None:  # type: ignore[no-untyped-def]
    """Written, stored and read back without a value changing."""
    rng = random.Random(5)
    for _ in range(200):
        mine = SparseVector({0: random_single(rng), 3: random_single(rng)}, 6)
        assert vectors.execute("select %s::sparsevec", (mine,)).fetchone()[0] == mine


class TestNamedDistances:
    """Six operators, two or three punctuation characters each, two differing by one character."""

    def test_every_operator_pgvector_defines_is_named(self) -> None:
        assert {d.value for d in Distance} == {"<->", "<#>", "<=>", "<+>", "<~>", "<%>"}

    def test_each_names_the_operator_class_an_index_needs(self) -> None:
        assert Distance.L2.operator_class == "vector_l2_ops"
        assert Distance.COSINE.operator_class == "vector_cosine_ops"
        assert Distance.INNER_PRODUCT.operator_class == "vector_ip_ops"
        assert Distance.L1.operator_class == "vector_l1_ops"

    def test_the_two_that_measure_bits_say_so(self) -> None:
        assert Distance.HAMMING.is_for_bits
        assert Distance.JACCARD.is_for_bits
        assert not Distance.L2.is_for_bits

    def test_it_is_usable_straight_into_a_statement(self) -> None:
        assert f"v {Distance.COSINE} %s" == "v <=> %s"

    @pytest.mark.parametrize("distance", [d for d in Distance if not d.is_for_bits])
    def test_the_server_has_every_one_of_them(self, vectors, distance) -> None:  # type: ignore[no-untyped-def]
        row = vectors.execute(
            f"select '[1,2,3,4]'::vector(4) {distance} '[4,3,2,1]'::vector(4)"
        ).fetchone()
        assert isinstance(row[0], float)

    @pytest.mark.parametrize("distance", [d for d in Distance if d.is_for_bits])
    def test_and_the_bit_ones_too(self, vectors, distance) -> None:  # type: ignore[no-untyped-def]
        row = vectors.execute(f"select '1010'::bit(4) {distance} '1100'::bit(4)").fetchone()
        assert isinstance(row[0], float)

    @pytest.mark.parametrize("distance", [d for d in Distance if not d.is_for_bits])
    def test_and_an_index_can_be_built_for_each(self, vectors, distance) -> None:  # type: ignore[no-untyped-def]
        graph = vectors.label_table.graph
        vectors.execute(
            f'create index on "{graph}".emb using hnsw (v {distance.operator_class})'
        )


class TestSendingAVector:
    """Reading needs nothing like this. Sending does, because the alternatives format text."""

    def test_it_holds_what_it_was_given(self) -> None:
        v = Vector([1.0, 2.0, 3.0])
        assert len(v) == 3
        assert list(v.values) == [1.0, 2.0, 3.0]

    def test_it_is_a_value(self) -> None:
        assert Vector([1.0]) == Vector([1.0])
        assert hash(Vector([1.0])) == hash(Vector([1.0]))
        with pytest.raises(AttributeError):
            Vector([1.0]).values = [2.0]  # type: ignore[misc]

    def test_the_wire_form_is_the_dimension_then_the_numbers(self) -> None:
        raw = Vector([1.0, 2.0]).to_bytes()
        assert len(raw) == 4 + 2 * 4
        assert struct.unpack_from(">HH", raw, 0) == (2, 0)
        assert struct.unpack_from(">2f", raw, 4) == (1.0, 2.0)

    def test_the_bytes_are_a_third_of_the_text(self) -> None:
        """Measured at 1536 dimensions: 6,148 bytes against 17,595."""
        values = [i * 0.0012345678 for i in range(1536)]
        text = "[" + ",".join(f"{v:.9g}" for v in values) + "]"
        assert len(Vector(values).to_bytes()) < len(text) / 2

    def test_a_value_sent_this_way_survives_exactly(self, vectors) -> None:  # type: ignore[no-untyped-def]
        rng = random.Random(6)
        values = [random_single(rng) for _ in range(4)]
        sent = Vector(values)
        back = vectors.execute("select %b", (sent,)).fetchone()[0]
        assert back == values

    def test_it_can_be_written_through_cypher(self, vectors) -> None:  # type: ignore[no-untyped-def]
        vectors.execute("create (:emb {v: %s})", (Vector([1.0, 2.0, 3.0, 4.0]),))
        (stored,) = vectors.execute_query("match (n:emb) return n.v").records[0]
        assert stored == EMBEDDING

    def test_it_sends_far_less_than_a_list_does(self, vectors) -> None:  # type: ignore[no-untyped-def]
        """The mechanism behind the speed, asserted deterministically rather than by a clock.

        A list is formatted as decimal text: 1536 numbers at up to nine significant digits each.
        Packed, each is four bytes. The wall-clock consequence -- 0.32 ms against 2.55 ms for one
        1536-dimension embedding, and 31,000 rows a second against 2,000 in bulk -- is recorded in
        the module rather than asserted here, because a clock in a test suite is a flake waiting to
        happen and the byte count is the same every time.
        """
        values = [i * 0.0012345678 for i in range(1536)]
        as_text = "[" + ",".join(f"{v:.9g}" for v in values) + "]"
        as_bytes = Vector(values).to_bytes()
        assert len(as_bytes) == 4 + 4 * 1536
        assert len(as_bytes) < len(as_text) / 2.5
        # And it is the same value either way, which is what makes the saving free.
        sent = vectors.execute("select %b", (Vector(values),)).fetchone()[0]
        assert sent == array("f", values).tolist()


class TestTheExtensionsVersion:
    def test_it_is_reported_as_numbers(self, agens) -> None:  # type: ignore[no-untyped-def]
        """A version rather than a boolean, because pgvector gates its own features on it."""
        if not agens.has_vectors():
            pytest.skip("the vector extension is not created in this database")
        version = agens.vector_version()
        assert version is not None
        assert version >= (0, 5), "sparse vectors and half precision arrived in 0.7.0"
        assert all(isinstance(part, int) for part in version)

    def test_it_can_be_compared_to_decide_on_a_feature(self, agens) -> None:  # type: ignore[no-untyped-def]
        if not agens.has_vectors():
            pytest.skip("the vector extension is not created in this database")
        version = agens.vector_version()
        assert version is not None
        iterative_scans = version >= (0, 8)
        assert isinstance(iterative_scans, bool)


class TestVectorIsLazyAndStillBehavesLikeASequence:
    """The bargain: nothing is unpacked until asked, and it still compares equal to a list.

    An array of singles would have been four bytes a number and fast, and would have called itself
    unequal to ``[1.0, 2.0]`` -- quietly, since comparing an array to a list is simply false. This
    keeps the speed and the equality both.
    """

    def test_it_knows_its_length_without_unpacking_anything(self) -> None:
        """The dimension is in the first two bytes, so counting is free."""
        raw = Vector([1.0, 2.0, 3.0]).to_bytes()
        held = Vector.from_wire(raw)
        assert len(held) == 3
        assert held._values is None, "still nothing unpacked"

    def test_reading_a_number_unpacks_once_and_keeps_it(self) -> None:
        held = Vector.from_wire(Vector([1.0, 2.0]).to_bytes())
        assert held[0] == 1.0
        assert held._values is not None
        assert held._raw is None, "the bytes are let go once the numbers exist"

    def test_it_equals_a_list_of_the_same_numbers(self) -> None:
        assert Vector([1.0, 2.0]) == [1.0, 2.0]
        assert Vector([1.0, 2.0]) == (1.0, 2.0)
        assert Vector([1.0, 2.0]) == Vector([1.0, 2.0])

    def test_and_differs_from_one_that_is_not(self) -> None:
        assert Vector([1.0, 2.0]) != [1.0, 3.0]
        assert Vector([1.0, 2.0]) != [1.0]
        assert Vector([1.0]) != "one"

    def test_the_sequence_operations_all_work(self) -> None:
        v = Vector([1.0, 2.0, 3.0])
        assert len(v) == 3
        assert v[0] == 1.0
        assert v[-1] == 3.0
        assert list(v[0:2]) == [1.0, 2.0]
        assert list(v) == [1.0, 2.0, 3.0]
        assert 2.0 in v
        assert v.index(2.0) == 1
        assert v.count(2.0) == 1
        assert sum(v) == 6.0
        assert [x * 2 for x in v] == [2.0, 4.0, 6.0]

    def test_the_numbers_are_an_array_numpy_can_take_without_copying(self) -> None:
        v = Vector([1.0, 2.0])
        assert v.values.typecode == "f"
        assert len(memoryview(v.values)) == 2

    def test_it_is_a_value(self) -> None:
        assert hash(Vector([1.0])) == hash(Vector([1.0]))
        assert {Vector([1.0]): "seen"}[Vector([1.0])] == "seen"
        with pytest.raises(AttributeError):
            Vector([1.0]).values = array("f", [2.0])  # type: ignore[misc]

    def test_sending_back_what_was_read_reuses_the_bytes(self) -> None:
        """Nothing is unpacked to send a vector straight back."""
        raw = Vector([1.0, 2.0]).to_bytes()
        held = Vector.from_wire(raw)
        assert held.to_bytes() == raw
        assert held._values is None, "and it still has not unpacked"

    def test_a_text_reading_is_lazy_too(self) -> None:
        """Parsing is the expensive half of the text form, so it waits as well."""
        held = Vector.from_wire_text(b"[1,2,3]")
        assert held._values is None
        assert held == [1.0, 2.0, 3.0]
        assert held._values is not None

    def test_both_readings_of_a_vector_are_equal_to_each_other(self, vectors) -> None:  # type: ignore[no-untyped-def]
        rng = random.Random(9)
        values = [random_single(rng) for _ in range(8)]
        literal = "[" + ",".join(f"{v:.9g}" for v in values) + "]"
        text = vectors.execute("select %s::vector(8)", (literal,)).fetchone()[0]
        binary = vectors.execute("select %s::vector(8)", (literal,), binary=True).fetchone()[0]
        assert text == binary
        assert text == values
        assert text.tolist() == binary.tolist()


class TestTuningASearch:
    """The seven settings a vector search takes, and refusing anything that is not one of them."""

    def test_a_whole_number_setting(self) -> None:
        assert search_option_statements({"hnsw.ef_search": 100}) == [
            "set local hnsw.ef_search = 100"
        ]

    def test_a_real_setting(self) -> None:
        assert search_option_statements({"hnsw.scan_mem_multiplier": 2.5}) == [
            "set local hnsw.scan_mem_multiplier = 2.5"
        ]

    def test_an_enumerated_setting_is_quoted(self) -> None:
        assert search_option_statements({"hnsw.iterative_scan": "relaxed_order"}) == [
            "set local hnsw.iterative_scan = 'relaxed_order'"
        ]

    def test_it_is_local_to_the_transaction_unless_asked_otherwise(self) -> None:
        assert search_option_statements({"ivfflat.probes": 10}, local=False) == [
            "set ivfflat.probes = 10"
        ]

    def test_every_documented_setting_is_accepted(self) -> None:
        from agensgraph.vector import SEARCH_OPTION_VALUES, SEARCH_OPTIONS

        for name, kind in SEARCH_OPTIONS.items():
            value: object = 1 if kind is not str else SEARCH_OPTION_VALUES[name][0]
            assert search_option_statements({name: value})

    def test_and_the_server_accepts_every_one_of_them(self, vectors) -> None:  # type: ignore[no-untyped-def]
        """Which is the point of checking a word against a list rather than quoting it."""
        from agensgraph.vector import SEARCH_OPTION_VALUES, SEARCH_OPTIONS

        vectors.execute("create (:loose {v: [1,2,3,4]})")
        for name, kind in SEARCH_OPTIONS.items():
            for value in [1] if kind is not str else SEARCH_OPTION_VALUES[name]:
                vectors.vector_search_options({name: value})

    def test_a_name_that_is_not_one_of_them_is_refused(self) -> None:
        """The server takes an unknown ``hnsw.`` name without complaint, so a typo would look as
        though it had been applied."""
        with pytest.raises(ValueError, match="not a vector search option"):
            search_option_statements({"hnsw.ef_serach": 100})

    def test_a_setting_given_the_wrong_kind_of_value_is_refused(self) -> None:
        with pytest.raises(ValueError, match="takes a whole number"):
            search_option_statements({"hnsw.ef_search": "lots"})

    def test_the_server_accepts_them_and_a_search_still_answers(self, vectors) -> None:  # type: ignore[no-untyped-def]
        vectors.execute(vector_index("loose", "v", dimensions=4))
        vectors.execute("create (:loose {v: [1,2,3,4]})")
        vectors.vector_search_options(
            {"hnsw.ef_search": 100, "hnsw.iterative_scan": "relaxed_order"}
        )
        result = vectors.execute_query(
            nearest("loose", "v", dimensions=4, limit=1), ("[1,2,3,4]",)
        )
        assert len(result.records) == 1

    def test_an_unknown_name_never_reaches_the_server(self, vectors) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="not a vector search option"):
            vectors.vector_search_options({"hnsw.nonsense": 1})


class TestBinaryQuantisation:
    """Reachable from Cypher as a function, and no further: the cast it needs is not Cypher's."""

    def test_the_function_runs_and_returns_a_bit_string(self, vectors) -> None:  # type: ignore[no-untyped-def]
        vectors.execute("create (:loose {v: [1,-2,3,-4]})")
        (value,) = vectors.execute_query(
            "match (n:loose) return binary_quantize(n.v::vector(4))"
        ).records[0]
        assert value == "1010"

    def test_but_the_bit_cast_is_not_something_cypher_can_spell(self, vectors) -> None:  # type: ignore[no-untyped-def]
        """So the hamming and jaccard operators need plain SQL against the label's table."""
        vectors.execute("create (:loose {v: [1,2,3,4]})")
        with pytest.raises(agensgraph.errors.Error):
            vectors.execute("match (n:loose) return binary_quantize(n.v::vector(4))::bit(4)")

    def test_and_it_works_in_sql_on_the_same_data(self, vectors) -> None:  # type: ignore[no-untyped-def]
        graph = vectors.label_table.graph
        vectors.execute("create (:loose {v: [1,-2,3,-4]})")
        (distance,) = vectors.execute(
            f"select binary_quantize((properties->>%s)::vector(4))::bit(4) "
            f"<~> binary_quantize('[1,-2,3,-4]'::vector(4))::bit(4) "
            f'from "{graph}".loose limit 1',
            ("v",),
        ).fetchone()
        assert distance == 0.0


class TestNothingCallerSuppliedReachesTheStatementUnchecked:
    """A label or a setting taken from somebody else's input is what these guard against.

    A name is quoted, because a name is an identifier and the server takes a quoted one. A method
    and an operator class are identifiers too. Everything else -- an operator, a type, a setting
    that takes a word -- is not an identifier and cannot be quoted, so it is checked against the
    values that exist.
    """

    HOSTILE = "x'); drop graph y cascade --"

    def test_a_search_option_takes_only_what_the_server_defines(self) -> None:
        with pytest.raises(ValueError, match="takes one of"):
            search_option_statements({"hnsw.iterative_scan": self.HOSTILE})
        with pytest.raises(ValueError, match="whole number"):
            search_option_statements({"hnsw.ef_search": self.HOSTILE})
        assert search_option_statements({"hnsw.iterative_scan": "relaxed_order"}) == [
            "set local hnsw.iterative_scan = 'relaxed_order'"
        ]

    def test_an_index_quotes_its_method_and_operator_class(self) -> None:
        built = vector_index("l", "p", dimensions=4, method=self.HOSTILE)
        assert f'using "{self.HOSTILE}"' in built
        built = vector_index("l", "p", dimensions=4, operator_class=self.HOSTILE)
        assert f'"{self.HOSTILE}"' in built

    def test_an_index_option_is_quoted_by_what_it_is(self) -> None:
        built = vector_index("l", "p", dimensions=4, options={"lists": "1') ; drop --"})
        assert "lists='1'') ; drop --'" in built
        built = vector_index("l", "p", dimensions=4, options={self.HOSTILE: 1})
        assert f'"{self.HOSTILE}"=1' in built

    def test_a_search_takes_only_a_distance_operator(self) -> None:
        with pytest.raises(ValueError, match="not a distance operator"):
            nearest("l", "p", operator="<=> ); drop --")
        for known in Distance:
            assert str(known) in nearest("l", "p", operator=known)

    def test_a_type_is_checked_even_with_no_dimension_to_cast_to(self) -> None:
        """The check sat after the early return for a property with a column of its own."""
        with pytest.raises(ValueError, match="expected one of"):
            nearest("l", "p", type="vector); drop --")
        with pytest.raises(ValueError, match="expected one of"):
            nearest("l", "p", dimensions=4, type="vector); drop --")

    def test_a_generated_column_quotes_its_name(self) -> None:
        assert generated_column(self.HOSTILE, 4).startswith(f'"{self.HOSTILE}"')

    def test_an_ordinary_call_is_unchanged(self) -> None:
        assert vector_index("movie", "embedding", dimensions=4) == (
            "create property index on movie using hnsw "
            "((embedding::vector(4)) vector_cosine_ops)"
        )
        assert nearest("movie", "embedding", dimensions=4) == (
            "match (n:movie) return n order by n.embedding::vector(4) <=> "
            "%s::vector(4) limit 10"
        )
