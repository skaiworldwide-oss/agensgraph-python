"""Reading and indexing embedding vectors.

Skipped where the extension is not created, since it is created per database rather than per
server. The two things worth pinning are that promotion changes what a vector property reads as --
which is the whole reason this module exists -- and that the dimension in a cast is the difference
between an index that exists and one the server refuses.
"""

from __future__ import annotations

import random
import struct

import pytest
import pytest_asyncio

import agensgraph
from agensgraph.vector import (
    TYPES,
    SparseVector,
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

    def test_a_sparse_column_can_be_asked_for(self) -> None:
        assert generated_column("v", 6, type="sparsevec") == "v sparsevec(6) generated"

    @pytest.mark.parametrize("unknown", ["bit", "float8", "vector4", ""])
    def test_a_type_this_driver_cannot_read_is_refused(self, unknown: str) -> None:
        with pytest.raises(ValueError, match="expected one of"):
            generated_column("v", 4, type=unknown)

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
