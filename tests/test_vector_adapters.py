"""Sending every vector type, in every format psycopg asks for.

These pin that a ``COPY`` into a half or sparse column finds its dumper, a list of vectors can be
sent, a sparse vector goes into a SQL column of its own type without a cast, and a text ``COPY`` of
a plain ``text`` column keeps its strings away from the sparse vector's dumper once the vector
types are registered.
"""

from __future__ import annotations

import random
import struct
from array import array

import psycopg
import pytest
import pytest_asyncio

import agensgraph
from agensgraph.vector import Bits, SparseVector, Vector, parse_vector_values

DIMENSIONS = 4


def singles(count: int, seed: int) -> list[float]:
    """Numbers a single holds exactly, so a value read back can be compared with ``==``."""
    rng = random.Random(seed)
    return [struct.unpack(">f", struct.pack(">f", rng.uniform(-1, 1)))[0] for _ in range(count)]


def halves(values: list[float]) -> list[float]:
    return [struct.unpack(">e", struct.pack(">e", value))[0] for value in values]


class TestMakingAVector:
    """From whatever a caller has, copying a packed buffer rather than iterating it."""

    def test_from_a_list(self) -> None:
        assert Vector([1.0, 2.5]).tolist() == [1.0, 2.5]

    def test_from_an_array_of_singles_or_doubles(self) -> None:
        assert Vector(array("f", [1.0, 2.5])).tolist() == [1.0, 2.5]
        assert Vector(array("d", [1.0, 2.5])).tolist() == [1.0, 2.5]

    def test_from_numpy_singles_doubles_and_halves(self) -> None:
        np = pytest.importorskip("numpy")
        expected = singles(16, seed=1)
        for dtype in ("float32", "float64"):
            assert Vector(np.array(expected, dtype=dtype)).tolist() == expected
        assert Vector(np.array([1.0, 0.5], dtype="float16")).tolist() == [1.0, 0.5]

    def test_from_a_slice_that_is_not_contiguous(self) -> None:
        np = pytest.importorskip("numpy")
        values = np.arange(8, dtype="float32")
        assert Vector(values[::2]).tolist() == [0.0, 2.0, 4.0, 6.0]

    def test_from_another_byte_order(self) -> None:
        np = pytest.importorskip("numpy")
        assert Vector(np.array([1.0, 2.5], dtype=">f4")).tolist() == [1.0, 2.5]

    def test_a_buffer_of_more_than_one_dimension_is_refused(self) -> None:
        np = pytest.importorskip("numpy")
        with pytest.raises(ValueError, match="one dimension"):
            Vector(np.zeros((2, 2), dtype="float32"))

    def test_the_buffer_is_copied_not_kept(self) -> None:
        """A caller reusing its array for the next row must not change the one already made."""
        np = pytest.importorskip("numpy")
        values = np.array([1.0, 2.0], dtype="float32")
        made = Vector(values)
        values[0] = 9.0
        assert made.tolist() == [1.0, 2.0]

    def test_copying_the_buffer_beats_iterating_it(self) -> None:
        """A floor, not a benchmark: the copy has to be the cheaper route by a wide margin."""
        import time

        np = pytest.importorskip("numpy")
        values = np.random.default_rng(3).random(1536, dtype="float32")
        as_list = values.tolist()
        copied = iterated = 0.0
        for _ in range(5):
            started = time.perf_counter()
            for _ in range(200):
                Vector(values)
            copied += time.perf_counter() - started
            started = time.perf_counter()
            for _ in range(200):
                array("f", iter(as_list))
            iterated += time.perf_counter() - started
        assert copied * 3 < iterated


class TestTheWireForms:
    """Each form written here is read back by the loader the driver already had."""

    def test_the_text_form_reads_back_to_the_same_singles(self) -> None:
        from agensgraph.vector import _vector_text

        values = singles(1536, seed=2)
        assert parse_vector_values(_vector_text(Vector(values))).tolist() == values

    def test_the_half_form_reads_back_through_the_half_loader(self) -> None:
        from agensgraph.vector import HalfVectorBinaryLoader, _half_bytes

        values = singles(8, seed=3)
        loaded = HalfVectorBinaryLoader(0).load(_half_bytes(Vector(values)))
        assert loaded.tolist() == halves(values)

    def test_the_sparse_form_reads_back_through_the_sparse_loader(self) -> None:
        from agensgraph.vector import SparseVectorBinaryLoader, _sparse_bytes

        sparse = SparseVector({0: 1.5, 3: -2.0, 9: 0.25}, 10)
        assert SparseVectorBinaryLoader(0).load(_sparse_bytes(sparse)) == sparse

    def test_an_empty_sparse_vector_has_a_form_too(self) -> None:
        from agensgraph.vector import SparseVectorBinaryLoader, _sparse_bytes

        sparse = SparseVector({}, 3)
        assert SparseVectorBinaryLoader(0).load(_sparse_bytes(sparse)) == sparse


def copy_in(conn, table: str, type: str, rows) -> None:  # type: ignore[no-untyped-def]
    """A binary ``COPY`` told its column types, which is what finds a dumper by the type's oid."""
    with (
        conn.cursor() as cursor,
        cursor.copy(f"copy {table} from stdin (format binary)") as copy,
    ):
        copy.set_types(["graphid", type])
        for row in rows:
            copy.write_row(row)


@pytest.fixture
def side(agens):  # type: ignore[no-untyped-def]
    """A label, some of its vertices, and a temporary table of each vector type keyed by them."""
    if not agens.has_vectors():
        pytest.skip("the vector extension is not created in this database")
    agens.register_vectors()
    agens.execute("create vlabel doc")
    ids = agens.load_vertices_with_ids("doc", [{"n": n} for n in range(3)])
    agens.execute(
        f"create temp table dense (id graphid primary key, embedding vector({DIMENSIONS}))"
    )
    agens.execute(
        f"create temp table half (id graphid primary key, embedding halfvec({DIMENSIONS}))"
    )
    agens.execute("create temp table sparse (id graphid primary key, embedding sparsevec(10))")
    agens.ids = ids
    return agens


@pytest.mark.server
class TestCopyingEachType:
    """Each type's binary dumper, found by its oid alone, in a ``COPY`` told its column types."""

    def test_dense_vectors_go_in_and_read_back_in_both_renderings(self, side) -> None:  # type: ignore[no-untyped-def]
        rows = [
            (identity, Vector(singles(DIMENSIONS, seed=n)))
            for n, identity in enumerate(side.ids)
        ]
        copy_in(side, "dense", "vector", rows)
        query = "select id, embedding from dense order by id"
        for binary in (False, True):
            read = side.execute(query, binary=binary).fetchall()
            assert [(i, v.tolist()) for i, v in read] == [(i, v.tolist()) for i, v in rows]

    def test_half_vectors_go_in(self, side) -> None:  # type: ignore[no-untyped-def]
        rows = [
            (identity, Vector(singles(DIMENSIONS, seed=n)))
            for n, identity in enumerate(side.ids)
        ]
        copy_in(side, "half", "halfvec", rows)
        read = side.execute("select embedding from half order by id", binary=True).fetchall()
        assert [v.tolist() for (v,) in read] == [halves(v.tolist()) for _, v in rows]

    def test_sparse_vectors_go_in(self, side) -> None:  # type: ignore[no-untyped-def]
        rows = [
            (identity, SparseVector({n: 1.5, 9: -1.0}, 10))
            for n, identity in enumerate(side.ids)
        ]
        copy_in(side, "sparse", "sparsevec", rows)
        read = side.execute("select embedding from sparse order by id").fetchall()
        assert [v for (v,) in read] == [value for _, value in rows]

    def test_a_wrong_length_is_the_server_s_refusal(self, side) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(psycopg.Error) as caught:
            copy_in(side, "dense", "vector", [(side.ids[0], Vector([1.0, 2.0]))])
        assert caught.value.sqlstate == "22000"


@pytest.mark.server
class TestSendingThemAsParameters:
    def test_a_dense_vector_in_text_with_t(self, side) -> None:  # type: ignore[no-untyped-def]
        (read,) = side.execute(
            f"select %t::vector({DIMENSIONS})", (Vector([1.0, 2.5, 3, 4]),)
        ).fetchone()
        assert read.tolist() == [1.0, 2.5, 3.0, 4.0]

    def test_a_list_of_vectors(self, side) -> None:  # type: ignore[no-untyped-def]
        sent = [Vector([1.0, 2.0]), Vector([3.0, 4.5])]
        read = side.execute("select unnest(%s::vector[])", (sent,)).fetchall()
        assert [v.tolist() for (v,) in read] == [[1.0, 2.0], [3.0, 4.5]]

    def test_a_sparse_vector_into_its_own_sql_column_with_b(self, side) -> None:  # type: ignore[no-untyped-def]
        sparse = SparseVector({2: 7.0}, 10)
        side.execute("insert into sparse values (%s, %b)", (side.ids[0], sparse))
        assert side.execute("select embedding from sparse").fetchone() == (sparse,)

    def test_and_as_text_into_a_cypher_property(self, side) -> None:  # type: ignore[no-untyped-def]
        sparse = SparseVector({2: 7.0}, 10)
        side.execute("create (:doc {s: %s})", (sparse,))
        (read,) = side.execute_query("match (n:doc) where n.s is not null return n.s").records[
            0
        ]
        assert read == sparse.to_text()


@pytest.mark.server
class TestRegisteringTheTypes:
    def test_one_statement_finds_every_type(self, side, monkeypatch) -> None:  # type: ignore[no-untyped-def]
        """psycopg's own lookup is one round trip per type."""

        def refuse(*args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("looked up one type at a time")

        monkeypatch.setattr(psycopg.types.TypeInfo, "fetch", refuse)
        assert side.register_vectors() == ("vector", "halfvec", "sparsevec")

    def test_a_text_copy_of_plain_text_works_after_registering(self, side) -> None:  # type: ignore[no-untyped-def]
        side.register_vectors()
        side.execute("create temp table words (w text)")
        with side.cursor() as cursor, cursor.copy("copy words (w) from stdin") as copy:
            copy.set_types(["text"])
            copy.write_row(["hello"])
        assert side.execute("select w from words").fetchall() == [("hello",)]

    def test_a_text_copy_of_vectors_works(self, side) -> None:  # type: ignore[no-untyped-def]
        side.register_vectors()
        with (
            side.cursor() as cursor,
            cursor.copy("copy dense (id, embedding) from stdin") as copy,
        ):
            copy.set_types(["graphid", "vector"])
            copy.write_row([side.ids[0], Vector([1.0, 2.0, 3.0, 4.5])])
        (read,) = side.execute("select embedding from dense").fetchone()
        assert read.tolist() == [1.0, 2.0, 3.0, 4.5]


class TestBits:
    """A bit string: what binary quantisation produces and HAMMING and JACCARD measure."""

    def test_it_is_the_text_the_server_prints(self) -> None:
        assert Bits("1010") == "1010"
        assert Bits([True, False, True, False]) == "1010"
        assert isinstance(Bits("1"), str)

    def test_anything_but_zeros_and_ones_is_refused(self) -> None:
        with pytest.raises(ValueError, match="only 0 and 1"):
            Bits("10a1")

    @pytest.mark.parametrize("text", ["", "1", "10110010", "101100101", "1" * 64 + "0"])
    def test_the_packed_form_reads_back(self, text: str) -> None:
        bits = Bits(text)
        assert Bits.from_bytes(bits.to_bytes(), len(bits)) == bits

    def test_the_packed_form_is_the_server_s(self) -> None:
        """Most significant first, the last byte padded: B'101100101' is b2 80."""
        assert Bits("101100101").to_bytes() == b"\xb2\x80"

    @pytest.mark.server
    def test_it_goes_both_ways_and_is_measured_without_a_cast(self, side) -> None:  # type: ignore[no-untyped-def]
        (read,) = side.execute("select B'101100101'::bit(9)", binary=True).fetchone()
        assert read == Bits("101100101")
        assert isinstance(read, Bits)
        (text_read,) = side.execute("select B'1010'::bit(4)").fetchone()
        assert isinstance(text_read, Bits)
        (distance,) = side.execute(
            "select binary_quantize('[1,-2,3,-4]'::vector(4))::bit(4) <~> %s", (Bits("1001"),)
        ).fetchone()
        assert distance == 2.0

    @pytest.mark.server
    def test_a_connection_that_did_not_register_reads_bits_as_text(self, dsn: str) -> None:
        with agensgraph.connect(dsn, autocommit=True) as conn:
            (read,) = conn.execute("select B'1010'::bit(4)").fetchone()
        assert read == "1010"
        assert not isinstance(read, Bits)


class TestTheAwaitingInterface:
    @pytest_asyncio.fixture
    async def conn(self, dsn: str):  # type: ignore[no-untyped-def]
        name = "vector_adapters_async"
        connection = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
        async with connection:
            if not await connection.has_vectors():
                pytest.skip("the vector extension is not created in this database")
            await connection.execute(f'drop graph if exists "{name}" cascade')
            await connection.execute(f'create graph "{name}"')
            await connection.graph(name)
            try:
                yield connection
            finally:
                await connection.execute("reset graph_path")
                await connection.execute(f'drop graph "{name}" cascade')

    @pytest.mark.server
    @pytest.mark.asyncio
    async def test_copying_vectors(self, conn) -> None:  # type: ignore[no-untyped-def]
        await conn.register_vectors()
        await conn.execute("create vlabel doc")
        ids = await conn.load_vertices_with_ids("doc", [{"n": 1}, {"n": 2}])
        await conn.execute("create temp table t (id graphid, embedding halfvec(2))")
        async with (
            conn.cursor() as cursor,
            cursor.copy("copy t from stdin (format binary)") as copy,
        ):
            copy.set_types(["graphid", "halfvec"])
            for identity in ids:
                await copy.write_row([identity, Vector([1.0, 2.0])])
        rows = await (await conn.execute("select count(*) from t")).fetchone()
        assert rows == (2,)
