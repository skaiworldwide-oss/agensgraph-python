"""Handing a result to something columnar, and loading one back.

Each backend is skipped where it is not installed, since none of them is a dependency. What is
asserted is the shape and the type of every column a graph result can produce, that a chunked export
carries one schema throughout, and that the copy stream built for the other direction is the format
the server reads.
"""

from __future__ import annotations

import struct
import subprocess
import sys
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID

import pytest

import agensgraph
from agensgraph import bulk
from agensgraph.columnar import (
    Layout,
    Plan,
    batches,
    columns,
    edge_payloads,
    reader,
    to_arrow,
    to_pandas,
    to_polars,
    vertex_payloads,
)
from agensgraph.vector import SparseVector, Vector

pyarrow = pytest.importorskip("pyarrow", reason="pyarrow is not installed")
pandas = pytest.importorskip("pandas", reason="pandas is not installed")
polars = pytest.importorskip("polars", reason="polars is not installed")

JSONB_OID = 3802
GRAPHID_OID = 7002
VERTEX_OID = 7012
EDGE_OID = 7022
GRAPHPATH_OID = 7032


def vertex(locid: int = 1, **properties: Any) -> agensgraph.Vertex:
    return agensgraph.Vertex(agensgraph.GraphId(3, locid), "person", properties)


def raw_vertex(locid: int, payload: bytes) -> agensgraph.Vertex:
    """A vertex holding its property map as the bytes it arrived in, undecoded."""
    return agensgraph.Vertex(agensgraph.GraphId(3, locid), "person", payload)


def edge(locid: int = 1, start: int = 1, end: int = 2) -> agensgraph.Edge:
    return agensgraph.Edge(
        agensgraph.GraphId(4, locid),
        "knows",
        agensgraph.GraphId(3, start),
        agensgraph.GraphId(3, end),
        {"since": 2001},
    )


def wire_vector(*numbers: float) -> Vector:
    """A vector holding the bytes the wire carries: a header, then big-endian singles."""
    packed = struct.pack(">HH", len(numbers), 0) + struct.pack(f">{len(numbers)}f", *numbers)
    return Vector.from_wire(packed)


class TestTurningRowsIntoColumns:
    def test_a_plain_result(self) -> None:
        assert columns([(1, "a"), (2, "b")], ["n", "s"]) == {"n": [1, 2], "s": ["a", "b"]}

    def test_no_rows_still_gives_the_columns(self) -> None:
        assert columns([], ["n", "s"]) == {"n": [], "s": []}

    def test_no_columns_at_all(self) -> None:
        assert columns([], []) == {}

    def test_a_row_of_the_wrong_width_is_refused(self) -> None:
        with pytest.raises(ValueError, match="against 2 column names"):
            columns([(1,)], ["a", "b"])

    def test_rows_of_differing_widths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="widths against"):
            columns([(1, 2), (1,)], ["a", "b"])

    def test_an_identity_becomes_its_text_form(self) -> None:
        assert columns([(agensgraph.GraphId(3, 1),)], ["id"]) == {"id": ["3.1"]}

    def test_a_vector_becomes_a_list_of_numbers(self) -> None:
        assert columns([(Vector([1.0, 2.0]),)], ["v"]) == {"v": [[1.0, 2.0]]}

    def test_a_sparse_vector_becomes_its_text_form(self) -> None:
        assert columns([(SparseVector({0: 1.0}, 4),)], ["v"]) == {"v": ["{1:1}/4"]}

    def test_a_property_map_is_carried_through(self) -> None:
        assert columns([({"a": 1},)], ["m"]) == {"m": [{"a": 1}]}

    def test_a_null_is_carried_through(self) -> None:
        assert columns([(None,)], ["x"]) == {"x": [None]}

    def test_a_vertex_becomes_a_mapping_of_its_parts(self) -> None:
        held = columns([(vertex(1, name="a"),)], ["n"])
        assert held == {"n": [{"id": "3.1", "label": "person", "properties": {"name": "a"}}]}

    def test_an_edge_carries_the_identities_it_joins(self) -> None:
        held = columns([(edge(),)], ["e"])["e"][0]
        assert held["start"] == "3.1"
        assert held["end"] == "3.2"

    def test_a_path_becomes_its_vertices_and_edges(self) -> None:
        path = agensgraph.Path((vertex(1), vertex(2)), (edge(),))
        held = columns([(path,)], ["p"])["p"][0]
        assert [each["id"] for each in held["vertices"]] == ["3.1", "3.2"]
        assert len(held["edges"]) == 1


class TestTheColumnTypes:
    def test_scalars(self) -> None:
        table = to_arrow([(1, 2.5, "a", True, None)], ["i", "f", "s", "b", "n"])
        assert [field.type for field in table.schema] == [
            pyarrow.int64(),
            pyarrow.float64(),
            pyarrow.utf8(),
            pyarrow.bool_(),
            pyarrow.null(),
        ]

    def test_an_oid_settles_the_width_a_value_cannot(self) -> None:
        table = to_arrow([(1, 1)], ["small", "big"], oids=[23, 20])
        assert table.schema.field("small").type == pyarrow.int32()
        assert table.schema.field("big").type == pyarrow.int64()

    def test_a_uuid_is_written_out(self) -> None:
        value = UUID("550e8400-e29b-41d4-a716-446655440000")
        table = to_arrow([(value,)], ["u"], oids=[2950])
        assert table.column("u").to_pylist() == [str(value)]

    def test_the_types_psycopg_hands_over_for_a_date_and_a_decimal(self) -> None:
        rows = [(date(2026, 8, 5), datetime(2026, 8, 5, tzinfo=UTC), Decimal("1.500"))]
        table = to_arrow(rows, ["d", "t", "n"], oids=[1082, 1184, 1700])
        assert table.schema.field("d").type == pyarrow.date32()
        assert table.schema.field("t").type == pyarrow.timestamp("us", "UTC")
        assert table.column("n").to_pylist() == [Decimal("1.500")]

    def test_an_integer_too_large_for_arrow_becomes_its_digits(self) -> None:
        table = to_arrow([(10**30,), (2,)], ["big"], oids=[JSONB_OID])
        assert table.schema.field("big").type == pyarrow.utf8()
        assert table.column("big").to_pylist() == ["1" + "0" * 30, "2"]

    def test_a_column_of_integers_and_floats_holds_both(self) -> None:
        table = to_arrow([(1,), (2.5,)], ["mixed"], oids=[JSONB_OID])
        assert table.column("mixed").to_pylist() == [1.0, 2.5]

    def test_a_map_becomes_json_text(self) -> None:
        table = to_arrow([({"a": 1},)], ["m"], oids=[JSONB_OID])
        assert table.schema.field("m").type == pyarrow.utf8()
        assert table.column("m").to_pylist() == ['{"a":1}']

    def test_a_list_of_maps_becomes_json_text(self) -> None:
        table = to_arrow([([{"a": 1}],)], ["m"], oids=[JSONB_OID])
        assert table.column("m").to_pylist() == ['[{"a":1}]']

    def test_a_list_of_numbers_becomes_a_list_column(self) -> None:
        table = to_arrow([([1, 2],)], ["l"], oids=[JSONB_OID])
        assert table.schema.field("l").type == pyarrow.list_(pyarrow.int64())

    def test_bytes_become_binary(self) -> None:
        table = to_arrow([(b"\x01\x02",)], ["b"], oids=[17])
        assert table.column("b").to_pylist() == [b"\x01\x02"]


class TestIdentities:
    def test_the_packed_value_is_what_a_graphid_holds(self) -> None:
        table = to_arrow([(agensgraph.GraphId(3, 1),)], ["id"])
        assert table.schema.field("id").type == pyarrow.uint64()
        assert table.column("id").to_pylist() == [(3 << 48) | 1]

    def test_the_label_id_is_the_top_sixteen_bits(self) -> None:
        packed = to_arrow([(agensgraph.GraphId(7, 9),)], ["id"]).column("id").to_pylist()[0]
        assert (packed >> 48, packed & ((1 << 48) - 1)) == (7, 9)

    def test_the_text_form_is_asked_for(self) -> None:
        table = to_arrow([(agensgraph.GraphId(3, 1),)], ["id"], layout=Layout(identity="text"))
        assert table.column("id").to_pylist() == ["3.1"]

    def test_a_null_identity_stays_null(self) -> None:
        table = to_arrow([(agensgraph.GraphId(3, 1),), (None,)], ["id"], oids=[GRAPHID_OID])
        assert table.column("id").to_pylist() == [(3 << 48) | 1, None]

    def test_an_array_of_identities(self) -> None:
        rows = [([agensgraph.GraphId(3, 1), agensgraph.GraphId(3, 2)],)]
        table = to_arrow(rows, ["ids"], oids=[7001])
        assert table.schema.field("ids").type == pyarrow.list_(pyarrow.uint64())
        assert table.column("ids").to_pylist() == [[(3 << 48) | 1, (3 << 48) | 2]]


class TestGraphElements:
    def test_a_vertex_is_a_struct_of_its_parts(self) -> None:
        table = to_arrow([(vertex(1, name="a"),)], ["n"])
        kind = table.schema.field("n").type
        assert [field.name for field in kind] == ["id", "label", "properties"]
        assert table.column("n").to_pylist() == [
            {"id": (3 << 48) | 1, "label": "person", "properties": '{"name":"a"}'}
        ]

    def test_the_property_map_is_the_bytes_it_arrived_in(self) -> None:
        payload = b'{"name": "a", "age": 30}'
        table = to_arrow([(raw_vertex(1, payload),)], ["n"])
        assert table.column("n").to_pylist()[0]["properties"] == payload.decode()

    def test_a_label_is_dictionary_encoded(self) -> None:
        table = to_arrow([(vertex(1),), (vertex(2),)], ["n"])
        labels = table.schema.field("n").type.field("label")
        assert labels.type == pyarrow.dictionary(pyarrow.int32(), pyarrow.utf8())

    def test_a_label_as_plain_text_is_asked_for(self) -> None:
        table = to_arrow([(vertex(1),)], ["n"], layout=Layout(labels="text"))
        assert table.schema.field("n").type.field("label").type == pyarrow.utf8()

    def test_the_parts_can_be_columns_of_their_own(self) -> None:
        table = to_arrow([(vertex(1, name="a"),)], ["n"], layout=Layout(elements="columns"))
        assert table.column_names == ["n.id", "n.label", "n.properties"]

    def test_the_properties_can_be_left_out(self) -> None:
        table = to_arrow([(vertex(1, name="a"),)], ["n"], layout=Layout(properties="skip"))
        assert [field.name for field in table.schema.field("n").type] == ["id", "label"]

    def test_the_properties_can_be_a_struct_the_caller_names(self) -> None:
        wanted = pyarrow.struct([("name", pyarrow.utf8()), ("age", pyarrow.int32())])
        table = to_arrow(
            [(vertex(1, name="a", age=30),)], ["n"], layout=Layout(properties=wanted)
        )
        held = table.column("n").to_pylist()[0]["properties"]
        assert held == {"name": "a", "age": 30}

    def test_an_edge_carries_the_identities_it_joins(self) -> None:
        table = to_arrow([(edge(1, 1, 2),)], ["e"])
        assert [field.name for field in table.schema.field("e").type] == [
            "id",
            "label",
            "start",
            "end",
            "properties",
        ]
        held = table.column("e").to_pylist()[0]
        assert held["start"] == (3 << 48) | 1
        assert held["end"] == (3 << 48) | 2

    def test_an_edge_joins_a_vertex_on_the_same_type(self) -> None:
        vertices = to_arrow([(vertex(1),)], ["n"])
        edges = to_arrow([(edge(1, 1, 2),)], ["e"])
        assert (
            vertices.schema.field("n").type.field("id").type
            == edges.schema.field("e").type.field("start").type
        )

    def test_a_path_is_a_struct_of_two_lists(self) -> None:
        path = agensgraph.Path((vertex(1), vertex(2)), (edge(),))
        table = to_arrow([(path,)], ["p"])
        kind = table.schema.field("p").type
        assert [field.name for field in kind] == ["vertices", "edges"]
        held = table.column("p").to_pylist()[0]
        assert len(held["vertices"]) == 2
        assert len(held["edges"]) == 1

    def test_a_path_can_be_two_columns(self) -> None:
        path = agensgraph.Path((vertex(1),), ())
        table = to_arrow([(path,)], ["p"], layout=Layout(elements="columns"))
        assert table.column_names == ["p.vertices", "p.edges"]
        assert table.column("p.edges").to_pylist() == [[]]

    def test_an_array_of_vertices_is_a_list_of_structs(self) -> None:
        table = to_arrow([([vertex(1), vertex(2)],)], ["ns"], oids=[7011])
        assert table.column("ns").to_pylist()[0][0]["id"] == (3 << 48) | 1
        assert len(table.column("ns").to_pylist()[0]) == 2

    def test_a_null_element_is_a_null_struct(self) -> None:
        table = to_arrow([(vertex(1),), (None,)], ["n"], oids=[VERTEX_OID])
        assert table.column("n").to_pylist()[1] is None

    def test_an_element_built_from_a_dict_is_written_back_out(self) -> None:
        table = to_arrow([(vertex(1, name="a"),)], ["n"])
        assert table.column("n").to_pylist()[0]["properties"] == '{"name":"a"}'


class TestVectors:
    def test_a_vector_is_a_fixed_size_list_of_singles(self) -> None:
        table = to_arrow([(wire_vector(1.5, -2.25, 3.0),)], ["v"])
        assert table.schema.field("v").type == pyarrow.list_(pyarrow.float32(), 3)
        assert table.column("v").to_pylist() == [[1.5, -2.25, 3.0]]

    def test_the_numbers_are_the_ones_the_vector_holds(self) -> None:
        held = wire_vector(0.1, 0.2, 0.3)
        table = to_arrow([(held,)], ["v"])
        assert table.column("v").to_pylist()[0] == held.tolist()

    def test_a_vector_built_from_numbers_reads_the_same(self) -> None:
        table = to_arrow([(Vector([1.5, -2.25, 3.0]),)], ["v"])
        assert table.column("v").to_pylist() == [[1.5, -2.25, 3.0]]

    def test_reading_the_numbers_first_does_not_change_the_answer(self) -> None:
        warm = wire_vector(1.5, -2.25, 3.0)
        assert warm.tolist() == [1.5, -2.25, 3.0]
        assert to_arrow([(warm,)], ["v"]).column("v").to_pylist() == [[1.5, -2.25, 3.0]]

    def test_a_null_vector_keeps_the_width_of_the_column(self) -> None:
        table = to_arrow([(wire_vector(1.0, 2.0),), (None,)], ["v"])
        assert table.schema.field("v").type == pyarrow.list_(pyarrow.float32(), 2)
        assert table.column("v").to_pylist() == [[1.0, 2.0], None]

    def test_vectors_of_two_widths_are_refused_and_say_what_to_pass(self) -> None:
        rows = [(wire_vector(1.0, 2.0),), (wire_vector(1.0, 2.0, 3.0),)]
        with pytest.raises(ValueError, match="Layout\\(vectors='list'\\)"):
            to_arrow(rows, ["v"])

    def test_a_list_column_holds_vectors_of_two_widths(self) -> None:
        rows = [
            (wire_vector(1.0, 2.0),),
            (
                wire_vector(
                    3.0,
                ),
            ),
        ]
        table = to_arrow(rows, ["v"], layout=Layout(vectors="list"))
        assert table.schema.field("v").type == pyarrow.list_(pyarrow.float32())
        assert table.column("v").to_pylist() == [[1.0, 2.0], [3.0]]

    def test_a_null_in_a_list_column(self) -> None:
        rows = [(wire_vector(1.0, 2.0),), (None,)]
        table = to_arrow(rows, ["v"], layout=Layout(vectors="list"))
        assert table.column("v").to_pylist() == [[1.0, 2.0], None]

    def test_a_column_holds_four_bytes_a_number(self) -> None:
        table = to_arrow([(wire_vector(*[1.0] * 8),)], ["v"])
        assert table.nbytes == 32

    def test_a_vector_of_no_dimensions_says_what_to_pass(self) -> None:
        with pytest.raises(ValueError, match="no dimensions"):
            to_arrow([(wire_vector(),)], ["v"])

    def test_a_vector_of_no_dimensions_in_a_list_column(self) -> None:
        table = to_arrow([(wire_vector(),)], ["v"], layout=Layout(vectors="list"))
        assert table.column("v").to_pylist() == [[]]

    def test_a_sparse_vector_is_a_struct_of_its_parts(self) -> None:
        table = to_arrow([(SparseVector({0: 1.5, 3: 2.5}, 8),)], ["v"])
        assert table.column("v").to_pylist() == [
            {"dimensions": 8, "indices": [0, 3], "values": [1.5, 2.5]}
        ]

    def test_a_sparse_vector_as_text(self) -> None:
        table = to_arrow([(SparseVector({0: 1.0}, 4),)], ["v"], layout=Layout(sparse="text"))
        assert table.column("v").to_pylist() == ["{1:1}/4"]

    def test_a_sparse_vector_expanded(self) -> None:
        table = to_arrow([(SparseVector({0: 1.0}, 4),)], ["v"], layout=Layout(sparse="dense"))
        assert table.column("v").to_pylist() == [[1.0, 0.0, 0.0, 0.0]]


class TestNullsAndEmptyResults:
    def test_no_rows_and_no_types_gives_null_columns(self) -> None:
        table = to_arrow([], ["a"])
        assert table.num_rows == 0
        assert table.schema.field("a").type == pyarrow.null()

    def test_no_rows_keeps_the_type_an_oid_settles(self) -> None:
        table = to_arrow([], ["n", "i"], oids=[VERTEX_OID, 23])
        assert [field.name for field in table.schema.field("n").type] == [
            "id",
            "label",
            "properties",
        ]
        assert table.schema.field("i").type == pyarrow.int32()

    def test_no_rows_keeps_a_declared_schema(self) -> None:
        wanted = pyarrow.schema(
            [("name", pyarrow.utf8()), ("v", pyarrow.list_(pyarrow.float32(), 384))]
        )
        table = to_arrow([], ["name", "v"], schema=wanted)
        assert table.schema == wanted
        assert table.num_rows == 0

    def test_no_columns_at_all(self) -> None:
        assert to_arrow([], []).num_rows == 0

    def test_a_column_of_nothing_but_nulls(self) -> None:
        table = to_arrow([(None,), (None,)], ["x"])
        assert table.column("x").to_pylist() == [None, None]


class TestADeclaredSchema:
    def test_a_declared_type_is_what_the_column_is_built_as(self) -> None:
        wanted = pyarrow.schema([("i", pyarrow.int32())])
        table = to_arrow([(1,)], ["i"], schema=wanted)
        assert table.schema == wanted

    def test_a_declared_width_settles_an_empty_vector_column(self) -> None:
        wanted = pyarrow.schema([("v", pyarrow.list_(pyarrow.float32(), 4))])
        table = to_arrow([], ["v"], schema=wanted)
        assert table.schema.field("v").type.list_size == 4

    def test_a_schema_naming_something_else_is_refused(self) -> None:
        wanted = pyarrow.schema([("other", pyarrow.int32())])
        with pytest.raises(ValueError, match="no column for"):
            to_arrow([(1,)], ["i"], schema=wanted)

    def test_a_schema_disagreeing_with_a_built_column_is_refused(self) -> None:
        wanted = pyarrow.schema([("n", pyarrow.utf8())])
        with pytest.raises(ValueError, match="is built as"):
            to_arrow([(vertex(1),)], ["n"], schema=wanted)


class TestChunking:
    def test_a_batch_per_chunk(self) -> None:
        rows = [(i,) for i in range(10)]
        built = list(batches(rows, ["i"], size=4))
        assert [batch.num_rows for batch in built] == [4, 4, 2]

    def test_every_chunk_carries_the_same_schema(self) -> None:
        rows = [(i,) for i in range(10)]
        built = list(batches(rows, ["i"], size=4))
        assert len({batch.schema for batch in built} - set()) == 1

    def test_the_chunks_make_one_table(self) -> None:
        rows = [(i,) for i in range(10)]
        table = to_arrow(rows, ["i"], size=4)
        assert table.column("i").to_pylist() == list(range(10))

    def test_a_chunk_that_does_not_fit_the_first_is_reported(self) -> None:
        rows = [(1,), (2,), (2.5,), (3,)]
        stream = batches(rows, ["i"], size=2, oids=[JSONB_OID])
        assert next(stream).num_rows == 2
        with pytest.raises(ValueError, match="pass a schema to settle it"):
            next(stream)

    def test_a_first_chunk_of_nothing_but_nulls_is_reported(self) -> None:
        rows = [(None,), (None,), (1,), (2,)]
        with pytest.raises(ValueError, match="held null in the first chunk"):
            to_arrow(rows, ["x"], size=2)

    def test_a_later_chunk_of_nothing_but_nulls_is_fine(self) -> None:
        rows = [(1,), (2,), (None,), (None,)]
        assert to_arrow(rows, ["x"], size=2).column("x").to_pylist() == [1, 2, None, None]

    def test_a_declared_schema_holds_across_chunks(self) -> None:
        rows = [(1,), (2,), (3,), (4,)]
        wanted = pyarrow.schema([("i", pyarrow.float64())])
        table = to_arrow(rows, ["i"], size=2, schema=wanted)
        assert table.column("i").to_pylist() == [1.0, 2.0, 3.0, 4.0]

    def test_an_iterator_of_rows_is_read_a_chunk_at_a_time(self) -> None:
        table = to_arrow(iter([(i,) for i in range(5)]), ["i"], size=2)
        assert table.num_rows == 5

    def test_an_empty_iterator_still_carries_a_schema(self) -> None:
        table = to_arrow(iter([]), ["i"], oids=[23])
        assert table.num_rows == 0
        assert table.schema.field("i").type == pyarrow.int32()

    def test_a_reader_declares_its_schema_before_it_is_read(self) -> None:
        rows = [(wire_vector(1.0, 2.0),) for _ in range(5)]
        pulled = reader(rows, ["v"], size=2)
        assert pulled.schema.field("v").type == pyarrow.list_(pyarrow.float32(), 2)
        assert pulled.read_all().num_rows == 5

    def test_a_plan_has_no_schema_until_a_chunk_settles_it(self) -> None:
        plan = Plan(["i"])
        with pytest.raises(ValueError, match="settled by the first chunk"):
            _ = plan.schema
        plan.batch([(1,)])
        assert plan.schema.field("i").type == pyarrow.int64()


class TestTheBackends:
    def test_arrow(self) -> None:
        table = to_arrow([(1, "a"), (2, "b")], ["n", "s"])
        assert table.column_names == ["n", "s"]
        assert table.num_rows == 2
        assert table.column("n").to_pylist() == [1, 2]

    def test_pandas(self) -> None:
        frame = to_pandas([(1, "a"), (2, "b")], ["n", "s"])
        assert list(frame.columns) == ["n", "s"]
        assert frame["n"].tolist() == [1, 2]

    def test_pandas_is_arrow_backed(self) -> None:
        frame = to_pandas([(1,)], ["n"])
        assert isinstance(frame["n"].dtype, pandas.ArrowDtype)

    def test_pandas_can_be_asked_for_its_own_dtypes(self) -> None:
        frame = to_pandas([(1,)], ["n"], dtypes="numpy")
        assert not isinstance(frame["n"].dtype, pandas.ArrowDtype)

    def test_pandas_keeps_a_vector_column_in_arrow(self) -> None:
        frame = to_pandas([(wire_vector(1.0, 2.0),)], ["v"])
        assert isinstance(frame["v"].dtype, pandas.ArrowDtype)

    def test_polars(self) -> None:
        frame = to_polars([(1, "a"), (2, "b")], ["n", "s"])
        assert frame.columns == ["n", "s"]
        assert frame["n"].to_list() == [1, 2]

    def test_polars_reads_a_vector_column_as_an_array(self) -> None:
        frame = to_polars([(wire_vector(1.0, 2.0),)], ["v"])
        assert frame.schema["v"] == polars.Array(polars.Float32, 2)

    def test_polars_reads_a_vertex_column_as_a_struct(self) -> None:
        frame = to_polars([(vertex(1, name="a"),)], ["n"])
        assert set(frame.schema["n"].to_schema()) == {"id", "label", "properties"}

    def test_an_empty_result_makes_an_empty_table(self) -> None:
        assert to_arrow([], ["n"]).num_rows == 0
        assert len(to_pandas([], ["n"])) == 0
        assert to_polars([], ["n"]).height == 0


class TestWhatAResultCarries:
    def test_a_result_says_what_its_columns_are_called(self) -> None:
        result = agensgraph.Result([(1,)], ["i"], agensgraph.GraphWriteCounts.unknown(), (23,))
        table = to_arrow(result)
        assert table.column_names == ["i"]
        assert table.schema.field("i").type == pyarrow.int32()

    def test_rows_on_their_own_need_their_names(self) -> None:
        with pytest.raises(TypeError, match="do not say what the columns are called"):
            to_arrow([(1,)])


class TestTheCopyStream:
    def read_stream(self, blocks: list[bytes]) -> list[list[bytes | None]]:
        """Take a binary copy stream apart again, so the framing is checked and not assumed."""
        data = b"".join(blocks)
        assert data.startswith(b"PGCOPY\n\xff\r\n\0" + b"\0" * 8)
        at = 19
        rows: list[list[bytes | None]] = []
        while True:
            (count,) = struct.unpack_from("!h", data, at)
            at += 2
            if count == -1:
                break
            row: list[bytes | None] = []
            for _ in range(count):
                (length,) = struct.unpack_from("!i", data, at)
                at += 4
                if length == -1:
                    row.append(None)
                    continue
                row.append(data[at : at + length])
                at += length
            rows.append(row)
        assert at == len(data)
        return rows

    def test_a_vertex_stream_carries_a_jsonb_field_per_row(self) -> None:
        rows = self.read_stream(list(bulk.vertex_blocks([b'{"a":1}', b'{"a":2}'])))
        assert rows == [[b'\x01{"a":1}'], [b'\x01{"a":2}']]

    def test_a_vertex_stream_of_nothing_is_a_signature_and_a_trailer(self) -> None:
        assert self.read_stream(list(bulk.vertex_blocks([]))) == []

    def test_an_edge_stream_carries_two_identities_and_a_map(self) -> None:
        rows = self.read_stream(list(bulk.edge_blocks([(1, 2, b"{}")])))
        assert rows == [[struct.pack("!Q", 1), struct.pack("!Q", 2), b"\x01{}"]]

    def test_a_stream_is_broken_into_blocks(self) -> None:
        payloads = [b'{"a":%d}' % i for i in range(4000)]
        blocks = list(bulk.vertex_blocks(payloads, block=1024))
        assert len(blocks) > 1
        assert len(self.read_stream(blocks)) == 4000


class TestWritingPayloads:
    def test_a_table_of_columns_becomes_a_map_per_row(self) -> None:
        table = pyarrow.table({"a": [1, 2], "b": ["x", None]})
        assert list(vertex_payloads(table)) == [[b'{"a":1,"b":"x"}', b'{"a":2,"b":null}']]

    def test_a_polars_frame_reads_the_same(self) -> None:
        frame = polars.DataFrame({"a": [1, 2], "b": ["x", None]})
        table = pyarrow.table({"a": [1, 2], "b": ["x", None]})
        assert list(vertex_payloads(frame)) == list(vertex_payloads(table))

    def test_a_pandas_frame_reads_the_same(self) -> None:
        frame = pandas.DataFrame({"a": [1, 2]})
        assert list(vertex_payloads(frame)) == [[b'{"a":1}', b'{"a":2}']]

    def test_a_mapping_of_columns_reads_the_same(self) -> None:
        assert list(vertex_payloads({"a": [1]})) == [[b'{"a":1}']]

    def test_a_properties_column_is_taken_as_the_map_itself(self) -> None:
        table = pyarrow.table({"properties": ['{"a":1}']})
        assert list(vertex_payloads(table)) == [[b'{"a":1}']]

    def test_a_column_named_otherwise_is_a_property(self) -> None:
        table = pyarrow.table({"name": ["a"]})
        assert list(vertex_payloads(table)) == [[b'{"name":"a"}']]

    def test_a_chunk_at_a_time(self) -> None:
        table = pyarrow.table({"a": list(range(10))})
        assert [len(chunk) for chunk in vertex_payloads(table, size=4)] == [4, 4, 2]

    def test_endpoints_from_packed_identities(self) -> None:
        table = pyarrow.table({"start": [1], "end": [2], "w": [1.5]})
        assert list(edge_payloads(table)) == [[(1, 2, b'{"w":1.5}')]]

    def test_endpoints_from_the_text_form(self) -> None:
        table = pyarrow.table({"start": ["3.1"], "end": ["3.2"]})
        assert list(edge_payloads(table)) == [[((3 << 48) | 1, (3 << 48) | 2, b"{}")]]

    def test_endpoints_of_the_caller_s_own_names(self) -> None:
        table = pyarrow.table({"from": [1], "to": [2]})
        assert list(edge_payloads(table, start="from", end="to")) == [[(1, 2, b"{}")]]

    def test_a_null_endpoint_is_refused(self) -> None:
        table = pyarrow.table({"start": [None], "end": [2]}, schema=None)
        with pytest.raises(ValueError, match="needs both of its endpoints"):
            list(edge_payloads(table))

    def test_an_endpoint_of_the_wrong_type_is_refused(self) -> None:
        table = pyarrow.table({"start": [1.5], "end": [2.5]})
        with pytest.raises(TypeError, match="not an identity"):
            list(edge_payloads(table))


class TestWhatIsNotADependency:
    """Run in a subprocess, because this one has already imported all three backends."""

    SOURCE = """
import sys
for name in ("pyarrow", "pandas", "polars"):
    sys.modules[name] = None

import agensgraph
from agensgraph import columnar
from agensgraph.bulk import vertex_blocks

assert not [n for n in ("pyarrow", "pandas", "polars") if sys.modules[n] is not None]
assert columnar.columns([(agensgraph.GraphId(3, 1),)], ["id"]) == {"id": ["3.1"]}
assert len(b"".join(vertex_blocks([b'{"a":1}']))) == 35
for call in ("to_arrow", "to_pandas", "to_polars"):
    try:
        getattr(columnar, call)([(1,)], ["i"])
    except ImportError:
        pass
    else:
        raise AssertionError(call + " did not need its backend")
print("ok")
"""

    def test_no_backend_is_imported_until_it_is_used(self) -> None:
        finished = subprocess.run(
            [sys.executable, "-c", self.SOURCE], capture_output=True, text=True, check=False
        )
        assert finished.returncode == 0, finished.stderr
        assert finished.stdout.strip() == "ok"


class TestAnUndecodedPropertyMap:
    def test_the_bytes_are_handed_back_without_decoding(self) -> None:
        payload = b'{"a": 1}'
        held = raw_vertex(1, payload)
        assert held.properties_json() == payload

    def test_a_map_that_was_decoded_is_written_out_again(self) -> None:
        held = raw_vertex(1, b'{"a": 1}')
        assert held.properties == {"a": 1}
        assert held.properties_json() == b'{"a":1}'

    def test_a_map_supplied_as_a_dict(self) -> None:
        assert vertex(1, a=1).properties_json() == b'{"a":1}'


@pytest.mark.server
class TestAgainstAServer:
    @pytest.fixture
    def graph(self, agens):  # type: ignore[no-untyped-def]
        agens.execute("create vlabel person")
        agens.execute("create elabel knows")
        agens.execute("create (:person {name: 'a', age: 30})")
        agens.execute("create (:person {name: 'b', age: 40})")
        agens.execute(
            "match (a:person {name: 'a'}), (b:person {name: 'b'}) create (a)-[:knows]->(b)"
        )
        return agens

    def test_a_projection_exports(self, graph) -> None:  # type: ignore[no-untyped-def]
        frame = to_pandas(graph.execute_query("match (n:person) return n.name as name"))
        assert sorted(frame["name"].tolist()) == ["a", "b"]

    def test_a_result_carries_the_column_types(self, graph) -> None:  # type: ignore[no-untyped-def]
        result = graph.execute_query("match (n:person) return n, id(n)")
        assert result.oids == (VERTEX_OID, GRAPHID_OID)

    def test_a_whole_vertex_exports_as_a_struct(self, graph) -> None:  # type: ignore[no-untyped-def]
        table = to_arrow(graph.execute_query("match (n:person) return n"))
        held = table.column("n").to_pylist()
        assert {each["label"] for each in held} == {"person"}
        assert all('"name"' in each["properties"] for each in held)

    def test_an_edge_exports_with_the_identities_it_joins(self, graph) -> None:  # type: ignore[no-untyped-def]
        vertices = to_arrow(graph.execute_query("match (n:person) return n"))
        edges = to_arrow(graph.execute_query("match ()-[e:knows]->() return e"))
        ids = set(vertices.column("n").combine_chunks().field("id").to_pylist())
        held = edges.column("e").to_pylist()[0]
        assert held["start"] in ids
        assert held["end"] in ids

    def test_a_path_exports(self, graph) -> None:  # type: ignore[no-untyped-def]
        table = to_arrow(graph.execute_query("match p = (:person)-[:knows]->() return p"))
        held = table.column("p").to_pylist()[0]
        assert len(held["vertices"]) == 2
        assert len(held["edges"]) == 1

    def test_an_empty_result_keeps_the_shape_of_its_columns(self, graph) -> None:  # type: ignore[no-untyped-def]
        result = graph.execute_query("match (n:person) where n.age = -1 return n")
        table = to_arrow(result)
        assert table.num_rows == 0
        assert [field.name for field in table.schema.field("n").type] == [
            "id",
            "label",
            "properties",
        ]

    def test_an_unmatched_optional_element_is_null(self, graph) -> None:  # type: ignore[no-untyped-def]
        result = graph.execute_query(
            "match (n:person) optional match (n)-[:knows]->(m:person) return m order by m"
        )
        assert None in to_arrow(result).column("m").to_pylist()

    def test_a_server_cursor_exports_a_chunk_at_a_time(self, graph) -> None:  # type: ignore[no-untyped-def]
        table = f'"{graph.label_table.graph}".person'
        with graph.transaction(), graph.cursor(name="colaudit") as cursor:
            cursor.execute(f"select id, properties from {table}")
            built = list(batches(cursor, size=1))
        assert sum(batch.num_rows for batch in built) == 2
        assert len({batch.schema for batch in built}) == 1

    def test_a_frame_loads_and_reads_back(self, graph) -> None:  # type: ignore[no-untyped-def]
        graph.execute("create vlabel loaded")
        table = pyarrow.table({"name": ["x", "y"], "n": [1, 2]})
        assert graph.load_vertex_frame("loaded", table) == 2
        back = to_arrow(graph.execute_query("match (n:loaded) return n.name as name"))
        assert sorted(back.column("name").to_pylist()) == ["x", "y"]

    def test_a_vertex_column_loads_back_into_another_label(self, graph) -> None:  # type: ignore[no-untyped-def]
        graph.execute("create vlabel copied")
        table = to_arrow(graph.execute_query("match (n:person) return n"))
        maps = pyarrow.table(
            {"properties": table.column("n").combine_chunks().field("properties")}
        )
        assert graph.load_vertex_frame("copied", maps) == 2
        back = to_arrow(graph.execute_query("match (n:copied) return n.name as name"))
        assert sorted(back.column("name").to_pylist()) == ["a", "b"]

    def test_edges_load_from_the_identities_an_export_gave(self, graph) -> None:  # type: ignore[no-untyped-def]
        graph.execute("create elabel linked")
        ids = to_arrow(graph.execute_query("match (n:person) return id(n) as id"))
        packed = ids.column("id").to_pylist()
        edges = pyarrow.table({"start": [packed[0]], "end": [packed[1]], "w": [1.5]})
        assert graph.load_edge_frame("linked", edges) == 1
        back = graph.execute_query("match (a)-[e:linked]->(b) return e")
        held = back.records[0][0]
        assert held.properties == {"w": 1.5}
        assert held.start.packed == packed[0]

    def test_a_polars_frame_loads(self, graph) -> None:  # type: ignore[no-untyped-def]
        graph.execute("create vlabel from_polars")
        frame = polars.DataFrame({"k": [1, 2, 3]})
        assert graph.load_vertex_frame("from_polars", frame) == 3


@pytest.mark.server
class TestVectorsAgainstAServer:
    @pytest.fixture
    def graph(self, agens):  # type: ignore[no-untyped-def]
        if not agens.has_vectors():
            pytest.skip("pgvector is not created in this database")
        agens.register_vectors()
        agens.execute("create vlabel emb (v vector(4) generated)")
        agens.execute("create (:emb {v: [1.5, -2.25, 3.0, 4.0]})")
        agens.execute("create (:emb {v: [0.5, 0.25, 0.125, 0.0625]})")
        return agens

    def test_a_vector_column_from_the_binary_rendering(self, graph) -> None:  # type: ignore[no-untyped-def]
        result = graph.execute_query("match (n:emb) return n.v as v", binary_=True)
        table = to_arrow(result)
        assert table.schema.field("v").type == pyarrow.list_(pyarrow.float32(), 4)
        assert [1.5, -2.25, 3.0, 4.0] in table.column("v").to_pylist()

    def test_the_two_renderings_agree(self, graph) -> None:  # type: ignore[no-untyped-def]
        query = "match (n:emb) return n.v as v order by n.v"
        binary = to_arrow(graph.execute_query(query, binary_=True))
        text = to_arrow(graph.execute_query(query))
        assert binary.column("v").to_pylist() == text.column("v").to_pylist()

    def test_the_numbers_are_the_ones_the_driver_read(self, graph) -> None:  # type: ignore[no-untyped-def]
        result = graph.execute_query("match (n:emb) return n.v as v", binary_=True)
        table = to_arrow(result)
        assert table.column("v").to_pylist() == [held.tolist() for (held,) in result.records]
