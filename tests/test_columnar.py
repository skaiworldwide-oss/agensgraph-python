"""Handing a result to something that works in columns.

Each backend is skipped where it is not installed, since none of them is a dependency. What is
asserted without one is the turning of rows into columns, and the refusal of a value that has no
columnar form.
"""

from __future__ import annotations

import pytest

import agensgraph
from agensgraph.columnar import columns, to_arrow, to_pandas, to_polars
from agensgraph.vector import SparseVector, Vector

pyarrow = pytest.importorskip("pyarrow", reason="pyarrow is not installed")
pandas = pytest.importorskip("pandas", reason="pandas is not installed")
polars = pytest.importorskip("polars", reason="polars is not installed")


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


class TestWhatHasNoColumnarForm:
    @pytest.mark.parametrize(
        "value",
        [
            agensgraph.Vertex(agensgraph.GraphId(3, 1), "person", {}),
            agensgraph.Edge(
                agensgraph.GraphId(4, 1),
                "knows",
                agensgraph.GraphId(3, 1),
                agensgraph.GraphId(3, 2),
                {},
            ),
            agensgraph.Path((), ()),
        ],
    )
    def test_it_is_refused_and_says_what_to_do_instead(self, value: object) -> None:
        with pytest.raises(TypeError, match="no columnar form"):
            columns([(value,)], ["n"])

    def test_the_message_names_a_projection(self) -> None:
        vertex = agensgraph.Vertex(agensgraph.GraphId(3, 1), "person", {})
        with pytest.raises(TypeError, match=r"return id\(n\), label\(n\), n\.name"):
            columns([(vertex,)], ["n"])


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

    def test_polars(self) -> None:
        frame = to_polars([(1, "a"), (2, "b")], ["n", "s"])
        assert frame.columns == ["n", "s"]
        assert frame["n"].to_list() == [1, 2]

    def test_a_vector_column_in_arrow_is_a_list_column(self) -> None:
        table = to_arrow([(Vector([1.0, 2.0]),), (Vector([3.0, 4.0]),)], ["v"])
        assert table.column("v").to_pylist() == [[1.0, 2.0], [3.0, 4.0]]

    def test_an_empty_result_makes_an_empty_table(self) -> None:
        assert to_arrow([], ["n"]).num_rows == 0
        assert len(to_pandas([], ["n"])) == 0
        assert to_polars([], ["n"]).height == 0


@pytest.mark.server
class TestAgainstAServer:
    @pytest.fixture
    def graph(self, agens):  # type: ignore[no-untyped-def]
        agens.execute("create vlabel person")
        agens.execute("create (:person {name: 'a', age: 30})")
        agens.execute("create (:person {name: 'b', age: 40})")
        return agens

    def test_a_projection_exports(self, graph) -> None:  # type: ignore[no-untyped-def]
        result = graph.execute_query("match (n:person) return n.name as name, n.age as age")
        frame = to_pandas(result.records, result.keys)
        assert sorted(frame["name"].tolist()) == ["a", "b"]
        assert sorted(frame["age"].tolist()) == [30, 40]

    def test_an_identity_exports_as_its_text_form(self, graph) -> None:  # type: ignore[no-untyped-def]
        result = graph.execute_query("match (n:person) return id(n) as id")
        table = to_arrow(result.records, result.keys)
        assert all("." in value for value in table.column("id").to_pylist())

    def test_returning_a_whole_vertex_is_refused(self, graph) -> None:  # type: ignore[no-untyped-def]
        result = graph.execute_query("match (n:person) return n")
        with pytest.raises(TypeError, match="no columnar form"):
            to_arrow(result.records, result.keys)
