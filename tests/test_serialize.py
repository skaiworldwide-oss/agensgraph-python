"""Handing a graph value to something that wants JSON.

Most of this needs no server: a value in, a shape out. The live half asserts the one thing a pure
test cannot, which is that the shape describes what the server actually sent.
"""

from __future__ import annotations

import json

import msgspec
import pytest

from agensgraph import GraphId, json_default, to_builtins, to_json
from agensgraph.types import Edge, Path, Vertex
from agensgraph.vector import SparseVector, Vector, generated_column

VERTEX = Vertex(GraphId(3, 1), "person", b'{"a": 1, "b": "x"}')
OTHER = Vertex(GraphId(3, 2), "person", b'{"a": 2}')
EDGE = Edge(GraphId(4, 1), "knows", GraphId(3, 1), GraphId(3, 2), b'{"weight": 2}')
HOP = Path((VERTEX, OTHER), (EDGE,))


class TestTheShape:
    def test_a_vertex_carries_its_identity_label_and_properties(self) -> None:
        assert to_builtins(VERTEX) == {
            "id": "3.1",
            "label": "person",
            "properties": {"a": 1, "b": "x"},
        }

    def test_an_edge_carries_both_ends(self) -> None:
        assert to_builtins(EDGE) == {
            "id": "4.1",
            "label": "knows",
            "start": "3.1",
            "end": "3.2",
            "properties": {"weight": 2},
        }

    def test_a_path_is_its_two_lists(self) -> None:
        assert to_builtins(HOP) == {
            "vertices": [to_builtins(VERTEX), to_builtins(OTHER)],
            "edges": [to_builtins(EDGE)],
        }

    def test_a_path_of_one_vertex_has_no_edges(self) -> None:
        assert to_builtins(Path((VERTEX,), ())) == {
            "vertices": [to_builtins(VERTEX)],
            "edges": [],
        }

    def test_an_identity_is_the_text_the_server_prints(self) -> None:
        """Rather than a pair of numbers, which invites building one that was never sent."""
        assert to_builtins(GraphId(3, 1)) == "3.1"

    def test_a_vector_is_its_numbers(self) -> None:
        assert to_builtins(Vector([1.0, 2.0])) == [1.0, 2.0]

    def test_a_sparse_vector_keeps_its_dimension(self) -> None:
        """Expanding it would discard the reason the type exists."""
        assert to_builtins(SparseVector({0: 1.0, 3: 2.0}, 6)) == {
            "dimensions": 6,
            "indices": [0, 3],
            "values": [1.0, 2.0],
        }

    def test_something_that_is_not_a_graph_value_says_so(self) -> None:
        with pytest.raises(TypeError, match="cannot describe"):
            to_builtins(object())


class TestThePrivateFieldsNeverAppear:
    """What this exists to prevent.

    A vertex is a struct, so an encoder that knows how to encode one writes its fields: ``_id`` as
    a nested pair, next to ``_raw`` and ``_props``. That publishes names the driver means to keep
    to itself, in a shape nobody would choose.
    """

    @pytest.mark.parametrize("value", [VERTEX, EDGE, HOP, GraphId(3, 1)])
    def test_no_underscore_name_reaches_the_output(self, value) -> None:  # type: ignore[no-untyped-def]
        for text in (to_json(value).decode(), json.dumps(value, default=json_default)):
            assert "_id" not in text
            assert "_raw" not in text
            assert "_props" not in text
            assert "labid" not in text

    def test_encoding_the_value_directly_is_what_goes_wrong(self) -> None:
        """The behaviour this guards against, asserted so the guard has a reason on the record."""
        leaked = msgspec.json.encode(VERTEX).decode()
        assert "_id" in leaked
        assert "labid" in leaked


class TestTheTwoRoutesAgree:
    @pytest.mark.parametrize("value", [VERTEX, EDGE, HOP])
    def test_the_fast_route_and_the_standard_one_produce_the_same_json(self, value) -> None:  # type: ignore[no-untyped-def]
        assert json.loads(to_json(value)) == json.loads(json.dumps(value, default=json_default))


class TestAWholeResultGoesThroughInOneCall:
    def test_rows_of_mixed_values(self) -> None:
        rows = [(VERTEX, "a label", 3), (VERTEX, None, 4)]
        assert json.loads(to_json(rows)) == [
            [to_builtins(VERTEX), "a label", 3],
            [to_builtins(VERTEX), None, 4],
        ]

    def test_a_mapping_is_walked_too(self) -> None:
        assert json.loads(to_json({"n": VERTEX})) == {"n": to_builtins(VERTEX)}

    def test_something_with_no_graph_value_in_it_passes_through(self) -> None:
        assert json.loads(to_json({"a": [1, 2], "b": None})) == {"a": [1, 2], "b": None}


class TestItDoesNotDecodeWhatNobodyRead:
    """The reason to reach for the fast route.

    A property map arrives as bytes and is decoded when something reads it. Writing it back out is
    not reading it, so the bytes go straight to the output.
    """

    def test_the_map_is_still_undecoded_afterwards(self) -> None:
        fresh = Vertex(GraphId(3, 1), "person", b'{"a": 1}')
        to_json(fresh)
        assert fresh._props is None, "nothing decoded it"
        assert fresh._raw is not None, "and the bytes are still there"

    def test_a_map_already_read_is_written_from_what_was_read(self) -> None:
        fresh = Vertex(GraphId(3, 1), "person", b'{"a": 1}')
        assert fresh.properties == {"a": 1}
        assert json.loads(to_json(fresh))["properties"] == {"a": 1}

    def test_it_beats_the_route_that_has_to_decode(self) -> None:
        """Not a benchmark -- a floor. The gap is widest on a map nobody wanted, which is what a
        result full of embeddings is."""
        import time

        raw = json.dumps({"v": [0.5] * 1024}).encode()

        def fresh() -> list[Vertex]:
            return [Vertex(GraphId(3, i), "doc", raw) for i in range(100)]

        fast = decoding = 0.0
        for _ in range(3):
            started = time.monotonic()
            to_json(fresh())
            fast += time.monotonic() - started
            started = time.monotonic()
            json.dumps(fresh(), default=json_default)
            decoding += time.monotonic() - started
        assert fast < decoding


@pytest.mark.server
class TestAgainstAServer:
    def test_what_the_server_sent_serializes_to_the_documented_shape(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel person")
        agens.execute("create elabel knows")
        agens.execute("create (:person {name: 'a'})-[:knows {w: 1}]->(:person {name: 'b'})")
        agens.refresh_labels()
        (path,) = agens.execute_query(
            "match p = (:person)-[:knows]->(:person) return p"
        ).records[0]
        described = json.loads(to_json(path))
        assert sorted(described) == ["edges", "vertices"]
        assert described["vertices"][0]["label"] == "person"
        assert described["edges"][0]["start"] == described["vertices"][0]["id"]

    def test_an_identity_written_out_reads_back_in(self, agens) -> None:  # type: ignore[no-untyped-def]
        """Which is the argument for the text form: it goes back to the server unchanged."""
        agens.execute("create vlabel person")
        agens.execute("create (:person {name: 'a'})")
        agens.refresh_labels()
        (vertex,) = agens.execute_query("match (n:person) return n").records[0]
        written = json.loads(to_json(vertex))["id"]
        (again,) = agens.execute_query(
            "match (n:person) where id(n) = %s::graphid return n", (written,)
        ).records[0]
        assert again.id == vertex.id

    def test_a_vector_column_read_from_the_server_serializes(self, agens) -> None:  # type: ignore[no-untyped-def]
        """Read from a column of its own, which is the shape that comes back as a vector.

        The same numbers left in the property map are not one: they read back as a list where the
        server can promote a property and as the text they print as where it cannot, so what a
        column gives is the case worth pinning here.
        """
        if not agens.has_vectors():
            pytest.skip("the vector extension is not created in this database")
        if not agens.can_promote_properties():
            pytest.skip("this server cannot store a property in a column of its own")
        agens.register_vectors()
        agens.execute(f"create vlabel emb ({generated_column('v', 3)})")
        agens.execute("create (:emb {v: %s})", (Vector([1.0, 2.0, 3.0]),))
        agens.refresh_labels()
        (value,) = agens.execute_query("match (n:emb) return n.v").records[0]
        assert isinstance(value, Vector)
        assert json.loads(to_json(value)) == [1.0, 2.0, 3.0]
