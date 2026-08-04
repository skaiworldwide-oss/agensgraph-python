from __future__ import annotations

import pytest

from agensgraph import Edge, GraphId, Label, Path, Vertex


def v(labid=5, locid=1, label="person", props=None):
    return Vertex(GraphId(labid, locid), label, props if props is not None else {})


def e(labid=6, locid=1, start=(5, 1), end=(5, 2), label="knows", props=None):
    return Edge(
        GraphId(labid, locid),
        label,
        GraphId(*start),
        GraphId(*end),
        props if props is not None else {},
    )


class TestIdentity:
    def test_equality_uses_identity_alone(self):
        """The server orders and compares elements by identity, so this matches it."""
        a = v(props={"name": "a"})
        b = v(props={"name": "b"}, label="other")
        assert a == b

    def test_different_identity_is_unequal(self):
        assert v(locid=1) != v(locid=2)

    def test_hashable_and_usable_in_a_set(self):
        assert len({v(locid=1), v(locid=1), v(locid=2)}) == 2
        assert {v(): "x"}[v()] == "x"

    def test_a_vertex_and_an_edge_with_the_same_id_are_not_equal(self):
        assert v(labid=5, locid=1) != e(labid=5, locid=1)

    def test_comparison_with_a_foreign_type_does_not_raise(self):
        vertex = v()
        assert vertex != object()
        assert vertex is not None
        assert vertex not in [object(), None, "x"]


class TestImmutability:
    def test_cannot_assign(self):
        with pytest.raises(AttributeError):
            v().label = "other"  # type: ignore[misc]

    def test_cannot_delete(self):
        with pytest.raises(AttributeError):
            del v()._label  # type: ignore[attr-defined]


class TestLazyProperties:
    def test_raw_slice_is_decoded_on_first_access(self):
        vertex = Vertex(GraphId(5, 1), "person", b'{"name": "a", "n": 1}')
        assert vertex.properties == {"name": "a", "n": 1}

    def test_decoding_happens_once(self):
        vertex = Vertex(GraphId(5, 1), "person", b'{"n": 1}')
        first = vertex.properties
        assert vertex.properties is first

    def test_untouched_properties_are_never_decoded(self):
        """Holding the slice is the point: an unread map costs nothing to carry."""
        vertex = Vertex(GraphId(5, 1), "person", b"not json at all")
        assert vertex.id == GraphId(5, 1)
        assert vertex.label == "person"
        with pytest.raises(Exception):
            _ = vertex.properties

    def test_get_reads_one_property(self):
        vertex = Vertex(GraphId(5, 1), "person", b'{"n": 1}')
        assert vertex.get("n") == 1
        assert vertex.get("missing") is None
        assert vertex.get("missing", 7) == 7

    def test_a_non_object_property_map_is_rejected(self):
        for bad in [b"null", b"0", b'"x"', b"[]"]:
            with pytest.raises(ValueError):
                _ = Vertex(GraphId(5, 1), "person", bad).properties

    def test_metadata_and_properties_are_separate_namespaces(self):
        """A property named like a metadata field must not shadow it."""
        vertex = Vertex(GraphId(5, 1), "person", b'{"label": "spoofed", "id": "spoofed"}')
        assert vertex.label == "person"
        assert vertex.id == GraphId(5, 1)
        assert vertex.properties["label"] == "spoofed"


class TestEdge:
    def test_endpoints(self):
        edge = e(start=(5, 1), end=(5, 2))
        assert edge.start == GraphId(5, 1)
        assert edge.end == GraphId(5, 2)


class TestPath:
    def test_single_vertex_path_is_truthy(self):
        """Counting only edges would make a legal one-vertex path read as empty."""
        p = Path((v(),), ())
        assert len(p) == 1
        assert bool(p) is True
        assert p.length == 0

    def test_empty_path(self):
        p = Path((), ())
        assert len(p) == 0
        assert bool(p) is False
        assert p.start is None
        assert p.end is None

    def test_length_is_hops_and_len_is_elements(self):
        p = Path((v(locid=1), v(locid=2), v(locid=3)), (e(locid=1), e(locid=2)))
        assert p.length == 2
        assert len(p) == 5

    def test_elements_interleave_in_order(self):
        p = Path((v(locid=1), v(locid=2)), (e(locid=1),))
        got = list(p)
        assert isinstance(got[0], Vertex)
        assert isinstance(got[1], Edge)
        assert isinstance(got[2], Vertex)
        assert got[0].id == GraphId(5, 1)
        assert got[2].id == GraphId(5, 2)

    def test_indexable_and_iterable(self):
        p = Path((v(locid=1), v(locid=2)), (e(),))
        assert p[0].id == GraphId(5, 1)
        assert len(list(p)) == 3

    def test_mismatched_counts_are_rejected(self):
        with pytest.raises(ValueError):
            Path((v(),), (e(), e()))
        with pytest.raises(ValueError):
            Path((v(locid=1), v(locid=2)), ())

    def test_hashable(self):
        p = Path((v(),), ())
        assert len({p, Path((v(),), ())}) == 1


class TestLabel:
    def test_is_a_string_subclass_so_it_reads_naturally(self):
        assert Label("person") == "person"
        assert Label("person").upper() == "PERSON"

    def test_repr_marks_it_as_an_identifier(self):
        assert repr(Label("person")) == "Label('person')"
