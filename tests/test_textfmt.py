from __future__ import annotations

import time

import pytest

from agensgraph._protocol import decode, textfmt

from .corpus import EDGES, PATHS, REJECTED_VERTICES, VERTICES


@pytest.mark.parametrize("buf,label,labid,locid,props", VERTICES)
def test_vertex(buf, label, labid, locid, props):
    v = decode.vertex_from_text(buf)
    assert v.label == label
    assert v.id.labid == labid
    assert v.id.locid == locid
    assert v.properties == props


@pytest.mark.parametrize("buf,label,ident,start,end,props", EDGES)
def test_edge(buf, label, ident, start, end, props):
    e = decode.edge_from_text(buf)
    assert e.label == label
    assert (e.id.labid, e.id.locid) == ident
    assert (e.start.labid, e.start.locid) == start
    assert (e.end.labid, e.end.locid) == end
    assert e.properties == props


@pytest.mark.parametrize("buf,nvertices,nedges", PATHS)
def test_path(buf, nvertices, nedges):
    p = decode.path_from_text(buf)
    assert len(p.vertices) == nvertices
    assert len(p.edges) == nedges
    assert p.length == nedges


def test_empty_path_is_legal():
    """The server writes this for a path with no vertices; it is not an error."""
    p = decode.path_from_text(b"[]")
    assert len(p) == 0
    assert p.vertices == ()
    assert p.start is None


def test_single_vertex_path_is_truthy():
    """A path of one vertex and no edges has one element, so it must not read as empty."""
    p = decode.path_from_text(b"[n[7.3]{}]")
    assert len(p) == 1
    assert bool(p) is True
    assert p.length == 0


def test_null_elements_map_to_none():
    """An element array may hold nulls, and the server writes them as a bare word."""
    assert decode.vertices_from_text(b"[NULL,NULL,NULL]") == [None, None, None]
    got = decode.vertices_from_text(b"[v[5.1]{},NULL]")
    assert got[0] is not None
    assert got[1] is None


def test_a_list_cannot_mix_a_vertex_and_an_edge():
    """Which is why nothing here classifies an array element by its shape: the server has an
    array type for vertices and one for edges and no type for both, and refuses to build one --
    ``graph object cannot be list element``."""
    assert not hasattr(decode, "elements_from_text")


def test_null_inside_a_path_is_rejected():
    """A path with a hole in it is not a path, so it fails rather than losing an element."""
    with pytest.raises(ValueError):
        decode.path_from_text(b"[NULL]")
    with pytest.raises(ValueError):
        decode.path_from_text(b"[v[5.1]{},NULL,v[5.2]{}]")


def test_even_element_count_is_rejected():
    with pytest.raises(ValueError):
        decode.path_from_text(b"[v[5.1]{},v[5.5]{}]")


@pytest.mark.parametrize("buf", REJECTED_VERTICES)
def test_bad_vertex_is_rejected_as_a_value_error(buf):
    """Every rejection is one exception type, so a caller can catch it in one clause."""
    with pytest.raises(ValueError):
        decode.vertex_from_text(buf)


def test_property_text_cannot_fabricate_endpoints():
    """A vertex whose property value contains an endpoint-shaped run stays a vertex.

    Reading this as an edge would invent a start and an end out of property text, so the
    endpoint group has to be validated rather than merely located.
    """
    buf = b'v[7.9]{"a": "][1.1,2.2]"}'
    v = decode.vertex_from_text(buf)
    assert v.properties == {"a": "][1.1,2.2]"}
    with pytest.raises(ValueError):
        decode.edge_from_text(buf)


def test_brace_in_a_label_does_not_swallow_a_path():
    """Tracking brace depth alone would never return to zero and merge every element."""
    p = decode.path_from_text(b"[a{b[7.3]{},r[5.7][7.3,7.9]{},a{b[7.9]{}]")
    assert [v.label for v in p.vertices] == ["a{b", "a{b"]


def test_comma_in_a_label_does_not_split_an_element():
    p = decode.path_from_text(b"[a,b[7.3]{},r[5.7][7.3,7.9]{},a,b[7.9]{}]")
    assert [v.label for v in p.vertices] == ["a,b", "a,b"]
    assert p.edges[0].label == "r"


def test_comma_inside_a_property_string_does_not_split():
    p = decode.path_from_text(b'[v[5.1]{"k": "a,b"},e[6.1][5.1,5.5]{},v[5.5]{}]')
    assert p.vertices[0].properties == {"k": "a,b"}


def test_escaped_quote_does_not_end_a_property_string():
    v = decode.vertex_from_text(b'n[7.3]{"s": "[}\\""}')
    assert v.properties == {"s": '[}"'}


def test_truncated_element_list_is_rejected():
    for bad in [b'[v[5.1]{"a": "x}]', b"[v[5.1]{]", b"v[5.1]{}"]:
        with pytest.raises(ValueError):
            textfmt.split_elements(bad)


def test_a_boundary_inside_a_property_string_does_not_split():
    """The close of a map, a comma and a label-shaped run, all inside one string value."""
    p = decode.path_from_text(b'[v[5.1]{"a": "},Company[1.1]{"},e[6.1][5.1,5.5]{},v[5.5]{}]')
    assert len(p.vertices) == 2
    assert p.vertices[0].properties == {"a": "},Company[1.1]{"}


@pytest.mark.parametrize(
    ("buf", "labels", "ids"),
    [
        # A property value holding whole elements, beside a label holding a comma.
        (
            b'[a[3.1]{"v": "[5.5]{},B[7.7]{},C[9.9]{}"},a,b[4.1]{}]',
            ["a", "a,b"],
            ["3.1", "4.1"],
        ),
        # The same, where the run inside the string does not close its own brace.
        (
            b'[a[3.1]{"v": "[5.5]{},B[7.7]{"},a,b[4.1]{}]',
            ["a", "a,b"],
            ["3.1", "4.1"],
        ),
        (b'[a[3.1]{"v": "x[1.1]{}"},plain[4.1]{}]', ["a", "plain"], ["3.1", "4.1"]),
    ],
)
def test_a_property_value_cannot_fabricate_elements(buf, labels, ids):
    """Reading one of these as several invents both labels and identities."""
    vertices = decode.vertices_from_text(buf)
    assert [v.label for v in vertices] == labels
    assert [str(v.id) for v in vertices] == ids


def test_the_map_is_not_decoded_until_it_is_read():
    """A boundary check asks whether the map is JSON, not what it holds."""
    from agensgraph import numbers

    calls = 0
    real = numbers._decode

    def counting(data):
        nonlocal calls
        calls += 1
        return real(data)

    numbers._decode = counting
    try:
        v = decode.vertex_from_text(b'a[3.1]{"k": 1, "s": "text"}')
        assert calls == 0
        assert v.properties == {"k": 1, "s": "text"}
        assert calls == 1
    finally:
        numbers._decode = real


# Two vertices of a label whose own name holds a vertex rendering and a comma. Measured from the
# left this is four vertices of labels `x` and `y`, and nothing in the bytes says otherwise. A
# label table is a dict here: `get` and `len` are all a reading asks of one.
AMBIGUOUS_LIST = b"[x[3.1]{},y[9.1]{},x[3.1]{},y[9.1]{}]"
AMBIGUOUS_PATH = b"[x[3.1]{},y[7.2]{},e[8.1][7.2,7.3]{},x[3.1]{},y[7.3]{}]"
AMBIGUOUS_EDGES = b"[k[2.1][1.1,1.2]{},f[4.1][3.1,3.2]{},k[2.1][1.1,1.2]{},f[4.1][3.1,3.2]{}]"


class TestALabelHoldingAnElementRendering:
    def test_the_shortest_reading_stands_without_a_table(self):
        assert [v.label for v in decode.vertices_from_text(AMBIGUOUS_LIST)] == [
            "x",
            "y",
            "x",
            "y",
        ]

    def test_the_table_settles_which_reading_it_is(self):
        names = {9: "x[3.1]{},y", 3: "other"}
        assert [v.label for v in decode.vertices_from_text(AMBIGUOUS_LIST, names)] == [
            "x[3.1]{},y",
            "x[3.1]{},y",
        ]

    def test_a_table_holding_nothing_is_not_asked(self):
        """Which is the state after the session moves to another graph, and asking a table that
        knows no labels would make every list read as one the labels are missing from."""
        assert len(decode.vertices_from_text(AMBIGUOUS_LIST, {})) == 4

    def test_a_table_without_the_label_leaves_the_shortest_reading(self):
        """A label created after the table was filled. Nothing here can do better than the bytes."""
        assert len(decode.vertices_from_text(AMBIGUOUS_LIST, {4: "other"})) == 4

    def test_an_ordinary_list_is_read_the_same_either_way(self):
        buf = b"[p[3.1]{},p[3.2]{}]"
        plain = decode.vertices_from_text(buf)
        assert [v.label for v in plain] == ["p", "p"]
        assert decode.vertices_from_text(buf, {3: "p"}) == plain

    def test_a_null_slot_survives_the_reading(self):
        found = decode.vertices_from_text(b"[x[3.1]{},y[9.1]{},NULL]", {9: "x[3.1]{},y"})
        assert [None if v is None else v.label for v in found] == ["x[3.1]{},y", None]

    def test_edges_read_the_same_way(self):
        assert len(decode.edges_from_text(AMBIGUOUS_EDGES)) == 4
        names = {4: "k[2.1][1.1,1.2]{},f"}
        assert [e.label for e in decode.edges_from_text(AMBIGUOUS_EDGES, names)] == [
            "k[2.1][1.1,1.2]{},f"
        ] * 2

    def test_a_path_is_read_by_the_table_too(self):
        names = {7: "x[3.1]{},y", 8: "e"}
        path = decode.path_from_text(AMBIGUOUS_PATH, names)
        assert [element.label for element in path.elements] == ["x[3.1]{},y", "e", "x[3.1]{},y"]
        assert path.edges[0].start == path.vertices[0].id
        assert path.edges[0].end == path.vertices[1].id

    def test_a_path_without_a_table_is_refused_rather_than_misread(self):
        """The reading the bytes give puts an edge where a vertex belongs, and says so."""
        with pytest.raises(ValueError, match="not a vertex"):
            decode.path_from_text(AMBIGUOUS_PATH)

    def test_a_malformed_path_is_not_read_as_one_vertex(self):
        """Even with a table: the one-element reading names a label the table does not hold."""
        with pytest.raises(ValueError, match="odd number"):
            decode.path_from_text(b"[v[5.1]{},v[5.5]{}]", {5: "person"})


class TestPreferringOneElementOverAnother:
    def test_the_element_the_test_takes_is_the_one_read(self):
        buf = b"[a[3.1]{},b[3.2]{}]"
        assert textfmt.split_elements(buf) == [b"a[3.1]{}", b"b[3.2]{}"]
        assert textfmt.split_elements(buf, lambda index, element: b"," in element) == [
            b"a[3.1]{},b[3.2]{}"
        ]

    def test_the_shortest_is_read_where_it_takes_none(self):
        buf = b"[a[3.1]{},b[3.2]{}]"
        assert textfmt.split_elements(buf, lambda index, element: False) == (
            textfmt.split_elements(buf)
        )

    def test_the_test_is_told_where_the_element_sits(self):
        buf = b"[a[3.1]{},b[3.2]{},c[3.3]{}]"
        seen = []

        def prefer(index, element):
            seen.append((index, element))
            return b"," not in element

        assert len(textfmt.split_elements(buf, prefer)) == 3
        assert [index for index, _ in seen] == [0, 1, 2]

    def test_an_ordinary_list_costs_one_test_an_element(self):
        """Every element of such a list also reads as the head of a longer one, so a reading
        that had to try them all would be quadratic. The first element taken ends the search."""
        asked = 0

        def prefer(index, element):
            nonlocal asked
            asked += 1
            return b"," not in element

        buf = b"[" + b",".join(b"a[3.%d]{}" % i for i in range(1, 501)) + b"]"
        assert len(textfmt.split_elements(buf, prefer)) == 500
        assert asked == 500


class TestTheLabelNameTable:
    """Label names are decoded once per distinct name rather than once per element."""

    def test_the_same_bytes_give_the_same_string_object(self):
        a = decode.vertex_from_text(b'person[3.1]{"k": 1}')
        b = decode.vertex_from_text(b'person[3.2]{"k": 2}')
        assert a.label == b.label == "person"
        assert a.label is b.label

    def test_a_label_the_table_has_never_seen_still_reads(self):
        assert decode.vertex_from_text("사람[3.1]{}".encode()).label == "사람"
        assert decode.vertex_from_text(b"a,b[3.1]{}").label == "a,b"

    def test_an_edge_reads_its_label_through_the_same_table(self):
        edge = decode.edge_from_text(b"knows[4.1][3.1,3.2]{}")
        assert edge.label == "knows"
        assert edge.label is decode.vertex_from_text(b"knows[3.9]{}").label

    def test_the_table_does_not_grow_without_bound(self):
        """It holds label names, which are schema rather than data -- but a process that
        somehow meets more of them must not accumulate them for ever."""
        names = decode._label_names
        names.clear()
        for i in range(decode.LABEL_NAMES_MAX + 50):
            assert names[b"label-%d" % i] == f"label-{i}"
        assert len(names) <= decode.LABEL_NAMES_MAX
        names.clear()


class TestTheMechanismsBehindTheNumbers:
    """Assertions on how the work is done, not only on what comes out.

    Every defect this suite missed once passed the type gates and every behavioural test: what
    they could not see was a mechanism -- a map decoded that should not have been, a scan whose
    cost grew with what it was scanning past. Each of those is cheap to assert directly, and
    what follows asserts it.
    """

    @staticmethod
    def element(index: int, size: int) -> bytes:
        return b'p[3.%d]{"pad": "%s"}' % (index, b"x" * size)

    def test_splitting_costs_the_same_per_element_however_many_there_are(self) -> None:
        """A splitter that measures each element in turn is linear; one that rescans is not.

        The property map is a kilobyte and a half, which is where a quadratic scan shows: it is
        the bytes between one separator and the next that a wrong implementation walks again.
        """

        def cost(count: int) -> float:
            buf = b"[" + b",".join(self.element(i, 1500) for i in range(count)) + b"]"
            best = float("inf")
            for _ in range(5):
                started = time.perf_counter()
                textfmt.split_elements(buf)
                best = min(best, time.perf_counter() - started)
            return best

        small, large = cost(50), cost(400)
        assert large < small * 20, "eight times the elements cost more than eight times as much"

    def test_a_map_is_decoded_once_per_element_and_not_twice(self) -> None:
        """The two renderings each decode; neither may decode what the other already did."""
        from agensgraph import numbers

        buf = b"[" + b",".join(self.element(i, 8) for i in range(20)) + b"]"
        calls = 0
        real = numbers._decode

        def counting(data: bytes) -> object:
            nonlocal calls
            calls += 1
            return real(data)

        numbers._decode = counting
        try:
            vertices = decode.vertices_from_text(buf)
            assert calls == 0, "nothing is decoded until a map is read"
            assert [v.properties["pad"] for v in vertices] == ["x" * 8] * 20
            assert calls == 20
            assert [v.properties["pad"] for v in vertices] == ["x" * 8] * 20
            assert calls == 20, "a map read twice is decoded once"
        finally:
            numbers._decode = real
