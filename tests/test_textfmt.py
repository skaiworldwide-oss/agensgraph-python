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


class TestTheLabelNameTable:
    """Label names are decoded once per distinct name rather than once per element."""

    def test_the_same_bytes_give_the_same_string_object(self):
        a = decode.vertex_from_text(b'person[3.1]{"k": 1}')
        b = decode.vertex_from_text(b'person[3.2]{"k": 2}')
        assert a.label == b.label == "person"
        assert a.label is b.label

    def test_a_label_the_table_has_never_seen_still_reads(self):
        assert decode.vertex_from_text('사람[3.1]{}'.encode()).label == "사람"
        assert decode.vertex_from_text(b'a,b[3.1]{}').label == "a,b"

    def test_an_edge_reads_its_label_through_the_same_table(self):
        edge = decode.edge_from_text(b'knows[4.1][3.1,3.2]{}')
        assert edge.label == "knows"
        assert edge.label is decode.vertex_from_text(b'knows[3.9]{}').label

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
