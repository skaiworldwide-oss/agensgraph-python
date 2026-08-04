from __future__ import annotations

import pytest

from agensgraph._protocol import decode, textfmt

from .corpus import EDGES, ELEMENT_ARRAYS, PATHS, REJECTED_VERTICES, VERTICES


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


@pytest.mark.parametrize("buf,count", ELEMENT_ARRAYS)
def test_element_array(buf, count):
    got = decode.elements_from_text(buf)
    assert len(got) == count


def test_null_elements_map_to_none():
    """An element array may hold nulls, and the server writes them as a bare word."""
    assert decode.elements_from_text(b"[NULL,NULL,NULL]") == [None, None, None]
    got = decode.elements_from_text(b"[v[5.1]{},NULL]")
    assert got[0] is not None
    assert got[1] is None


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


def test_fallback_is_counted_when_the_guess_does_not_hold():
    """The cheap boundary search is an accelerator, so its misses have to be visible."""
    textfmt.reset_fallback_count()
    decode.path_from_text(b"[n[7.3]{},r[5.7][7.3,7.9]{},n[7.9]{}]")
    assert textfmt.fallback_count() == 0

    textfmt.reset_fallback_count()
    p = decode.path_from_text(b'[v[5.1]{"a": "},Company[1.1]{"},e[6.1][5.1,5.5]{},v[5.5]{}]')
    assert len(p.vertices) == 2
    assert p.vertices[0].properties == {"a": "},Company[1.1]{"}
    assert textfmt.fallback_count() == 1
