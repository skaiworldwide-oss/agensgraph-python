"""Checking the two renderings against each other.

The same value can arrive as text or as a composite, and the two are decoded by
independent code. Feeding one logical value through both must produce the same object,
so a disagreement is a defect in one of them and the comparison needs no other oracle.

Comparisons here are deliberately strict about type. ``Decimal('1.5') == 1.5`` and
``True == 1`` are both true in Python, so a decoder that returned the wrong numeric
type would pass an ordinary equality check while quietly losing precision. The type is
therefore checked before the value, and never after.
"""

from __future__ import annotations

import math

import msgspec
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agensgraph._protocol import decode
from agensgraph.types import Edge, Path, Vertex

from . import wire
from .corpus import EDGES, PATHS, VERTICES


def same(a: object, b: object) -> bool:
    """Structural equality that does not paper over the traps of ``==``."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        assert isinstance(b, float)
        if math.isnan(a) and math.isnan(b):
            return True
        if a == 0.0 and b == 0.0:
            # -0.0 == 0.0 is true, so plain equality would hide a lost sign.
            return math.copysign(1.0, a) == math.copysign(1.0, b)
        return a == b
    if isinstance(a, dict):
        assert isinstance(b, dict)
        if list(a.keys()) != list(b.keys()):
            return False
        return all(same(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        assert isinstance(b, list | tuple)
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)


def same_element(a: object, b: object) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, Vertex):
        assert isinstance(b, Vertex)
        return a.id == b.id and a.label == b.label and same(a.properties, b.properties)
    if isinstance(a, Edge):
        assert isinstance(b, Edge)
        return (
            a.id == b.id
            and a.label == b.label
            and a.start == b.start
            and a.end == b.end
            and same(a.properties, b.properties)
        )
    if isinstance(a, Path):
        assert isinstance(b, Path)
        return len(a) == len(b) and all(same_element(x, y) for x, y in zip(a, b, strict=True))
    raise TypeError(f"not a graph value: {type(a)}")


class TestSameHelper:
    """The comparison helper has to be right, or every test below is vacuous."""

    def test_nan_agrees_with_nan(self):
        assert same(float("nan"), float("nan"))

    def test_negative_zero_differs_from_zero(self):
        assert not same(-0.0, 0.0)
        assert same(-0.0, -0.0)

    def test_int_and_float_are_not_interchangeable(self):
        assert not same(1, 1.0)

    def test_bool_and_int_are_not_interchangeable(self):
        assert not same(True, 1)

    def test_key_order_matters(self):
        assert not same({"a": 1, "b": 2}, {"b": 2, "a": 1})


@pytest.mark.parametrize("buf,label,labid,locid,props", VERTICES)
def test_vertex_agrees(buf, label, labid, locid, props):
    from_text = decode.vertex_from_text(buf)
    body = msgspec.json.encode(props)
    from_binary = decode.vertex_from_binary(
        wire.vertex(labid, locid, body), lambda _labid: label
    )
    assert same_element(from_text, from_binary)


@pytest.mark.parametrize("buf,label,ident,start,end,props", EDGES)
def test_edge_agrees(buf, label, ident, start, end, props):
    from_text = decode.edge_from_text(buf)
    body = msgspec.json.encode(props)
    from_binary = decode.edge_from_binary(
        wire.edge(ident[0], ident[1], start, end, body), lambda _labid: label
    )
    assert same_element(from_text, from_binary)


@pytest.mark.parametrize("buf,nvertices,nedges", PATHS)
def test_path_agrees(buf, nvertices, nedges):
    from_text = decode.path_from_text(buf)
    names = {v.id.labid: v.label for v in from_text.vertices}
    names.update({e.id.labid: e.label for e in from_text.edges})
    from_binary = decode.path_from_binary(
        wire.path(
            [
                wire.vertex(v.id.labid, v.id.locid, msgspec.json.encode(v.properties))
                for v in from_text.vertices
            ],
            [
                wire.edge(
                    e.id.labid,
                    e.id.locid,
                    (e.start.labid, e.start.locid),
                    (e.end.labid, e.end.locid),
                    msgspec.json.encode(e.properties),
                )
                for e in from_text.edges
            ],
        ),
        names.get,
    )
    assert same_element(from_text, from_binary)


json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2**53), max_value=2**53),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(max_size=40),
)
json_objects = st.dictionaries(st.text(min_size=1, max_size=12), json_scalars, max_size=6)


@given(
    labid=st.integers(min_value=1, max_value=65535),
    locid=st.integers(min_value=0, max_value=2**40),
    label=st.text(min_size=1, max_size=12).filter(lambda s: "\x00" not in s),
    props=json_objects,
)
@settings(max_examples=300, deadline=None)
def test_generated_vertices_agree(labid, locid, label, props):
    """Generate a value, render it both ways, and require the two readers to agree."""
    body = msgspec.json.encode(props)
    text = label.encode() + f"[{labid}.{locid}]".encode() + body
    from_text = decode.vertex_from_text(text)
    from_binary = decode.vertex_from_binary(
        wire.vertex(labid, locid, body), lambda _labid: label
    )
    assert same_element(from_text, from_binary)
