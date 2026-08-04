from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from agensgraph import LABID_MAX, LOCID_MAX, GraphId
from agensgraph._protocol import graphid

from .corpus import REJECTED_GRAPHIDS

labids = st.integers(min_value=0, max_value=LABID_MAX)
locids = st.integers(min_value=0, max_value=LOCID_MAX)


def test_parts_round_trip():
    gid = GraphId(3, 1)
    assert gid.labid == 3
    assert gid.locid == 1
    assert str(gid) == "3.1"


@given(labids, locids)
def test_pack_unpack_round_trip(labid, locid):
    gid = GraphId(labid, locid)
    assert graphid.unpack(graphid.pack(gid)) == gid


@given(labids, locids)
def test_text_round_trip(labid, locid):
    gid = GraphId(labid, locid)
    assert graphid.parse_text(str(gid)) == gid


def test_wire_form_is_eight_bytes_big_endian():
    assert graphid.pack(GraphId(3, 1)) == b"\x00\x03\x00\x00\x00\x00\x00\x01"
    assert graphid.unpack(b"\x00\x03\x00\x00\x00\x00\x00\x01") == GraphId(3, 1)


def test_value_is_unsigned():
    """A label id in the top half of the range must not read as a negative number."""
    gid = GraphId(LABID_MAX, LOCID_MAX)
    assert gid.packed == 0xFFFF_FFFF_FFFF_FFFF
    assert graphid.unpack(graphid.pack(gid)) == gid
    assert GraphId(32768, 1).packed > 0


def test_sorts_by_label_then_serial():
    """A label occupies a contiguous range, which is what makes a range test a label test."""
    ids = [GraphId(2, 5), GraphId(1, 9), GraphId(2, 1), GraphId(1, 1)]
    assert sorted(ids) == [GraphId(1, 1), GraphId(1, 9), GraphId(2, 1), GraphId(2, 5)]


def test_usable_as_a_key():
    gid = GraphId(3, 1)
    assert {gid: "x"}[GraphId(3, 1)] == "x"
    assert len({GraphId(3, 1), GraphId(3, 1)}) == 1


def test_immutable():
    gid = GraphId(3, 1)
    with pytest.raises(AttributeError):
        gid.labid = 4  # type: ignore[misc]


def test_comparison_with_other_types_is_not_an_error():
    """A membership test against a mixed list must not raise."""
    gid = GraphId(3, 1)
    assert gid != object()
    assert gid != "3.1"
    assert gid != (3, 1)
    assert gid is not None
    assert gid not in [object(), "3.1", None]


@pytest.mark.parametrize(
    "labid,locid", [(-1, 0), (LABID_MAX + 1, 0), (0, -1), (0, LOCID_MAX + 1)]
)
def test_out_of_range_is_rejected(labid, locid):
    with pytest.raises(ValueError):
        GraphId(labid, locid)


@pytest.mark.parametrize("text", REJECTED_GRAPHIDS)
def test_bad_text_is_rejected(text):
    """Trailing characters are an error, not something to discard.

    A truncated value must not pass as a valid identity, or a misaligned parse produces
    a plausible-looking wrong answer instead of a failure.
    """
    with pytest.raises(ValueError):
        graphid.parse_text(text)


@pytest.mark.parametrize("size", [0, 1, 4, 7, 9, 16])
def test_wrong_wire_length_is_rejected(size):
    with pytest.raises(ValueError):
        graphid.unpack(b"\x00" * size)
