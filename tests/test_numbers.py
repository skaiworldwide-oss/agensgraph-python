"""How a number in a property map is read.

jsonb keeps an arbitrary-precision decimal and Python's float does not, so the two part company at
some point. These tests say exactly where, against the server rather than from a reading of either
specification.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

import agensgraph
from agensgraph import GraphId, Vertex
from agensgraph.numbers import read_numbers_exactly

pytestmark = pytest.mark.server


@pytest.fixture
def graph(agens):  # type: ignore[no-untyped-def]
    agens.execute("create vlabel doc")
    agens.refresh_labels()
    return agens


@pytest.fixture(autouse=True)
def read_as_floats():  # type: ignore[no-untyped-def]
    """Whatever a test asks for, the next one starts from the default again."""
    yield
    agensgraph.read_numbers_exactly(False)


def stored(graph, literal: str):  # type: ignore[no-untyped-def]
    graph.execute(f"create (:doc {{v: {literal}}})")
    return graph.execute_query("match (n:doc) return n").records[0][0].properties["v"]


class TestWhatSurvivesAsAFloat:
    @pytest.mark.parametrize(
        "literal",
        [
            "12345678901234567890123456789",
            "9007199254740993",
            "1e400",
            "-12345678901234567890123456789",
            "0",
        ],
    )
    def test_an_integer_of_any_size_is_read_exactly(self, graph, literal: str) -> None:  # type: ignore[no-untyped-def]
        """Including one written as an exponent: the server holds 1e400 as an exact integer of four
        hundred and one digits, not as an infinity."""
        value = stored(graph, literal)
        assert isinstance(value, int)
        assert value == int(Decimal(literal))

    def test_a_non_integer_keeps_about_seventeen_digits(self, graph) -> None:  # type: ignore[no-untyped-def]
        value = stored(graph, "3.141592653589793238462643383279")
        assert value == 3.141592653589793
        assert isinstance(value, float)

    def test_a_number_too_small_for_a_float_becomes_zero(self, graph) -> None:  # type: ignore[no-untyped-def]
        assert stored(graph, "1e-400") == 0.0

    def test_an_ordinary_float_is_unharmed(self, graph) -> None:  # type: ignore[no-untyped-def]
        assert stored(graph, "1.5") == 1.5


class TestWhatTheServerItselfChanges:
    def test_a_negative_zero_loses_its_sign_before_the_driver_sees_it(self, graph) -> None:  # type: ignore[no-untyped-def]
        """Not a decoding matter: the server stores it as a positive zero."""
        graph.execute("create (:doc {v: -0.0})")
        printed = graph.execute_query("match (n:doc) return n.v::text").records[0][0]
        assert printed == "0.0"


class TestReadingThemExactly:
    def test_a_long_decimal_keeps_every_digit(self, graph) -> None:  # type: ignore[no-untyped-def]
        agensgraph.read_numbers_exactly()
        assert stored(graph, "3.141592653589793238462643383279") == Decimal(
            "3.141592653589793238462643383279"
        )

    def test_a_number_too_small_for_a_float_survives(self, graph) -> None:  # type: ignore[no-untyped-def]
        agensgraph.read_numbers_exactly()
        assert stored(graph, "1e-400") == Decimal("1e-400")

    def test_an_integer_is_still_an_integer(self, graph) -> None:  # type: ignore[no-untyped-def]
        agensgraph.read_numbers_exactly()
        value = stored(graph, "12345678901234567890")
        assert isinstance(value, int)
        assert value == 12345678901234567890

    def test_it_can_be_turned_back_off(self, graph) -> None:  # type: ignore[no-untyped-def]
        agensgraph.read_numbers_exactly()
        assert agensgraph.reading_numbers_exactly()
        agensgraph.read_numbers_exactly(False)
        assert not agensgraph.reading_numbers_exactly()
        assert isinstance(stored(graph, "1.5"), float)

    def test_it_reaches_the_composite_rendering_too(self, graph) -> None:  # type: ignore[no-untyped-def]
        """Both renderings decode the map through the same reader, so both honour the choice."""
        agensgraph.read_numbers_exactly()
        graph.execute("create (:doc {v: 3.141592653589793238462643383279})")
        result = graph.execute_query("match (n:doc) return n", binary_=True)
        assert result.records[0][0].properties["v"] == Decimal(
            "3.141592653589793238462643383279"
        )

    def test_a_decimal_can_be_written_back(self, graph) -> None:  # type: ignore[no-untyped-def]
        """A decimal renders as a JSON string, so a round trip is not identity -- worth knowing
        rather than discovering."""
        graph.execute("create (:doc %s)", ({"v": Decimal("1.5")},))
        assert (
            graph.execute_query("match (n:doc) return n").records[0][0].properties["v"] == "1.5"
        )


def test_a_map_is_decoded_when_it_is_first_read_not_when_its_row_arrived() -> None:
    """Which is why the setting is meant to be chosen once, at startup.

    The text path holds the property map as the bytes it arrived as and decodes it on first
    access, so what a row holds depends on when it was touched rather than when it was read.
    """
    read_numbers_exactly(False)
    late = Vertex(GraphId(3, 1), "p", b'{"x": 1.5}')
    early = Vertex(GraphId(3, 2), "p", b'{"x": 1.5}')
    assert early.properties["x"] == 1.5
    assert type(early.properties["x"]) is float
    try:
        read_numbers_exactly(True)
        assert type(late.properties["x"]) is Decimal
        assert type(early.properties["x"]) is float
    finally:
        read_numbers_exactly(False)
