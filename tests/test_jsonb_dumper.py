"""Rendering a property map on the way to the server.

Two things are asserted: that what arrives is what was sent, and that a float jsonb cannot hold is
refused rather than turned into a null.
"""

from __future__ import annotations

import json
import math

import pytest

import agensgraph
from agensgraph.adapters import dump_jsonb


class TestWhatItRenders:
    @pytest.mark.parametrize(
        "value",
        [
            {},
            {"a": 1},
            {"a": None},
            {"a": "text", "b": 1, "c": 1.5, "d": True, "e": None},
            {"nested": {"deep": {"deeper": [1, 2, 3]}}},
            {"list": [1, "two", None, {"three": 3}]},
            {"unicode": "안녕 héllo 🎉"},
            {"quotes": 'he said "hi"'},
            {"backslash": "back\\slash"},
            {"newline": "a\nb\tc"},
            {"empty string": ""},
            {"big": 2**62},
            {"negative": -0.0},
            {"exponent": 1e300},
        ],
    )
    def test_it_means_the_same_as_the_standard_library(self, value: object) -> None:
        assert json.loads(dump_jsonb(value)) == json.loads(json.dumps(value))

    def test_a_tuple_becomes_a_list(self) -> None:
        assert json.loads(dump_jsonb({"a": (1, 2)})) == {"a": [1, 2]}

    def test_the_output_is_bytes(self) -> None:
        assert isinstance(dump_jsonb({"a": 1}), bytes)


class TestTheFloatsJsonbCannotHold:
    """These encode as ``null`` unguarded, which stores the wrong value without saying so."""

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_refused_at_the_top(self, bad: float) -> None:
        with pytest.raises(ValueError, match="NaN or an infinity"):
            dump_jsonb({"a": bad})

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_refused_inside_a_list(self, bad: float) -> None:
        with pytest.raises(ValueError, match="NaN or an infinity"):
            dump_jsonb({"a": [1.0, bad, 3.0]})

    def test_refused_deep_inside(self) -> None:
        with pytest.raises(ValueError, match="NaN or an infinity"):
            dump_jsonb({"a": {"b": [{"c": [math.nan]}]}})

    def test_refused_inside_a_set(self) -> None:
        with pytest.raises(ValueError, match="NaN or an infinity"):
            dump_jsonb({"a": {math.inf}})

    def test_a_legitimate_null_is_not_refused(self) -> None:
        """The guard runs when the output holds a null, and a null is usually just a null."""
        assert json.loads(dump_jsonb({"a": None, "b": [None, 1]})) == {
            "a": None,
            "b": [None, 1],
        }

    def test_a_string_that_reads_like_a_null_is_not_refused(self) -> None:
        assert json.loads(dump_jsonb({"a": "null"})) == {"a": "null"}

    def test_a_float_at_the_edge_of_what_is_finite_is_kept(self) -> None:
        biggest = 1.7976931348623157e308
        assert json.loads(dump_jsonb({"a": biggest})) == {"a": biggest}


@pytest.mark.server
class TestRoundTripping:
    @pytest.fixture
    def graph(self, agens):  # type: ignore[no-untyped-def]
        agens.execute("create vlabel doc")
        # The composite rendering names a label from the table, which a label created since it was
        # filled is not in.
        agens.refresh_labels()
        return agens

    @pytest.mark.parametrize(
        "value",
        [
            {"a": 1},
            {"a": "text"},
            {"quotes": 'he said "hi"'},
            {"backslash": "back\\slash"},
            {"unicode": "안녕 héllo 🎉"},
            {"nested": {"deep": [1, {"two": 2}]}},
            {"big": 2**62},
            {"float": 1.5},
            {"null": None},
        ],
    )
    def test_what_arrives_is_what_was_sent(self, graph, value: dict[str, object]) -> None:  # type: ignore[no-untyped-def]
        graph.execute("create (:doc %s)", (value,))
        (vertex,) = graph.execute_query("match (n:doc) return n").records[0]
        # A null property is stripped on the way in unless the server is told otherwise, so it
        # is absent rather than present and null.
        expected = {key: item for key, item in value.items() if item is not None}
        assert vertex.properties == expected

    def test_a_thousand_floats_survive_the_trip(self, graph) -> None:  # type: ignore[no-untyped-def]
        embedding = [i / 7 for i in range(1024)]
        graph.execute("create (:doc %s)", ({"v": embedding},))
        (vertex,) = graph.execute_query("match (n:doc) return n").records[0]
        assert vertex.properties["v"] == embedding

    def test_a_non_finite_float_is_refused_before_it_is_sent(self, graph) -> None:  # type: ignore[no-untyped-def]
        """The server refuses one too, but with the map already on the wire and the transaction
        left needing a rollback."""
        with pytest.raises(ValueError, match="NaN or an infinity"):
            graph.execute("create (:doc %s)", ({"v": [1.0, math.nan]},))
        assert graph.execute_query("match (n:doc) return count(*)").records[0][0] == 0

    def test_the_binary_rendering_carries_the_same_map(self, graph) -> None:  # type: ignore[no-untyped-def]
        value = {"a": 1, "b": "two", "c": [3.5]}
        graph.execute("create (:doc %s)", (value,))
        result = graph.execute_query("match (n:doc) return n", binary_=True)
        assert result.records[0][0].properties == value

    def test_a_wrapped_map_still_takes_its_own_renderer(self, graph) -> None:  # type: ignore[no-untyped-def]
        """psycopg's wrapper may carry a renderer of its own, which is not displaced."""
        wrapped = agensgraph.Jsonb({"a": 1}, dumps=lambda obj: '{"a": 99}')
        graph.execute("create (:doc %s)", (wrapped,))
        (vertex,) = graph.execute_query("match (n:doc) return n").records[0]
        assert vertex.properties == {"a": 99}
