"""What a string parameter means, and what it costs to say so.

The driver declares a string to be text rather than leaving its type to be worked out. That
reaches the server's own text-to-jsonb conversion, which keeps a string a string, instead of
jsonb's input function, which parses it. The difference is not cosmetic: left to be worked
out, a search for a value that happens to read as JSON matched nothing and raised nothing.

Both halves are tested here -- the values that now work, and the SQL positions that now want
a cast -- because the second is the price of the first and a change to either should fail
something.
"""

from __future__ import annotations

import datetime
import decimal
import uuid

import psycopg
import pytest

import agensgraph
from agensgraph.errors import STRING_TYPE_HINT

pytestmark = pytest.mark.server

# Every one of these was accepted and matched nothing before, except the last two, which
# raised. None of them is exotic: an order number, a version, a flag, a serialised payload.
USED_TO_BE_WRONG = [
    "123",
    "0",
    "-1",
    "1.5",
    "1e5",
    "null",
    "true",
    "false",
    "[]",
    "[1,2]",
    '{"a":1}',
    '"quoted"',
    "Arthur",
    'he said "hi"',
]

ALSO_AWKWARD = [
    "back\\slash",
    "",
    " leading and trailing ",
    "line\nbreak",
    "tab\there",
    "ünïcødé",
    "사람",
    "x" * 5000,
    "%s",
    "$1",
    "'; drop graph g; --",
]


@pytest.fixture
def graph(agens):  # type: ignore[no-untyped-def]
    agens.execute("create vlabel thing")
    return agens


def store_and_find(conn, value: str) -> list[object]:  # type: ignore[no-untyped-def]
    """Store a property whose value really is *value*, then look it up by that value."""
    conn.execute("create (:thing {v: %s})", (value,))
    return conn.execute_query("match (n:thing) where n.v = %s return n", (value,)).records


@pytest.mark.parametrize("value", USED_TO_BE_WRONG)
def test_a_value_that_reads_as_json_is_still_a_string(graph, value: str) -> None:  # type: ignore[no-untyped-def]
    assert len(store_and_find(graph, value)) == 1


@pytest.mark.parametrize("value", ALSO_AWKWARD)
def test_a_value_that_is_awkward_for_other_reasons(graph, value: str) -> None:  # type: ignore[no-untyped-def]
    assert len(store_and_find(graph, value)) == 1


@pytest.mark.parametrize("value", [*USED_TO_BE_WRONG, *ALSO_AWKWARD])
def test_a_value_survives_the_round_trip_unchanged(graph, value: str) -> None:  # type: ignore[no-untyped-def]
    """Not merely found, but the same characters coming back."""
    graph.execute("create (:thing {v: %s})", (value,))
    (stored,) = graph.execute_query(
        "match (n:thing) where n.v = %s return n.v", (value,)
    ).records[0]
    assert stored == value


def test_a_number_is_still_a_number(graph) -> None:  # type: ignore[no-untyped-def]
    """Declaring strings text must not have turned every value into one."""
    graph.execute("create (:thing {v: %s})", (42,))
    (stored,) = graph.execute_query("match (n:thing) where n.v = %s return n.v", (42,)).records[
        0
    ]
    assert stored == 42
    assert isinstance(stored, int)


def test_the_string_and_the_number_are_told_apart(graph) -> None:  # type: ignore[no-untyped-def]
    """The whole point: '123' and 123 are different values and must not match each other."""
    graph.execute("create (:thing {v: %s})", ("123",))
    graph.execute("create (:thing {v: %s})", (123,))
    as_string = graph.execute_query(
        "match (n:thing) where n.v = %s return n.v", ("123",)
    ).records
    as_number = graph.execute_query("match (n:thing) where n.v = %s return n.v", (123,)).records
    assert [r[0] for r in as_string] == ["123"]
    assert [r[0] for r in as_number] == [123]


def test_a_mapping_needs_no_wrapping(graph) -> None:  # type: ignore[no-untyped-def]
    graph.execute("create (:thing %s)", ({"v": "in a map"},))
    assert len(graph.execute_query("match (n:thing {v: 'in a map'}) return n").records) == 1


def test_a_wrapped_string_still_works(graph) -> None:  # type: ignore[no-untyped-def]
    """So that anyone who wrapped one before does not have to stop."""
    graph.execute("create (:thing {v: 'wrapped'})")
    result = graph.execute_query(
        "match (n:thing) where n.v = %s return n", (agensgraph.Jsonb("wrapped"),)
    )
    assert len(result.records) == 1


class TestTheCostInSql:
    """A string standing for another type wants a cast now. That is the whole price."""

    @pytest.fixture
    def table(self, agens):  # type: ignore[no-untyped-def]
        agens.execute("drop table if exists params")
        agens.execute(
            "create table params(name text, code varchar(10), tag name, d date,"
            " ts timestamptz, u uuid, n integer, amt numeric, b bytea)"
        )
        agens.execute(
            "insert into params values ('Arthur', 'A1', 't1', '2026-08-05',"
            " '2026-08-05 12:00+00', '11111111-1111-1111-1111-111111111111', 42, 9.5, '\\x00')"
        )
        yield agens
        agens.execute("drop table params")

    @pytest.mark.parametrize(
        ("column", "value"),
        [
            ("d", "2026-08-05"),
            ("u", "11111111-1111-1111-1111-111111111111"),
            ("n", "42"),
            ("amt", "9.5"),
        ],
    )
    def test_a_string_for_another_type_is_refused(self, table, column: str, value: str) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(psycopg.Error) as caught:
            table.execute_query(f"select 1 from params where {column} = %s", (value,))
        assert "text" in str(caught.value)

    def test_and_the_refusal_says_why_the_parameter_was_text(self, table) -> None:  # type: ignore[no-untyped-def]
        """The server's own message cannot know; the driver adds the one thing it does."""
        with pytest.raises(psycopg.Error) as caught:
            table.execute_query("select 1 from params where d = %s", ("2026-08-05",))
        notes = getattr(caught.value, "__notes__", [])
        assert any("Unspecified" in note for note in notes)
        assert any("%s::date" in note for note in notes)

    @pytest.mark.parametrize(
        ("column", "cast", "value"),
        [
            ("d", "date", "2026-08-05"),
            ("u", "uuid", "11111111-1111-1111-1111-111111111111"),
            ("n", "int", "42"),
            ("amt", "numeric", "9.5"),
        ],
    )
    def test_a_cast_fixes_it(self, table, column: str, cast: str, value: str) -> None:  # type: ignore[no-untyped-def]
        rows = table.execute(
            f"select 1 from params where {column} = %s::{cast}", (value,)
        ).fetchall()
        assert len(rows) == 1

    @pytest.mark.parametrize(
        ("column", "value"),
        [
            ("d", datetime.date(2026, 8, 5)),
            ("u", uuid.UUID("11111111-1111-1111-1111-111111111111")),
            ("n", 42),
            ("amt", decimal.Decimal("9.5")),
            ("b", b"\x00"),
        ],
    )
    def test_passing_the_values_own_type_is_untouched(
        self, table, column: str, value: object
    ) -> None:  # type: ignore[no-untyped-def]
        """Which is the ordinary, correct thing to do, and is what most callers already do."""
        rows = table.execute(f"select 1 from params where {column} = %s", (value,)).fetchall()
        assert len(rows) == 1

    @pytest.mark.parametrize("column", ["name", "code", "tag"])
    def test_a_text_like_column_is_untouched(self, table, column: str) -> None:  # type: ignore[no-untyped-def]
        value = {"name": "Arthur", "code": "A1", "tag": "t1"}[column]
        rows = table.execute(f"select 1 from params where {column} = %s", (value,)).fetchall()
        assert len(rows) == 1

    def test_the_usual_string_work_is_untouched(self, table) -> None:  # type: ignore[no-untyped-def]
        assert table.execute("select 1 from params where name like %s", ("Art%",)).fetchall()
        assert table.execute(
            "select 1 from pg_type where typname = any(%s)", (["vertex", "edge"],)
        ).fetchall()

    def test_a_position_with_nothing_to_infer_from_now_works(self, table) -> None:  # type: ignore[no-untyped-def]
        """An unspecified parameter here could not be resolved at all before."""
        assert table.execute("select concat(%s, %s)", ("a", "b")).fetchone() == ("ab",)


class TestUnspecified:
    """The escape hatch, for a string that should have its type worked out as before."""

    def test_it_restores_the_inference(self, agens) -> None:  # type: ignore[no-untyped-def]
        rows = agens.execute(
            "select 1 where current_date = %s",
            (agensgraph.Unspecified("2026-08-05"),),
        ).fetchall()
        assert rows in ([], [(1,)])  # depends on today's date; the point is it did not raise

    def test_it_is_still_a_string_everywhere_else(self) -> None:
        value = agensgraph.Unspecified("abc")
        assert value == "abc"
        assert value.upper() == "ABC"
        assert "Unspecified(" in repr(value)

    def test_it_carries_no_dictionary(self) -> None:
        with pytest.raises(AttributeError):
            agensgraph.Unspecified("x").anything = 1  # type: ignore[attr-defined]


def test_the_hint_is_not_added_to_an_unrelated_failure(agens) -> None:  # type: ignore[no-untyped-def]
    """Or every syntax error would carry advice about string types."""
    with pytest.raises(psycopg.Error) as caught:
        agens.execute_query("match (n) return")
    assert STRING_TYPE_HINT not in getattr(caught.value, "__notes__", [])
