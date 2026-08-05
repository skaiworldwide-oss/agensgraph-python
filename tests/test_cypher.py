"""Quoting an identifier, and refusing a statement the server would misread."""

from __future__ import annotations

import pytest

import agensgraph
from agensgraph import DesiredIndex, Unique
from agensgraph.cypher import (
    check_bindable_positions,
    quote_identifier,
    quote_string,
    without_literals,
)

# Every shape the server accepts and reads as something other than what it says. Each was
# confirmed against a live server: the statement prepares, reports its parameter as jsonb,
# and matches a walk of any length.
MISREAD = [
    "match (a)-[r*1..$1]->(b) return a",
    "match (a)-[r*..$1]->(b) return a",
    "match (a)-[r*$1]->(b) return a",
    "match (a)-[r:knows*1..$1]->(b) return a",
    "match (a)-[*1..$1]->(b) return a",
    "match (a)-[r * 1 .. $2]->(b) return a",
    "MATCH (a)-[r*1..$1]->(b) RETURN a",
    "match (a)-[r*10..$1]->(b) return a",
]

# Shapes with nothing wrong with them, including several written to look as though there is.
FINE = [
    "match (n) return n",
    "match (n) where n.a = $1 return n",
    "match (a)-[r*1..2]->(b) return a",
    "match (a)-[r*1..2]->(b) where $1 > 0 return a",
    "return 3 * 2",
    "match (n) return n limit $1",
    "create (:person $1)",
    "match (n) where id(n) = $1 return n",
    # The shape, but inside something the lexer does not read as syntax.
    "match (n) where n.s = 'r*1..$1' return n",
    "match (n) where n.s = '][*..$9]' return n",
    'match (n:"a*1..$1b") return n',
    "-- match (a)-[r*1..$1]->(b)\nmatch (n) return n",
    "/* [r*1..$1] */ match (n) return n",
    "/* /* [r*1..$1] */ */ match (n) return n",
    "return $$ [r*1..$1] $$",
    "return $tag$ [r*1..$1] $tag$",
]


@pytest.mark.parametrize("statement", MISREAD)
def test_a_parameter_where_a_walk_length_belongs_is_refused(statement: str) -> None:
    with pytest.raises(ValueError, match="length of a variable-length relationship"):
        check_bindable_positions(statement)


@pytest.mark.parametrize("statement", FINE)
def test_everything_else_is_sent(statement: str) -> None:
    check_bindable_positions(statement)


def test_the_refusal_says_what_the_server_would_have_done() -> None:
    """A caller told only that it is invalid would reasonably think the server said so."""
    with pytest.raises(ValueError) as caught:
        check_bindable_positions("match (a)-[r*1..$1]->(b) return a")
    message = str(caught.value)
    assert "property map" in message
    assert "any length" in message


class TestWithoutLiterals:
    @pytest.mark.parametrize(
        ("statement", "expected"),
        [
            ("a 'b' c", "a     c"),
            ('a "b" c', "a     c"),
            ("a 'it''s' c", "a         c"),
            ('a "b""c" d', "a        d"),
            ("a -- b", "a     "),
            ("a /* b */ c", "a         c"),
            ("a /* /* b */ */ c", "a               c"),
            ("a $$b$$ c", "a       c"),
            ("a $t$b$t$ c", "a         c"),
        ],
    )
    def test_what_is_blanked(self, statement: str, expected: str) -> None:
        assert without_literals(statement) == expected

    def test_positions_are_kept(self) -> None:
        """So that anything reported about what is left points at the right place."""
        statement = "match (n) where n.s = 'r*1..$1' return n"
        assert len(without_literals(statement)) == len(statement)

    def test_a_parameter_is_not_mistaken_for_a_dollar_quote(self) -> None:
        assert without_literals("where a = $1 and b = $2") == "where a = $1 and b = $2"

    def test_a_line_comment_keeps_its_newline(self) -> None:
        """Or every line after it would be read as part of it."""
        assert without_literals("-- x\nmatch") == "    \nmatch"

    def test_an_unterminated_string_swallows_the_rest(self) -> None:
        """Which is what the server's own lexer does with it."""
        assert without_literals("match 'unclosed").strip() == "match"


class TestQuoteIdentifier:
    @pytest.mark.parametrize("name", ["person", "_x", "a1", "x_2", "a_very_long_one"])
    def test_a_lower_case_name_is_left_plain(self, name: str) -> None:
        assert quote_identifier(name) == name

    @pytest.mark.parametrize("name", ["Person", "MixedKey", "UPPER", "aB", "eMail"])
    def test_a_name_holding_a_capital_is_quoted(self, name: str) -> None:
        """The lexer lowers an unquoted name, so bare it would reach the server as another."""
        assert quote_identifier(name) == f'"{name}"'

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("my label", '"my label"'),
            ("a,b", '"a,b"'),
            ("a}b{c", '"a}b{c"'),
            ('a"b', '"a""b"'),
            ('a""b', '"a""""b"'),
            ("사람", '"사람"'),
            ("1abc", '"1abc"'),
            ("a-b", '"a-b"'),
            ("", '""'),
        ],
    )
    def test_everything_else_is_quoted(self, name: str, expected: str) -> None:
        assert quote_identifier(name) == expected

    @pytest.mark.parametrize("name", ["match", "MATCH", "Match", "return", "where", "set"])
    def test_a_word_the_grammar_uses_is_quoted(self, name: str) -> None:
        """A label may be called MATCH, and unquoted it would be read as the clause."""
        assert quote_identifier(name) == f'"{name}"'

    def test_a_name_holding_a_null_byte_is_refused(self) -> None:
        """The server's lexer stops at one, so the statement would end early and silently."""
        with pytest.raises(ValueError, match="null byte"):
            quote_identifier("a\x00b")

    def test_a_quote_cannot_escape_the_quoting(self) -> None:
        """Which is the whole point: no name may close its own quoting and add syntax."""
        hostile = 'x" return 1 as "y'
        quoted = quote_identifier(hostile)
        assert quoted.count('"') == 2 + hostile.count('"') * 2
        assert quoted.startswith('"')
        assert quoted.endswith('"')
        assert quoted[1:-1].replace('""', "").count('"') == 0


class TestQuoteString:
    def test_a_quote_is_doubled(self) -> None:
        assert quote_string("it's") == "'it''s'"

    def test_a_plain_string(self) -> None:
        assert quote_string("abc") == "'abc'"

    def test_a_string_holding_a_null_byte_is_refused(self) -> None:
        with pytest.raises(ValueError, match="null byte"):
            quote_string("a\x00b")

    def test_a_quote_cannot_escape_the_quoting(self) -> None:
        quoted = quote_string("' or true --")
        assert quoted == "''' or true --'"


@pytest.mark.server
class TestCasePreservationAgainstAServer:
    """The lexer lowers an unquoted identifier, so what the driver builds has to be quoted."""

    def test_a_label_keeps_the_case_it_was_created_with(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute(f"create vlabel {quote_identifier('MyLabel')}")
        agens.refresh_labels()
        assert "MyLabel" in [label.name for label in agens.labels()]

    def test_two_keys_differing_only_in_case_stay_two(self, agens) -> None:  # type: ignore[no-untyped-def]
        """Written bare they collapse into one, the later value winning."""
        agens.execute("create vlabel doc")
        agens.refresh_labels()
        agens.execute("create (:doc %s)", ({"MixedKey": "a", "mixedkey": "b"},))
        (vertex,) = agens.execute_query("match (n:doc) return n").records[0]
        assert vertex.properties == {"MixedKey": "a", "mixedkey": "b"}

    def test_a_mixed_case_property_reads_back_through_a_built_statement(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel doc")
        agens.refresh_labels()
        agens.execute("create (:doc %s)", ({"MixedKey": "a", "mixedkey": "b"},))
        key = quote_identifier("MixedKey")
        (got,) = agens.execute_query(f"match (n:doc) return n.{key}").records[0]
        assert got == "a"

    def test_a_uniqueness_assertion_on_a_mixed_case_property_is_enforced(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel doc")
        agens.refresh_labels()
        agens.ensure_constraints([Unique("doc", "MixedKey")])
        agens.execute("create (:doc %s)", ({"MixedKey": "one"},))
        with pytest.raises(agensgraph.errors.Error):
            agens.execute("create (:doc %s)", ({"MixedKey": "one"},))

    def test_reconciling_a_mixed_case_name_twice_does_nothing_the_second_time(
        self, agens
    ) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel doc")
        agens.refresh_labels()
        indexes = [DesiredIndex("doc", ("MixedKey",))]
        assert len(agens.ensure_indexes(indexes)) == 1
        assert agens.ensure_indexes(indexes) == []
        constraints = [Unique("doc", "MixedKey")]
        assert len(agens.ensure_constraints(constraints)) == 1
        assert agens.ensure_constraints(constraints) == []

    def test_an_identity_map_reads_the_label_and_key_it_was_given(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute(f"create vlabel {quote_identifier('MyLabel')}")
        agens.refresh_labels()
        agens.execute(f"create (:{quote_identifier('MyLabel')} %s)", ({"Tag": "x"},))
        assert set(agens.identity_map("MyLabel", "Tag")) == {"x"}

    def test_a_channel_holding_a_capital_carries_a_notification(self, agens, dsn: str) -> None:  # type: ignore[no-untyped-def]
        """``listen`` quotes the channel and ``pg_notify`` takes it as a parameter, so the two
        have to agree about its case."""
        seen: list[agensgraph.Notify] = []
        agens.add_notify_handler(seen.append)
        agens.listen("MyChannel")
        with agensgraph.Connection.connect(dsn, autocommit=True) as other:
            other.notify("MyChannel", "hello")
        for _ in range(200):
            agens.execute("select 1")
            if seen:
                break
        assert [(item.channel, item.payload) for item in seen] == [("MyChannel", "hello")]
