"""Quoting an identifier, and refusing a statement the server would misread."""

from __future__ import annotations

import pytest

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
    @pytest.mark.parametrize("name", ["person", "Person", "_x", "a1", "x_2"])
    def test_a_plain_name_is_left_plain(self, name: str) -> None:
        assert quote_identifier(name) == name

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
