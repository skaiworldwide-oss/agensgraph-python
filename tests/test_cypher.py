"""Quoting an identifier, and refusing a statement the server would misread."""

from __future__ import annotations

import contextlib

import psycopg
import pytest

import agensgraph
from agensgraph import DesiredIndex, Unique
from agensgraph.cypher import (
    check_bindable_positions,
    check_single_statement,
    quote_identifier,
    quote_string,
    without_literals,
    writable_counters,
)

# Every shape the server accepts and reads as something other than what it says. Each was
# confirmed against a live server: the statement prepares, reports its parameter as jsonb,
# and matches a walk of any length.
MISREAD = [
    "match (a)-[r*1..$1]->(b) return a",
    "match (a)<-[r*1..$1]-(b) return a",
    'match (a)-[r:"my label"*1..$1]->(b) return a',
    "match (a)-[r*1..%s]->(b) return a",
    "match (a)-[r*1..%(bound)s]->(b) return a",
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
    # A star that is multiplication or a count, which the server runs and answers.
    "return 2 * $1",
    "match (n) return n.price * $1",
    "match (n) where n.qty * $1 > 10 return n",
    "match (n) return count(*) * $1",
    "match (n) where n.tags[0] * $1 > 1 return n",
    "return [2 * $1]",
    "match (a)-[r*1..2]->(b) return count(*) * $1",
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


class TestTheRefusalReachesEveryWayOfRunningAStatement:
    """The shape is as reachable through a cursor or through bytes as through the one method
    a caller was shown first, so the check lives on the cursor every route builds."""

    BAD = "match (a)-[r*1..%s]->(b) return a"

    def test_every_route_refuses_it(self, agens) -> None:  # type: ignore[no-untyped-def]
        from psycopg import sql

        routes = [
            lambda: agens.execute(self.BAD, (1,)),
            lambda: agens.cursor().execute(self.BAD, (1,)),
            lambda: agens.cursor().executemany(self.BAD, [(1,)]),
            lambda: agens.execute_query(self.BAD, (1,)),
            lambda: agens.execute_query(self.BAD.encode(), (1,)),
            lambda: agens.execute(self.BAD.encode(), (1,)),
            lambda: agens.execute(sql.SQL(self.BAD), (1,)),  # type: ignore[arg-type]
            lambda: agens.execute(
                sql.SQL("match (a)-[r*1..{}]->(b) return a").format(sql.SQL("%s")), (1,)
            ),
        ]
        for run in routes:
            with pytest.raises(ValueError, match="length of a variable-length relationship"):
                run()

    def test_a_bound_written_into_the_statement_is_not_refused(self, agens) -> None:  # type: ignore[no-untyped-def]
        """Which is what the refusal tells a caller to do, so it has to be allowed."""
        from psycopg import sql

        agens.execute("match (a)-[r*1..2]->(b) return a")
        agens.execute(sql.SQL("match (a)-[r*1..{}]->(b) return a").format(sql.Literal(2)))


@pytest.mark.server
class TestMultiplicationByAParameter:
    """Every one of these was refused as a walk length, and the server runs them all."""

    @pytest.mark.parametrize(
        ("statement", "expected"),
        [
            ("return 2 * %s", 6),
            ("match (n:m) return n.price * %s", 30),
            ("match (n:m) return count(*) * %s", 3),
            ("match (n:m) where n.tags[0] * %s > 1 return n.qty", 4),
        ],
    )
    def test_the_server_answers_it(self, agens, statement: str, expected: int) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel m")
        agens.refresh_labels()
        agens.execute("create (:m %s)", ({"price": 10, "qty": 4, "tags": [7]},))
        (got,) = agens.execute_query(statement, (3,)).records[0]
        assert got == expected


class TestMoreThanOneStatement:
    """Sent with no parameters, a string of statements runs all of them and reports the first.

    So a read with a write after a semicolon runs the write and looks like the read, which is why
    this one is read from the text: the server does not refuse it, because running the whole string
    is what the simple query protocol is for.
    """

    @pytest.mark.parametrize(
        "statement",
        [
            "select 1; create table t(i int)",
            "match (n) return n; create (:x)",
            "select 1;;",
            "-- a comment\nselect 1; drop table t",
        ],
    )
    def test_a_second_statement_is_refused(self, statement: str) -> None:
        with pytest.raises(ValueError, match="more than one statement"):
            check_single_statement(statement)

    @pytest.mark.parametrize(
        "statement",
        [
            "match (n) return n",
            "match (n) return n;",
            "match (n) return n;   \n",
            "match (n) where n.s = 'a;b' return n",
            "match (n) return n -- trailing ; here",
            "match (n) return n /* ; */",
            "match (n) where n.s = $$a;b$$ return n",
        ],
    )
    def test_one_statement_is_sent(self, statement: str) -> None:
        """A terminator is not a second statement, and a semicolon in a literal separates nothing."""
        check_single_statement(statement)

    def test_the_refusal_shows_what_came_after(self) -> None:
        """A caller has to be able to see which part of its own text is the reason."""
        with pytest.raises(ValueError) as caught:
            check_single_statement("match (n) return n; create (:sneaked)")
        assert "create (:sneaked)" in str(caught.value)


@pytest.mark.server
class TestWhatTheServerDoesWithMoreThanOne:
    """The two halves of the argument above, each asserted rather than assumed."""

    def test_without_parameters_it_runs_them_all(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create temp table smuggled(i int)")
        agens.execute("select 1; insert into smuggled values (7)")
        (count,) = agens.execute("select count(*) from smuggled").fetchone()
        assert count == 1, "the statement after the semicolon ran"

    def test_binding_a_parameter_is_what_refuses_it(self, agens) -> None:  # type: ignore[no-untyped-def]
        """Which is the other half of the answer, and the half that holds if the reading is wrong."""
        agens.execute("create temp table bound(i int)")
        with pytest.raises(psycopg.Error) as caught:
            agens.execute("select %s; insert into bound values (8)", ("x",))
        assert caught.value.sqlstate == "42601"
        (count,) = agens.execute("select count(*) from bound").fetchone()
        assert count == 0


class TestAnEscapeStringEndsWhereTheServerEndsIt:
    """``E'...'`` honours a backslash escape whatever ``standard_conforming_strings`` says.

    Read as though it did not, the quote a backslash escaped is taken for a closing one and the
    next for an opening one, so everything after the string is swallowed -- and a second statement
    hiding there is neither refused nor counted.
    """

    @pytest.mark.parametrize(
        ("statement", "refused"),
        [
            (r"SELECT E'a\''; CREATE TABLE t(i int)", True),
            (r"SELECT 'a'''; CREATE TABLE t(i int)", True),
            (r"SELECT Ename'x'; CREATE TABLE t(i int)", True),
            (r"MATCH (n) WHERE n.s = E'x\'' RETURN n", False),
            (r"MATCH (n) WHERE n.s = 'a\b' RETURN n", False),
            (r"MATCH (n) WHERE n.s = E'a\\' RETURN n", False),
        ],
    )
    def test_a_second_statement_after_one_is_still_found(
        self, statement: str, refused: bool
    ) -> None:
        if refused:
            with pytest.raises(ValueError, match="more than one statement"):
                check_single_statement(statement)
        else:
            check_single_statement(statement)

    def test_a_write_hiding_after_one_still_counts_as_a_write(self) -> None:
        assert writable_counters(r"SELECT E'a\''; CREATE TABLE t(i int)")

    @pytest.mark.server
    @pytest.mark.parametrize(
        ("statement", "expected"),
        [(r"SELECT E'a\''", "a'"), (r"SELECT 'a\b'", r"a\b")],
    )
    def test_the_server_reads_them_the_same_way(self, agens, statement, expected) -> None:  # type: ignore[no-untyped-def]
        """Asserted against the server, since the whole point is agreeing with its lexer."""
        (value,) = agens.execute(statement).fetchone()
        assert value == expected


class TestWhichSettingTheServerIsOnIsNotKnownHere:
    """A plain string escapes too where ``standard_conforming_strings`` is off.

    Each setting hides a second statement the other does not, so reading the text one way leaves
    the other way's open. Both readings are taken and either one finding a second statement is
    enough, which is an answer that does not depend on a setting the caller may not control.
    """

    ON_HIDES_IT = r"SELECT 'a\'; CREATE TABLE t(i int)"
    OFF_HIDES_IT = r"SELECT 'a\''; CREATE TABLE t(i int)"

    @pytest.mark.parametrize("statement", [ON_HIDES_IT, OFF_HIDES_IT])
    def test_a_second_statement_either_setting_would_run_is_refused(
        self, statement: str
    ) -> None:
        with pytest.raises(ValueError, match="more than one statement"):
            check_single_statement(statement)

    @pytest.mark.parametrize(
        "statement",
        [
            r"SELECT 'a\b'",
            r"SELECT 'C:\path\'",
            r"SELECT 'a;b'",
            r"MATCH (n) WHERE n.s = 'a\b' RETURN n",
            r"SELECT E'a;b\''",
        ],
    )
    def test_and_one_statement_is_still_one_under_either(self, statement: str) -> None:
        """Reading twice must not cost a caller whose text was never more than one statement."""
        check_single_statement(statement)

    @pytest.mark.server
    @pytest.mark.parametrize("setting", ["on", "off"])
    def test_the_server_runs_the_one_its_setting_hides(self, agens, setting: str) -> None:  # type: ignore[no-untyped-def]
        """So neither is hypothetical: each setting leaves the table behind for its own statement."""
        hidden = self.ON_HIDES_IT if setting == "on" else self.OFF_HIDES_IT
        agens.execute(f"set standard_conforming_strings = {setting}")
        try:
            agens.execute("drop table if exists smuggled_here")
            with contextlib.suppress(psycopg.Error):
                agens.execute(hidden.replace("t(i int)", "smuggled_here(i int)"))
            (made,) = agens.execute(
                "select to_regclass('smuggled_here') is not null"
            ).fetchone()
        finally:
            agens.execute("drop table if exists smuggled_here")
            agens.execute("reset standard_conforming_strings")
        assert made, "the server ran it, so the driver refusing it is not a false positive"
