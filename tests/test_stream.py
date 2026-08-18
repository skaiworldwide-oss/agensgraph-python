"""Reading a result a chunk at a time.

`DECLARE ... CURSOR FOR MATCH` is a syntax error, so the only route to a server-side cursor over
Cypher is to put the statement where a subquery goes. That wrap takes only the read-only subset,
which is the constraint everything here is about.
"""

from __future__ import annotations

import psycopg
import pytest
import pytest_asyncio

import agensgraph
from agensgraph.cypher import check_can_wrap, wrap_for_cursor

pytestmark = pytest.mark.server

ROWS = 250


@pytest.fixture
def many(agens):  # type: ignore[no-untyped-def]
    agens.execute("create vlabel thing")
    agens.execute("create elabel links")
    agens.execute(f"unwind range(1, {ROWS})::jsonb as i create (:thing {{n: i}})")
    # The labels were made after the graph was selected, so the table the composite rendering
    # resolves names through has not heard of them yet.
    agens.refresh_labels()
    agens.autocommit = False
    yield agens
    agens.rollback()
    agens.autocommit = True


class TestTheWrap:
    def test_it_adds_an_alias(self) -> None:
        """Without one the server reports that Cypher in a FROM needs an alias."""
        assert (
            wrap_for_cursor("match (n) return n")
            == "select * from (\nmatch (n) return n\n) as t"
        )

    def test_a_trailing_semicolon_is_dropped(self) -> None:
        assert wrap_for_cursor("match (n) return n;").endswith("return n\n) as t")

    @pytest.mark.parametrize(
        "statement",
        [
            "match (n) set n.a = 1 return n",
            "create (:thing)",
            "match (n) detach delete n",
            "match (n) merge (:other) return n",
            "match (n) remove n.a return n",
        ],
    )
    def test_a_statement_that_writes_is_refused_by_name(self, statement: str) -> None:
        with pytest.raises(ValueError, match="cannot be read in chunks"):
            check_can_wrap(statement)

    @pytest.mark.parametrize(
        "statement",
        [
            "match (n) return n",
            "match (n) return n order by n.a",
            "match (n) return n limit 5",
            "match (n) where n.s = 'create' return n",
            "-- create\nmatch (n) return n",
            'match (n:"created") return n',
        ],
    )
    def test_a_statement_that_only_reads_is_allowed(self, statement: str) -> None:
        check_can_wrap(statement)


class TestReadingInChunks:
    def test_every_row_arrives(self, many) -> None:  # type: ignore[no-untyped-def]
        seen = list(many.stream("match (n:thing) return n"))
        assert len(seen) == ROWS

    def test_the_values_are_decoded_as_usual(self, many) -> None:  # type: ignore[no-untyped-def]
        (first,) = next(iter(many.stream("match (n:thing) return n order by n.n")))
        assert isinstance(first, agensgraph.Vertex)
        assert first.properties["n"] == 1

    def test_the_order_asked_for_is_kept(self, many) -> None:  # type: ignore[no-untyped-def]
        got = [
            v.properties["n"] for (v,) in many.stream("match (n:thing) return n order by n.n")
        ]
        assert got == list(range(1, ROWS + 1))

    @pytest.mark.parametrize("size", [1, 7, 100, 1000])
    def test_the_chunk_size_changes_nothing_about_the_answer(self, many, size: int) -> None:  # type: ignore[no-untyped-def]
        assert len(list(many.stream("match (n:thing) return n", size=size))) == ROWS

    def test_a_trailing_limit_works(self, many) -> None:  # type: ignore[no-untyped-def]
        assert len(list(many.stream("match (n:thing) return n limit 10"))) == 10

    def test_a_parameter_works(self, many) -> None:  # type: ignore[no-untyped-def]
        rows = list(many.stream("match (n:thing) where n.n = %s return n", (42,)))
        assert len(rows) == 1

    def test_the_composite_rendering_works(self, many) -> None:  # type: ignore[no-untyped-def]
        rows = list(many.stream("match (n:thing) return n", size=50, binary_=True))
        assert len(rows) == ROWS
        assert rows[0][0].label == "thing"

    def test_stopping_early_leaves_the_connection_usable(self, many) -> None:  # type: ignore[no-untyped-def]
        """Which is what requiring a transaction buys: leaving it closes the cursor with it."""
        for _ in many.stream("match (n:thing) return n"):
            break
        many.rollback()
        assert many.execute_query("match (n:thing) return count(*)").records

    def test_a_write_is_refused_before_anything_is_sent(self, many) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="cannot be read in chunks"):
            next(iter(many.stream("match (n:thing) set n.a = 1 return n")))

    def test_a_mid_query_limit_is_the_servers_to_refuse(self, many) -> None:  # type: ignore[no-untyped-def]
        """Left to the server rather than kept as a second copy of its grammar.

        Which is the point of leaving it there: 2.18 refuses this and 2.17 runs it, so a driver
        holding its own copy of the boundary would be wrong on one of them.
        """
        rows = many.stream("match (n:thing) with n limit 2 return n")
        if many.capabilities.has_gql_clauses():
            with pytest.raises(psycopg.Error):
                next(iter(rows))
        else:
            assert len(list(rows)) == 2

    @pytest.mark.parametrize("size", [0, -1])
    def test_a_chunk_has_to_hold_something(self, many, size: int) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="at least one row"):
            next(iter(many.stream("match (n:thing) return n", size=size)))

    def test_the_shape_the_server_would_misread_is_still_refused(self, many) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="length of a variable-length relationship"):
            next(iter(many.stream("match (a)-[r*1..%s]->(b) return a")))


class TestItReallyStreams:
    def test_the_rows_are_not_all_fetched_at_once(self, many) -> None:  # type: ignore[no-untyped-def]
        """A server-side cursor is the point; buffering everything would defeat it.

        Checked by watching the cursor: after one row has been taken, the server still reports a
        cursor open on this connection, which it would not if the result had been drained.
        """
        rows = many.stream("match (n:thing) return n", size=10, name="watch_me")
        next(iter(rows))
        (open_cursors,) = many.execute(
            "select count(*) from pg_cursors where name = %s", ("watch_me",)
        ).fetchone()
        assert open_cursors == 1
        rows.close()

    def test_and_the_cursor_is_gone_afterwards(self, many) -> None:  # type: ignore[no-untyped-def]
        for _ in many.stream("match (n:thing) return n", size=10, name="gone_after"):
            pass
        (open_cursors,) = many.execute(
            "select count(*) from pg_cursors where name = %s", ("gone_after",)
        ).fetchone()
        assert open_cursors == 0


class TestTheAwaitingInterface:
    @pytest_asyncio.fixture
    async def amany(self, dsn: str):  # type: ignore[no-untyped-def]
        graph = "stream_async"
        conn = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
        async with conn:
            await conn.execute(f'drop graph if exists "{graph}" cascade')
            await conn.execute(f'create graph "{graph}"')
            await conn.graph(graph)
            await conn.execute("create vlabel thing")
            await conn.execute(f"unwind range(1, {ROWS})::jsonb as i create (:thing {{n: i}})")
            await conn.set_autocommit(False)
            try:
                yield conn
            finally:
                await conn.rollback()
                await conn.set_autocommit(True)
                await conn.execute("reset graph_path")
                await conn.execute(f'drop graph "{graph}" cascade')

    @pytest.mark.asyncio
    async def test_every_row_arrives(self, amany) -> None:  # type: ignore[no-untyped-def]
        seen = [row async for row in amany.stream("match (n:thing) return n")]
        assert len(seen) == ROWS

    @pytest.mark.asyncio
    async def test_the_order_asked_for_is_kept(self, amany) -> None:  # type: ignore[no-untyped-def]
        got = [
            v.properties["n"]
            async for (v,) in amany.stream("match (n:thing) return n order by n.n", size=25)
        ]
        assert got == list(range(1, ROWS + 1))

    @pytest.mark.asyncio
    async def test_a_write_is_refused(self, amany) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="cannot be read in chunks"):
            async for _ in amany.stream("match (n:thing) set n.a = 1 return n"):
                pass

    @pytest.mark.asyncio
    async def test_stopping_early_leaves_the_connection_usable(self, amany) -> None:  # type: ignore[no-untyped-def]
        async for _ in amany.stream("match (n:thing) return n"):
            break
        await amany.rollback()
        result = await amany.execute_query("match (n:thing) return count(*)")
        assert result.records


class TestAWordThatIsAlsoAPropertyName:
    """Every write clause is also a legal property name, label and map key."""

    @pytest.mark.parametrize(
        "statement",
        [
            "match (n:doc) return n.set",
            "match (n:doc) return n.create",
            "match (n:doc) where n.delete = 1 return n",
            "match (n:doc) return n.merge, n.remove, n.detach",
            'match (n:doc) return n."set"',
            "match (n:set) return n",
            "match (n) return {set: 1, create: 2}",
            "match (n:doc) return n.setting",
        ],
    )
    def test_a_read_is_not_refused_for_holding_one(self, statement: str) -> None:
        check_can_wrap(statement)

    @pytest.mark.parametrize(
        "statement",
        [
            "match (n:doc) set n.a = 1",
            "create (:doc)",
            "match (n) detach delete n",
            "match (n) remove n.a",
            "merge (n:doc)",
            "MATCH (n) SET n.a = 1",
            "match (n) with n set n.a = 1",
        ],
    )
    def test_a_write_still_is(self, statement: str) -> None:
        with pytest.raises(ValueError, match="cannot be read in chunks"):
            check_can_wrap(statement)


class TestWhatTheWrapItselfTakes:
    """Asserted against the server, since the boundary is the grammar's rather than the driver's."""

    @pytest.mark.server
    @pytest.mark.parametrize(
        "statement",
        [
            "match (n:doc) return n",
            "match (n:doc) return n order by n.a",
            "match (n:doc) return n limit 1",
            "match (n:doc) with n return n",
            "unwind [1,2] as x return x",
            "(match (n:doc) return 1) union (match (n:doc) return 2)",
            "(match (n:doc) return 1) intersect (match (n:doc) return 1)",
            "(match (n:doc) return 1) except (match (n:doc) return 2)",
        ],
    )
    def test_it_is_accepted(self, agens, statement: str) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel doc")
        agens.execute("create (:doc {a: 1})")
        agens.execute(wrap_for_cursor(statement))

    @pytest.mark.server
    @pytest.mark.parametrize(
        "statement",
        [
            "let x = 1 return x",
            "for x in [1,2] return x",
            "call { match (n:doc) return n as v } return v",
            "match (n:doc) finish",
        ],
    )
    def test_a_gql_clause_is_accepted_where_the_grammar_has_one(self, agens, statement) -> None:  # type: ignore[no-untyped-def]
        """These are the clauses 2.18 added, so on an older server the wrap is smaller."""
        agens.execute("create vlabel doc")
        agens.execute("create (:doc {a: 1})")
        if not agens.capabilities.has_gql_clauses():
            with pytest.raises(agensgraph.errors.Error):
                agens.execute(wrap_for_cursor(statement))
            return
        agens.execute(wrap_for_cursor(statement))

    @pytest.mark.server
    @pytest.mark.parametrize(
        "statement",
        [
            "match (n:doc) filter n.a > 0 return n",
            "call generate_series(1,2) yield generate_series as g return g",
        ],
    )
    def test_it_is_refused_by_the_server(self, agens, statement: str) -> None:  # type: ignore[no-untyped-def]
        """None of these is a write, so the driver lets them through and the server judges."""
        agens.execute("create vlabel doc")
        with pytest.raises(agensgraph.errors.Error):
            agens.execute(wrap_for_cursor(statement))

    @pytest.mark.server
    def test_a_limit_that_is_not_last_is_refused_only_where_it_is(self, agens) -> None:  # type: ignore[no-untyped-def]
        """2.18 refuses a `LIMIT` the wrap cannot carry; 2.17 takes the same statement.

        So the boundary belongs to the server's grammar and not to the driver, which is why
        nothing here tries to predict it -- the statement goes as written and the server judges.
        """
        agens.execute("create vlabel doc")
        statement = wrap_for_cursor("match (n:doc) with n limit 1 return n")
        if agens.capabilities.has_gql_clauses():
            with pytest.raises(agensgraph.errors.Error):
                agens.execute(statement)
        else:
            agens.execute(statement)


class TestTheWrapAroundAwkwardStatements:
    def test_a_statement_ending_in_a_line_comment_keeps_its_bracket(self) -> None:
        """On one line the closing bracket lands inside the comment."""
        wrapped = wrap_for_cursor("match (n:t) return n -- why\n")
        assert wrapped.endswith("\n) as t")
        assert "-- why" in wrapped

    def test_a_trailing_semicolon_is_taken_off(self) -> None:
        assert (
            wrap_for_cursor("match (n) return n;")
            == "select * from (\nmatch (n) return n\n) as t"
        )

    @pytest.mark.parametrize(
        "statement",
        [
            "match (n) where n.s = 'a;b' return n",
            "match (n) return n -- ends with ;\n",
            "match (n) return n /* ; */",
        ],
    )
    def test_a_semicolon_that_terminates_nothing_is_left_alone(self, statement: str) -> None:
        assert statement.rstrip() in wrap_for_cursor(statement)

    def test_insert_is_refused_as_create_is(self) -> None:
        """It is the same clause under another name, so it writes."""
        with pytest.raises(ValueError, match="cannot be read in chunks"):
            check_can_wrap("insert (:t {x: 1}) return 1")

    @pytest.mark.parametrize(
        "statement",
        [
            "match (n) return n.inserted",
            "match (n:insert) return n",
            "match (n) return n AS inserter",
        ],
    )
    def test_and_a_name_that_merely_holds_the_word_is_not(self, statement: str) -> None:
        check_can_wrap(statement)

    @pytest.mark.parametrize(
        "statement",
        [
            "match (n:doc) return n.name as create",
            "match (n:doc) return n.a as set, n.b as delete",
            "match (n:doc) return 1 as merge",
        ],
    )
    def test_a_word_naming_a_column_is_not_the_clause_it_spells(self, statement: str) -> None:
        """``AS`` puts a name next, and the server takes a reserved one there."""
        check_can_wrap(statement)

    @pytest.mark.parametrize(
        "statement",
        [
            "match (n:doc) with n.a as alias create (:made {v: alias})",
            "match (n:doc) with n.a as has create (:made {v: has})",
            "match (n:doc) with n.a as was delete n",
            "match (n:doc) with n.a as _as set n.b = 1",
        ],
    )
    def test_a_word_merely_ending_in_as_does_not_make_the_next_one_a_name(
        self, statement: str
    ) -> None:
        """The word before is read too, so the name is ``alias`` and what follows is the clause."""
        with pytest.raises(ValueError, match="cannot be read in chunks"):
            check_can_wrap(statement)

    @pytest.mark.parametrize(
        "statement",
        [
            "match (n:doc) return n.a as\n  create",
            "match (n:doc) return n.a as\tcreate",
            "match (n:doc) return n.a AS create",
        ],
    )
    def test_the_name_is_found_however_it_is_spaced_from_as(self, statement: str) -> None:
        check_can_wrap(statement)

    @pytest.mark.server
    def test_the_server_agrees_the_alias_is_a_read(self, agens) -> None:  # type: ignore[no-untyped-def]
        """Asserted against the server, so the driver is not alone in calling it one."""
        agens.execute("create vlabel doc")
        agens.execute("create (:doc {name: 'a'})")
        (row,) = agens.execute(
            wrap_for_cursor("match (n:doc) return n.name as create")
        ).fetchall()
        assert row[0] == "a"

    @pytest.mark.server
    def test_and_agrees_the_one_after_a_word_ending_in_as_writes(self, agens) -> None:  # type: ignore[no-untyped-def]
        """Which is what the refusal is protecting: it would take the wrap and write a row."""
        agens.execute("create vlabel doc")
        agens.execute("create vlabel made")
        agens.execute("create (:doc {a: 1})")
        agens.execute("match (n:doc) with n.a as alias create (:made {v: alias})")
        (count,) = agens.execute("match (m:made) return count(*)").fetchone()
        assert count == 1


@pytest.mark.server
class TestWhatAStreamNeeds:
    def test_autocommit_is_refused_by_name(self, dsn: str) -> None:
        from agensgraph.errors import NoEnclosingTransaction

        with (
            agensgraph.Connection.connect(dsn, autocommit=True) as conn,
            pytest.raises(NoEnclosingTransaction, match="transaction"),
        ):
            list(conn.stream("match (n) return n"))

    def test_what_is_wrong_with_the_statement_is_said_first(self, dsn: str) -> None:
        """A caller with both faults is not sent round twice."""
        with (
            agensgraph.Connection.connect(dsn, autocommit=True) as conn,
            pytest.raises(ValueError, match="cannot be read in chunks"),
        ):
            list(conn.stream("insert (:t {x: 1}) return 1"))

    def test_two_streams_at_once_do_not_collide(self, agens, dsn: str) -> None:  # type: ignore[no-untyped-def]
        """One name for both aborts the transaction, which takes down the reader as well."""
        agens.execute("create vlabel t")
        agens.execute("create (:t {x: 1}), (:t {x: 2})")
        with agensgraph.Connection.connect(dsn) as conn:
            conn.execute(f"set graph_path = {agens.label_table.graph}")
            first = conn.stream("match (n:t) return n.x order by n.x")
            second = conn.stream("match (n:t) return n.x order by n.x desc")
            assert next(iter(first)) == (1,)
            assert next(iter(second)) == (2,)
            conn.rollback()

    def test_a_statement_ending_in_a_comment_streams(self, agens, dsn: str) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel t")
        agens.execute("create (:t {x: 1}), (:t {x: 2})")
        with agensgraph.Connection.connect(dsn) as conn:
            conn.execute(f"set graph_path = {agens.label_table.graph}")
            rows = list(conn.stream("match (n:t) return n.x order by n.x -- why\n"))
            assert [row[0] for row in rows] == [1, 2]
            conn.rollback()
