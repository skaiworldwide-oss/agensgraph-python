"""Reporting what a write changed, and declining to report what it did not.

The numbers in these tests are the ones a live server produced, recorded in one session:
a ``CREATE`` of two vertices, then a ``SET ... RETURN`` of two properties which reports the
earlier statement's two insertions alongside its own two updates.
"""

from __future__ import annotations

import pytest

from agensgraph.summary import COUNTER_COLUMNS, GraphWriteCounts


def test_the_columns_are_in_the_order_the_server_returns_them() -> None:
    assert COUNTER_COLUMNS == (
        "insertedvertices",
        "insertededges",
        "deletedvertices",
        "deletededges",
        "updatedproperties",
    )


class TestExact:
    """A write with no ``RETURN`` zeroes all five before it runs, so all five are its own."""

    def test_all_five_are_reported(self) -> None:
        counts = GraphWriteCounts.exact([2, 1, 0, 0, 0])
        assert counts.inserted_vertices == 2
        assert counts.inserted_edges == 1
        assert counts.deleted_vertices == 0
        assert counts.complete
        assert counts.total == 3

    def test_a_write_that_changed_nothing_reports_zeros_and_not_nothing(self) -> None:
        """Which is a different statement from one that was never asked."""
        counts = GraphWriteCounts.exact([0, 0, 0, 0, 0])
        assert counts.complete
        assert counts.total == 0

    @pytest.mark.parametrize("wrong", [[], [1], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]])
    def test_the_wrong_number_of_counters_is_refused(self, wrong: list[int]) -> None:
        with pytest.raises(ValueError, match="five"):
            GraphWriteCounts.exact(wrong)


class TestBetween:
    """A write that returned rows zeroes only what its own clauses touch."""

    def test_a_counter_the_statement_inherited_is_not_reported(self) -> None:
        """The two insertions belong to an earlier statement, not to this one."""
        counts = GraphWriteCounts.between([2, 0, 0, 0, 0], [2, 0, 0, 0, 2])
        assert counts.inserted_vertices is None
        assert counts.updated_properties == 2
        assert not counts.complete

    def test_a_counter_that_changed_is_reported(self) -> None:
        counts = GraphWriteCounts.between([1, 0, 0, 0, 0], [1, 0, 1, 0, 0])
        assert counts.deleted_vertices == 1
        assert counts.inserted_vertices is None

    def test_a_counter_that_was_already_zero_is_reported(self) -> None:
        """Zeroed and counted as zero, or never zeroed and already zero: either way, zero."""
        counts = GraphWriteCounts.between([0, 0, 0, 0, 0], [0, 0, 0, 0, 3])
        assert counts.complete
        assert counts.inserted_vertices == 0
        assert counts.updated_properties == 3

    def test_a_counter_that_went_down_is_reported(self) -> None:
        """Only the statement could have lowered it, by zeroing it and counting less."""
        counts = GraphWriteCounts.between([5, 0, 0, 0, 0], [1, 0, 0, 0, 0])
        assert counts.inserted_vertices == 1

    def test_an_unreported_counter_makes_the_total_unreportable(self) -> None:
        """Rather than a sum missing one of its terms, which would read as a smaller total."""
        counts = GraphWriteCounts.between([2, 0, 0, 0, 0], [2, 0, 0, 0, 2])
        assert counts.total is None

    @pytest.mark.parametrize(
        ("before", "after"), [([1], [1, 0, 0, 0, 0]), ([1, 0, 0, 0, 0], [1, 0])]
    )
    def test_the_wrong_number_of_counters_is_refused(
        self, before: list[int], after: list[int]
    ) -> None:
        with pytest.raises(ValueError, match="five"):
            GraphWriteCounts.between(before, after)


class TestUnknown:
    def test_nothing_read_is_nothing_reported(self) -> None:
        counts = GraphWriteCounts.unknown()
        assert not counts.complete
        assert counts.total is None
        assert all(value is None for value in counts)

    def test_it_is_not_zeros(self) -> None:
        """A statement that changed nothing and a statement never asked about are different."""
        assert GraphWriteCounts.unknown() != GraphWriteCounts.exact([0, 0, 0, 0, 0])


class TestWhichCountersAStatementCouldMove:
    """The server zeroes a counter only for a clause that can write it."""

    @pytest.mark.parametrize(
        ("statement", "expected"),
        [
            ("create (:t)", {0, 1}),
            ("insert (:t)", {0, 1}),
            ("merge (:t {a: 1})", {0, 1, 4}),
            ("match (n) delete n", {2, 3}),
            ("match (n) detach delete n", {2, 3}),
            ("match (n) set n.a = 1", {4}),
            ("match (n) remove n.a", {4}),
            ("match (n) return n", set()),
            ("select 1", set()),
            ("match (n) where n.s = 'create' return n", set()),
            ("-- create (:t)\nmatch (n) return n", set()),
        ],
    )
    def test_the_clauses_are_read_from_the_statement(
        self, statement: str, expected: set[int]
    ) -> None:
        from agensgraph.cypher import writable_counters

        assert writable_counters(statement) == expected


class TestForStatement:
    """A counter no clause of the statement can write is nought for that statement."""

    def test_a_counter_no_clause_names_is_nought(self) -> None:
        counts = GraphWriteCounts.for_statement([9, 9, 9, 9, 9], [9, 9, 9, 9, 9], {0, 1})
        assert counts.deleted_vertices == 0
        assert counts.deleted_edges == 0
        assert counts.updated_properties == 0

    def test_a_counter_that_moved_is_reported_whatever_the_clauses_say(self) -> None:
        """A statement can write without naming a clause, by calling something that does, and
        nothing else ran between the two readings."""
        counts = GraphWriteCounts.for_statement([0, 0, 0, 0, 0], [1, 0, 0, 0, 0], set())
        assert counts.inserted_vertices == 1

    def test_a_counter_a_clause_names_is_read_as_between_reads_it(self) -> None:
        counts = GraphWriteCounts.for_statement([9, 0, 0, 0, 0], [4, 0, 0, 0, 0], {0, 1})
        assert counts.inserted_vertices == 4
        counts = GraphWriteCounts.for_statement([9, 0, 0, 0, 0], [9, 0, 0, 0, 0], {0, 1})
        assert counts.inserted_vertices is None

    def test_a_statement_with_no_write_clause_reports_five_zeros(self) -> None:
        counts = GraphWriteCounts.for_statement([2, 3, 4, 5, 6], [2, 3, 4, 5, 6], set())
        assert tuple(counts) == (0, 0, 0, 0, 0)
        assert counts.total == 0

    def test_five_counters_each_are_required(self) -> None:
        with pytest.raises(ValueError, match="five counters each"):
            GraphWriteCounts.for_statement([0], [0, 0, 0, 0, 0], set())


@pytest.mark.server
class TestCountersAgainstAServer:
    """The counters live on the session, so what a statement is credited with matters."""

    def test_a_statement_that_wrote_no_graph_elements_reports_none_of_them(self, agens) -> None:  # type: ignore[no-untyped-def]
        """An ordinary SQL update reports its command as an update, exactly as a graph write
        does, and its row count has nothing to do with these counters."""
        agens.execute("create vlabel t")
        agens.refresh_labels()
        agens.execute("create table plain (x int)")
        agens.execute("insert into plain values (1), (2), (3)")
        try:
            written = agens.execute_query("create (:t {a: 1}), (:t {a: 2})", counts_=True)
            assert written.counts.inserted_vertices == 2

            plain = agens.execute_query("update plain set x = x + 1", counts_=True)
            assert plain.counts.inserted_vertices == 0
            assert plain.counts.total == 0
        finally:
            agens.execute("drop table plain")

    def test_a_read_reports_five_zeros(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel t")
        agens.refresh_labels()
        agens.execute("create (:t {a: 1})")
        counts = agens.execute_query("match (n:t) return n", counts_=True).counts
        assert tuple(counts) == (0, 0, 0, 0, 0)

    def test_a_write_is_not_credited_with_an_earlier_statement_s_numbers(self, agens) -> None:  # type: ignore[no-untyped-def]
        """A write with RETURN zeroes only the counters its own clauses touch, so the rest
        still hold whatever the statement before it left."""
        agens.execute("create vlabel t")
        agens.refresh_labels()
        created = agens.execute_query("create (:t {a: 1}), (:t {a: 2})", counts_=True)
        assert created.counts.inserted_vertices == 2

        updated = agens.execute_query(
            "match (n:t) where n.a = 1 set n.b = 9 return n", counts_=True
        ).counts
        assert updated.inserted_vertices == 0
        assert updated.updated_properties == 1

    def test_a_delete_reports_only_what_it_deleted(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel t")
        agens.refresh_labels()
        agens.execute("create (:t {a: 1}), (:t {a: 2})")
        counts = agens.execute_query("match (n:t) where n.a = 1 delete n", counts_=True).counts
        assert counts.deleted_vertices == 1
        assert counts.inserted_vertices == 0


@pytest.mark.server
class TestAWriteThatNamesNoClause:
    """A statement's text is not the only way it can write."""

    def test_a_function_that_writes_is_still_counted(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel t")
        agens.refresh_labels()
        agens.execute(
            "create or replace function writes_one() returns void as $$ "
            "begin execute 'create (:t {a: 1})'; end $$ language plpgsql"
        )
        try:
            counts = agens.execute_query("select writes_one()", counts_=True).counts
            assert counts.inserted_vertices == 1
        finally:
            agens.execute("drop function writes_one()")


@pytest.mark.server
class TestHowManyStatementsCountingCosts:
    """The counters live on the session and are read by a statement of their own, so asking for
    them costs round trips. How many depends on what the server zeroes."""

    def sent(self, conn, run) -> list[str]:  # type: ignore[no-untyped-def]
        from agensgraph import connection as connection_module

        seen: list[str] = []
        real = connection_module.Cursor.execute

        def counting(self, query, params=None, **kwargs):  # type: ignore[no-untyped-def]
            seen.append(str(query))
            return real(self, query, params, **kwargs)

        connection_module.Cursor.execute = counting  # type: ignore[method-assign, assignment]
        try:
            run()
        finally:
            connection_module.Cursor.execute = real  # type: ignore[method-assign]
        return seen

    def test_a_write_that_returns_nothing_is_read_once(self, agens) -> None:  # type: ignore[no-untyped-def]
        """The server zeroes all five for it, so the reading afterwards is the whole answer."""
        agens.execute("create vlabel t")
        sent = self.sent(agens, lambda: agens.execute_query("create (:t {n: 1})", counts_=True))
        assert len(sent) == 2, sent

    def test_a_write_that_returns_rows_is_read_twice(self, agens) -> None:  # type: ignore[no-untyped-def]
        """Only the groups its clauses can move are zeroed, so the rest need a reading before."""
        agens.execute("create vlabel t")
        sent = self.sent(
            agens, lambda: agens.execute_query("create (:t {n: 1}) return 1", counts_=True)
        )
        assert len(sent) == 3, sent

    def test_a_statement_with_no_write_clause_is_read_twice(self, agens) -> None:  # type: ignore[no-untyped-def]
        """It may still move a counter by calling something that writes, and nothing is zeroed
        for that -- so only the difference between two readings finds it."""
        sent = self.sent(agens, lambda: agens.execute_query("select 1", counts_=True))
        assert len(sent) == 3, sent

    def test_it_still_counts_what_the_write_did(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel t")
        result = agens.execute_query("create (:t {n: 1}), (:t {n: 2})", counts_=True)
        assert result.counts.inserted_vertices == 2
