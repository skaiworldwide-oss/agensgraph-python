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
