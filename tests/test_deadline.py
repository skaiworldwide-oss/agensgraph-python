"""A budget spent across every stage of an operation.

Time is faked where a real wait would make the test slow or flaky, and read for real only
where the point is that the clock is the monotonic one.
"""

from __future__ import annotations

import time

import pytest

from agensgraph.deadline import Deadline, Expired


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """A monotonic clock that only moves when told to."""

    class Clock:
        def __init__(self) -> None:
            self.now = 1000.0

        def advance(self, seconds: float) -> None:
            self.now += seconds

    fake = Clock()
    monkeypatch.setattr(time, "monotonic", lambda: fake.now)
    return fake


class TestALimit:
    def test_what_is_left_shrinks_as_time_passes(self, clock) -> None:  # type: ignore[no-untyped-def]
        budget = Deadline(10.0)
        assert budget.remaining() == pytest.approx(10.0)
        clock.advance(4.0)
        assert budget.remaining() == pytest.approx(6.0)
        assert budget.spent() == pytest.approx(4.0)

    def test_it_expires_when_it_runs_out(self, clock) -> None:  # type: ignore[no-untyped-def]
        budget = Deadline(1.0)
        assert not budget.expired
        clock.advance(1.0)
        assert budget.expired
        clock.advance(5.0)
        assert budget.remaining() == pytest.approx(-5.0)

    def test_expiring_names_what_ran_out(self, clock) -> None:  # type: ignore[no-untyped-def]
        budget = Deadline(2.0)
        clock.advance(3.0)
        with pytest.raises(Expired) as caught:
            budget.check("reading a result")
        assert "reading a result" in str(caught.value)
        assert caught.value.total == pytest.approx(2.0)
        assert caught.value.spent == pytest.approx(3.0)

    def test_expiring_is_a_timeout(self) -> None:
        """So that everything classifying failures already knows what it means."""
        assert issubclass(Expired, TimeoutError)
        assert isinstance(Expired.after(1.0, 1.0, doing="x"), OSError)

    def test_a_wait_cannot_outlast_the_budget(self, clock) -> None:  # type: ignore[no-untyped-def]
        """A per-read timeout is not a limit on the whole; clamping is what makes it one."""
        budget = Deadline(5.0)
        assert budget.bounded(10.0) == pytest.approx(5.0)
        assert budget.bounded(1.0) == pytest.approx(1.0)
        clock.advance(4.5)
        assert budget.bounded(10.0) == pytest.approx(0.5)
        clock.advance(1.0)
        assert budget.bounded(10.0) == 0.0

    @pytest.mark.parametrize("bad", [0, -1, -0.001])
    def test_a_budget_of_nothing_is_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="more than nothing"):
            Deadline(bad)


class TestNoLimit:
    def test_it_never_expires(self) -> None:
        budget = Deadline.none()
        assert budget.unlimited
        assert budget.remaining() is None
        assert not budget.expired
        assert budget.available() is None
        budget.check("anything")

    def test_it_clamps_nothing(self) -> None:
        assert Deadline.none().bounded(10.0) == pytest.approx(10.0)
        assert Deadline.none().bounded(None) is None

    def test_it_asks_the_server_for_no_limit_either(self) -> None:
        assert Deadline.none().statement_timeout_ms() is None


class TestTheCommitReserve:
    """Never let the budget run out mid-commit; that manufactures the unresolvable failure."""

    def test_a_statement_gets_the_budget_less_the_reserve(self, clock) -> None:  # type: ignore[no-untyped-def]
        budget = Deadline(10.0, commit_reserve=2.0)
        assert budget.available() == pytest.approx(8.0)
        assert budget.remaining() == pytest.approx(10.0)

    def test_committing_is_allowed_while_the_reserve_is_intact(self, clock) -> None:  # type: ignore[no-untyped-def]
        budget = Deadline(10.0, commit_reserve=2.0)
        assert budget.can_commit
        clock.advance(7.9)
        assert budget.can_commit
        clock.advance(0.2)
        assert not budget.can_commit

    def test_with_no_limit_there_is_always_room(self) -> None:
        assert Deadline(None, commit_reserve=5.0).can_commit

    @pytest.mark.parametrize("bad", [-1, -0.5])
    def test_a_negative_reserve_is_refused(self, bad: float) -> None:
        with pytest.raises(ValueError, match="negative"):
            Deadline(1.0, commit_reserve=bad)


class TestWhatTheServerIsAsked:
    def test_it_is_told_to_stop_before_the_caller_gives_up(self, clock) -> None:  # type: ignore[no-untyped-def]
        """Otherwise the caller walks away leaving a statement running on the connection."""
        budget = Deadline(10.0)
        asked = budget.statement_timeout_ms(gap=0.5)
        assert asked is not None
        assert asked < 10_000
        assert asked == 9500

    def test_the_reserve_comes_off_it_too(self, clock) -> None:  # type: ignore[no-untyped-def]
        budget = Deadline(10.0, commit_reserve=2.0)
        assert budget.statement_timeout_ms(gap=0.5) == 7500

    def test_a_budget_with_nothing_left_is_refused_rather_than_floored(self, clock) -> None:  # type: ignore[no-untyped-def]
        """Zero means no limit to the server, and one millisecond is a limit no statement can
        meet -- and it outlives the caller it was set for, on a pooled connection. Neither is
        worth sending, so the caller is told the budget has gone."""
        budget = Deadline(0.1)
        clock.advance(0.2)
        with pytest.raises(Expired):
            budget.statement_timeout_ms()


def test_the_clock_is_the_monotonic_one() -> None:
    """A wall clock moves when the machine is corrected, which would move a deadline."""
    budget = Deadline(0.05)
    time.sleep(0.06)
    assert budget.expired
