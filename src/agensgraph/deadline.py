"""A budget for how long something may take, spent across every stage of it.

A timeout given per operation is not a limit on the whole. A read timeout of five seconds is
satisfied by a peer that sends one byte every four, forever; a timeout that restarts at each
round trip is a promise the caller cannot hold anyone to. So the caller's limit is turned into
a moment, and every stage asks how much of it is left rather than being handed the whole thing
again.

Time is read from the monotonic clock and never from the wall clock, which moves when the
machine is corrected and would otherwise make a deadline arrive early, late, or twice.

One thing is set aside. A budget that runs out while a transaction is committing manufactures
the single failure that cannot be retried -- a commit whose outcome nobody knows -- so a
commit reserve is held back, and arriving at the commit with less than that left is worth
either refusing before sending or overshooting the budget. Overshooting is recoverable and an
unresolvable write is not.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["Deadline", "Expired"]


class Expired(TimeoutError):
    """The budget ran out.

    A ``TimeoutError``, so it is a socket timeout as far as anything classifying failures is
    concerned -- which is right: a statement whose time ran out says nothing about whether the
    server ran it.
    """

    spent: float | None = None
    total: float | None = None

    @classmethod
    def after(cls, spent: float, total: float, *, doing: str) -> Expired:
        """Build the report, naming what ran out of time and how much it had."""
        exc = cls(f"{doing} ran out of its {total:.3f}s budget after {spent:.3f}s")
        exc.spent = spent
        exc.total = total
        return exc


class Deadline:
    """A moment by which something must be finished.

    A deadline with no limit never expires and always has room, so a caller that did not ask
    for one does not have to be written differently from one that did.
    """

    __slots__ = ("_at", "_reserve", "_total")

    _at: float | None
    _total: float | None
    _reserve: float

    def __init__(self, seconds: float | None = None, *, commit_reserve: float = 0.0) -> None:
        if seconds is not None and seconds <= 0:
            raise ValueError(f"a budget must be more than nothing, got {seconds}")
        if commit_reserve < 0:
            raise ValueError(f"a commit reserve cannot be negative, got {commit_reserve}")
        self._total = seconds
        self._at = None if seconds is None else time.monotonic() + seconds
        self._reserve = commit_reserve

    @classmethod
    def none(cls) -> Deadline:
        """A budget with no limit, for a caller that asked for none."""
        return cls(None)

    @property
    def total(self) -> float | None:
        """What the budget was, or ``None`` if there is no limit."""
        return self._total

    @property
    def commit_reserve(self) -> float:
        """How much is held back so that a commit is never the thing that runs out."""
        return self._reserve

    @property
    def unlimited(self) -> bool:
        return self._at is None

    def remaining(self) -> float | None:
        """How much is left, which may be zero or less. ``None`` if there is no limit."""
        if self._at is None:
            return None
        return self._at - time.monotonic()

    def spent(self) -> float:
        """How long has gone, whether or not there is a limit."""
        if self._total is None or self._at is None:
            return 0.0
        return self._total - (self._at - time.monotonic())

    @property
    def expired(self) -> bool:
        remaining = self.remaining()
        return remaining is not None and remaining <= 0

    def available(self) -> float | None:
        """What may be spent before the commit reserve, which is what a statement gets.

        A statement is given the budget less the reserve, so that arriving at the commit with
        nothing left is a thing the caller decided rather than a thing that happened.
        """
        remaining = self.remaining()
        if remaining is None:
            return None
        # Never below nothing. A negative wait is not a short one: handed to a socket it means
        # no limit at all, which is the opposite of a budget that has run out.
        return max(remaining - self._reserve, 0.0)

    @property
    def can_commit(self) -> bool:
        """Whether there is enough left to commit within the budget.

        A caller arriving here with less has to choose: refuse before sending anything, or
        commit and overshoot. Both are recoverable, which is the point -- committing with no
        budget left and being cut off partway through is not.
        """
        remaining = self.remaining()
        return remaining is None or remaining >= self._reserve

    def check(self, doing: str) -> None:
        """Raise if the budget has run out, naming what was being done."""
        if self.expired:
            total = self._total
            assert total is not None
            raise Expired.after(self.spent(), total, doing=doing)

    def bounded(self, seconds: float | None) -> float | None:
        """A wait clamped to what is left, so no single wait can outlast the budget."""
        remaining = self.remaining()
        if remaining is None:
            return seconds
        remaining = max(remaining, 0.0)
        return remaining if seconds is None else min(seconds, remaining)

    def statement_timeout_ms(self, *, gap: float = 0.5) -> int | None:
        """What to ask the server to stop at, in milliseconds, or ``None`` for no limit.

        Set below what the caller is waiting for, so that the server is the one to give up and
        reports it as a cancelled statement, rather than the caller giving up first and leaving
        a statement running on a connection it no longer holds.
        """
        available = self.available()
        if available is None:
            return None
        limit = int((available - gap) * 1000)
        if limit < 1:
            # A floor of one millisecond is a limit no statement can meet, and it outlives the
            # caller it was set for. Nothing is left, so the caller is told rather than sent.
            raise Expired.after(self.spent(), self._total or 0.0, doing="setting a limit")
        return limit

    def __repr__(self) -> str:
        if self._at is None:
            return "Deadline(unlimited)"
        return f"Deadline({self._total}s, {self.remaining():.3f}s left)"


def stages(deadline: Deadline, names: tuple[str, ...]) -> Iterator[tuple[str, float | None]]:
    """Walk a series of stages, handing each what is left rather than the whole budget.

    Reading the remaining time between one stage and the next is the difference between a
    limit on the whole and a limit repeated -- which is not a limit.
    """
    for name in names:
        deadline.check(name)
        yield name, deadline.available()
