"""Deciding whether to try again, and how long to wait first.

Three things, kept apart because they fail in different ways.

*Whether* to try again is :mod:`agensgraph.errors`, which answers with one of six recoveries
rather than a yes or a no.

*How long to wait* is full jitter: cap the delay, then pick uniformly below it. Re-running
Marc Brooker's simulation with a hundred clients, exponential backoff with no jitter took
63,819 ms and 1,861 calls to drain the work; full jitter took 4,868 ms and 796; equal jitter
took 6,600 ms. The order of the two operations is not incidental -- capping *after* jittering,
which is a mistake that has shipped, makes the cap reachable only by chance and the
distribution not the one the name describes.

*How often* to try again is a budget, not a counter. A counter is per call site, and four
layers each willing to try three times is eighty-one attempts from one action. A token bucket
is per process: a failed attempt costs tokens, a success returns a fraction of one, and while
the bucket is low nothing retries at all. The asymmetry is the tuning surface. A transient
failure costs more than a rejection, because a rejection is one request being turned away
while a transient failure is usually the whole service in trouble and retrying is the worst
thing to do to it.

The clock starts *after* the first attempt. The first attempt is not a retry, and counting it
against the budget makes a single slow attempt look like an exhausted one.
"""

from __future__ import annotations

import random
import threading
from typing import TYPE_CHECKING

from .errors import Retryability, attach_retry_history, retryability

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "SHARED_ALLOWANCE",
    "Attempt",
    "RetryPolicy",
    "TokenBucket",
    "full_jitter",
]

TRANSIENT_COST = 14
"""What a transient failure costs the bucket.

More than a rejection, because a connection failure or an internal error is usually the whole
service and not this one request. With three attempts allowed, sustained transient failure
above roughly a fifth of all calls drains the bucket and stops the retrying.
"""

REJECTION_COST = 5
"""What a rejection costs. Less, because being turned away is about this request."""

SUCCESS_CREDIT = 1
"""What a success returns. So fourteen successes pay for one transient retry and five for
one rejection, which is the ratio those two costs are chosen to give."""


def full_jitter(
    attempt: int,
    *,
    base: float = 0.05,
    cap: float = 2.0,
    rng: random.Random | None = None,
) -> float:
    """How long to wait before attempt number *attempt*, counting the first as one.

    The ceiling doubles per attempt and is capped, and the wait is drawn uniformly below the
    capped ceiling. Capping first is what makes the cap a cap.
    """
    if attempt < 1:
        raise ValueError(f"an attempt is numbered from one, got {attempt}")
    ceiling = min(cap, base * (2 ** (attempt - 1)))
    pick = rng.uniform if rng is not None else random.uniform
    return pick(0.0, ceiling)


class TokenBucket:
    """A process-wide allowance for retrying, so that a counter cannot be multiplied.

    Held at a fixed size, drained by a failed attempt and refilled a little by each success.
    While it is at or below half, nothing retries -- which is the point: a service that is
    failing should see the ordinary load and not the ordinary load plus every retry.
    """

    __slots__ = ("_capacity", "_lock", "_tokens")

    def __init__(self, capacity: int = 100) -> None:
        if capacity <= 0:
            raise ValueError(f"a bucket has to hold something, got {capacity}")
        self._capacity = capacity
        self._tokens = float(capacity)
        # Not relying on any container being atomic, which is a description of an
        # implementation and not a promise.
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def tokens(self) -> float:
        with self._lock:
            return self._tokens

    @property
    def allows_retry(self) -> bool:
        """Whether there is enough left to be worth spending."""
        with self._lock:
            return self._tokens > self._capacity / 2

    def spend(self, recovery: Retryability) -> None:
        """Take what an attempt of this kind costs."""
        cost = REJECTION_COST if recovery is Retryability.BACKPRESSURE else TRANSIENT_COST
        with self._lock:
            self._tokens = max(0.0, self._tokens - cost)

    def credit(self) -> None:
        """Return what a success is worth."""
        with self._lock:
            self._tokens = min(float(self._capacity), self._tokens + SUCCESS_CREDIT)

    def __repr__(self) -> str:
        return f"TokenBucket({self.tokens():.0f}/{self._capacity})"


SHARED_ALLOWANCE = TokenBucket()
"""The allowance every policy draws on unless it is given one of its own.

One per process, because the multiplication it exists to bound happens *between* layers: four
of them each willing to try three times is eighty-one attempts from one action, and four
separate allowances would each say yes to their own three.
"""


class Attempt:
    """What was decided about one failure, and why.

    Returned rather than acted on, so that the deciding and the waiting are separable and the
    deciding is testable without a clock.
    """

    __slots__ = ("delay", "number", "reason", "recovery", "retry")

    def __init__(
        self,
        *,
        number: int,
        recovery: Retryability,
        retry: bool,
        delay: float,
        reason: str,
    ) -> None:
        self.number = number
        self.recovery = recovery
        self.retry = retry
        self.delay = delay
        self.reason = reason

    def __repr__(self) -> str:
        verdict = f"retry in {self.delay:.3f}s" if self.retry else f"give up: {self.reason}"
        return f"Attempt({self.number}, {self.recovery.name}, {verdict})"


class RetryPolicy:
    """How many times, how long between, and out of whose allowance.

    ``attempts`` counts the first try, so three means one attempt and at most two retries. A
    library that means something else by it is a library whose users are surprised once.
    """

    __slots__ = ("_attempts", "_base", "_bucket", "_cap", "_rng")

    def __init__(
        self,
        *,
        attempts: int = 3,
        base: float = 0.05,
        cap: float = 2.0,
        bucket: TokenBucket | None = None,
        rng: random.Random | None = None,
    ) -> None:
        if attempts < 1:
            raise ValueError(f"there is always a first attempt, got {attempts}")
        if base <= 0:
            raise ValueError(f"a delay has to grow from something, got {base}")
        if cap < base:
            raise ValueError(f"a cap below the base is not a cap, got {cap} under {base}")
        self._attempts = attempts
        self._base = base
        self._cap = cap
        # The allowance is shared by every policy that does not ask for its own, which is what
        # makes it a budget: four layers each holding their own would multiply the retries the
        # budget exists to bound.
        self._bucket = bucket if bucket is not None else SHARED_ALLOWANCE
        self._rng = rng

    @property
    def attempts(self) -> int:
        """How many tries in total, counting the first."""
        return self._attempts

    @property
    def bucket(self) -> TokenBucket:
        """The allowance this policy draws on, which may be shared with others."""
        return self._bucket

    def decide(
        self,
        exc: BaseException,
        *,
        number: int,
        wrote: bool = False,
        merging: bool = False,
        remaining: float | None = None,
    ) -> Attempt:
        """What to do about a failure on attempt *number*.

        ``wrote`` says whether the transaction had written anything, which decides whether a
        lost connection is worth reconnecting for or is a commit whose outcome is now unknown.
        ``remaining`` is what is left of the caller's budget; a wait that would not leave time
        for another attempt is not worth taking. ``merging`` says the statement creates only what
        is missing, which is what makes another writer having got there first a reason to run it
        again rather than a failure.
        """
        if number < 1:
            raise ValueError(f"an attempt is numbered from one, got {number}")
        recovery = retryability(exc, wrote=wrote, merging=merging)
        delay = 0.0

        if not recovery.is_retryable:
            reason = (
                "the outcome is not known and has to be resolved rather than repeated"
                if recovery is Retryability.UNKNOWN
                else "another attempt would fail the same way"
            )
            return Attempt(
                number=number, recovery=recovery, retry=False, delay=delay, reason=reason
            )

        if number >= self._attempts:
            return Attempt(
                number=number,
                recovery=recovery,
                retry=False,
                delay=delay,
                reason=f"reached max retries: {self._attempts}",
            )

        if not self._bucket.allows_retry:
            return Attempt(
                number=number,
                recovery=recovery,
                retry=False,
                delay=delay,
                reason="the retry allowance is spent, so the server is being left alone",
            )

        # A server short of something is arrived at later, and doubling the base doubles the
        # ceiling the wait is drawn below -- before the cap is applied, so the cap stays one.
        # Doubling the drawn value instead put the wait at twice the cap.
        base = self._base * 2 if recovery.wants_longer_delay else self._base
        delay = full_jitter(number, base=base, cap=self._cap, rng=self._rng)
        if remaining is not None and delay >= remaining:
            return Attempt(
                number=number,
                recovery=recovery,
                retry=False,
                delay=delay,
                reason="what is left of the budget would not cover the wait and another attempt",
            )

        self._bucket.spend(recovery)
        return Attempt(
            number=number,
            recovery=recovery,
            retry=True,
            delay=delay,
            reason=f"retrying: {recovery.name.lower()}",
        )

    def succeeded(self) -> None:
        """Report a success, which pays a little of the allowance back."""
        self._bucket.credit()

    def exhausted(
        self, exc: BaseException, *, attempts: int, previous: Sequence[BaseException]
    ) -> BaseException:
        """The failure to raise once no more attempts will be made.

        The number of attempts goes into the message as well as onto the exception, because a
        report saying a statement failed and one saying it failed every time it was tried lead
        to different places. This is the one place that can say so, being the one that knows
        no further attempt will be made.
        """
        attach_retry_history(exc, attempts=attempts, previous_errors=previous, exhausted=True)
        return exc

    def __repr__(self) -> str:
        return f"RetryPolicy(attempts={self._attempts}, base={self._base}, cap={self._cap})"
