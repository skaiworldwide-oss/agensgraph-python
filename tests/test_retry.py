"""Deciding whether to try again, and how long to wait first.

The deciding is separable from the waiting on purpose, so all of this runs without a clock.
Randomness is seeded, and where the point is the shape of a distribution it is asserted over
enough draws to mean something.
"""

from __future__ import annotations

import random

import psycopg.errors as pg
import pytest

from agensgraph.errors import Retryability
from agensgraph.retry import (
    REJECTION_COST,
    SUCCESS_CREDIT,
    TRANSIENT_COST,
    RetryPolicy,
    TokenBucket,
    full_jitter,
)


def failure(state: str) -> pg.Error:
    return pg.lookup(state)("failed")


CONFLICT = "40001"
LOST = "08006"
SYNTAX = "42601"
OVERLOADED = "53300"
STALE_STATEMENT = "26000"


class TestFullJitter:
    def test_the_ceiling_doubles_and_then_stops(self) -> None:
        rng = random.Random(1)
        for attempt, ceiling in [(1, 0.05), (2, 0.10), (3, 0.20), (4, 0.40), (5, 0.40)]:
            draws = [full_jitter(attempt, base=0.05, cap=0.4, rng=rng) for _ in range(4000)]
            assert max(draws) <= ceiling
            assert max(draws) > ceiling * 0.95, "the ceiling should be nearly reached"

    def test_the_cap_is_applied_before_the_jitter(self) -> None:
        """Capping afterwards makes the cap reachable only by chance, and has shipped."""
        rng = random.Random(2)
        draws = [full_jitter(20, base=0.05, cap=1.0, rng=rng) for _ in range(4000)]
        assert max(draws) <= 1.0
        # Uniform below the cap, so the mean sits near its middle. Capping after jittering
        # would pile almost every draw at the cap instead.
        assert 0.4 < sum(draws) / len(draws) < 0.6

    def test_the_whole_range_below_the_ceiling_is_used(self) -> None:
        rng = random.Random(3)
        draws = [full_jitter(4, base=0.05, cap=0.4, rng=rng) for _ in range(4000)]
        assert min(draws) < 0.02, "small waits should occur, which is what spreads clients out"
        assert max(draws) > 0.38

    def test_a_wait_is_never_negative(self) -> None:
        rng = random.Random(4)
        assert all(full_jitter(a, rng=rng) >= 0 for a in range(1, 12) for _ in range(200))

    @pytest.mark.parametrize("bad", [0, -1])
    def test_attempts_are_numbered_from_one(self, bad: int) -> None:
        with pytest.raises(ValueError, match="numbered from one"):
            full_jitter(bad)


class TestTokenBucket:
    def test_it_starts_full_and_allows_retrying(self) -> None:
        bucket = TokenBucket(100)
        assert bucket.tokens() == 100
        assert bucket.allows_retry

    def test_it_stops_at_half_and_not_at_empty(self) -> None:
        """A failing server should see the ordinary load, not the load plus every retry."""
        bucket = TokenBucket(100)
        spent = 0
        while bucket.allows_retry:
            bucket.spend(Retryability.RECONNECT)
            spent += 1
        assert spent == 4  # 100 -> 86 -> 72 -> 58 -> 44
        assert bucket.tokens() == 100 - 4 * TRANSIENT_COST

    def test_a_rejection_costs_less_than_a_transient_failure(self) -> None:
        """Being turned away is about this request; a transient failure is usually the service."""
        assert REJECTION_COST < TRANSIENT_COST
        rejected, transient = TokenBucket(100), TokenBucket(100)
        rejected.spend(Retryability.BACKPRESSURE)
        transient.spend(Retryability.RECONNECT)
        assert rejected.tokens() > transient.tokens()

    def test_successes_pay_it_back(self) -> None:
        bucket = TokenBucket(100)
        for _ in range(4):
            bucket.spend(Retryability.RECONNECT)
        assert not bucket.allows_retry
        for _ in range(TRANSIENT_COST):
            bucket.credit()
        assert bucket.allows_retry

    def test_it_never_goes_below_nothing_or_above_full(self) -> None:
        bucket = TokenBucket(10)
        for _ in range(50):
            bucket.spend(Retryability.RECONNECT)
        assert bucket.tokens() == 0
        for _ in range(500):
            bucket.credit()
        assert bucket.tokens() == 10

    def test_roughly_ten_successes_pay_for_one_retry(self) -> None:
        assert pytest.approx(14, abs=1) == TRANSIENT_COST / SUCCESS_CREDIT

    @pytest.mark.parametrize("bad", [0, -1])
    def test_a_bucket_has_to_hold_something(self, bad: int) -> None:
        with pytest.raises(ValueError, match="hold something"):
            TokenBucket(bad)


class TestDeciding:
    @pytest.fixture
    def policy(self) -> RetryPolicy:
        return RetryPolicy(attempts=3, rng=random.Random(5))

    def test_a_conflict_is_retried(self, policy: RetryPolicy) -> None:
        decision = policy.decide(failure(CONFLICT), number=1)
        assert decision.retry
        assert decision.recovery is Retryability.SAFE
        assert decision.delay > 0

    def test_a_conflict_is_retried_even_after_a_write(self, policy: RetryPolicy) -> None:
        """A conflict is the server reporting that it rolled the transaction back."""
        assert policy.decide(failure(CONFLICT), number=1, wrote=True).retry

    def test_a_lost_connection_is_retried_on_a_new_one(self, policy: RetryPolicy) -> None:
        decision = policy.decide(failure(LOST), number=1)
        assert decision.retry
        assert decision.recovery.needs_new_connection

    def test_a_lost_connection_after_a_write_is_not_retried(self, policy: RetryPolicy) -> None:
        """What was lost is the answer to whether the commit landed."""
        decision = policy.decide(failure(LOST), number=1, wrote=True)
        assert not decision.retry
        assert decision.recovery is Retryability.UNKNOWN
        assert "resolved" in decision.reason

    def test_a_syntax_error_is_never_retried(self, policy: RetryPolicy) -> None:
        decision = policy.decide(failure(SYNTAX), number=1)
        assert not decision.retry
        assert "the same way" in decision.reason

    def test_a_stale_statement_keeps_the_connection(self, policy: RetryPolicy) -> None:
        decision = policy.decide(failure(STALE_STATEMENT), number=1)
        assert decision.retry
        assert decision.recovery.clears_prepared_statements
        assert decision.recovery.keeps_connection

    def test_an_overloaded_server_is_given_longer(self, policy: RetryPolicy) -> None:
        same_seed = RetryPolicy(attempts=3, rng=random.Random(9))
        overloaded = same_seed.decide(failure(OVERLOADED), number=1)
        other_seed = RetryPolicy(attempts=3, rng=random.Random(9))
        conflict = other_seed.decide(failure(CONFLICT), number=1)
        assert overloaded.delay > conflict.delay
        assert overloaded.recovery.wants_longer_delay

    def test_the_last_attempt_gives_up_and_says_how_many(self, policy: RetryPolicy) -> None:
        decision = policy.decide(failure(CONFLICT), number=3)
        assert not decision.retry
        assert "reached max retries: 3" in decision.reason

    def test_three_attempts_means_one_try_and_two_retries(self, policy: RetryPolicy) -> None:
        """Stated because libraries disagree about what the number counts."""
        assert policy.attempts == 3
        assert policy.decide(failure(CONFLICT), number=1).retry
        assert policy.decide(failure(CONFLICT), number=2).retry
        assert not policy.decide(failure(CONFLICT), number=3).retry

    def test_a_spent_allowance_stops_it(self) -> None:
        bucket = TokenBucket(20)
        policy = RetryPolicy(attempts=10, bucket=bucket, rng=random.Random(6))
        assert policy.decide(failure(LOST), number=1).retry
        decision = policy.decide(failure(LOST), number=2)
        assert not decision.retry
        assert "allowance" in decision.reason

    def test_a_budget_too_small_for_the_wait_stops_it(self, policy: RetryPolicy) -> None:
        decision = policy.decide(failure(CONFLICT), number=1, remaining=0.0001)
        assert not decision.retry
        assert "budget" in decision.reason

    def test_a_budget_with_room_does_not(self, policy: RetryPolicy) -> None:
        assert policy.decide(failure(CONFLICT), number=1, remaining=30.0).retry

    def test_deciding_not_to_retry_spends_nothing(self, policy: RetryPolicy) -> None:
        before = policy.bucket.tokens()
        policy.decide(failure(SYNTAX), number=1)
        assert policy.bucket.tokens() == before

    def test_deciding_to_retry_spends(self, policy: RetryPolicy) -> None:
        before = policy.bucket.tokens()
        policy.decide(failure(CONFLICT), number=1)
        assert policy.bucket.tokens() < before

    def test_a_success_pays_back(self, policy: RetryPolicy) -> None:
        policy.decide(failure(CONFLICT), number=1)
        before = policy.bucket.tokens()
        policy.succeeded()
        assert policy.bucket.tokens() > before

    def test_a_shared_allowance_is_shared(self) -> None:
        """A counter is per call site; four layers of three attempts is sixty-four tries."""
        bucket = TokenBucket(100)
        one = RetryPolicy(bucket=bucket, rng=random.Random(7))
        two = RetryPolicy(bucket=bucket, rng=random.Random(8))
        one.decide(failure(LOST), number=1)
        assert two.bucket is bucket
        assert two.bucket.tokens() == 100 - TRANSIENT_COST

    @pytest.mark.parametrize("bad", [0, -1])
    def test_there_is_always_a_first_attempt(self, bad: int) -> None:
        with pytest.raises(ValueError, match="first attempt"):
            RetryPolicy(attempts=bad)


class TestGivingUp:
    def test_the_failure_carries_what_happened(self) -> None:
        policy = RetryPolicy()
        earlier = [failure(CONFLICT), failure(CONFLICT)]
        last = failure(CONFLICT)
        raised = policy.exhausted(last, attempts=3, previous=earlier)
        assert raised is last
        assert raised.attempts == 3  # type: ignore[attr-defined]
        assert raised.previous_errors == tuple(earlier)  # type: ignore[attr-defined]
        assert "reached max retries: 3" in str(raised)

    def test_a_single_attempt_says_nothing_extra(self) -> None:
        policy = RetryPolicy()
        before = str(failure(CONFLICT))
        raised = policy.exhausted(failure(CONFLICT), attempts=1, previous=[])
        assert str(raised) == before


def test_a_decision_reads_clearly() -> None:
    """It goes into logs, so it has to say what was decided and why in one line."""
    policy = RetryPolicy(rng=random.Random(11))
    assert "retry in" in repr(policy.decide(failure(CONFLICT), number=1))
    assert "give up" in repr(policy.decide(failure(SYNTAX), number=1))
    assert "RetryPolicy(attempts=3" in repr(policy)
    assert "/100" in repr(TokenBucket(100))
