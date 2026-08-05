"""The state that is shared between threads, and whether sharing it is safe.

Three things in this driver are reachable from more than one thread at once: the retry allowance,
the adapters template every connection derives from, and the list of query loggers. A module-level
cache is the documented first hazard of a free-threaded build, and the canonical broken shape is
check-then-fill, so each is exercised here.

The switch interval is dropped to a microsecond, which forces interleaving without needing a
free-threaded interpreter -- one of the two techniques that find these races on an ordinary build.
Nothing here relies on a container's internal lock, because the documentation is explicit that
those are a description of the current implementation and not a promise.
"""

from __future__ import annotations

import sys
import threading

import pytest

from agensgraph._core import GRAPH_ADAPTERS
from agensgraph.errors import Retryability
from agensgraph.observability import (
    QueryRecord,
    add_query_logger,
    remove_query_logger,
    report,
)
from agensgraph.retry import REJECTION_COST, TokenBucket

THREADS = 8
EACH = 2000


@pytest.fixture(autouse=True)
def interleave_aggressively():
    """Switch threads as often as the interpreter will, so a race has a chance to happen."""
    before = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    yield
    sys.setswitchinterval(before)


def run_on_threads(work) -> None:  # type: ignore[no-untyped-def]
    threads = [threading.Thread(target=work) for _ in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def test_the_retry_allowance_loses_no_updates() -> None:
    """Read, modify, write -- so it holds a lock rather than trusting an operation to be atomic."""
    capacity = THREADS * EACH * REJECTION_COST
    bucket = TokenBucket(capacity)

    def spend() -> None:
        for _ in range(EACH):
            bucket.spend(Retryability.BACKPRESSURE)

    run_on_threads(spend)
    assert bucket.tokens() == 0, "an update was lost, so the allowance is not actually shared"


def test_paying_back_loses_nothing_either() -> None:
    bucket = TokenBucket(THREADS * EACH)
    for _ in range(THREADS * EACH):
        bucket.spend(Retryability.BACKPRESSURE)
    assert bucket.tokens() == 0

    def credit() -> None:
        for _ in range(EACH):
            bucket.credit()

    run_on_threads(credit)
    assert bucket.tokens() == THREADS * EACH


def test_the_allowance_never_reads_as_a_number_it_never_held() -> None:
    """A torn read would show a value between two states, which no observer should ever see."""
    bucket = TokenBucket(1000)
    seen: list[float] = []
    stop = threading.Event()

    def watch() -> None:
        while not stop.is_set():
            seen.append(bucket.tokens())

    watcher = threading.Thread(target=watch)
    watcher.start()
    try:
        for _ in range(200):
            bucket.spend(Retryability.BACKPRESSURE)
    finally:
        stop.set()
        watcher.join()
    assert seen
    assert all(0 <= value <= 1000 for value in seen)
    assert all(value % REJECTION_COST == 0 for value in seen), "a value mid-update was observed"


def test_the_adapters_template_can_be_derived_from_at_once() -> None:
    """Every connection derives its own map from one shared template, on whatever thread it is on."""
    from psycopg.adapt import AdaptersMap

    failures: list[BaseException] = []

    def derive() -> None:
        try:
            for _ in range(500):
                AdaptersMap(GRAPH_ADAPTERS)
        except BaseException as exc:
            failures.append(exc)

    run_on_threads(derive)
    assert not failures, f"deriving a map from the template failed: {failures[0]!r}"


def test_no_query_record_is_dropped() -> None:
    collected: list[QueryRecord] = []
    guard = threading.Lock()

    def observer(record: QueryRecord) -> None:
        with guard:
            collected.append(record)

    add_query_logger(observer)
    try:

        def emit() -> None:
            for _ in range(500):
                report(QueryRecord(1, "x", 0.0, 0, False))

        run_on_threads(emit)
    finally:
        remove_query_logger(observer)
    assert len(collected) == THREADS * 500


def test_the_build_can_be_told_apart_from_the_state_of_the_lock() -> None:
    """Two different questions, and code that confuses them is wrong on a build with PYTHON_GIL=1.

    The build is what decides whether shared use has to be safe at all; whether the lock is off
    right now can change per process and is not what a decision about the code should read.
    """
    import sysconfig

    build = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
    right_now = getattr(sys, "_is_gil_enabled", lambda: True)()
    assert isinstance(build, bool)
    assert isinstance(right_now, bool)
    if not build:
        assert right_now, "a build with the lock cannot be running without it"
