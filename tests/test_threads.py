"""The state that is shared between threads, and whether sharing it is safe.

Five things in this driver are reachable from more than one thread at once: the retry allowance,
the adapters template every connection derives from, the list of query loggers, the map each
connection derives from that template, and the label table. A cache is the documented first hazard
of a free-threaded build, and the canonical broken shape is check-then-fill, so each is exercised
here. The prepared-statement cache is psycopg's own and is exercised through it.

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
from agensgraph._protocol.labels import LabelCache
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


class TestTheLabelTable:
    """Which names a label by an id whose meaning depends on which graph the table came from.

    Label ids restart per graph, so names read beside the wrong graph name do not fail -- they
    name a different label, and the caller is told a vertex is something it is not. The names and
    the graph they came from are therefore one field, replaced in a single assignment.

    Racing for that is not how it is checked. The window between two assignments is one bytecode
    wide, and eight readers against two thousand writes at a microsecond switch interval never
    land in it -- the same test passes against the shape that holds the two apart. So the count of
    assignments is asserted directly, and the interleaving test below is kept for what it does
    show, which is that hammering the table from eight threads answers correctly or not at all.
    """

    def test_replacing_the_table_is_one_assignment(self) -> None:
        class Recording(LabelCache):
            """Whose published field records how many times it is written."""

            def __init__(self) -> None:
                self.writes = 0
                super().__init__()

            @property
            def _table(self) -> tuple[str | None, dict[int, str]]:
                return self._written

            @_table.setter
            def _table(self, value: tuple[str | None, dict[int, str]]) -> None:
                self.writes += 1
                self._written = value

        cache = Recording()
        cache.writes = 0
        cache.load("first", [(3, "alpha")])
        assert cache.writes == 1
        cache.writes = 0
        cache.invalidate()
        assert cache.writes == 1

    def test_a_lookup_is_answered_correctly_or_not_at_all(self) -> None:
        cache = LabelCache()
        cache.load("first", [(3, "alpha")])
        states = [
            lambda c: c.load("first", [(3, "alpha")]),
            lambda c: c.load("second", [(4, "beta")]),
            lambda c: c.invalidate(),
        ]
        seen: list[tuple[str, str | None]] = []
        stop = threading.Event()

        def write() -> None:
            for i in range(EACH):
                states[i % len(states)](cache)
            stop.set()

        def read() -> None:
            """At least one read before looking at the flag, or a reader that starts late
            collects nothing and the assertion below has nothing to assert."""
            while True:
                try:
                    seen.append(("named", cache.name(3)))
                except KeyError as exc:
                    seen.append(("refused", str(exc.args[0]).rpartition("graph ")[2]))
                if stop.is_set():
                    return

        writer = threading.Thread(target=write)
        readers = [threading.Thread(target=read) for _ in range(THREADS)]
        writer.start()
        for reader in readers:
            reader.start()
        writer.join()
        for reader in readers:
            reader.join()

        assert seen
        assert set(seen) <= {("named", "alpha"), ("refused", "'second'"), ("refused", "None")}


@pytest.mark.server
class TestWhatAConnectionFillsForItself:
    """The two per-connection copies, and the gate that decides whether to keep the connection.

    A field that fills itself on first use is filled twice when two callers arrive together, and
    one of the two copies is dropped -- taking with it whatever the losing caller registered on it
    or loaded into it. So none of them fills itself on first use: the two copies are built with the
    connection, and the gate is answered inside connect before the connection is handed over. That
    is asserted directly for the same reason as the label table -- the window is too narrow to race
    for, and a test that raced for it would pass either way.
    """

    def test_both_are_there_before_the_connection_has_been_used(self, dsn) -> None:  # type: ignore[no-untyped-def]
        """Read behind the properties, since reading through them is what would fill them."""
        import agensgraph

        fresh = agensgraph.Connection.connect(dsn)
        try:
            assert isinstance(fresh._agens_adapters, type(GRAPH_ADAPTERS))
            assert isinstance(fresh._agens_labels, LabelCache)
        finally:
            fresh.close()

    def test_both_copies_are_the_same_object_from_every_thread(self, agens) -> None:  # type: ignore[no-untyped-def]
        maps: list[int] = []
        tables: list[int] = []

        def look() -> None:
            for _ in range(EACH):
                maps.append(id(agens.adapters))
                tables.append(id(agens.label_table))

        run_on_threads(look)
        assert len(set(maps)) == 1
        assert len(set(tables)) == 1

    def test_the_map_is_this_connection_s_own(self, agens, dsn) -> None:  # type: ignore[no-untyped-def]
        import agensgraph

        other = agensgraph.Connection.connect(dsn)
        try:
            assert agens.adapters is not other.adapters
            assert agens.adapters is not GRAPH_ADAPTERS
            assert agens.label_table is not other.label_table
        finally:
            other.close()

    def test_the_version_gate_is_answered_before_anybody_holds_the_connection(
        self, agens
    ) -> None:  # type: ignore[no-untyped-def]
        """So the one field that is still filled on first use cannot be reached by two callers."""
        assert agens._agens_capabilities is not None

    def test_the_prepared_statement_cache_takes_eight_threads(self, agens) -> None:  # type: ignore[no-untyped-def]
        """psycopg's own, not this driver's, and pinned here because it is shared all the same."""
        agens.execute("create (:cached {n: 1})")
        answers: list[int] = []

        def ask() -> None:
            for _ in range(20):
                answers.append(
                    agens.execute_query("match (n:cached) return n.n", prepare_=True).records[
                        0
                    ][0]
                )

        run_on_threads(ask)
        assert answers == [1] * (THREADS * 20)
