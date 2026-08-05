"""Watching what the driver does, and paying nothing when nobody is.

The governing claim is that off costs a boolean test, so the tests that matter are the ones
asserting nothing happens when nothing is configured -- a logger that is never called, a timer
that never reads a clock, and a tracer that is never imported.
"""

from __future__ import annotations

import sys

import psycopg
import pytest

import agensgraph
from agensgraph.observability import (
    Notice,
    QueryRecord,
    Timer,
    add_query_logger,
    disable_tracing,
    logging_wanted,
    notice_from_diagnostic,
    query_span,
    remove_query_logger,
    report,
    tracing_enabled,
)


@pytest.fixture(autouse=True)
def nothing_left_listening():
    """Every test starts and ends with nothing configured, since all of it is module state."""
    disable_tracing()
    yield
    disable_tracing()


class TestCostingNothing:
    def test_no_logger_means_no_work(self) -> None:
        assert not logging_wanted()
        report(QueryRecord(1, "x", 0.0, 0, False))  # must not raise, must do nothing

    def test_a_timer_reads_no_clock_when_nobody_asks(self) -> None:
        """A clock read per statement is small, and small per statement is not nothing."""
        assert Timer().elapsed == 0.0

    def test_and_reads_one_when_somebody_does(self) -> None:
        seen: list[QueryRecord] = []
        add_query_logger(seen.append)
        try:
            assert Timer().elapsed >= 0.0
            assert logging_wanted()
        finally:
            remove_query_logger(seen.append)

    def test_a_span_is_not_taken_when_tracing_is_off(self) -> None:
        assert not tracing_enabled()
        with query_span("match (n) return n"):
            pass

    def test_the_tracing_package_is_not_imported_by_importing_the_driver(self) -> None:
        """Only the api, only if asked for, and never the sdk."""
        assert "opentelemetry" not in sys.modules
        assert "opentelemetry.sdk" not in sys.modules


class TestTheQueryLogger:
    def test_it_is_told_and_can_be_untold(self) -> None:
        seen: list[QueryRecord] = []
        add_query_logger(seen.append)
        report(QueryRecord(1, "a", 0.5, 3, False))
        remove_query_logger(seen.append)
        report(QueryRecord(1, "b", 0.5, 3, False))
        assert [record.statement for record in seen] == ["a"]

    def test_one_that_raises_does_not_break_the_statement(self, caplog) -> None:  # type: ignore[no-untyped-def]
        """A statement's outcome is the statement's, not its observer's."""

        def broken(record: QueryRecord) -> None:
            raise RuntimeError("the observer is broken")

        seen: list[QueryRecord] = []
        add_query_logger(broken)
        add_query_logger(seen.append)
        try:
            report(QueryRecord(1, "a", 0.0, 0, False))
        finally:
            remove_query_logger(broken)
            remove_query_logger(seen.append)
        assert len(seen) == 1, "the others are still told"

    def test_removing_one_never_added_is_harmless(self) -> None:
        remove_query_logger(lambda record: None)


class TestNotices:
    def test_the_untranslated_severity_is_the_one_kept(self) -> None:
        """The other passes through the server's message locale, so comparing it is a bug."""

        class Diag:
            severity = "HINWEIS"
            severity_nonlocalized = "NOTICE"
            sqlstate = "00000"
            message_primary = "something happened"
            message_detail = "in detail"
            message_hint = "try this"

        notice = notice_from_diagnostic(Diag())  # type: ignore[arg-type]
        assert notice.severity == "NOTICE"
        assert notice.code == "00000"
        assert notice.detail == "in detail"
        assert notice.hint == "try this"

    def test_it_reads_as_a_sentence_but_is_not_one(self) -> None:
        notice = Notice("WARNING", "01000", "look out")
        assert str(notice) == "WARNING: look out"
        assert notice.code == "01000"


@pytest.mark.server
class TestAgainstAServer:
    def test_a_statement_is_reported(self, agens) -> None:  # type: ignore[no-untyped-def]
        seen: list[QueryRecord] = []
        add_query_logger(seen.append)
        try:
            agens.execute("create vlabel thing")
            agens.execute_query("match (n:thing) return n")
        finally:
            remove_query_logger(seen.append)
        assert len(seen) == 1, "execute_query is reported; psycopg's own execute is not"
        record = seen[0]
        assert record.statement == "match (n:thing) return n"
        assert not record.failed
        assert record.elapsed > 0
        assert record.rows == 0

    def test_the_rows_are_counted(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel thing")
        agens.execute("create (:thing), (:thing)")
        seen: list[QueryRecord] = []
        add_query_logger(seen.append)
        try:
            agens.execute_query("match (n:thing) return n")
        finally:
            remove_query_logger(seen.append)
        assert seen[0].rows == 2

    def test_a_failure_is_reported_too(self, agens) -> None:  # type: ignore[no-untyped-def]
        seen: list[QueryRecord] = []
        add_query_logger(seen.append)
        try:
            with pytest.raises(psycopg.Error):
                agens.execute_query("match (n) return")
        finally:
            remove_query_logger(seen.append)
        assert len(seen) == 1
        assert seen[0].failed
        assert seen[0].error is not None
        assert seen[0].rows == 0

    def test_the_connection_is_a_number_and_not_its_settings(self, agens) -> None:  # type: ignore[no-untyped-def]
        """A driver that hands its logger the connection's settings hands it the password."""
        seen: list[QueryRecord] = []
        add_query_logger(seen.append)
        try:
            agens.execute_query("match (n) return n limit 0")
        finally:
            remove_query_logger(seen.append)
        assert isinstance(seen[0].connection, int)
        assert "password" not in repr(seen[0])

    def test_a_span_is_taken_when_tracing_is_on(self, agens) -> None:  # type: ignore[no-untyped-def]
        """Checked with a tracer of our own, so no package has to be installed to test it."""
        taken: list[str] = []

        class FakeSpan:
            def set_attribute(self, key: str, value: object) -> None:
                taken.append(f"{key}={value}")

            def __enter__(self) -> FakeSpan:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

        class FakeTracer:
            def start_as_current_span(self, name: str) -> FakeSpan:
                taken.append(f"span={name}")
                return FakeSpan()

        agensgraph.enable_tracing(FakeTracer())
        try:
            agens.execute_query("match (n) return n limit 0")
        finally:
            disable_tracing()
        assert "span=query" in taken
        assert "db.system.name=postgresql" in taken
        assert any(item.startswith("db.query.text=match (n) return n") for item in taken)
        assert not any("password" in item for item in taken)

    def test_no_parameter_ever_reaches_a_span(self, agens) -> None:  # type: ignore[no-untyped-def]
        taken: list[str] = []

        class FakeSpan:
            def set_attribute(self, key: str, value: object) -> None:
                taken.append(f"{key}={value}")

            def __enter__(self) -> FakeSpan:
                return self

            def __exit__(self, *exc: object) -> None:
                return None

        class FakeTracer:
            def start_as_current_span(self, name: str) -> FakeSpan:
                return FakeSpan()

        agens.execute("create vlabel thing")
        agensgraph.enable_tracing(FakeTracer())
        try:
            agens.execute_query("match (n:thing) where n.secret = %s return n", ("hunter2",))
        finally:
            disable_tracing()
        assert taken
        assert not any("hunter2" in item for item in taken)
