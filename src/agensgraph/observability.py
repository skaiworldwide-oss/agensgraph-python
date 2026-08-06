"""Watching what the driver does, without paying for it when nobody is watching.

Three things, and the same rule governs all of them: off costs almost nothing, and the number is
written down below rather than described.

**A span per statement.** Guarded by a module-level flag set once when tracing is configured,
not by asking the tracer whether it is recording -- by the time it has been asked the cost has
been paid. Research collected for this driver measured a *no-op* span at 38,734 instructions
against 13,603 to parse an entire vertex, and 5.37 µs of wall clock against 7.8 ns for a flag.
That figure is not reproduced here, because ``opentelemetry-api`` is not installed in this
tree. What is measured here is what a caller with no tracing pays to ask for a span: **316
nanoseconds**, being a call, a flag read and entering an object that does nothing --
278 on a later run, so read it as a few hundred rather than as a figure. The flag
itself reads in 40, and closing that gap would mean putting the test at every call site and
writing the body twice -- 270 nanoseconds against a round trip of about 87 microseconds. A span
is taken per statement and never per row.

Only ``opentelemetry-api`` is ever imported, never the SDK, and only if a caller asks for
tracing. The official line that the API alone does nothing and costs nothing is right about
emission and wrong about cost: its own documentation says every operation is a no-op *except
context propagation*, and context propagation is the part that is not free.

Asking for a :class:`Timer` costs a further 232 nanoseconds whether or not anybody is
listening, which is the larger half of what an unobserved statement pays.

**A record per statement, for a caller that would rather have data than a span.** It carries an
opaque connection id rather than the connection's own settings, because a driver that hands its
logger the connection parameters hands it the password -- and one does. What ``elapsed``
measures is written down, because a figure that silently includes preparing a statement and a
catalog lookup on a cache miss is not the figure it appears to be.

**Notices, structured.** The server says things during a statement that are worth having --
``regather_graphmeta()`` alone says two -- and every dialect that flattens them into a log line
at INFO throws away the severity, the code and the position. They are drained per statement, or
they arrive attached to the next one.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from psycopg.errors import Diagnostic

__all__ = [
    "Notice",
    "QueryRecord",
    "add_notice_listener",
    "add_query_logger",
    "disable_tracing",
    "enable_tracing",
    "notice_from_diagnostic",
    "notices_wanted",
    "remove_notice_listener",
    "remove_query_logger",
    "report_notice",
    "tracing_enabled",
]

logger = logging.getLogger("agensgraph")

# Read before every span site, and the only thing a caller who has not asked for tracing pays.
# Not a call, not an attribute lookup on an object, and not a question put to the tracer.
_tracing = False
_tracer: Any = None

_query_loggers: list[Callable[[QueryRecord], None]] = []

SYSTEM = "postgresql"
"""What to report as the kind of database.

There is no registered value for AgensGraph, and inventing one would put this driver's spans
outside every dashboard and query that already knows what to do with PostgreSQL's.
"""


class QueryRecord(NamedTuple):
    """One statement, after it finished or failed."""

    connection: int
    """An opaque number identifying the connection. Not its settings, which hold the password."""

    statement: str
    elapsed: float
    """Seconds from just before the statement was sent to just after its rows were read.

    Includes preparing it, if the server had not seen it before, and reading the write counters
    if they were asked for. It does not include waiting for a connection.
    """

    rows: int
    failed: bool
    error: BaseException | None = None


class Notice(NamedTuple):
    """Something the server said during a statement.

    Kept whole rather than flattened into a sentence. The severity is the one that is safe to
    compare -- the other is translated -- and the code is what a caller should match on.
    """

    severity: str
    code: str | None
    message: str
    detail: str | None = None
    hint: str | None = None

    def __str__(self) -> str:
        return f"{self.severity}: {self.message}"


def notice_from_diagnostic(diag: Diagnostic) -> Notice:
    """Read a notice out of what the server sent.

    The untranslated severity is used, because the other one passes through the server's
    message locale and comparing it works only in English.
    """
    return Notice(
        severity=diag.severity_nonlocalized or diag.severity or "NOTICE",
        code=diag.sqlstate,
        message=diag.message_primary or "",
        detail=diag.message_detail,
        hint=diag.message_hint,
    )


def enable_tracing(tracer: Any = None) -> None:
    """Start taking a span per statement.

    Imports ``opentelemetry-api`` and nothing else. Pass a tracer to use one already configured;
    otherwise one is asked for under this driver's name.
    """
    global _tracing, _tracer
    if tracer is None:
        try:
            from opentelemetry import trace
        except ImportError as exc:  # pragma: no cover - depends on what is installed
            raise RuntimeError(
                "tracing needs opentelemetry-api, which is not installed: "
                "pip install 'agensgraph-python[otel]'"
            ) from exc
        tracer = trace.get_tracer("agensgraph")
    _tracer = tracer
    _tracing = True


def disable_tracing() -> None:
    """Stop taking spans, and go back to costing a boolean test."""
    global _tracing, _tracer
    _tracing = False
    _tracer = None


def tracing_enabled() -> bool:
    """Whether spans are being taken."""
    return _tracing


class _NoSpan:
    """What a statement gets when nobody is listening.

    One object for the process, entered and left without building anything. A generator
    decorated with ``@contextmanager`` costs the whole generator-and-protocol dance before its
    first line runs, so a flag read inside one is not the cheap gate it looks like: 1.27
    microseconds against the 46 nanoseconds this is.
    """

    __slots__ = ()

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None


_NO_SPAN = _NoSpan()


@contextmanager
def _real_span(statement: str, graph: str | None) -> Generator[None]:
    """A span around one statement, for when tracing has been asked for."""
    with _tracer.start_as_current_span("query") as span:
        span.set_attribute("db.system.name", SYSTEM)
        span.set_attribute("db.query.text", statement)
        if graph is not None:
            span.set_attribute("db.namespace", graph)
        yield


def query_span(statement: str, *, graph: str | None = None) -> Any:
    """A span around one statement, or nothing at all.

    The statement text is attached as it will be sent. A value the caller wrote into the
    statement is part of the statement and appears there; a *parameter* is never attached,
    which is the whole reason to pass one. That is also the position the semantic conventions
    take.

    Without tracing this returns a shared object that does nothing, so a caller who never asked
    for tracing pays a flag read and nothing else.
    """
    if not _tracing:
        return _NO_SPAN
    return _real_span(statement, graph)


_notice_handlers: list[Callable[[Notice], None]] = []


def add_notice_listener(callback: Callable[[Notice], None]) -> None:
    """Be told about every notice the server sends, as a :class:`Notice`.

    psycopg delivers its own diagnostic object; this is that read into the shape above, with the
    severity that is safe to compare and the code to match on. ``regather_graphmeta()`` alone
    sends two, and planner and index notices are worth acting on.

    Registered per process and delivered by every connection this driver makes. An exception
    raised here is swallowed, as psycopg swallows one raised in its own handler.
    """
    _notice_handlers.append(callback)


def remove_notice_listener(callback: Callable[[Notice], None]) -> None:
    """Stop being told. Removing something never added is harmless."""
    while callback in _notice_handlers:
        _notice_handlers.remove(callback)


def notices_wanted() -> bool:
    """Whether anybody has asked to be told, which is what a connection checks."""
    return bool(_notice_handlers)


def report_notice(diag: Diagnostic) -> None:
    """Hand one notice to everybody listening, letting none of them stop the others."""
    if not _notice_handlers:
        return
    notice = notice_from_diagnostic(diag)
    for handler in tuple(_notice_handlers):
        try:
            handler(notice)
        except Exception:
            logger.exception("a notice listener raised")


def add_query_logger(callback: Callable[[QueryRecord], None]) -> None:
    """Be told about every statement the driver runs, after it finishes or fails.

    Called on the thread or task that ran the statement, so it should not block. An exception
    raised in it is logged and swallowed rather than replacing the statement's own outcome.
    """
    _query_loggers.append(callback)


def remove_query_logger(callback: Callable[[QueryRecord], None]) -> None:
    """Stop being told. Removing something never added is harmless.

    Compared by equality rather than identity, because a bound method is a fresh object every
    time it is referred to -- so ``remove(obj.method)`` after ``add(obj.method)`` would never
    match anything if identity were the test, and the caller would go on being told.
    """
    _query_loggers[:] = [existing for existing in _query_loggers if existing != callback]


def logging_wanted() -> bool:
    """Whether anything is listening. The only cost when nothing is."""
    return bool(_query_loggers)


def report(record: QueryRecord) -> None:
    """Hand a record to everything listening, and let none of them break the statement."""
    for callback in _query_loggers:
        try:
            callback(record)
        except Exception:
            logger.exception("a query logger raised; the statement itself was unaffected")


class Timer:
    """How long a statement took, started only when somebody is going to ask."""

    __slots__ = ("_started",)

    def __init__(self) -> None:
        self._started = time.monotonic() if logging_wanted() or _tracing else 0.0

    @classmethod
    def start(cls) -> Timer:
        """A timer, or the one shared timer that never started, when nobody will ask.

        Every statement makes one, so when nothing is listening the allocation is the whole
        cost: 229 nanoseconds for an object against 30 for reading two module flags.
        """
        return cls() if logging_wanted() or _tracing else _IDLE

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started if self._started else 0.0


_IDLE = Timer()
"""Handed out while nothing is listening, and reporting nothing because it never started."""
