"""What a failure is, and what to do about it.

Two questions with different answers, kept apart. *What kind* of error something is
comes from psycopg: the server sends a SQLSTATE, psycopg raises the class registered
against it, and a graph failure is caught by the same PEP-249 class as any other, with
nothing of ours in the way. *What to do* about it is asked of this module instead,
because the class does not answer it -- a write conflict and a lock timeout are both an
``OperationalError``, and the name says nothing about whether running the statement again
would help, or whether the connection it ran on is still usable.

The server mints no SQLSTATE of its own. A graph write conflict is reported as 40001,
the code an ordinary row conflict uses, so there is nothing to register and every retry
loop that already understands PostgreSQL's codes already understands a graph write.

Three failures do not fit that, and each is named in :func:`translate`: two settings whose
refusal arrives as an internal error carrying nothing but a message, and a read-only
transaction refusing a graph write under a message with a literal ``???`` in it.

Every class here takes the message as its only argument, with the rest of its fields
class-level defaults assigned on the instance. Unpickling an exception calls the class
with ``args``, which holds the message and nothing else, so a required second parameter
would make the exception unpicklable -- and psycopg carries the instance dictionary
through, so fields assigned this way survive the round trip.
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING

import psycopg_pool as _pool
from psycopg import conninfo
from psycopg import errors as _pg
from psycopg.errors import (
    DatabaseError as DatabaseError,
)
from psycopg.errors import (
    DataError as DataError,
)
from psycopg.errors import (
    Error as Error,
)
from psycopg.errors import (
    IntegrityError as IntegrityError,
)
from psycopg.errors import (
    InterfaceError as InterfaceError,
)
from psycopg.errors import (
    InternalError as InternalError,
)
from psycopg.errors import (
    NotSupportedError as NotSupportedError,
)
from psycopg.errors import (
    OperationalError as OperationalError,
)
from psycopg.errors import (
    ProgrammingError as ProgrammingError,
)
from psycopg.errors import (
    Warning as Warning,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "STRING_TYPE_HINT",
    "BatchFailed",
    "CapabilityError",
    "ConfigurationError",
    "DataError",
    "DatabaseError",
    "Error",
    "IntegrityError",
    "InterfaceError",
    "InternalError",
    "NetworkError",
    "NetworkTimeout",
    "NotSupportedError",
    "OperationalError",
    "ProgrammingError",
    "ReadOnlyGraphWrite",
    "Retryability",
    "StaleGeneration",
    "StaleLabelCache",
    "UnresolvedCommit",
    "Warning",
    "attach_query",
    "attach_retry_history",
    "explain_string_type",
    "from_os_error",
    "is_retryable",
    "mask_dsn",
    "retryability",
    "translate",
]


class Retryability(enum.Enum):
    """What recovery a failure admits.

    A single boolean cannot carry this. An invalid prepared statement name and a lost
    connection are both worth another attempt, but one keeps the connection and empties
    the statement cache while the other throws the connection away; treating them alike
    means a stale cache is "fixed" by reconnecting, which hides the bug that keeps
    refilling it. Resource exhaustion is worth retrying too, but sooner is worse than
    later, so it cannot share a delay with a write conflict.

    Anything unrecognised is :attr:`FATAL`. A failure wrongly called retryable can apply
    a write twice; one wrongly called fatal only surfaces an error the caller can see.
    """

    SAFE = "safe"
    """The transaction is known to have aborted. Run it again on the same connection."""

    RECONNECT = "reconnect"
    """The connection is gone. Take a new one, then run it again."""

    BACKPRESSURE = "backpressure"
    """The server is short of something. Run it again, later than usual."""

    RESET_STATE = "reset_state"
    """Session state the statement relied on is gone. Clear it and keep the connection."""

    UNKNOWN = "unknown"
    """Whether the transaction committed is not known. Resolve it before deciding."""

    FATAL = "fatal"
    """Another attempt would fail the same way."""

    @property
    def is_retryable(self) -> bool:
        """Whether another attempt is sound without first establishing anything else."""
        return self in _RETRYABLE

    @property
    def needs_new_connection(self) -> bool:
        """Whether the connection must be replaced before another attempt."""
        return self is Retryability.RECONNECT

    @property
    def keeps_connection(self) -> bool:
        """Whether the connection remains usable."""
        return self in _KEEPS_CONNECTION

    @property
    def clears_prepared_statements(self) -> bool:
        """Whether the statement cache has to be emptied first."""
        return self is Retryability.RESET_STATE

    @property
    def wants_longer_delay(self) -> bool:
        """Whether the usual delay is too short, because the server is the constraint."""
        return self is Retryability.BACKPRESSURE


_RETRYABLE = frozenset(
    {
        Retryability.SAFE,
        Retryability.RECONNECT,
        Retryability.BACKPRESSURE,
        Retryability.RESET_STATE,
    }
)
_KEEPS_CONNECTION = frozenset(
    {Retryability.SAFE, Retryability.BACKPRESSURE, Retryability.RESET_STATE}
)

_BY_STATE: dict[str, Retryability] = {
    # A graph write conflict lands here, and it is the ordinary outcome of two writers
    # touching one element rather than an exotic one: SET and MERGE report a concurrent
    # update this way and DELETE a concurrent delete. In every case the transaction has
    # already been rolled back, so the same connection can run it again.
    "40001": Retryability.SAFE,
    "40P01": Retryability.SAFE,
    # Named for saying that the statement's completion is not known, which is the one
    # thing a safe retry may not assume.
    "40003": Retryability.UNKNOWN,
    "08007": Retryability.UNKNOWN,
    "57P01": Retryability.RECONNECT,
    "57P02": Retryability.RECONNECT,
    "57P03": Retryability.RECONNECT,
    "57P05": Retryability.RECONNECT,
    "25P03": Retryability.RECONNECT,
    "53200": Retryability.BACKPRESSURE,
    "53300": Retryability.BACKPRESSURE,
    "55006": Retryability.BACKPRESSURE,
    "55P03": Retryability.BACKPRESSURE,
    "26000": Retryability.RESET_STATE,
    # Cancellation has four causes that all report this one code, and which of them it
    # was cannot be read out of the message, which is translated. Only the driver knows
    # whether it asked for the cancellation, so this is fatal here and the caller that
    # cancelled decides otherwise for itself.
    "57014": Retryability.FATAL,
}

# A class-wide answer only where every code in the class shares it. Class 40 does not:
# a serialization failure aborted, an unknown completion did not say. The full code is
# looked up first, so a class listed here does not overrule a code named above.
_BY_CLASS: dict[str, Retryability] = {
    "08": Retryability.RECONNECT,
    "53": Retryability.BACKPRESSURE,
    "57": Retryability.RECONNECT,
}


def retryability(exc: BaseException, *, wrote: bool = False) -> Retryability:
    """Classify a failure.

    Pass ``wrote=True`` when the transaction had issued a write. A connection lost
    mid-transaction is ordinarily worth reconnecting for, but if the transaction had
    written then whether its commit reached the server is exactly what has been lost, so
    it becomes :attr:`Retryability.UNKNOWN` and wants resolving rather than repeating.
    A conflict stays safe either way, because a conflict is the server saying it rolled
    the transaction back.
    """
    for kind, answer in _OURS:
        if isinstance(exc, kind):
            # Nothing raised here carries a SQLSTATE, so what it means has to be said
            # rather than looked up, and saying nothing would leave every one of them
            # reading as a lost connection.
            found = answer
            break
    else:
        found = _classify_foreign(exc)

    if wrote and found is Retryability.RECONNECT:
        return Retryability.UNKNOWN
    return found


def _classify_foreign(exc: BaseException) -> Retryability:
    if isinstance(exc, _pg.Error):
        state = exc.sqlstate
        if state is None:
            # No SQLSTATE means the server never answered, which is a transport failure
            # if it is reported as one and a driver fault otherwise.
            found = (
                Retryability.RECONNECT
                if isinstance(exc, _pg.OperationalError)
                else Retryability.FATAL
            )
        else:
            found = _BY_STATE.get(state) or _BY_CLASS.get(state[:2], Retryability.FATAL)
    elif isinstance(exc, TimeoutError):
        # Checked before OSError, which it inherits from. A read that timed out says
        # nothing about whether the server ran the statement.
        found = Retryability.UNKNOWN
    elif isinstance(exc, OSError):
        found = Retryability.RECONNECT
    else:
        found = Retryability.FATAL
    return found


def is_retryable(exc: BaseException, *, wrote: bool = False) -> bool:
    """Whether another attempt is sound. See :func:`retryability` for the rest."""
    return retryability(exc, wrote=wrote).is_retryable


class CapabilityError(_pg.NotSupportedError):
    """A feature the connected server does not have."""

    feature: str | None = None
    required: str | None = None
    found: str | None = None

    @classmethod
    def for_feature(cls, feature: str, *, required: str, found: str) -> CapabilityError:
        """Build the refusal, naming the version that would carry the feature."""
        exc = cls(f"{feature} needs AgensGraph {required} or later; this server is {found}")
        exc.feature = feature
        exc.required = required
        exc.found = found
        return exc


class ConfigurationError(_pg.OperationalError):
    """A setting, rather than the statement, is what refused the work."""

    setting: str | None = None


class BatchFailed(_pg.Error):
    """A statement in a pipelined batch failed, and which one is not known.

    A pipeline reports an error against the wrong statement. Measured with four statements of which
    only the second was bad: the *first* raised the error and the rest raised with no SQLSTATE at
    all. So the batch is reported as a whole.

    ``statements`` holds what was sent, in order, and the failure the server reported is the
    ``__cause__``. Running them one at a time is how to find the one at fault -- not done here,
    because replaying a write would apply it twice.
    """

    statements: tuple[str, ...] = ()


class ReadOnlyGraphWrite(_pg.ReadOnlySqlTransaction):
    """A graph write in a transaction that was opened read-only."""


class UnresolvedCommit(_pg.OperationalError):
    """Whether a transaction committed could not be established.

    Raised when the connection was lost with a commit in flight and the server could not
    afterwards say what became of the transaction. The transaction id it was asked about
    is kept, so a caller can ask again later.
    """

    transaction_id: int | None = None

    @classmethod
    def for_transaction(cls, transaction_id: int | None, *, reason: str) -> UnresolvedCommit:
        """Build the report, naming the transaction whose fate is open."""
        exc = cls(f"cannot establish whether the transaction committed: {reason}")
        exc.transaction_id = transaction_id
        return exc


class StaleLabelCache(_pg.OperationalError):
    """A label the connection's label table cannot name.

    Only the composite rendering can reach this, because only it leaves the label name out
    and asks for it to be resolved from the label id. Either the label was created after the
    table was filled, or the session was moved to another graph and the table was dropped
    because the ids of one graph mean nothing in another. Both are answered by filling the
    table again, with ``refresh_labels()``, and running the statement once more.
    """

    labid: int | None = None
    graph: str | None = None

    @classmethod
    def for_label(cls, labid: int, *, graph: str | None) -> StaleLabelCache:
        """Build the report, naming the label id that could not be resolved."""
        if graph is None:
            reason = (
                "the label table names no graph, so either it was never filled or the "
                "session was moved to another graph"
            )
        else:
            reason = f"it is not a label of graph {graph!r}, which the table was filled from"
        exc = cls(f"label id {labid} cannot be named: {reason}. Call refresh_labels() first")
        exc.labid = labid
        exc.graph = graph
        return exc


class StaleGeneration(_pg.OperationalError):
    """A pooled connection belonging to a generation the pool has retired.

    Raised by the pool's own reset hook, which is how psycopg is told to close a connection
    rather than reuse it. It is not a failure a caller sees: by the time it is raised the
    caller has finished and let the connection go.
    """

    generation: int | None = None
    current: int | None = None

    @classmethod
    def for_connection(cls, generation: int | None, *, current: int) -> StaleGeneration:
        """Build the report, naming the generation the connection came from."""
        exc = cls(
            f"this connection belongs to generation {generation}, and the pool has moved on "
            f"to {current}, so it is closed rather than reused"
        )
        exc.generation = generation
        exc.current = current
        return exc


class NetworkError(_pg.OperationalError):
    """A socket failure, carrying the number the operating system gave it."""

    errno: int | None = None


class NetworkTimeout(NetworkError):
    """A socket operation that ran out of time."""


def from_os_error(exc: OSError, *, what: str) -> NetworkError:
    """Wrap a socket failure, keeping its error number.

    The number is the first thing anyone asks for and it is lost by every wrapper that
    only carries the message across, so it is kept on the wrapper as well. A timeout gets
    a class of its own, because it leaves the statement's fate open where a refused or
    reset connection does not.
    """
    cls = NetworkTimeout if isinstance(exc, TimeoutError) else NetworkError
    wrapped = cls(f"{what}: {exc}")
    wrapped.errno = exc.errno
    return wrapped


# Most specific first, since a class is matched by the first entry it belongs to.
#
# The pool's three failures are here because they are all an OperationalError carrying no
# SQLSTATE, so classifying by code calls every one of them a lost connection -- and then a
# blanket "reconnect on an OperationalError" treats a pool with no free connections as a dead
# socket. None of them is that. There is no connection to replace, so replacing one is
# meaningless; waiting for the pool is worth doing later rather than sooner; and a pool that
# has been closed will never hand anything over however long anyone waits.
_OURS: tuple[tuple[type[BaseException], Retryability], ...] = (
    (NetworkTimeout, Retryability.UNKNOWN),
    (NetworkError, Retryability.RECONNECT),
    (StaleLabelCache, Retryability.RESET_STATE),
    (StaleGeneration, Retryability.RECONNECT),
    (UnresolvedCommit, Retryability.UNKNOWN),
    (ConfigurationError, Retryability.FATAL),
    (CapabilityError, Retryability.FATAL),
    (_pool.PoolClosed, Retryability.FATAL),
    (_pool.PoolTimeout, Retryability.BACKPRESSURE),
    (_pool.TooManyRequests, Retryability.BACKPRESSURE),
)


# The message the server gives is the only thing that distinguishes these two from a
# genuine internal fault, because both are reported without a SQLSTATE of their own and
# so arrive as XX000. The match is against the whole message, so anything the server
# words differently stays an internal error rather than being read as a setting.
_CONFIGURATION_GATES: tuple[tuple[str, str, str], ...] = (
    (
        "DML query to graph objects is not allowed",
        "enable_graph_dml",
        "plain SQL data modification of a label table is refused while enable_graph_dml "
        "is off, which is its default and which only a superuser may change; write "
        "through Cypher instead",
    ),
    (
        "eagerness plan is not allowed.",
        "enable_eager",
        "a write of more than one clause needs enable_eager on; with it off the statement "
        "is refused outright rather than planned another way",
    ),
)


def translate(exc: BaseException) -> _pg.Error | None:
    """A better exception for the failures the server describes badly, or ``None``.

    Three of them. Two settings refuse work through a path that attaches no SQLSTATE, so
    they arrive as an internal error and read as a driver or server fault when they are
    neither -- those become a :class:`ConfigurationError` naming the setting. And a graph
    write in a read-only transaction is classified correctly but described as ``cannot
    execute ??? in a read-only transaction``, because the server has no name for the
    command; that message cannot be shown to anyone, so it is replaced.

    The replacement is always a subclass of what psycopg would have raised, so an
    ``except`` clause written against psycopg keeps matching.
    """
    if not isinstance(exc, _pg.Error):
        return None
    state = exc.sqlstate
    message = str(exc).strip()

    if state == "XX000":
        for text, setting, explanation in _CONFIGURATION_GATES:
            if message == text:
                replacement = ConfigurationError(f"{text}: {explanation}")
                replacement.setting = setting
                return replacement
        return None

    if state == "25006" and "???" in message:
        return ReadOnlyGraphWrite(
            "cannot write to a graph in a read-only transaction "
            "(the server has no name for the command and reports it as '???')"
        )

    return None


# Reported when a string reached a position that wanted another type: no operator exists for
# the pair, or a column of one type was given an expression of another.
_WRONG_TYPE = frozenset({"42883", "42804"})

STRING_TYPE_HINT = (
    "this driver sends a string as text rather than leaving its type to be worked out, so "
    "that looking a property up by name reads the string as a string. In plain SQL a string "
    "standing for another type wants a cast -- %s::date, %s::uuid, %s::int -- or pass the "
    "value's own type, or wrap it in agensgraph.Unspecified to have the server work the type "
    "out as it did before."
)


def explain_string_type(exc: BaseException) -> str | None:
    """Advice for a failure that a string's declared type caused, or ``None``.

    Kept apart from :func:`translate`, which replaces an exception. Nothing is wrong with the
    class or the code here -- the server is right that there is no ``date = text`` operator --
    so what is added is the one thing the message cannot know, which is why the parameter was
    text in the first place. That is advice for a person rather than data a caller matches on,
    which is what a note is for.
    """
    if not isinstance(exc, _pg.Error) or exc.sqlstate not in _WRONG_TYPE:
        return None
    if " text" not in str(exc):
        return None
    return STRING_TYPE_HINT


def attach_query(exc: BaseException, *, statement: str | None, params: object = None) -> None:
    """Record what was running when a failure happened, without rendering it.

    Kept as attributes and left out of the message. The server already puts row data in
    a failure's detail field -- a unique violation names the conflicting value -- so an
    error is a place secrets arrive at whether or not the driver puts them there, and
    nothing here adds to that by writing parameters into the text that ordinary logging
    will print.
    """
    exc.statement = statement  # type: ignore[attr-defined]
    exc.params = params  # type: ignore[attr-defined]


def attach_retry_history(
    exc: BaseException, *, attempts: int, previous_errors: Sequence[BaseException] = ()
) -> None:
    """Record how many attempts a failure survived, and what the earlier ones were.

    Typed attributes rather than notes, because this is data a caller reads and matches
    on rather than advice for a person. The count also goes into the message, which is
    the difference between a report saying a statement failed and one saying it failed
    every time it was tried.
    """
    exc.attempts = attempts  # type: ignore[attr-defined]
    exc.previous_errors = tuple(previous_errors)  # type: ignore[attr-defined]
    if attempts > 1 and exc.args:
        first = f"{exc.args[0]} (reached max retries: {attempts})"
        exc.args = (first, *exc.args[1:])


def mask_dsn(dsn: str | None) -> str:
    """A connection string safe to write down.

    Every value whose key names a password is replaced. A string that will not parse
    yields a placeholder rather than itself, because the reason it will not parse may be
    that it is not a connection string at all.
    """
    if not dsn:
        return ""
    try:
        parts = conninfo.conninfo_to_dict(dsn)
        masked = {
            key: ("***" if "password" in key else str(value))
            for key, value in parts.items()
            if value is not None
        }
        return conninfo.make_conninfo(**masked)
    except Exception:
        return "<unreadable connection string>"
