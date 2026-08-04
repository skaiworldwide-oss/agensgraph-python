"""Classifying failures."""

from __future__ import annotations

import pickle

import psycopg.errors as pg
import pytest

from agensgraph import errors as E
from agensgraph.errors import Retryability
from agensgraph.errors import Retryability as R

# Every entry is a failure the driver will meet and the recovery it admits.
BY_STATE = [
    ("40001", R.SAFE),
    ("40P01", R.SAFE),
    ("40003", R.UNKNOWN),
    ("08007", R.UNKNOWN),
    ("08006", R.RECONNECT),
    ("08003", R.RECONNECT),
    ("08000", R.RECONNECT),
    ("57P01", R.RECONNECT),
    ("57P02", R.RECONNECT),
    ("57P03", R.RECONNECT),
    ("57P05", R.RECONNECT),
    ("25P03", R.RECONNECT),
    ("53200", R.BACKPRESSURE),
    ("53300", R.BACKPRESSURE),
    ("55006", R.BACKPRESSURE),
    ("55P03", R.BACKPRESSURE),
    ("26000", R.RESET_STATE),
    ("57014", R.FATAL),
    ("42601", R.FATAL),
    ("22P02", R.FATAL),
    ("23505", R.FATAL),
    ("XX000", R.FATAL),
]


def error_for(state: str) -> pg.Error:
    """The exception psycopg raises for a SQLSTATE, built without a server."""
    return pg.lookup(state)("failed")


class UnregisteredCode(pg.DatabaseError):
    """A failure carrying a SQLSTATE psycopg has no class of its own for.

    The server is free to send any code, and a fork is free to add one. psycopg then
    raises whatever its two-character fallback picks while the diagnostics still carry the
    real code, which is the shape this stands in for -- ``lookup`` cannot build it, because
    ``lookup`` only knows codes it has a class for.
    """

    def __init__(self, state: str) -> None:
        # Before the base constructor, which reads the code while building its message.
        self._state = state
        super().__init__("failed")

    @property
    def sqlstate(self) -> str:
        return self._state


@pytest.mark.parametrize(("state", "expected"), BY_STATE)
def test_sqlstate_classification(state: str, expected: Retryability) -> None:
    assert E.retryability(error_for(state)) is expected


@pytest.mark.parametrize("state", ["08P01", "5300A", "57ZZZ"])
def test_unlisted_code_falls_back_to_its_class(state: str) -> None:
    by_class = {"08": R.RECONNECT, "53": R.BACKPRESSURE, "57": R.RECONNECT}
    assert E.retryability(UnregisteredCode(state)) is by_class[state[:2]]


@pytest.mark.parametrize("state", ["99999", "ZZ001", "39000"])
def test_an_unknown_class_is_fatal(state: str) -> None:
    assert E.retryability(UnregisteredCode(state)) is R.FATAL


def test_a_conflict_stays_safe_after_a_write() -> None:
    """A conflict is the server reporting that it rolled the transaction back."""
    assert E.retryability(error_for("40001"), wrote=True) is R.SAFE


def test_a_lost_connection_becomes_unknown_after_a_write() -> None:
    """What was lost is the answer to whether the commit landed."""
    assert E.retryability(error_for("08006")) is R.RECONNECT
    assert E.retryability(error_for("08006"), wrote=True) is R.UNKNOWN


def test_a_connection_failure_with_no_sqlstate_is_a_reconnect() -> None:
    assert E.retryability(pg.OperationalError("could not connect")) is R.RECONNECT


def test_a_client_side_failure_with_no_sqlstate_is_fatal() -> None:
    assert E.retryability(pg.InterfaceError("the connection is closed")) is R.FATAL


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ConnectionResetError(104, "reset"), R.RECONNECT),
        (BrokenPipeError(32, "broken pipe"), R.RECONNECT),
        (TimeoutError(), R.UNKNOWN),
        (OSError(101, "unreachable"), R.RECONNECT),
        (ValueError("not a failure of the server"), R.FATAL),
        (KeyboardInterrupt(), R.FATAL),
    ],
)
def test_classification_of_failures_that_are_not_the_servers(
    exc: BaseException, expected: Retryability
) -> None:
    assert E.retryability(exc) is expected


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (E.StaleLabelCache.for_label(9, graph="g"), R.RESET_STATE),
        (E.UnresolvedCommit.for_transaction(7, reason="lost"), R.UNKNOWN),
        (E.NetworkTimeout("timed out"), R.UNKNOWN),
        (E.NetworkError("reset"), R.RECONNECT),
        (E.ConfigurationError("a setting refused it"), R.FATAL),
        (E.CapabilityError.for_feature("x", required="2.18", found="2.16"), R.FATAL),
    ],
)
def test_the_drivers_own_failures_are_classified_by_class(
    exc: BaseException, expected: Retryability
) -> None:
    """None of these carries a SQLSTATE, so classifying by code would call them all lost."""
    assert E.retryability(exc) is expected


class TestRecoveryProperties:
    def test_only_the_undecided_and_the_hopeless_are_not_retryable(self) -> None:
        assert not R.UNKNOWN.is_retryable
        assert not R.FATAL.is_retryable
        assert all(r.is_retryable for r in (R.SAFE, R.RECONNECT, R.BACKPRESSURE, R.RESET_STATE))

    def test_exactly_one_recovery_replaces_the_connection(self) -> None:
        assert R.RECONNECT.needs_new_connection
        assert not R.RECONNECT.keeps_connection
        assert all(
            r.keeps_connection and not r.needs_new_connection
            for r in (R.SAFE, R.BACKPRESSURE, R.RESET_STATE)
        )

    def test_exactly_one_recovery_empties_the_statement_cache(self) -> None:
        assert R.RESET_STATE.clears_prepared_statements
        assert not R.SAFE.clears_prepared_statements

    def test_exactly_one_recovery_waits_longer(self) -> None:
        assert R.BACKPRESSURE.wants_longer_delay
        assert not R.SAFE.wants_longer_delay


class TestTranslate:
    """The three failures the server reports in terms a caller cannot act on."""

    def test_a_dml_refusal_names_the_setting(self) -> None:
        exc = error_for("XX000")
        exc.args = ("DML query to graph objects is not allowed",)
        replacement = E.translate(exc)
        assert isinstance(replacement, E.ConfigurationError)
        assert replacement.setting == "enable_graph_dml"
        assert "enable_graph_dml" in str(replacement)

    def test_an_eagerness_refusal_names_the_setting(self) -> None:
        exc = error_for("XX000")
        exc.args = ("eagerness plan is not allowed.",)
        replacement = E.translate(exc)
        assert isinstance(replacement, E.ConfigurationError)
        assert replacement.setting == "enable_eager"

    def test_another_internal_failure_is_left_alone(self) -> None:
        exc = error_for("XX000")
        exc.args = ("unrecognized node type: 703",)
        assert E.translate(exc) is None

    def test_a_read_only_refusal_loses_the_question_marks(self) -> None:
        exc = error_for("25006")
        exc.args = ("cannot execute ??? in a read-only transaction",)
        replacement = E.translate(exc)
        assert isinstance(replacement, E.ReadOnlyGraphWrite)
        assert "graph" in str(replacement)

    def test_the_replacement_still_matches_what_psycopg_would_have_raised(self) -> None:
        exc = error_for("25006")
        exc.args = ("cannot execute ??? in a read-only transaction",)
        replacement = E.translate(exc)
        assert isinstance(replacement, pg.ReadOnlySqlTransaction)

    def test_an_ordinary_read_only_refusal_is_left_alone(self) -> None:
        exc = error_for("25006")
        exc.args = ("cannot execute INSERT in a read-only transaction",)
        assert E.translate(exc) is None

    def test_something_that_is_not_a_database_failure_is_left_alone(self) -> None:
        assert E.translate(ValueError("elsewhere")) is None


class TestMaskDsn:
    @pytest.mark.parametrize(
        "dsn",
        [
            "postgresql://alice:s3cr3t@db.example.com:5432/graph",
            "host=db user=alice password=s3cr3t dbname=graph",
            "host=db sslpassword=s3cr3t",
        ],
    )
    def test_no_secret_survives(self, dsn: str) -> None:
        assert "s3cr3t" not in E.mask_dsn(dsn)

    def test_what_is_not_a_secret_survives(self) -> None:
        masked = E.mask_dsn("host=db port=5432 dbname=graph password=s3cr3t")
        assert "host=db" in masked
        assert "dbname=graph" in masked

    @pytest.mark.parametrize("dsn", ["", None])
    def test_nothing_in_nothing_out(self, dsn: str | None) -> None:
        assert E.mask_dsn(dsn) == ""

    def test_something_unreadable_is_not_echoed_back(self) -> None:
        """The reason it will not parse may be that it is not a connection string."""
        secret = "s3cr3t not a dsn ==="
        assert secret not in E.mask_dsn(secret)


class TestAttachments:
    def test_a_statement_is_kept_and_not_printed(self) -> None:
        exc = error_for("42601")
        E.attach_query(exc, statement="MATCH (n) RETURN n", params=("alice@example.com",))
        assert exc.statement == "MATCH (n) RETURN n"
        assert exc.params == ("alice@example.com",)
        assert "alice@example.com" not in str(exc)
        assert "MATCH" not in str(exc)

    def test_retry_history_is_kept_as_data(self) -> None:
        earlier = [error_for("40001"), error_for("40001")]
        exc = error_for("40001")
        E.attach_retry_history(exc, attempts=3, previous_errors=earlier)
        assert exc.attempts == 3
        assert exc.previous_errors == tuple(earlier)

    def test_the_message_says_how_many_attempts_there_were(self) -> None:
        exc = error_for("40001")
        E.attach_retry_history(exc, attempts=4)
        assert "reached max retries: 4" in str(exc)

    def test_a_single_attempt_leaves_the_message_alone(self) -> None:
        exc = error_for("40001")
        before = str(exc)
        E.attach_retry_history(exc, attempts=1)
        assert str(exc) == before
        assert exc.attempts == 1


class TestPickling:
    """A failure crossing a process boundary has to arrive whole."""

    @pytest.mark.parametrize(
        "exc",
        [
            E.CapabilityError.for_feature("f", required="2.18", found="2.16"),
            E.StaleLabelCache.for_label(9, graph="g"),
            E.UnresolvedCommit.for_transaction(77, reason="the connection was lost"),
            E.ConfigurationError("a setting refused it"),
        ],
    )
    def test_the_fields_survive(self, exc: pg.Error) -> None:
        restored = pickle.loads(pickle.dumps(exc))
        assert type(restored) is type(exc)
        assert str(restored) == str(exc)
        assert restored.__dict__ == exc.__dict__

    def test_a_wrapped_socket_failure_survives(self) -> None:
        exc = E.from_os_error(ConnectionResetError(104, "reset"), what="reading a result")
        restored = pickle.loads(pickle.dumps(exc))
        assert restored.errno == 104


class TestFromOsError:
    def test_the_error_number_is_kept(self) -> None:
        wrapped = E.from_os_error(ConnectionResetError(104, "reset"), what="reading")
        assert wrapped.errno == 104
        assert "reading" in str(wrapped)

    def test_a_timeout_gets_a_class_of_its_own(self) -> None:
        assert isinstance(E.from_os_error(TimeoutError(), what="reading"), E.NetworkTimeout)
        assert not isinstance(
            E.from_os_error(ConnectionResetError(104, "x"), what="reading"), E.NetworkTimeout
        )

    def test_a_wrapped_failure_is_still_an_operational_one(self) -> None:
        """So that an except clause written against psycopg keeps matching."""
        wrapped = E.from_os_error(OSError(101, "unreachable"), what="connecting")
        assert isinstance(wrapped, pg.OperationalError)


def test_the_pep_249_names_are_here() -> None:
    """A caller importing them from the driver should not have to reach for psycopg."""
    for name in (
        "Warning",
        "Error",
        "InterfaceError",
        "DatabaseError",
        "DataError",
        "OperationalError",
        "IntegrityError",
        "InternalError",
        "ProgrammingError",
        "NotSupportedError",
    ):
        assert getattr(E, name) is getattr(pg, name)
