"""Classifying failures."""

from __future__ import annotations

import pickle
from types import SimpleNamespace

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


class TestSomebodyElseGotThereFirst:
    """A statement that creates only what is missing has a different answer for a conflict."""

    @pytest.mark.parametrize("state", ["23505", "42P07"])
    def test_it_is_fatal_unless_the_caller_says_it_was_merging(self, state: str) -> None:
        """Only the caller knows whether a duplicate is its own mistake or another writer."""
        assert E.retryability(error_for(state)) is R.FATAL
        assert E.retryability(error_for(state), merging=True) is R.SAFE
        assert E.is_retryable(error_for(state), merging=True)

    @pytest.mark.parametrize("state", ["23503", "23502", "42601", "42501"])
    def test_and_nothing_else_becomes_retryable(self, state: str) -> None:
        """A missing reference or a bad statement is not somebody arriving first."""
        assert E.retryability(error_for(state), merging=True) is R.FATAL

    def test_a_write_before_it_does_not_change_the_answer(self) -> None:
        assert E.retryability(error_for("23505"), wrote=True, merging=True) is R.SAFE

    def test_the_policy_authorises_the_attempt(self) -> None:
        from agensgraph import RetryPolicy

        policy = RetryPolicy(attempts=3)
        assert policy.decide(error_for("23505"), number=1, wrote=True).retry is False
        assert policy.decide(error_for("23505"), number=1, wrote=True, merging=True).retry


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
        (E.ConfigurationError("a setting refused it"), R.FATAL),
        (E.CapabilityError.for_feature("x", required="2.18", found="2.17"), R.FATAL),
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

    def test_the_message_says_so_only_when_the_limit_was_reached(self) -> None:
        """A failure that stopped for another reason used some of its attempts, not all of
        them, and saying it reached the maximum answers the wrong question about why."""
        exhausted = error_for("40001")
        E.attach_retry_history(exhausted, attempts=4, exhausted=True)
        assert "reached max retries: 4" in str(exhausted)

        stopped = error_for("40001")
        E.attach_retry_history(stopped, attempts=2)
        assert "reached max retries" not in str(stopped)
        assert stopped.attempts == 2

    def test_recording_it_twice_does_not_say_it_twice(self) -> None:
        exc = error_for("40001")
        E.attach_retry_history(exc, attempts=4, exhausted=True)
        E.attach_retry_history(exc, attempts=4, exhausted=True)
        assert str(exc).count("reached max retries") == 1

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
            E.CapabilityError.for_feature("f", required="2.18", found="2.17"),
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


def test_the_driver_never_sees_a_raw_socket_failure() -> None:
    """Which is why nothing wraps one: psycopg is the transport, and by the time a failure
    reaches this driver it is already one of psycopg's own classes."""
    import agensgraph

    with pytest.raises(pg.OperationalError) as caught:
        agensgraph.Connection.connect("host=127.0.0.1 port=59999 connect_timeout=2")
    assert not isinstance(caught.value, OSError)


class TestKeepingRowDataOutOfAMessage:
    """PostgreSQL puts row data in DETAIL, so a plain ``logger.exception`` writes it to a log."""

    class Faked(pg.UniqueViolation):
        """psycopg's own class, since that is what redaction must leave a caller holding.

        Only ``diag`` is supplied, because the real one reads a result this test has not got.
        """

        def __init__(self, primary: str, detail: str) -> None:
            super().__init__(f"{primary}\nDETAIL:  {detail}" if detail else primary)
            self._fake = SimpleNamespace(message_primary=primary, message_detail=detail)

        @property
        def diag(self) -> SimpleNamespace:  # type: ignore[override]
            return self._fake

    def make(self, primary: str, detail: str) -> pg.Error:
        return self.Faked(primary, detail)

    def test_the_detail_is_cut_from_the_message(self) -> None:
        exc = self.make(
            "duplicate key value", "Key (email)=(alice@example.com) already exists."
        )
        assert "alice@example.com" in str(exc)
        E.redact_details(exc)
        assert str(exc) == "duplicate key value"
        assert "alice@example.com" not in str(exc)

    def test_the_data_is_still_there_to_look_at(self) -> None:
        """Redact the rendering, keep the attribute -- a post-mortem still needs the value."""
        exc = self.make(
            "duplicate key value", "Key (email)=(alice@example.com) already exists."
        )
        E.redact_details(exc)
        assert exc.diag.message_detail == "Key (email)=(alice@example.com) already exists."

    def test_it_stays_the_class_psycopg_raised(self) -> None:
        """Or every except clause written against psycopg stops matching."""
        exc = self.make("duplicate key value", "Key (email)=(x) already exists.")
        assert E.redact_details(exc) is exc
        assert isinstance(exc, pg.UniqueViolation)

    def test_doing_it_twice_changes_nothing(self) -> None:
        exc = self.make("duplicate key value", "Key (email)=(x) already exists.")
        E.redact_details(exc)
        once = str(exc)
        E.redact_details(exc)
        assert str(exc) == once

    def test_a_message_that_is_only_the_primary_is_left_alone(self) -> None:
        exc = self.make("syntax error", "")
        assert str(E.redact_details(exc)) == "syntax error"

    def test_a_caller_can_ask_for_the_detail_back(self) -> None:
        E.show_error_details(True)
        try:
            assert E.showing_error_details()
            exc = self.make(
                "duplicate key value", "Key (email)=(alice@example.com) already exists."
            )
            E.redact_details(exc)
            assert "alice@example.com" in str(exc)
        finally:
            E.show_error_details(False)
        assert not E.showing_error_details()


@pytest.mark.server
class TestARealMergeRace:
    """The classification above, against two writers rather than a constructed exception."""

    def test_the_conflict_is_the_one_merging_answers_for(self, agens, dsn: str) -> None:  # type: ignore[no-untyped-def]
        import threading
        import time

        import agensgraph
        from agensgraph import DesiredIndex, RetryPolicy

        graph = agens.label_table.graph
        agens.execute("create vlabel m")
        agens.ensure_indexes([DesiredIndex("m", ("name",), unique=True)])

        caught: dict[str, BaseException] = {}
        retried: dict[str, bool] = {}

        def writer(tag: str, hold: bool) -> None:
            conn = agensgraph.connect(dsn, autocommit=False)
            conn.graph(graph)
            try:
                conn.execute("merge (n:m {name: 'shared'})")
                if hold:
                    time.sleep(0.5)
                conn.commit()
            except Exception as exc:
                caught[tag] = exc
                conn.rollback()
                # Selecting a graph is transactional, so the rollback undid it.
                conn.graph(graph)
                policy = RetryPolicy(attempts=3)
                attempt = policy.decide(exc, number=1, wrote=True, merging=True)
                retried[tag] = attempt.retry
                if attempt.retry:
                    conn.execute("merge (n:m {name: 'shared'})")
                    conn.commit()
                    policy.succeeded()
            finally:
                conn.close()

        first = threading.Thread(target=writer, args=("a", True))
        second = threading.Thread(target=writer, args=("b", False))
        first.start()
        time.sleep(0.05)
        second.start()
        first.join()
        second.join()

        if not caught:
            pytest.skip("the two writers did not overlap, so there is no conflict to classify")
        (tag, exc), *_ = caught.items()
        assert exc.sqlstate in {"23505", "42P07"}  # type: ignore[attr-defined]
        assert E.retryability(exc) is R.FATAL, "not retryable without the caller saying why"
        assert E.retryability(exc, merging=True) is R.SAFE
        assert retried[tag] is True
        (count,) = agens.execute("match (n:m {name: 'shared'}) return count(*)").fetchone()
        assert count == 1, "the retry found what the other writer made rather than adding to it"
