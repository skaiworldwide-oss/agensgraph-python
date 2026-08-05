"""Establishing what became of a commit nobody saw finish.

Every other failure this driver classifies has an answer. This one does not: a connection lost
with a commit in flight leaves the question of whether the write landed unanswerable from the
connection that asked it, and both possible guesses are wrong some of the time -- retrying
applies it twice, giving up loses it.

The server can be asked. A transaction that wrote has an id, the id outlives the connection,
and another connection can ask what became of it. So the test that matters here is the
destructive one: kill the backend during a commit, then ask, and check that what the server
says agrees with what is actually in the graph. Anything less is testing the accessor.
"""

from __future__ import annotations

import time

import pytest

import agensgraph
from agensgraph.summary import CommitOutcome, read_outcome

pytestmark = pytest.mark.server


class TestReadingWhatTheServerSaid:
    """No server needed for these; the words are the server's, recorded."""

    @pytest.mark.parametrize(
        ("reported", "expected"),
        [
            ("committed", CommitOutcome.COMMITTED),
            ("aborted", CommitOutcome.ABORTED),
            ("in progress", CommitOutcome.IN_PROGRESS),
            (None, CommitOutcome.UNKNOWN),
            ("something new", CommitOutcome.UNKNOWN),
        ],
    )
    def test_the_reading(self, reported: str | None, expected: CommitOutcome) -> None:
        assert read_outcome(reported) is expected

    def test_nothing_reported_is_not_the_same_as_aborted(self) -> None:
        """A record truncated away is the server declining to say, not saying no."""
        assert read_outcome(None) is not CommitOutcome.ABORTED
        assert not read_outcome(None).safe_to_retry
        assert not read_outcome(None).is_settled

    def test_only_an_abort_is_safe_to_run_again(self) -> None:
        assert CommitOutcome.ABORTED.safe_to_retry
        for other in (
            CommitOutcome.COMMITTED,
            CommitOutcome.IN_PROGRESS,
            CommitOutcome.UNKNOWN,
        ):
            assert not other.safe_to_retry

    def test_only_a_finished_transaction_is_settled(self) -> None:
        assert CommitOutcome.COMMITTED.is_settled
        assert CommitOutcome.ABORTED.is_settled
        assert not CommitOutcome.IN_PROGRESS.is_settled
        assert not CommitOutcome.UNKNOWN.is_settled


class TestAskingTheServer:
    @pytest.fixture
    def writer(self, agens):  # type: ignore[no-untyped-def]
        agens.execute("create vlabel receipt")
        agens.autocommit = False
        yield agens
        agens.rollback()
        agens.autocommit = True

    @pytest.fixture
    def onlooker(self, dsn: str):  # type: ignore[no-untyped-def]
        """A second connection, because the point is to survive the first one."""
        with agensgraph.connect(dsn, autocommit=True) as conn:
            yield conn

    def test_a_read_is_given_no_transaction_id(self, writer) -> None:  # type: ignore[no-untyped-def]
        """So asking costs nothing on a transaction that has nothing to lose."""
        writer.execute("match (n:receipt) return n").fetchall()
        assert writer.transaction_id() is None

    def test_a_write_is_given_one(self, writer) -> None:  # type: ignore[no-untyped-def]
        writer.execute("create (:receipt {n: 1})")
        assert writer.transaction_id() is not None

    def test_one_can_be_taken_before_writing(self, writer) -> None:  # type: ignore[no-untyped-def]
        """Which is what a caller does when it intends to be able to ask afterwards."""
        assert writer.transaction_id() is None
        taken = writer.transaction_id(assign=True)
        assert taken is not None
        assert writer.transaction_id() == taken

    def test_a_transaction_still_open_reports_as_much(self, writer, onlooker) -> None:  # type: ignore[no-untyped-def]
        writer.execute("create (:receipt {n: 2})")
        xid = writer.transaction_id()
        assert xid is not None
        assert onlooker.resolve_commit(xid) is CommitOutcome.IN_PROGRESS

    def test_a_committed_one_reports_committed(self, writer, onlooker) -> None:  # type: ignore[no-untyped-def]
        writer.execute("create (:receipt {n: 3})")
        xid = writer.transaction_id()
        writer.commit()
        assert xid is not None
        assert onlooker.resolve_commit(xid) is CommitOutcome.COMMITTED

    def test_a_rolled_back_one_reports_aborted_and_is_safe_to_repeat(
        self, writer, onlooker
    ) -> None:  # type: ignore[no-untyped-def]
        writer.execute("create (:receipt {n: 4})")
        xid = writer.transaction_id()
        writer.rollback()
        assert xid is not None
        outcome = onlooker.resolve_commit(xid)
        assert outcome is CommitOutcome.ABORTED
        assert outcome.safe_to_retry

    def test_a_record_truncated_away_says_nothing_rather_than_guessing(self, onlooker) -> None:  # type: ignore[no-untyped-def]
        """Transaction 3 is old enough on any real cluster that its record has gone."""
        assert onlooker.resolve_commit(3) is CommitOutcome.UNKNOWN


class TestKillingTheBackendDuringCommit:
    """The test the whole mechanism exists for.

    The backend is killed while it commits, so the connection dies without saying what
    happened. Then the server is asked, and what it says is checked against what is actually
    in the graph -- because an answer that does not match the data would be worse than none.
    """

    @pytest.fixture
    def onlooker(self, dsn: str):  # type: ignore[no-untyped-def]
        with agensgraph.connect(dsn, autocommit=True) as conn:
            yield conn

    @pytest.mark.parametrize("delay", [0.0, 0.001, 0.005, 0.02])
    def test_what_the_server_says_matches_what_is_there(
        self, dsn: str, onlooker, delay: float
    ) -> None:  # type: ignore[no-untyped-def]
        import threading

        graph = f"kill_{int(delay * 1000)}"
        onlooker.execute(f'drop graph if exists "{graph}" cascade')
        onlooker.execute(f'create graph "{graph}"')
        try:
            setup = agensgraph.connect(dsn, autocommit=True)
            setup.graph(graph)
            setup.execute("create vlabel receipt")
            setup.close()

            writer = agensgraph.connect(dsn)
            writer.graph(graph)
            writer.execute("create (:receipt {mark: 'once'})")
            xid = writer.transaction_id()
            assert xid is not None
            victim = writer.info.backend_pid

            def kill() -> None:
                time.sleep(delay)
                onlooker.execute("select pg_terminate_backend(%s)", (victim,))

            killer = threading.Thread(target=kill)
            killer.start()
            try:
                writer.commit()
                committed_cleanly = True
            except Exception:
                committed_cleanly = False
            finally:
                killer.join()
                writer.close()

            outcome = onlooker.resolve_commit(xid)
            onlooker.graph(graph)
            found = onlooker.execute_query(
                "match (n:receipt) where n.mark = %s return n", ("once",)
            ).records

            # Whatever happened, the server's answer and the graph agree, and the row is
            # there exactly once or not at all -- never twice.
            assert len(found) in (0, 1)
            if outcome is CommitOutcome.COMMITTED:
                assert len(found) == 1
            elif outcome is CommitOutcome.ABORTED:
                assert len(found) == 0
            else:
                pytest.fail(f"the outcome should be settled by now, got {outcome}")
            # And a clean commit is only ever reported as committed.
            if committed_cleanly:
                assert outcome is CommitOutcome.COMMITTED
        finally:
            onlooker.execute("reset graph_path")
            onlooker.execute(f'drop graph "{graph}" cascade')
