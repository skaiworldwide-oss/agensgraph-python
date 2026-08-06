"""Committing a graph write in two phases.

A graph write survives ``PREPARE TRANSACTION`` and the later ``COMMIT PREPARED`` or
``ROLLBACK PREPARED``, including a write that returned rows, which the server plans as a select at
the top rather than as a graph write. Settled here rather than in prose, because a driver that
declined to support it would have to say so and this one has nothing to say.

Nothing in the driver implements this. psycopg's own two-phase methods work unchanged, and the one
case worth reporting well -- a server left at the default ``max_prepared_transactions = 0`` -- it
already reports as a ``NotSupportedError`` carrying the server's message and the name of the setting.
So these tests exist to pin the behaviour, not to cover code of ours.
"""

from __future__ import annotations

import pytest

import agensgraph

pytestmark = pytest.mark.server


@pytest.fixture
def forget_prepared(dsn):  # type: ignore[no-untyped-def]
    """Roll back anything left prepared, before the graph it wrote to is dropped.

    A prepared transaction holds its locks until it is finished, so one left behind blocks the drop
    the graph fixture does next -- which hangs rather than failing. Cleaning up from a connection of
    its own means a test that failed part way cannot take the suite with it.
    """
    yield
    with agensgraph.Connection.connect(dsn, autocommit=True) as conn:
        for row in conn.execute("select gid from pg_prepared_xacts").fetchall():
            conn.execute("rollback prepared " + _literal(row[0]))


@pytest.fixture
def two_phase(agens, dsn, forget_prepared):  # type: ignore[no-untyped-def]
    """A connection that is not in autocommit, which two-phase transactions need.

    Skips on a server that has prepared transactions turned off, which is the default.
    """
    allowed = agens.execute("show max_prepared_transactions").fetchone()[0]
    if int(allowed) == 0:
        pytest.skip("set max_prepared_transactions above zero to run the two-phase tests")
    graph = agens.label_table.graph
    agens.execute("create vlabel doc")
    conn = agensgraph.Connection.connect(dsn)
    conn.graph(graph)
    conn.commit()
    try:
        yield conn
    finally:
        # Closed rather than committed or rolled back: neither is allowed while a two-phase
        # transaction is open, and anything left prepared is finished by the fixture after this.
        conn.close()


def count(conn) -> int:  # type: ignore[no-untyped-def]
    return int(conn.execute_query("match (n:doc) return count(*)").records[0][0])


class TestAGraphWriteInTwoPhases:
    def test_a_prepared_write_is_not_visible_until_it_is_committed(
        self, two_phase, dsn
    ) -> None:  # type: ignore[no-untyped-def]
        graph = two_phase.label_table.graph
        two_phase.tpc_begin("gx_visible")
        two_phase.execute("create (:doc {a: 1})")
        two_phase.tpc_prepare()
        with agensgraph.Connection.connect(dsn, autocommit=True) as onlooker:
            onlooker.graph(graph)
            assert count(onlooker) == 0, "a prepared write was visible before it committed"
        two_phase.tpc_commit()
        with agensgraph.Connection.connect(dsn, autocommit=True) as onlooker:
            onlooker.graph(graph)
            assert count(onlooker) == 1

    def test_a_prepared_write_can_be_rolled_back(self, two_phase, dsn) -> None:  # type: ignore[no-untyped-def]
        graph = two_phase.label_table.graph
        two_phase.tpc_begin("gx_rollback")
        two_phase.execute("create (:doc {a: 2})")
        two_phase.tpc_prepare()
        two_phase.tpc_rollback()
        with agensgraph.Connection.connect(dsn, autocommit=True) as onlooker:
            onlooker.graph(graph)
            assert count(onlooker) == 0

    def test_a_write_that_returned_rows_survives_it_too(self, two_phase, dsn) -> None:  # type: ignore[no-untyped-def]
        """Such a write is a select at the top, so it takes a different path through the executor."""
        graph = two_phase.label_table.graph
        two_phase.tpc_begin("gx_returning")
        result = two_phase.execute_query("create (:doc {a: 3}) return 1")
        assert result.records == [(1,)]
        two_phase.tpc_prepare()
        two_phase.tpc_commit()
        with agensgraph.Connection.connect(dsn, autocommit=True) as onlooker:
            onlooker.graph(graph)
            assert count(onlooker) == 1

    def test_a_prepared_transaction_is_listed_while_it_waits(self, two_phase, dsn) -> None:  # type: ignore[no-untyped-def]
        """Listed from elsewhere, because a connection holding something prepared cannot be asked
        -- reading the list needs a transaction, and it already has one waiting."""
        two_phase.tpc_begin("gx_listed")
        two_phase.execute("create (:doc {a: 4})")
        two_phase.tpc_prepare()
        with agensgraph.Connection.connect(dsn) as other:
            assert "gx_listed" in [str(xid) for xid in other.tpc_recover()]

    def test_it_can_be_finished_from_another_connection(self, two_phase, dsn) -> None:  # type: ignore[no-untyped-def]
        """Which is the point of preparing: whoever commits need not be whoever wrote."""
        graph = two_phase.label_table.graph
        two_phase.tpc_begin("gx_elsewhere")
        two_phase.execute("create (:doc {a: 5})")
        two_phase.tpc_prepare()
        with agensgraph.Connection.connect(dsn, autocommit=True) as other:
            waiting = [xid for xid in other.tpc_recover() if str(xid) == "gx_elsewhere"]
            assert waiting
            other.tpc_commit(waiting[0])
            other.graph(graph)
            assert count(other) == 1


class TestWhatIsReportedWhenItIsTurnedOff:
    def test_the_message_names_the_setting(self, agens, dsn) -> None:  # type: ignore[no-untyped-def]
        """The default is zero, so this is what most servers answer."""
        allowed = int(agens.execute("show max_prepared_transactions").fetchone()[0])
        if allowed:
            pytest.skip("this server has prepared transactions turned on")
        conn = agensgraph.Connection.connect(dsn)
        try:
            conn.tpc_begin("gx_off")
            with pytest.raises(
                agensgraph.errors.NotSupportedError, match="max_prepared_transactions"
            ):
                conn.tpc_prepare()
        finally:
            # Closed rather than rolled back: the connection is still in a two-phase transaction
            # as far as psycopg is concerned, and a rollback is refused there.
            conn.close()


def _literal(text: str) -> str:
    """A single-quoted literal, for the one statement that cannot take a parameter."""
    return "'" + text.replace("'", "''") + "'"
