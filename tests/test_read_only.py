"""Running a statement that came from somewhere else.

The statements here are the ones a caller did not write: a model's answer, most often. What is
asserted is that the server refuses them rather than the driver, because a driver that reads the
text to decide has to recognise every way of writing and does not.

Every framework surveyed reads the text, and each of those readings lets `INSERT`, `TRUNCATE` and
`COPY` through -- none of which is Cypher, and all of which this server runs, because it is
PostgreSQL underneath.
"""

from __future__ import annotations

import contextlib

import psycopg
import pytest

import agensgraph
from agensgraph.errors import ConfigurationError, ReadOnlyGraphWrite

pytestmark = pytest.mark.server

PROBE_PASSWORD = "agens-probe-only"
"""Given to every role these tests make, so they connect as it whether or not the server trusts
local connections. Without one, a server that asks for a password skipped these tests."""


WRITES = [
    pytest.param("create (:thing {a: 1})", id="cypher-create"),
    pytest.param("match (n:thing) set n.a = 2", id="cypher-set"),
    pytest.param("match (n:thing) delete n", id="cypher-delete"),
    pytest.param("create vlabel sneaked", id="cypher-ddl"),
]


@pytest.fixture
def unprivileged(agens, dsn):  # type: ignore[no-untyped-def]
    """A connection as a role that cannot run a command on the server's host.

    The boundary is only a boundary for such a role, so the tests that assert it need one. A
    server that will not let this test make one has nothing to say about the question, and says
    so rather than passing.
    """
    name = "agens_read_only_probe"
    graph = agens.execute("select current_setting('graph_path')").fetchone()[0]
    # Left behind by a run that did not reach its own cleanup.
    with contextlib.suppress(psycopg.Error):
        agens.execute(f'drop owned by "{name}" cascade')
    try:
        agens.execute(f'drop role if exists "{name}"')
        agens.execute(f"create role \"{name}\" login password '{PROBE_PASSWORD}'")
        agens.execute(f'grant usage on schema "{graph}" to "{name}"')
        agens.execute(f'grant select on all tables in schema "{graph}" to "{name}"')
    except psycopg.Error:
        pytest.skip("this role cannot create another, so the boundary cannot be tested")
    # The point of the fixture is the other role, so the connection has to be made as it.
    theirs = psycopg.conninfo.make_conninfo(dsn, user=name, password=PROBE_PASSWORD)
    try:
        with agensgraph.connect(theirs, autocommit=False) as conn:
            yield conn
    except psycopg.OperationalError:
        pytest.skip("this server will not let the test connect as another role")
    finally:
        agens.execute(f'drop owned by "{name}" cascade')
        agens.execute(f'drop role if exists "{name}"')


class TestTheServerRefusesTheWrite:
    @pytest.mark.parametrize("statement", WRITES)
    def test_a_write_is_refused_whatever_it_is_spelled_like(self, agens, statement) -> None:  # type: ignore[no-untyped-def]
        """The driver never reads the statement to decide, so its spelling cannot matter."""
        agens.execute("create vlabel thing")
        agens.autocommit = False
        with (
            pytest.raises(psycopg.Error) as caught,
            agens.read_only_transaction(allow_server_programs=True),
        ):
            agens.execute(statement)
        assert caught.value.sqlstate == "25006"

    def test_a_read_is_not_refused(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel thing")
        agens.execute("create (:thing {a: 1})")
        agens.autocommit = False
        with agens.read_only_transaction(allow_server_programs=True):
            (row,) = agens.execute_query("match (n:thing) return n").records
        assert row[0].properties == {"a": 1}

    def test_the_refusal_says_what_happened(self, agens) -> None:  # type: ignore[no-untyped-def]
        """The server names the command that was refused, not what it was trying to do.

        A tag either way -- '???' up to 2.17, where a graph write had no name, and 'CYPHER' from
        2.18 -- so the driver says it in terms of the graph, which is the one part of this only a
        driver can supply. Where the server lets a Cypher write through a read-only transaction,
        2.14 and 2.15, the driver refuses it before sending and says that instead.
        """
        agens.execute("create vlabel thing")
        agens.autocommit = False
        with (
            pytest.raises(agensgraph.errors.ReadOnlyGraphWrite) as caught,
            agens.read_only_transaction(allow_server_programs=True),
        ):
            agens.execute("create (:thing)")
        if agens.capabilities.has("cypher_write_enforced"):
            assert "cannot write to a graph in a read-only transaction" in str(caught.value)
        else:
            assert "refused before it was sent" in str(caught.value)


class TestWhatAReadOnlyTransactionDoesNotStop:
    """``COPY ... TO PROGRAM`` runs a command on the server's host and is not a write.

    It takes rows out of the database rather than putting any in, so there is nothing for a
    read-only transaction to refuse. This is why the driver asks about the role instead.
    """

    def test_a_role_that_could_run_a_program_is_refused_the_transaction(self, agens) -> None:  # type: ignore[no-untyped-def]
        if not agens.can_run_server_programs():
            pytest.skip(
                "this role could not run a program anyway, so there is nothing to refuse"
            )
        agens.autocommit = False
        with pytest.raises(ConfigurationError) as caught, agens.read_only_transaction():
            pass
        assert "pg_execute_server_program" in str(caught.value)

    def test_the_answer_is_kept_rather_than_asked_twice(self, agens) -> None:  # type: ignore[no-untyped-def]
        """A privilege does not change while a connection is open, so it is asked about once."""
        seen: list[str] = []

        def record_statement(record) -> None:  # type: ignore[no-untyped-def]
            seen.append(record.statement)

        agensgraph.add_query_logger(record_statement)
        try:
            agens.can_run_server_programs()
            asked_once = sum("pg_execute_server_program" in text for text in seen)
            agens.can_run_server_programs()
            asked_again = sum("pg_execute_server_program" in text for text in seen)
        finally:
            agensgraph.remove_query_logger(record_statement)
        assert asked_once == 1
        assert asked_again == 1, "the second call reads what the first kept"


class TestTheBoundaryHoldsForAnOrdinaryRole:
    """The whole point, asserted as one piece: a role without the privilege cannot get out."""

    def test_a_write_and_a_program_are_both_refused(self, unprivileged, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel thing")
        agens.execute("create (:thing {a: 1})")
        graph = agens.execute("select current_setting('graph_path')").fetchone()[0]
        unprivileged.autocommit = True
        unprivileged.graph(graph)
        unprivileged.autocommit = False
        assert unprivileged.can_run_server_programs() is False

        refused: dict[str, str | None] = {}
        for name, statement in (
            ("write", "create (:thing)"),
            ("program", "copy (select 1) to program 'true'"),
        ):
            with (
                pytest.raises(psycopg.Error) as caught,
                unprivileged.read_only_transaction(),
            ):
                unprivileged.execute(statement)
            refused[name] = caught.value.sqlstate
        assert refused["write"] == "25006", "the transaction refuses the write"
        assert refused["program"] == "42501", "the privilege refuses the program"
        # One hidden behind another statement never reaches the server at all.
        with (
            pytest.raises(ValueError, match="more than one statement"),
            unprivileged.read_only_transaction(),
        ):
            unprivileged.execute("select 1; copy (select 1) to program 'true'")


class TestItLeavesTheSessionAsItFoundIt:
    """A read-only transaction allows ``SET``, and a setting outlives the transaction that
    committed it. On a pooled connection the rest of that session belongs to somebody else.
    """

    def test_a_setting_made_inside_does_not_reach_the_next_borrower(self, dsn: str) -> None:
        # One connection, so the next borrower is provably the same one.
        pool = agensgraph.ConnectionPool(dsn, min_size=1, max_size=1)
        pool.open(wait=True)
        try:
            with pool.connection() as conn:
                conn.set_autocommit(False)
                with conn.read_only_transaction(allow_server_programs=True):
                    conn.execute("set search_path = 'left_behind'")
            with pool.connection() as conn:
                (search_path,) = conn.execute("show search_path").fetchone()
        finally:
            pool.close()
        assert "left_behind" not in search_path

    def test_a_read_inside_still_answers(self, agens) -> None:  # type: ignore[no-untyped-def]
        """Rolling back at the end does not take back rows already read."""
        agens.execute("create vlabel thing")
        agens.execute("create (:thing {a: 1})")
        agens.autocommit = False
        with agens.read_only_transaction(allow_server_programs=True):
            (row,) = agens.execute_query("match (n:thing) return n.a").records
        assert row[0] == 1

    def test_it_rolls_back_itself_and_not_the_transaction_around_it(self, agens) -> None:  # type: ignore[no-untyped-def]
        """Nested, the block is a savepoint, so what it undoes is its own work."""
        agens.execute("create vlabel thing")
        agens.autocommit = False
        with agens.transaction():
            agens.execute("create (:thing {a: 1})")
            with agens.read_only_transaction(allow_server_programs=True):
                agens.execute("set search_path = 'inner_only'")
            (search_path,) = agens.execute("show search_path").fetchone()
            assert "inner_only" not in search_path, "the inner block undid its own setting"
            (count,) = agens.execute("match (n:thing) return count(*)").fetchone()
        assert count == 1, "and left the write in the transaction around it alone"


@pytest.fixture
def reaching(agens, dsn):  # type: ignore[no-untyped-def]
    """A connection as a role that can *reach* the privilege without holding it.

    A ``noinherit`` role's memberships carry nothing until ``set role`` names one. Such a role is as
    dangerous as one that holds the privilege outright, because naming a membership moves no rows and
    a read-only transaction has no reason to refuse it.
    """
    holder, name = "agens_program_holder", "agens_program_reacher"
    for role in (name, holder):
        with contextlib.suppress(psycopg.Error):
            agens.execute(f'drop owned by "{role}" cascade')
    try:
        agens.execute(f'drop role if exists "{name}"')
        agens.execute(f'drop role if exists "{holder}"')
        agens.execute(f'create role "{holder}"')
        agens.execute(f'grant pg_execute_server_program to "{holder}"')
        agens.execute(f"create role \"{name}\" login noinherit password '{PROBE_PASSWORD}'")
        agens.execute(f'grant "{holder}" to "{name}"')
    except psycopg.Error:
        pytest.skip("this role cannot grant a membership, so the reach cannot be tested")
    theirs = psycopg.conninfo.make_conninfo(dsn, user=name, password=PROBE_PASSWORD)
    try:
        with agensgraph.connect(theirs, autocommit=False) as conn:
            yield conn
    except psycopg.OperationalError:
        pytest.skip("this server will not let the test connect as another role")
    finally:
        for role in (name, holder):
            with contextlib.suppress(psycopg.Error):
                agens.execute(f'drop owned by "{role}" cascade')
            agens.execute(f'drop role if exists "{role}"')


class TestARoleThatCanReachThePrivilege:
    """Being able to name a membership is being able to use it.

    Measured: a role the driver called safe opened a read-only transaction, named its membership,
    and ran a command on the host. So the question the driver asks is what the role can reach.
    """

    def test_it_is_refused_the_transaction(self, reaching) -> None:  # type: ignore[no-untyped-def]
        assert reaching.can_run_server_programs() is True
        with pytest.raises(ConfigurationError) as caught, reaching.read_only_transaction():
            pass
        assert "pg_execute_server_program" in str(caught.value)

    def test_naming_the_membership_is_not_a_write(self, reaching) -> None:  # type: ignore[no-untyped-def]
        """Which is why holding the privilege at this moment is the wrong thing to ask about."""
        with reaching.read_only_transaction(allow_server_programs=True):
            reaching.execute('set role "agens_program_holder"')
            (held,) = reaching.execute(
                "select pg_has_role(current_user, 'pg_execute_server_program', 'usage')"
            ).fetchone()
        assert held is True, "the transaction allowed the role to take the privilege"


SMUGGLED = "select 1; create table smuggled_through(i int)"


def _run_by(path: str, conn, statement: str) -> None:  # type: ignore[no-untyped-def]
    if path == "execute":
        conn.execute(statement)
    elif path == "execute_query":
        conn.execute_query(statement)
    elif path == "cursor":
        with conn.cursor() as cursor:
            cursor.execute(statement)
    elif path == "stream":
        with conn.cursor() as cursor:
            for _ in cursor.stream(statement):
                pass
    else:
        with conn.cursor() as cursor, cursor.copy(statement):
            pass


class TestAConnectionThatPreparesEverything:
    """``prepare_threshold=0``, which an integration sets to prepare every statement on first use.

    No prepared statement can hold two statements in one string, so the block opens with one, and
    the driver's own statements are never prepared.
    """

    def test_the_block_opens_and_holds_every_time(self, dsn: str) -> None:
        with agensgraph.connect(dsn, prepare_threshold=0) as conn:
            for _ in range(3):
                with conn.read_only_transaction(allow_server_programs=True):
                    assert conn.execute("select 1").fetchone() == (1,)
                    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
                        conn.execute("create table not_here(i int)")
                with (
                    conn.read_only_transaction(allow_server_programs=True),
                    pytest.raises(psycopg.errors.ActiveSqlTransaction),
                ):
                    conn.execute("set transaction read write")

    def test_a_block_inside_a_block_opens_too(self, dsn: str) -> None:
        """Inside the outer block every statement goes over the extended protocol, so the inner
        one's opening has to be a single statement there as well."""
        with (
            agensgraph.connect(dsn, prepare_threshold=0) as conn,
            conn.read_only_transaction(allow_server_programs=True),
            conn.read_only_transaction(allow_server_programs=True),
        ):
            assert conn.execute("select 1").fetchone() == (1,)

    def test_the_driver_s_own_statements_take_no_prepared_slot(self, dsn: str) -> None:
        with agensgraph.connect(dsn, prepare_threshold=0, autocommit=True) as conn:
            for budget in (5.0, 6.0, 7.0):
                with conn.deadline(budget):
                    pass
            with conn.read_only_transaction(allow_server_programs=True):
                pass
            prepared = conn.execute(
                "select count(*) from pg_prepared_statements where statement not like '%%count%%'"
            ).fetchone()[0]
        assert prepared == 0


class TestNothingLeavesTheBlock:
    """The boundary holds on every way of running a statement, on every supported release."""

    @pytest.mark.parametrize("path", ["execute", "execute_query", "cursor", "stream", "copy"])
    def test_a_smuggled_statement_is_refused_and_nothing_persists(
        self, agens, path: str
    ) -> None:  # type: ignore[no-untyped-def]
        agens.autocommit = False
        statement = "copy (select 1) to stdout; create table smuggled_through(i int)"
        with (
            pytest.raises(ValueError, match="more than one statement"),
            agens.read_only_transaction(allow_server_programs=True),
        ):
            _run_by(path, agens, statement if path == "copy" else SMUGGLED)
        agens.rollback()
        assert agens.execute("select to_regclass('smuggled_through')").fetchone() == (None,)

    @pytest.mark.parametrize(
        "statement",
        [
            "commit",
            "rollback",
            "end",
            "abort",
            "start transaction",
            "savepoint s",
            "release s",
            # Back to a savepoint taken before the block, this would turn read-only off again.
            "rollback to savepoint s",
            "prepare transaction 'x'",
            "/* a comment does not hide it */ commit",
        ],
    )
    def test_ending_the_transaction_is_refused_before_it_is_sent(
        self, agens, statement: str
    ) -> None:  # type: ignore[no-untyped-def]
        agens.autocommit = False
        with agens.read_only_transaction(allow_server_programs=True):
            with pytest.raises(ValueError, match="end or split the transaction"):
                agens.execute(statement)
            with pytest.raises(ValueError, match="end or split the transaction"):
                _run_by("stream", agens, statement)
            with pytest.raises(psycopg.Error) as caught:
                agens.execute("create table after_the_attempt(i int)")
        assert caught.value.sqlstate == "25006", "the block is still read-only"

    @pytest.mark.parametrize(
        "statement",
        [
            "set transaction read write",
            "set transaction_read_only = off",
            "select set_config('transaction_read_only', 'off', true)",
        ],
    )
    def test_the_transaction_cannot_be_made_writable_again(self, agens, statement: str) -> None:  # type: ignore[no-untyped-def]
        """It took a snapshot as it opened, after which the server refuses read-write mode."""
        agens.autocommit = False
        with (
            pytest.raises(psycopg.Error) as caught,
            agens.read_only_transaction(allow_server_programs=True),
        ):
            agens.execute(statement)
        assert caught.value.sqlstate == "25001"

    def test_a_session_setting_is_not_mistaken_for_a_write(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.autocommit = False
        with agens.read_only_transaction(allow_server_programs=True):
            agens.execute("set search_path = 'still_a_read'")
            (path,) = agens.execute("show search_path").fetchone()
        assert "still_a_read" in path

    def test_a_cypher_write_never_reaches_a_server_that_would_run_it(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel thing")
        agens.execute("create (:thing {a: 1})")
        sent: list[str] = []

        def record_statement(record) -> None:  # type: ignore[no-untyped-def]
            sent.append(record.statement)

        agensgraph.add_query_logger(record_statement)
        agens.autocommit = False
        try:
            with (
                pytest.raises(ReadOnlyGraphWrite) as caught,
                agens.read_only_transaction(allow_server_programs=True),
            ):
                agens.execute("match (n:thing) set n.a = 2")
        finally:
            agensgraph.remove_query_logger(record_statement)
        agens.rollback()
        # A development build or a release candidate may predate the fix, so the table does not
        # take it to have it, and until the probe has said otherwise it is held to the reading.
        if agens.capabilities.has("cypher_write_enforced"):
            assert any("set n.a = 2" in text for text in sent)
        else:
            assert "refused before it was sent" in str(caught.value)
            assert not any("set n.a = 2" in text for text in sent)
        assert agens.execute("match (n:thing) return n.a").fetchone() == (1,)

    def test_the_driver_s_own_savepoints_go_through(self, agens) -> None:  # type: ignore[no-untyped-def]
        """``pipeline_query`` takes one around each batch, and ``describe`` reads through it.

        A savepoint the driver takes inside the block is released before the block closes, so it
        is not held to the checks the caller's text is, and a graph can be described read-only.
        """
        agens.execute("create vlabel thing")
        agens.execute("create (:thing {a: 1})")
        agens.autocommit = False
        with agens.read_only_transaction(allow_server_programs=True):
            first, second = agens.pipeline_query(["match (n:thing) return n.a", "select 2"])
            assert (first.records, second.records) == ([(1,)], [(2,)])
            assert "thing" in [label.name for label in agens.describe(sample=1).labels]
            with pytest.raises(psycopg.Error) as caught:
                agens.execute("create table after_the_savepoints(i int)")
        assert caught.value.sqlstate == "25006", "the block is still read-only"
        with (
            pytest.raises(psycopg.errors.DivisionByZero),
            agens.read_only_transaction(allow_server_programs=True),
        ):
            # A failed batch rolls back to the driver's savepoint before asking again singly.
            agens.pipeline_query(["select 1", "select 1/0"])
        agens.rollback()

    def test_the_probe_agrees_with_the_version(self, agens) -> None:  # type: ignore[no-untyped-def]
        """A zero-row write in a rolled-back read-only transaction, answered once per connection."""
        assert agens.read_only_is_enforced() is agens.capabilities.has("cypher_write_enforced")
        assert agens._agens_read_only_enforced is not None, "the answer is kept"


class TestTheAwaitingInterface:
    @pytest.mark.asyncio
    async def test_it_refuses_a_write_there_too(self, dsn: str) -> None:
        graph = "read_only_async"
        conn = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
        async with conn:
            await conn.execute(f'drop graph if exists "{graph}" cascade')
            await conn.execute(f'create graph "{graph}"')
            await conn.graph(graph)
            await conn.execute("create vlabel thing")
            await conn.set_autocommit(False)
            try:
                with pytest.raises(psycopg.Error) as caught:
                    async with conn.read_only_transaction(allow_server_programs=True):
                        await conn.execute("create (:thing)")
                assert caught.value.sqlstate == "25006"
            finally:
                await conn.rollback()
                await conn.set_autocommit(True)
                await conn.execute("reset graph_path")
                await conn.execute(f'drop graph "{graph}" cascade')
