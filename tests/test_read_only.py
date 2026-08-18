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
from agensgraph.errors import ConfigurationError

pytestmark = pytest.mark.server


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
        agens.execute(f'create role "{name}" login')
        agens.execute(f'grant usage on schema "{graph}" to "{name}"')
        agens.execute(f'grant select on all tables in schema "{graph}" to "{name}"')
    except psycopg.Error:
        pytest.skip("this role cannot create another, so the boundary cannot be tested")
    # The point of the fixture is the other role, so the connection has to be made as it.
    theirs = psycopg.conninfo.make_conninfo(dsn, user=name, password="")
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
        """The server has no name for a graph write and reports it as '???'.

        Which is unreadable, so the driver replaces it -- and that replacement is the one part of
        this only a driver can supply.
        """
        agens.execute("create vlabel thing")
        agens.autocommit = False
        with (
            pytest.raises(agensgraph.errors.ReadOnlyGraphWrite) as caught,
            agens.read_only_transaction(allow_server_programs=True),
        ):
            agens.execute("create (:thing)")
        assert "cannot write to a graph in a read-only transaction" in str(caught.value)


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

        refused = {}
        for name, statement in (
            ("write", "create (:thing)"),
            ("program", "copy (select 1) to program 'true'"),
            ("smuggled", "select 1; copy (select 1) to program 'true'"),
        ):
            with (
                pytest.raises(psycopg.Error) as caught,
                unprivileged.read_only_transaction(),
            ):
                unprivileged.execute(statement)
            refused[name] = caught.value.sqlstate
        assert refused["write"] == "25006", "the transaction refuses the write"
        assert refused["program"] == "42501", "the privilege refuses the program"
        assert refused["smuggled"] == "42501", "including one hidden behind another statement"


@pytest.fixture
def reaching(agens, dsn):  # type: ignore[no-untyped-def]
    """A connection as a role that can *reach* the privilege without holding it.

    A membership granted ``with inherit false`` carries nothing until ``set role`` names it. Such a
    role is as dangerous as one that holds the privilege outright, because naming a membership moves
    no rows and a read-only transaction has no reason to refuse it.
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
        agens.execute(f'create role "{name}" login')
        agens.execute(f'grant "{holder}" to "{name}" with inherit false')
    except psycopg.Error:
        pytest.skip("this role cannot grant a membership, so the reach cannot be tested")
    theirs = psycopg.conninfo.make_conninfo(dsn, user=name, password="")
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
