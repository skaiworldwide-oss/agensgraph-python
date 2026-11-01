"""Running a statement a model wrote: the reading before it is sent, and the run itself."""

from __future__ import annotations

import contextlib

import psycopg
import pytest
from psycopg.pq import TransactionStatus
from psycopg.rows import dict_row

import agensgraph
from agensgraph.errors import CapabilityError, ConfigurationError
from agensgraph.untrusted import check_columns, check_untrusted, role_problems

READS = [
    "MATCH (n:Person) RETURN n.name AS name ORDER BY name LIMIT 5",
    "MATCH (a)-[r:KNOWS*1..3]->(b) RETURN a, b",
    "MATCH (n) WHERE n.from = 'x' AND n.values > 1 RETURN n.set AS s, count(*) AS c",
    "OPTIONAL MATCH (n:Table {create: 1}) RETURN n",
    "UNWIND [1, 2, 3] AS x RETURN x",
    "WITH 1 AS x RETURN x",
    'MATCH p = (a)-[*..2]-(b) RETURN length(p) AS len, "end"(last(relationships(p))) AS e',
    "match (n) return n;",
    "MATCH (n) RETURN 'it''s' AS q, E'a\\'b' AS r",
    "MATCH (n) WHERE n.p =~ 'a\\d+' RETURN n",
    "RETURN *",
    # A name in any script is a name, as the server reads it.
    "MATCH (n:사람) RETURN n.이름 AS 이름",
    "MATCH (n) RETURN n.café AS c",
    # A map's value may call one of Cypher's own functions.
    "MATCH (n) RETURN {a: count(n), b: toupper(n.name)} AS m",
]

REFUSED = [
    pytest.param(
        "MATCH (n) RETURN n) AS t0 CROSS JOIN (TABLE pg_catalog.pg_stat_activity",
        id="wrapper-escape",
    ),
    pytest.param("MATCH (n) RETURN n -- the rest", id="line-comment"),
    pytest.param("MATCH (n) /* hidden */ RETURN n", id="block-comment"),
    pytest.param("RETURN $$x$$", id="dollar-quote"),
    pytest.param("MATCH (n) RETURN 'a\\')' AS x", id="quote-confusion"),
    pytest.param("MATCH (n) RETURN pg_sleep(10)", id="sql-function"),
    pytest.param('MATCH (n) RETURN "pg_advisory_lock"(1)', id="quoted-function"),
    pytest.param("MATCH (n) RETURN set_config('a', 'b', false)", id="set-config"),
    # After a colon a name looks like a label, and as a map's value it is a call all the same.
    pytest.param("MATCH (n) RETURN {a: pg_sleep(10)} AS m", id="call-as-a-map-value"),
    pytest.param(
        "RETURN {a: query_to_xml('select * from pg_stat_activity', true, false, '')} AS m",
        id="sql-through-a-map-value",
    ),
    pytest.param("RETURN pg_catalog.pg_sleep(10) AS x", id="schema-qualified"),
    pytest.param("RETURN pg_catalog.count(1) AS x", id="schema-qualified-allowed-name"),
    pytest.param("MATCH (n) SET n.x = 1", id="write"),
    pytest.param("MATCH (n) DETACH DELETE n", id="detach-delete"),
    pytest.param("SELECT 1", id="sql"),
    pytest.param("EXPLAIN ANALYZE MATCH (n) SET n.x = 1", id="explain-analyze"),
    pytest.param("MATCH (n) RETURN (SELECT 1) AS x", id="subquery"),
    pytest.param("MATCH (a)-[*]->(b) RETURN a", id="unbounded-walk"),
    pytest.param("MATCH (a)-[*2..]->(b) RETURN a", id="open-walk"),
    pytest.param("MATCH (a)-[*1..9]->(b) RETURN a", id="long-walk"),
    pytest.param("MATCH (n) RETURN n; COMMIT", id="second-statement"),
    pytest.param("COMMIT", id="commit"),
    pytest.param("CALL db.labels()", id="call"),
    pytest.param("LOAD FROM t AS r RETURN r", id="load"),
    pytest.param(
        "MATCH " + ", ".join(f"(n{i})" for i in range(40)) + " RETURN n0", id="patterns"
    ),
    pytest.param("MATCH (n) RETURN n" + " " * 9000, id="size"),
]


@pytest.mark.parametrize("text", READS)
def test_a_plain_read_is_accepted(text: str) -> None:
    assert check_untrusted(text).text


@pytest.mark.parametrize("text", REFUSED)
def test_what_a_read_has_no_use_for_is_refused_before_it_is_sent(text: str) -> None:
    with pytest.raises(ValueError, match="refused before it was sent"):
        check_untrusted(text)


def test_writes_are_admitted_only_when_asked_for() -> None:
    assert check_untrusted("MATCH (n:T) SET n.x = 1 RETURN n", allow_writes=True).writes
    with pytest.raises(ValueError, match="pg_sleep"):
        check_untrusted("MATCH (n) SET n.x = pg_sleep(1)", allow_writes=True)


@pytest.mark.parametrize(
    "text", ["CREATE (n:T {a: 1})", "MERGE (n:T {a: 1}) RETURN n", "CREATE (:사람 {이름: 'a'})"]
)
def test_a_write_may_start_with_the_clause_that_writes(text: str) -> None:
    assert check_untrusted(text, allow_writes=True).writes
    with pytest.raises(ValueError, match="a read starts with MATCH"):
        check_untrusted(text)


@pytest.mark.parametrize(
    ("text", "columns", "wrappable"),
    [
        ("MATCH (n) RETURN n.name AS name, count(*) AS c ORDER BY c", ("name", "c"), True),
        ("MATCH (n) RETURN n, n.x", (None, None), True),
        ("MATCH (n) RETURN DISTINCT n.x AS x", ("x",), True),
        ("RETURN *", None, True),
        ("WITH 1 AS x RETURN x AS y", ("y",), False),
    ],
)
def test_what_the_statement_returns_is_read_from_its_last_return(
    text: str, columns: tuple[str | None, ...] | None, wrappable: bool
) -> None:
    checked = check_untrusted(text)
    assert checked.columns == columns
    assert checked.wrappable is wrappable


def test_a_result_with_other_columns_is_refused() -> None:
    check_columns(("name", None), ("NAME", "n"))
    with pytest.raises(ValueError, match="columns"):
        check_columns(("name",), ("name", "rolname"))
    with pytest.raises(ValueError, match="where the statement returns"):
        check_columns(("name",), ("usename",))


def test_write_privileges_count_only_where_the_server_ignores_them() -> None:
    held = (False, False, False, False, False, False, True)
    assert role_problems(held, writes_enforced=False) == (
        "it may change tables in the schemas it was asked about",
    )
    assert role_problems(held, writes_enforced=True) == ()
    assert role_problems((True,) + (False,) * 6, writes_enforced=True) == ("it is a superuser",)


ROLE = "agens_untrusted_probe"
APP = "agens_untrusted_app"
PROBE_PASSWORD = "agens-probe-only"
"""So the test connects as the role whether or not the server trusts local connections."""


@pytest.fixture
def model(agens, dsn):  # type: ignore[no-untyped-def]
    """A connection as a role made for model-written Cypher: it may read the graph and no more."""
    graph = agens.execute("select current_setting('graph_path')").fetchone()[0]
    agens.execute("create vlabel thing")
    agens.execute("create (:thing {a: 1}), (:thing {a: 2}), (:thing {a: 3})")
    for name in (ROLE, APP):
        with contextlib.suppress(psycopg.Error):
            agens.execute(f'drop owned by "{name}" cascade')
        agens.execute(f'drop role if exists "{name}"')
    try:
        agens.execute(f"create role \"{ROLE}\" login password '{PROBE_PASSWORD}'")
        agens.execute(f'create role "{APP}"')
        agens.execute(f'grant usage on schema "{graph}" to "{ROLE}"')
        agens.execute(f'grant select on all tables in schema "{graph}" to "{ROLE}"')
    except psycopg.Error:
        pytest.skip("this role cannot create another, so the executor cannot be tested")
    theirs = psycopg.conninfo.make_conninfo(dsn, user=ROLE, password=PROBE_PASSWORD)
    try:
        with agensgraph.connect(theirs, autocommit=True) as conn:
            conn.graph(graph)
            yield conn
    except psycopg.OperationalError:
        pytest.skip("this server will not let the test connect as another role")
    finally:
        for name in (ROLE, APP):
            agens.execute(f'drop owned by "{name}" cascade')
            agens.execute(f'drop role if exists "{name}"')


def _logged(statements: list[str]):  # type: ignore[no-untyped-def]
    def record(record) -> None:  # type: ignore[no-untyped-def]
        statements.append(record.statement)

    return record


@pytest.mark.server
class TestRunningIt:
    def test_a_read_comes_back_in_one_round_trip(self, model) -> None:  # type: ignore[no-untyped-def]
        sent: list[str] = []
        listener = _logged(sent)
        agensgraph.add_query_logger(listener)
        try:
            result = model.run_untrusted("MATCH (n:thing) RETURN n.a AS a ORDER BY a")
        finally:
            agensgraph.remove_query_logger(listener)
        assert result.columns == ("a",)
        assert [row[0] for row in result.rows] == [1, 2, 3]
        assert result.truncated is False
        assert any("declare agensgraph_untrusted" in text for text in sent)
        assert model.pgconn.transaction_status == TransactionStatus.IDLE

    def test_a_statement_the_wrap_cannot_hold_is_streamed(self, model) -> None:  # type: ignore[no-untyped-def]
        """A LIMIT before the last clause: 2.18's wrap refuses it, earlier wraps take it."""
        sent: list[str] = []
        listener = _logged(sent)
        agensgraph.add_query_logger(listener)
        try:
            result = model.run_untrusted(
                "MATCH (n:thing) WITH n.a AS a ORDER BY a DESC LIMIT 2 RETURN a"
            )
        finally:
            agensgraph.remove_query_logger(listener)
        assert [row[0] for row in result.rows] == [3, 2]
        wrapped = any("declare agensgraph_untrusted" in text for text in sent)
        assert wrapped is not model.capabilities.has("gql_clauses")
        assert model.pgconn.transaction_status == TransactionStatus.IDLE

    @pytest.mark.parametrize(
        "text",
        [
            "MATCH (n:thing) RETURN n.a AS a ORDER BY a",
            # Streamed on 2.18, whose wrap refuses a LIMIT before the last clause.
            "MATCH (n:thing) WITH n.a AS a ORDER BY a LIMIT 3 RETURN a",
        ],
    )
    def test_rows_are_tuples_whatever_the_connection_reads_them_as(
        self, model, text: str
    ) -> None:  # type: ignore[no-untyped-def]
        """The result names its columns once, so every route hands back plain tuples."""
        model.row_factory = dict_row
        result = model.run_untrusted(text)
        assert result.rows
        assert all(type(row) is tuple for row in result.rows)

    def test_only_the_rows_asked_for_are_read(self, model) -> None:  # type: ignore[no-untyped-def]
        result = model.run_untrusted("MATCH (n:thing) RETURN n.a AS a ORDER BY a", max_rows=2)
        assert [row[0] for row in result.rows] == [1, 2]
        assert result.truncated is True

    def test_refused_text_never_reaches_the_server(self, model) -> None:  # type: ignore[no-untyped-def]
        sent: list[str] = []
        listener = _logged(sent)
        agensgraph.add_query_logger(listener)
        try:
            with pytest.raises(ValueError, match="refused before it was sent"):
                model.run_untrusted(
                    "MATCH (n) RETURN n) AS t0 CROSS JOIN (TABLE pg_catalog.pg_roles"
                )
        finally:
            agensgraph.remove_query_logger(listener)
        assert not any("pg_roles" in text for text in sent)

    def test_a_failing_statement_leaves_the_connection_usable(self, model) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(psycopg.Error):
            model.run_untrusted("MATCH (n:thing) RETURN n.a / 0 AS broken")
        assert model.pgconn.transaction_status == TransactionStatus.IDLE
        assert model.run_untrusted("MATCH (n:thing) RETURN count(n) AS c").rows == [(3,)]

    def test_a_superuser_is_refused(self, agens) -> None:  # type: ignore[no-untyped-def]
        if "superuser" not in " ".join(agens.untrusted_role_problems()):
            pytest.skip("the test's own role is not a superuser")
        with pytest.raises(ConfigurationError, match="superuser"):
            agens.run_untrusted("RETURN 1 AS one")

    def test_a_member_of_the_application_role_is_refused(self, agens, model) -> None:  # type: ignore[no-untyped-def]
        agens.execute(f'grant "{APP}" to "{ROLE}"')
        with pytest.raises(ConfigurationError, match="application's role"):
            model.run_untrusted("RETURN 1 AS one", app_role=APP)

    def test_writes_need_a_server_that_checks_them(self, agens, model) -> None:  # type: ignore[no-untyped-def]
        graph = agens.execute("select current_setting('graph_path')").fetchone()[0]
        agens.execute(f'grant update on all tables in schema "{graph}" to "{ROLE}"')
        statement = "MATCH (n:thing) WHERE n.a = 1 SET n.a = 10 RETURN n.a AS a"
        # A development or release-candidate build is not taken to enforce it whatever its line,
        # since what it enforces may still change; engine main reports 2.18-devel.
        if not agens.capabilities.has("cypher_write_enforced"):
            with pytest.raises((CapabilityError, ConfigurationError)):
                model.run_untrusted(statement, allow_writes=True)
            return
        result = model.run_untrusted(statement, allow_writes=True)
        assert result.rows == [(10,)]
        assert agens.execute("match (n:thing) where n.a = 10 return count(n)").fetchone() == (
            1,
        )
