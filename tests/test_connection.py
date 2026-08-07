"""A connection against a live server, in both interfaces.

Every test that matters here is written twice on purpose: the blocking interface is generated
from the awaiting one, so a test of only one of them would pass while the other was broken by
a construct the generator handled wrongly. These are the tests that would catch that.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

import agensgraph
from agensgraph.errors import ConfigurationError, StaleLabelCache

pytestmark = pytest.mark.server


@pytest.fixture
def name() -> str:
    return f"c_{os.getpid()}"


@pytest.fixture
def conn(dsn: str, name: str):  # type: ignore[no-untyped-def]
    with agensgraph.connect(dsn, autocommit=True) as connection:
        connection.execute(f'drop graph if exists "{name}" cascade')
        connection.execute(f'create graph "{name}"')
        connection.graph(name)
        connection.execute("create vlabel person")
        connection.execute("create elabel knows")
        connection.refresh_labels()
        connection.execute("create (:person {name: 'a'})-[:knows]->(:person {name: 'b'})")
        try:
            yield connection
        finally:
            connection.execute("reset graph_path")
            connection.execute(f'drop graph "{name}" cascade')


class TestConnecting:
    def test_the_version_is_read_at_connect(self, dsn: str) -> None:
        with agensgraph.connect(dsn) as conn:
            assert conn.capabilities.version >= agensgraph.MINIMUM_VERSION

    def test_the_graph_types_are_read_without_registering_anything(self, conn) -> None:  # type: ignore[no-untyped-def]
        (v,) = conn.execute("match (n:person) return n limit 1").fetchone()
        assert isinstance(v, agensgraph.Vertex)

    def test_a_plain_psycopg_connection_is_unaffected(self, dsn: str) -> None:
        """Registering the graph types must not reach a connection that did not ask."""
        import psycopg

        with agensgraph.connect(dsn), psycopg.connect(dsn) as plain:
            (value,) = plain.execute("select '3.1'::graphid").fetchone()
            assert isinstance(value, str)


class TestSelectingAGraph:
    def test_the_label_table_is_filled(self, conn, name: str) -> None:  # type: ignore[no-untyped-def]
        assert conn.label_table.graph == name
        assert conn.label_table.get(3) == "person"

    def test_selecting_the_same_graph_again_keeps_the_table(self, conn, name: str) -> None:  # type: ignore[no-untyped-def]
        before = len(conn.label_table)
        conn.graph(name)
        assert len(conn.label_table) == before

    @pytest.mark.parametrize("odd", ["g odd,name", "MATCH", "g space"])
    def test_a_name_needing_quoting(self, dsn: str, odd: str) -> None:
        """A graph name cannot be bound, so it is quoted, and this is what proves it works."""
        with agensgraph.connect(dsn, autocommit=True) as conn:
            conn.execute(f'drop graph if exists "{odd}" cascade')
            conn.execute(f'create graph "{odd}"')
            try:
                conn.graph(odd)
                assert conn.label_table.graph == odd
                assert len(conn.label_table) == 2
            finally:
                conn.execute("reset graph_path")
                conn.execute(f'drop graph "{odd}" cascade')


class TestExecuteQuery:
    def test_a_read(self, conn) -> None:  # type: ignore[no-untyped-def]
        result = conn.execute_query("match (n:person) return n order by n.name")
        assert result.keys == ["n"]
        assert [v.properties["name"] for (v,) in result.records] == ["a", "b"]

    def test_a_read_in_the_composite_rendering(self, conn) -> None:  # type: ignore[no-untyped-def]
        result = conn.execute_query("match (n:person) return n order by n.name", binary_=True)
        assert [v.properties["name"] for (v,) in result.records] == ["a", "b"]

    def test_both_renderings_agree(self, conn) -> None:  # type: ignore[no-untyped-def]
        query = "match p = (:person)-[:knows]->(:person) return p, nodes(p), relationships(p)"
        assert (
            conn.execute_query(query).records == conn.execute_query(query, binary_=True).records
        )

    def test_a_parameter(self, conn) -> None:  # type: ignore[no-untyped-def]
        """A plain string, because the driver says a string is text."""
        result = conn.execute_query("match (n:person) where n.name = %s return n", ("a",))
        assert len(result.records) == 1

    def test_a_wrapped_parameter_still_works(self, conn) -> None:  # type: ignore[no-untyped-def]
        """Anyone who wrapped one before should not have to stop."""
        result = conn.execute_query(
            "match (n:person) where n.name = %s return n", (agensgraph.Jsonb("a"),)
        )
        assert len(result.records) == 1

    def test_a_number_needs_no_wrapping(self, conn) -> None:  # type: ignore[no-untyped-def]
        conn.execute_query("match (n:person {name: 'a'}) set n.n = 1")
        result = conn.execute_query("match (n:person) where n.n = %s return n", (1,))
        assert len(result.records) == 1

    def test_a_property_map_needs_no_wrapping(self, conn) -> None:  # type: ignore[no-untyped-def]
        """psycopg cannot adapt a bare mapping at all, so the driver registers one that can."""
        conn.execute_query("create (:person %s)", ({"name": "mapped"},))
        result = conn.execute_query("match (n:person {name: 'mapped'}) return n")
        assert len(result.records) == 1

    def test_a_list_is_still_an_array(self, conn) -> None:  # type: ignore[no-untyped-def]
        """Because plain SQL on the same connection needs it to be."""
        rows = conn.execute(
            "select typname from pg_type where typname = any(%s)", (["vertex", "edge"],)
        ).fetchall()
        assert len(rows) == 2

    def test_a_statement_that_returns_nothing(self, conn) -> None:  # type: ignore[no-untyped-def]
        result = conn.execute_query("match (n:person) set n.seen = true")
        assert result.records == []
        assert result.keys == []

    def test_a_write_with_no_return_reports_every_counter(self, conn) -> None:  # type: ignore[no-untyped-def]
        result = conn.execute_query("create (:person {name: 'c'})", counts_=True)
        assert result.counts.complete
        assert result.counts.inserted_vertices == 1
        assert result.counts.total == 1

    def test_a_write_that_returns_rows_is_credited_with_its_own_clauses(self, conn) -> None:  # type: ignore[no-untyped-def]
        """The insert counters still hold the fixture's vertices, and a SET cannot have
        inserted anything, so they are nought for this statement whatever they hold."""
        conn.execute_query("create (:person {name: 'd'}), (:person {name: 'e'})")
        result = conn.execute_query("match (n:person) set n.x = 1 return n", counts_=True)
        assert result.counts.updated_properties == len(result.records)
        assert result.counts.inserted_vertices == 0
        assert result.counts.inserted_edges == 0

    def test_counters_are_not_read_unless_asked_for(self, conn) -> None:  # type: ignore[no-untyped-def]
        result = conn.execute_query("create (:person {name: 'f'})")
        assert not result.counts.complete
        assert result.counts.inserted_vertices is None

    def test_a_row_factory(self, conn) -> None:  # type: ignore[no-untyped-def]
        from psycopg.rows import scalar_row

        result = conn.execute_query(
            "match (n:person) return n.name order by n.name", row_=scalar_row
        )
        assert result.records == ["a", "b"]

    def test_a_statement_the_server_would_misread_never_leaves(self, conn) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="length of a variable-length relationship"):
            conn.execute_query("match (a)-[r*1..%s]->(b) return a")

    def test_a_setting_that_refused_the_work_is_named(self, conn, name: str) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ConfigurationError) as caught:
            conn.execute_query(f"insert into \"{name}\".person values (default, '{{}}'::jsonb)")
        assert caught.value.setting == "enable_graph_dml"


class TestTheLabelTableGoingStale:
    def test_a_label_created_afterwards_is_read_in_the_text_rendering(self, conn) -> None:  # type: ignore[no-untyped-def]
        conn.execute("create vlabel later")
        conn.execute("create (:later)")
        (v,) = conn.execute_query("match (n:later) return n").records[0]
        assert v.label == "later"

    def test_and_reported_in_the_composite_one(self, conn) -> None:  # type: ignore[no-untyped-def]
        conn.execute("create vlabel later")
        conn.execute("create (:later)")
        with pytest.raises(StaleLabelCache) as caught:
            conn.execute_query("match (n:later) return n", binary_=True)
        assert caught.value.labid is not None

    def test_until_the_table_is_filled_again(self, conn) -> None:  # type: ignore[no-untyped-def]
        conn.execute("create vlabel later")
        conn.execute("create (:later)")
        conn.refresh_labels()
        (v,) = conn.execute_query("match (n:later) return n", binary_=True).records[0]
        assert v.label == "later"


class TestTheAwaitingInterface:
    """The same behaviour, awaited, because the blocking one is generated from it."""

    @pytest_asyncio.fixture
    async def aconn(self, dsn: str, name: str):  # type: ignore[no-untyped-def]
        graph = f"{name}_a"
        conn = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
        async with conn:
            await conn.execute(f'drop graph if exists "{graph}" cascade')
            await conn.execute(f'create graph "{graph}"')
            await conn.graph(graph)
            await conn.execute("create vlabel person")
            await conn.execute("create elabel knows")
            await conn.refresh_labels()
            await conn.execute("create (:person {name: 'a'})-[:knows]->(:person {name: 'b'})")
            try:
                yield conn
            finally:
                await conn.execute("reset graph_path")
                await conn.execute(f'drop graph "{graph}" cascade')

    @pytest.mark.asyncio
    async def test_the_version_is_read_at_connect(self, dsn: str) -> None:
        conn = await agensgraph.AsyncConnection.connect(dsn)
        async with conn:
            assert conn.capabilities.version >= agensgraph.MINIMUM_VERSION

    @pytest.mark.asyncio
    async def test_a_read(self, aconn) -> None:  # type: ignore[no-untyped-def]
        result = await aconn.execute_query("match (n:person) return n order by n.name")
        assert [v.properties["name"] for (v,) in result.records] == ["a", "b"]

    @pytest.mark.asyncio
    async def test_both_renderings_agree(self, aconn) -> None:  # type: ignore[no-untyped-def]
        query = "match p = (:person)-[:knows]->(:person) return p, nodes(p)"
        first = await aconn.execute_query(query)
        second = await aconn.execute_query(query, binary_=True)
        assert first.records == second.records

    @pytest.mark.asyncio
    async def test_a_write_with_no_return_reports_every_counter(self, aconn) -> None:  # type: ignore[no-untyped-def]
        result = await aconn.execute_query("create (:person {name: 'c'})", counts_=True)
        assert result.counts.complete
        assert result.counts.inserted_vertices == 1

    @pytest.mark.asyncio
    async def test_a_write_that_returns_rows_is_credited_with_its_own_clauses(
        self, aconn
    ) -> None:  # type: ignore[no-untyped-def]
        await aconn.execute_query("create (:person {name: 'd'})")
        result = await aconn.execute_query(
            "match (n:person) set n.x = 1 return n", counts_=True
        )
        assert result.counts.inserted_vertices == 0
        assert result.counts.updated_properties == len(result.records)

    @pytest.mark.asyncio
    async def test_a_statement_the_server_would_misread_never_leaves(self, aconn) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="length of a variable-length relationship"):
            await aconn.execute_query("match (a)-[r*1..%s]->(b) return a")

    @pytest.mark.asyncio
    async def test_the_label_table_going_stale(self, aconn) -> None:  # type: ignore[no-untyped-def]
        await aconn.execute("create vlabel later")
        await aconn.execute("create (:later)")
        with pytest.raises(StaleLabelCache):
            await aconn.execute_query("match (n:later) return n", binary_=True)
        await aconn.refresh_labels()
        result = await aconn.execute_query("match (n:later) return n", binary_=True)
        assert result.records[0][0].label == "later"


class TestKeepaliveDefaults:
    """The connection string is as likely a place to set one as the arguments are, and psycopg
    lays the arguments over it -- so a default filled in blindly reverses what the string said."""

    def test_nothing_said_gets_the_defaults(self) -> None:
        from agensgraph._core import KEEPALIVE_DEFAULTS, with_keepalives

        assert with_keepalives({}, "host=h") == KEEPALIVE_DEFAULTS

    def test_turning_them_off_in_the_connection_string_is_respected(self) -> None:
        import psycopg

        from agensgraph._core import with_keepalives

        dsn = "host=h keepalives=0"
        merged = psycopg.conninfo.conninfo_to_dict(dsn, **with_keepalives({}, dsn))
        assert merged["keepalives"] == "0"
        assert "keepalives_idle" not in merged

    def test_turning_them_off_in_the_arguments_is_respected(self) -> None:
        from agensgraph._core import with_keepalives

        assert with_keepalives({"keepalives": 0}, "host=h") == {"keepalives": 0}

    def test_one_setting_in_the_string_is_not_displaced(self) -> None:
        import psycopg

        from agensgraph._core import with_keepalives

        dsn = "host=h keepalives_idle=99"
        merged = psycopg.conninfo.conninfo_to_dict(dsn, **with_keepalives({}, dsn))
        assert merged["keepalives_idle"] == "99"
        assert merged["keepalives"] == 1

    def test_a_url_connection_string_is_read_too(self) -> None:
        from agensgraph._core import with_keepalives

        assert with_keepalives({}, "postgresql://h/db?keepalives=0") == {}


def test_the_reserved_arguments_are_the_four_that_are_documented() -> None:
    """So that a fifth is a decision rather than an accident.

    Two a caller might look for are absent on purpose: a graph to read for this statement alone,
    and a time limit for it. Each would cost round trips that belong where they are paid once for
    many statements.
    """
    import inspect

    signature = inspect.signature(agensgraph.Connection.execute_query)
    reserved = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    assert reserved == ["binary_", "counts_", "prepare_", "row_"]
    positional = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    assert positional == ["self", "query", "params"]
