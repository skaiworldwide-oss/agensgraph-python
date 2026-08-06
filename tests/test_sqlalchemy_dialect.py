"""A SQLAlchemy dialect, as a test of the PEP 249 surface rather than as something shipped.

The claim being checked is that a dialect for this driver is a handful of lines because psycopg ships
a synchronous interface and so does this. If it were long, the surface would be wrong somewhere, and
the length of the class below is asserted so that it cannot quietly become long.

Nothing here is part of the package. It lives in the tests because what it proves is about the driver,
not about SQLAlchemy.
"""

from __future__ import annotations

import inspect

import pytest

import agensgraph

sqlalchemy = pytest.importorskip("sqlalchemy", reason="sqlalchemy is not installed")

from sqlalchemy import create_engine, text  # noqa: E402
from sqlalchemy.dialects import registry  # noqa: E402
from sqlalchemy.dialects.postgresql.psycopg import PGDialect_psycopg  # noqa: E402

pytestmark = pytest.mark.server


class AgensGraphDialect(PGDialect_psycopg):
    """Everything psycopg's dialect does, over this driver's connections.

    Named for the driver rather than for postgresql, since a dialect's name prefixes the arguments
    it accepts and two dialects claiming one name would collide.
    """

    driver = "agensgraph"
    supports_statement_cache = True

    @classmethod
    def import_dbapi(cls) -> object:
        return agensgraph.dbapi


registry.register("postgresql.agensgraph", __name__, "AgensGraphDialect")


@pytest.fixture
def engine(dsn: str):  # type: ignore[no-untyped-def]
    """An engine over this driver, from the same connection string the tests use."""
    from psycopg.conninfo import conninfo_to_dict

    parts = conninfo_to_dict(dsn)
    url = (
        f"postgresql+agensgraph://{parts.get('user', '')}@"
        f"{parts.get('host', 'localhost')}:{parts.get('port', 5432)}/{parts.get('dbname', '')}"
    )
    made = create_engine(url)
    try:
        yield made
    finally:
        made.dispose()


class TestTheDialectIsShort:
    def test_it_is_a_handful_of_lines(self) -> None:
        """Asserted so it cannot drift: a handful of lines, not two hundred.

        The length is the point. A dialect this short is what a complete PEP 249 surface buys, so
        a dialect that grew would mean something had gone missing from it."""
        lines = inspect.getsource(AgensGraphDialect).splitlines()
        code = [line for line in lines if line.strip() and not line.strip().startswith("#")]
        assert len(code) < 20, f"the dialect needed {len(code)} lines, so something is missing"

    def test_it_only_has_to_say_where_the_driver_is(self) -> None:
        """Everything else comes from psycopg's dialect unchanged."""
        overridden = {
            name for name, value in vars(AgensGraphDialect).items() if not name.startswith("__")
        }
        assert overridden == {"driver", "supports_statement_cache", "import_dbapi"}


class TestWhatTheDialectNeedsFromTheDriver:
    def test_the_exception_names_are_on_the_module(self) -> None:
        """``DBAPIError.instance()`` keys off these, so a missing one loses error classification."""
        required = [
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
        ]
        for name in required:
            assert hasattr(agensgraph.dbapi, name), name

    def test_a_connection_reports_closed_and_broken(self) -> None:
        """psycopg's whole ``is_disconnect`` is those two, which replaces matching on messages."""
        assert hasattr(agensgraph.Connection, "closed")
        assert hasattr(agensgraph.Connection, "broken")

    def test_the_paramstyle_is_the_one_the_dialect_expects(self) -> None:
        assert agensgraph.dbapi.paramstyle == "pyformat"

    def test_it_declares_the_api_level_and_thread_safety(self) -> None:
        assert agensgraph.dbapi.apilevel == "2.0"
        assert agensgraph.dbapi.threadsafety in (1, 2, 3)


class TestItActuallyDrivesAServer:
    def test_a_plain_query(self, engine) -> None:  # type: ignore[no-untyped-def]
        with engine.connect() as conn:
            assert conn.execute(text("select 1")).scalar() == 1

    def test_the_server_version_is_read_from_the_banner(self, engine) -> None:  # type: ignore[no-untyped-def]
        """Free: the server says ``PostgreSQL 18beta1 (AgensGraph 2.18-devel)``, which the
        PostgreSQL dialect's own regular expression already matches."""
        with engine.connect() as conn:
            assert conn.dialect.server_version_info is not None
            assert conn.dialect.server_version_info[0] >= 16

    def test_a_cypher_statement_runs_and_describes_its_columns(self, engine, agens) -> None:  # type: ignore[no-untyped-def]
        graph = agens.label_table.graph
        agens.execute("create vlabel person")
        agens.execute("create (:person {name: 'a'})")
        with engine.connect() as conn:
            conn.execute(text(f'set graph_path = "{graph}"'))
            rows = conn.execute(text(r"match (n\:person) return n.name as name")).fetchall()
            assert [row.name for row in rows] == ["a"]

    def test_a_vertex_comes_back_as_a_vertex(self, engine, agens) -> None:  # type: ignore[no-untyped-def]
        """The types are the driver's, not stringified on the way through the dialect."""
        graph = agens.label_table.graph
        agens.execute("create vlabel person")
        agens.execute("create (:person {name: 'a'})")
        with engine.connect() as conn:
            conn.execute(text(f'set graph_path = "{graph}"'))
            (vertex,) = conn.execute(text(r"match (n\:person) return n")).fetchone()
            assert isinstance(vertex, agensgraph.Vertex)
            assert vertex.properties == {"name": "a"}

    def test_a_parameter_is_bound_as_this_driver_binds_one(self, engine, agens) -> None:  # type: ignore[no-untyped-def]
        """Which is the correction this driver ships: a string is declared text, so a lookup for
        '123' finds the row rather than matching nothing."""
        graph = agens.label_table.graph
        agens.execute("create vlabel doc")
        agens.execute("create (:doc {code: '123'})")
        with engine.connect() as conn:
            conn.execute(text(f'set graph_path = "{graph}"'))
            found = conn.execute(
                text(r"match (n\:doc) where n.code = :code return n.code"), {"code": "123"}
            ).scalar()
            assert found == "123"

    def test_a_failure_is_classified_rather_than_read_from_its_message(self, engine) -> None:  # type: ignore[no-untyped-def]
        with engine.connect() as conn, pytest.raises(sqlalchemy.exc.ProgrammingError):
            conn.execute(text("select from where"))

    def test_a_label_reads_as_a_bind_parameter_unless_the_colon_is_escaped(
        self, engine, agens
    ) -> None:  # type: ignore[no-untyped-def]
        """The collision worth knowing about: ``(:doc)`` is Cypher's label syntax and SQLAlchemy's
        named-parameter syntax, so ``text()`` asks for a value for a parameter called ``doc``.
        Escaping the colon passes it through, and Cypher receives the label."""
        graph = agens.label_table.graph
        agens.execute("create vlabel doc")
        with engine.connect() as conn:
            conn.execute(text(f'set graph_path = "{graph}"'))
            with pytest.raises(sqlalchemy.exc.StatementError, match="bind parameter 'doc'"):
                conn.execute(text("create (:doc {n: 1})"))
            conn.rollback()
            conn.execute(text(f'set graph_path = "{graph}"'))
            conn.execute(text(r"create (\:doc {n: 1})"))
            assert conn.execute(text("match (n\\:doc) return count(*)")).scalar() == 1

    def test_a_transaction_commits_and_rolls_back(self, engine, agens) -> None:  # type: ignore[no-untyped-def]
        graph = agens.label_table.graph
        agens.execute("create vlabel doc")
        with engine.connect() as conn:
            conn.execute(text(f'set graph_path = "{graph}"'))
            conn.execute(text(r"create (\:doc {n: 1})"))
            conn.rollback()
            conn.execute(text(f'set graph_path = "{graph}"'))
            assert conn.execute(text(r"match (n\:doc) return count(*)")).scalar() == 0
            conn.execute(text(r"create (\:doc {n: 2})"))
            conn.commit()
        with engine.connect() as conn:
            conn.execute(text(f'set graph_path = "{graph}"'))
            assert conn.execute(text(r"match (n\:doc) return count(*)")).scalar() == 1

    def test_the_pool_hands_back_a_usable_connection(self, engine) -> None:  # type: ignore[no-untyped-def]
        """SQLAlchemy pools these itself, so a connection has to survive being returned and taken
        again."""
        for _ in range(5):
            with engine.connect() as conn:
                assert conn.execute(text("select 1")).scalar() == 1
