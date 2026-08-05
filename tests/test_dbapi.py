"""The interface a generic tool expects.

Everything a tool looks for by PEP 249's names has to be findable by those names, or it does not
matter that the behaviour underneath is right. Most of it is psycopg's, so most of these tests
check that it was actually re-exported rather than assumed.
"""

from __future__ import annotations

import datetime

import psycopg
import pytest

from agensgraph import dbapi

REQUIRED_MODULE_ATTRIBUTES = ["apilevel", "threadsafety", "paramstyle", "connect"]

REQUIRED_EXCEPTIONS = [
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

REQUIRED_CONSTRUCTORS = [
    "Date",
    "Time",
    "Timestamp",
    "DateFromTicks",
    "TimeFromTicks",
    "TimestampFromTicks",
    "Binary",
]

REQUIRED_TYPE_OBJECTS = ["STRING", "BINARY", "NUMBER", "DATETIME", "ROWID"]


@pytest.mark.parametrize(
    "name",
    [
        *REQUIRED_MODULE_ATTRIBUTES,
        *REQUIRED_EXCEPTIONS,
        *REQUIRED_CONSTRUCTORS,
        *REQUIRED_TYPE_OBJECTS,
    ],
)
def test_every_name_a_tool_looks_for_is_there(name: str) -> None:
    assert hasattr(dbapi, name), f"{name} is part of the interface and is missing"


def test_the_level_and_style_are_the_ones_underneath() -> None:
    """Not restated, because a restated constant is one that can drift from the truth."""
    assert dbapi.apilevel == psycopg.apilevel == "2.0"
    assert dbapi.paramstyle == psycopg.paramstyle
    assert dbapi.threadsafety == psycopg.threadsafety


@pytest.mark.parametrize("name", REQUIRED_EXCEPTIONS)
def test_an_exception_is_the_same_class_psycopg_raises(name: str) -> None:
    """A tool that catches these must catch what actually comes out of a statement."""
    assert getattr(dbapi, name) is getattr(psycopg, name)


def test_the_exceptions_nest_the_way_the_specification_says() -> None:
    assert issubclass(dbapi.DatabaseError, dbapi.Error)
    for name in (
        "DataError",
        "OperationalError",
        "IntegrityError",
        "InternalError",
        "ProgrammingError",
        "NotSupportedError",
    ):
        assert issubclass(getattr(dbapi, name), dbapi.DatabaseError)
    assert issubclass(dbapi.InterfaceError, dbapi.Error)


def test_the_constructors_build_what_they_say() -> None:
    assert dbapi.Date(2026, 8, 5) == datetime.date(2026, 8, 5)
    assert dbapi.Time(12, 30, 0) == datetime.time(12, 30)
    assert dbapi.Timestamp(2026, 8, 5, 12, 30, 0) == datetime.datetime(2026, 8, 5, 12, 30)
    assert dbapi.Binary(b"x") is not None


@pytest.mark.server
class TestAgainstAServer:
    def test_connecting_by_the_generic_name_reads_the_graph_types(self, dsn: str) -> None:
        """The one thing here that is not psycopg's, and the reason the module exists."""
        import agensgraph

        conn = dbapi.connect(dsn, autocommit=True)
        try:
            conn.execute("drop graph if exists dbapi_check cascade")
            conn.execute("create graph dbapi_check")
            conn.graph("dbapi_check")
            conn.execute("create vlabel thing")
            conn.execute("create (:thing {n: 1})")
            (v,) = conn.execute("match (n:thing) return n").fetchone()
            assert isinstance(v, agensgraph.Vertex)
            conn.execute("reset graph_path")
            conn.execute("drop graph dbapi_check cascade")
        finally:
            conn.close()

    def test_the_two_attributes_that_replace_matching_on_messages(self, dsn: str) -> None:
        """psycopg's entire judgement about a lost connection is these two, so a tool that asks
        them never has to read an error message -- which is what breaks on a reworded one."""
        conn = dbapi.connect(dsn)
        assert conn.closed is False
        assert conn.broken is False
        conn.close()
        assert conn.closed is True

    def test_a_cursor_describes_its_columns(self, dsn: str) -> None:
        """A tool reads the shape of a result from here, including for a Cypher one."""
        conn = dbapi.connect(dsn, autocommit=True)
        try:
            cursor = conn.execute("select 1 as one, 'x' as two")
            assert [column.name for column in cursor.description] == ["one", "two"]
        finally:
            conn.close()

    def test_the_standard_cursor_dance_works(self, dsn: str) -> None:
        conn = dbapi.connect(dsn, autocommit=True)
        try:
            cursor = conn.cursor()
            cursor.execute("select generate_series(1, 5) as n")
            assert cursor.fetchone() == (1,)
            assert cursor.fetchmany(2) == [(2,), (3,)]
            assert cursor.fetchall() == [(4,), (5,)]
            assert cursor.rowcount == 5
            cursor.close()
        finally:
            conn.close()

    def test_a_transaction_commits_and_rolls_back(self, dsn: str) -> None:
        conn = dbapi.connect(dsn)
        try:
            conn.execute("create temporary table t (n int)")
            conn.execute("insert into t values (1)")
            conn.commit()
            conn.execute("insert into t values (2)")
            conn.rollback()
            assert conn.execute("select count(*) from t").fetchone() == (1,)
        finally:
            conn.close()

    def test_the_version_gate_still_applies(self, dsn: str) -> None:
        """Connecting by the generic name must not be a way around the check."""
        conn = dbapi.connect(dsn)
        try:
            assert conn.capabilities.version >= (2, 16)
        finally:
            conn.close()
