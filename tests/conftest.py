"""Shared fixtures.

Everything that needs a server carries the ``server`` marker and is skipped when
``AGENSGRAPH_TEST_DSN`` is unset, so the ordinary suite runs against nothing.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

DSN_VARIABLE = "AGENSGRAPH_TEST_DSN"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip the server tests when there is no server to run them against."""
    if os.environ.get(DSN_VARIABLE):
        return
    skip = pytest.mark.skip(reason=f"set {DSN_VARIABLE} to run the tests that need a server")
    for item in items:
        if "server" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def dsn() -> str:
    value = os.environ.get(DSN_VARIABLE)
    if not value:
        pytest.skip(f"set {DSN_VARIABLE} to run the tests that need a server")
    return value


@pytest.fixture
def agens(dsn: str) -> Iterator[object]:
    """A driver connection, on its own scratch graph, with the graph already selected.

    Distinct from :func:`conn`, which is a plain psycopg connection with the graph loaders
    registered on it. Both are wanted: this one exercises the driver's own surface, and that
    one proves the loaders work on a connection the driver did not make.
    """
    import agensgraph

    with agensgraph.connect(dsn, autocommit=True) as connection:
        name = f"a_{os.getpid()}_{id(connection) % 100000}"
        connection.execute(f'create graph "{name}"')
        connection.graph(name)
        try:
            yield connection
        finally:
            connection.rollback()
            connection.execute("reset graph_path")
            connection.execute(f'drop graph "{name}" cascade')


@pytest.fixture
def second_graph(dsn: str) -> Iterator[str]:
    """A second graph to move a session to, holding one label of its own.

    Its first user label gets id three, the same as the first label of any other graph, which
    is what makes a table belonging to the wrong graph resolve to a name instead of failing.
    """
    import agensgraph

    with agensgraph.connect(dsn, autocommit=True) as connection:
        name = f"b_{os.getpid()}_{id(connection) % 100000}"
        connection.execute(f'create graph "{name}"')
        connection.execute(f'set graph_path = "{name}"')
        connection.execute("create vlabel account")
        connection.execute("create elabel owns")
        try:
            yield name
        finally:
            connection.execute("reset graph_path")
            connection.execute(f'drop graph "{name}" cascade')


@pytest.fixture
def conn(dsn: str) -> Iterator[object]:
    """A connection with the graph types registered, on its own scratch graph.

    A graph per test rather than one for the suite, so that a test creating a label cannot
    change what another one sees -- which matters here more than usual, because a label
    created after a label table was filled is one of the states under test.
    """
    import psycopg

    from agensgraph.adapters import register_text

    connection = psycopg.connect(dsn, autocommit=True)
    register_text(connection)
    name = f"t_{os.getpid()}_{id(connection) % 100000}"
    connection.execute(f'create graph "{name}"')
    connection.execute(f'set graph_path = "{name}"')
    try:
        yield connection
    finally:
        try:
            connection.rollback()
            connection.execute("reset graph_path")
            connection.execute(f'drop graph "{name}" cascade')
        finally:
            connection.close()
