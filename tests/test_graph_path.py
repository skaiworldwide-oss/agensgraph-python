"""Keeping the label table in step with the graph the session is reading.

The composite rendering carries a label id and not a label name, so the driver resolves the
name from a table it filled for one graph. Label ids restart at three in every graph, so a
table belonging to another graph does not fail to resolve -- it resolves to the wrong name.
Everything here is about making sure it cannot.
"""

from __future__ import annotations

import pytest

import agensgraph
from agensgraph.cypher import changes_graph_path
from agensgraph.errors import StaleLabelCache

MOVES = [
    "set graph_path = other",
    "SET GRAPH_PATH = other",
    "set graph_path to other",
    "reset graph_path",
    "set graph_path to default",
    "select set_config('graph_path', 'other', false)",
    "reset all",
    "discard all",
    "RESET ALL",
    "discard   all",
]

STAYS = [
    "select 1",
    "match (n:person) return n",
    "set role agens",
    "set session authorization agens",
    "discard plans",
    "discard sequences",
    "set statement_timeout = 0",
    "create (:person {name: 'a'})",
    "select 'graph' as a, 'path' as b",
]


@pytest.mark.parametrize("statement", MOVES)
def test_a_statement_that_moves_the_session_is_recognised(statement: str) -> None:
    assert changes_graph_path(statement)


@pytest.mark.parametrize("statement", STAYS)
def test_and_one_that_does_not_is_left_alone(statement: str) -> None:
    assert not changes_graph_path(statement)


@pytest.mark.server
class TestTheTableFollowsTheSession:
    """Each statement below was run against a server to see whether it moves the session."""

    @pytest.mark.parametrize("statement", MOVES)
    def test_the_table_is_dropped(self, agens, second_graph: str, statement: str) -> None:  # type: ignore[no-untyped-def]
        agens.execute(statement.replace("other", second_graph))
        assert agens.label_table.graph is None

    @pytest.mark.parametrize("statement", STAYS)
    def test_the_table_is_kept(self, agens, statement: str) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel person")
        agens.refresh_labels()
        held = agens.label_table.graph
        agens.execute(statement)
        assert agens.label_table.graph == held

    def test_a_failed_statement_changes_nothing(self, agens) -> None:  # type: ignore[no-untyped-def]
        held = agens.label_table.graph
        with pytest.raises(agensgraph.errors.Error):
            agens.execute("set graph_path = no_such_graph_exists")
        assert agens.label_table.graph == held


@pytest.mark.server
class TestABinaryReadAfterTheSessionMoves:
    """The defect this exists to prevent: a vertex of one graph named from another's table."""

    def test_it_refuses_rather_than_naming_the_wrong_label(self, agens, second_graph: str) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel person")
        agens.refresh_labels()
        agens.execute(f'set graph_path = "{second_graph}"')
        agens.execute("create (:account {n: 'x'})")
        assert [r[0].label for r in agens.execute_query("match (n) return n").records] == [
            "account"
        ]
        with pytest.raises(StaleLabelCache, match="refresh_labels"):
            agens.execute_query("match (n) return n", binary_=True)

    def test_refresh_labels_is_the_way_back_without_naming_the_graph(
        self, agens, second_graph: str
    ) -> None:  # type: ignore[no-untyped-def]
        """The driver never saw the name, so it asks the server which graph it is reading."""
        agens.execute(f'set graph_path = "{second_graph}"')
        agens.execute("create (:account {n: 'x'})-[:owns]->(:account {n: 'y'})")
        agens.refresh_labels()
        assert agens.label_table.graph == second_graph
        (vertex,) = agens.execute_query("match (n:account) return n", binary_=True).records[0]
        assert vertex.label == "account"
        (path,) = agens.execute_query(
            "match p = (:account)-[:owns]->() return p", binary_=True
        ).records[0]
        assert [element.label for element in path.elements] == ["account", "owns", "account"]

    def test_the_text_rendering_is_right_throughout(self, agens, second_graph: str) -> None:  # type: ignore[no-untyped-def]
        """It carries the name the server wrote, so it never depended on the table."""
        agens.execute(f'set graph_path = "{second_graph}"')
        agens.execute("create (:account {n: 'x'})")
        (vertex,) = agens.execute_query("match (n:account) return n").records[0]
        assert vertex.label == "account"


@pytest.mark.server
class TestRollingBackTheGraphPath:
    """Setting the graph path belongs to the transaction, so a rollback takes it back."""

    def test_a_graph_selected_in_a_rolled_back_transaction_drops_the_table(
        self, dsn: str, agens, second_graph: str
    ) -> None:  # type: ignore[no-untyped-def]
        first = agens.label_table.graph
        with agensgraph.Connection.connect(dsn) as conn:
            conn.graph(first)
            conn.commit()
            conn.graph(second_graph)
            assert conn.label_table.graph == second_graph
            conn.rollback()
            assert conn.execute("show graph_path").fetchone()[0] == first
            assert conn.label_table.graph is None
            conn.rollback()

    def test_committing_keeps_the_table(self, dsn: str, second_graph: str) -> None:
        with agensgraph.Connection.connect(dsn) as conn:
            conn.graph(second_graph)
            conn.commit()
            assert conn.label_table.graph == second_graph
            conn.rollback()


@pytest.mark.server
class TestTheCompositeRenderingWithNoTable:
    """The loaders for it are built around a label table, so without one nothing reads it."""

    def test_it_is_refused_rather_than_handing_back_the_wire_bytes(
        self, dsn: str, second_graph: str
    ) -> None:
        with agensgraph.Connection.connect(dsn, autocommit=True) as conn:
            conn.execute(f'set graph_path = "{second_graph}"')
            conn.execute("create (:account {n: 'x'})-[:owns]->(:account {n: 'y'})")
            assert conn.label_table.graph is None
            for statement in [
                "match (n:account) return n",
                "match p = (:account)-[:owns]->() return p",
                "match p = (:account)-[:owns]->() return nodes(p)",
            ]:
                with pytest.raises(StaleLabelCache, match="refresh_labels"):
                    conn.execute_query(statement, binary_=True)

    def test_reading_in_chunks_is_refused_too(self, dsn: str, second_graph: str) -> None:
        """In a transaction, which is what reading in chunks needs of its own accord."""
        with agensgraph.Connection.connect(dsn) as conn:
            conn.execute(f'set graph_path = "{second_graph}"')
            conn.execute("create (:account {n: 'x'})")
            with pytest.raises(StaleLabelCache, match="refresh_labels"):
                list(conn.stream("match (n:account) return n", binary_=True))
            conn.rollback()

    def test_the_text_rendering_needs_no_table(self, dsn: str, second_graph: str) -> None:
        with agensgraph.Connection.connect(dsn, autocommit=True) as conn:
            conn.execute(f'set graph_path = "{second_graph}"')
            conn.execute("create (:account {n: 'x'})")
            (vertex,) = conn.execute_query("match (n:account) return n").records[0]
            assert vertex.label == "account"

    def test_filling_the_table_makes_it_available(self, dsn: str, second_graph: str) -> None:
        with agensgraph.Connection.connect(dsn, autocommit=True) as conn:
            conn.execute(f'set graph_path = "{second_graph}"')
            conn.execute("create (:account {n: 'x'})")
            conn.refresh_labels()
            (vertex,) = conn.execute_query("match (n:account) return n", binary_=True).records[0]
            assert vertex.label == "account"
