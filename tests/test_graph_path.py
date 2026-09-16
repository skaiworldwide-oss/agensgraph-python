"""Keeping the label table in step with the graph the session is reading.

The composite rendering carries a label id and not a label name, so the driver resolves the
name from a table it filled for one graph. Label ids restart at three in every graph, so a
table belonging to another graph does not fail to resolve -- it resolves to the wrong name.
Everything here is about making sure it cannot.
"""

from __future__ import annotations

from types import SimpleNamespace

import psycopg
import pytest

import agensgraph
from agensgraph._core import GraphMixin, LabelCache
from agensgraph.cypher import changes_graph_path, quote_identifier, without_literals
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
    "set role myrole",
    "set session authorization myrole",
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


class _Session(GraphMixin):
    """Enough of a connection for the methods that read the reported graph path.

    Built here rather than run against a server because the servers this suite can reach do
    not all report it: 2.18.6 does, and 2.17 and 2.18.4 do not.
    """

    def __init__(self, reported: str | None) -> None:
        self.reported = reported

    @property
    def info(self) -> SimpleNamespace:  # type: ignore[override]
        return SimpleNamespace(parameter_status=lambda name: self.reported)


class TestReadingTheGraphPathTheServerReports:
    def test_a_server_that_does_not_report_it_is_read_as_not_reporting(self) -> None:
        session = _Session(None)
        session._note_graph_path()
        assert not session._agens_reports_graph_path

    def test_a_server_reporting_no_graph_at_all_still_reports(self) -> None:
        """It answers with an empty string at connect, which is an answer and not a silence."""
        session = _Session("")
        session._note_graph_path()
        assert session._agens_reports_graph_path
        assert session._agens_graph_path_seen == ""

    def test_a_move_is_seen_once_and_not_again(self) -> None:
        session = _Session("")
        session._note_graph_path()
        assert not session._graph_path_moved()
        session.reported = "other"
        assert session._graph_path_moved()
        assert not session._graph_path_moved()

    def test_a_move_back_is_a_move(self) -> None:
        """Which is what a rollback performs, and what reading the statement cannot see."""
        session = _Session("first")
        session._note_graph_path()
        session.reported = "second"
        assert session._graph_path_moved()
        session.reported = "first"
        assert session._graph_path_moved()

    def test_following_the_path_drops_the_table_once_per_move(self) -> None:
        session = _Session("first")
        session._note_graph_path()
        session._agens_labels = LabelCache()
        session.label_table.load("first", [])
        assert not session._follow_reported_graph_path()
        assert session.label_table.graph == "first"
        session.reported = "second"
        assert session._follow_reported_graph_path()
        assert session.label_table.graph is None
        session.label_table.load("second", [])
        assert not session._follow_reported_graph_path()
        assert session.label_table.graph == "second"

    def test_a_server_that_does_not_report_it_is_never_followed(self) -> None:
        """The text of the statements is the only signal there, and that is read elsewhere."""
        session = _Session(None)
        session._note_graph_path()
        session._agens_labels = LabelCache()
        session.label_table.load("first", [])
        session.reported = "second"
        assert not session._follow_reported_graph_path()
        assert session.label_table.graph == "first"


@pytest.mark.server
class TestTheReportedPathIsWhatTheServerWouldAnswer:
    """Where the server reports the path the driver reads it in place of asking, so the two
    must agree. On a server that does not report it both are the one query, as a control."""

    def test_where_the_session_is_read_free_is_where_the_server_says(
        self, agens, second_graph: str
    ) -> None:  # type: ignore[no-untyped-def]
        def asked() -> str:
            # Names the setting, so the text scan drops the table once this has run.
            return agens.execute("select current_setting('graph_path')").fetchone()[0]

        held = agens.label_table.graph
        assert agens._current_graph() == held
        assert asked() == held
        agens.execute(f'set graph_path = "{second_graph}"')
        assert agens._current_graph() == second_graph
        assert asked() == second_graph
        agens.execute("reset graph_path")
        assert agens._current_graph() is None
        assert asked() == ""


@pytest.mark.server
class TestTheTableFollowsTheSession:
    """Each statement below was run against a server to see whether it moves the session."""

    @pytest.mark.parametrize("statement", MOVES)
    def test_the_table_is_dropped(self, agens, second_graph: str, statement: str) -> None:  # type: ignore[no-untyped-def]
        agens.execute(statement.replace("other", second_graph))
        assert agens.label_table.graph is None

    @pytest.mark.parametrize("statement", STAYS)
    def test_the_table_is_kept(self, agens, statement: str) -> None:  # type: ignore[no-untyped-def]
        """``myrole`` stands for whoever is connected, since the two role statements name one and
        no server has a role this suite can assume."""
        agens.execute("create vlabel person")
        agens.refresh_labels()
        held = agens.label_table.graph
        agens.execute(statement.replace("myrole", quote_identifier(agens.info.user)))
        assert agens.label_table.graph == held

    def test_a_failed_statement_changes_nothing(self, agens) -> None:  # type: ignore[no-untyped-def]
        held = agens.label_table.graph
        with pytest.raises(agensgraph.errors.Error):
            agens.execute("set graph_path = no_such_graph_exists")
        assert agens.label_table.graph == held


@pytest.mark.server
class TestABinaryReadAfterTheSessionMoves:
    """The defect this exists to prevent: a vertex of one graph named from another's table."""

    def test_it_refuses_rather_than_naming_the_wrong_label(
        self, agens, second_graph: str
    ) -> None:  # type: ignore[no-untyped-def]
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

    def test_a_path_set_for_the_transaction_goes_back_at_commit(
        self, dsn: str, agens, second_graph: str
    ) -> None:  # type: ignore[no-untyped-def]
        """``SET LOCAL`` ends with its transaction, so a commit moves the session back as a
        rollback does. Seen where the server reports the path; the statement text says a path
        was set, not for how long."""
        first = agens.label_table.graph
        with agensgraph.Connection.connect(dsn) as conn:
            if not conn._agens_reports_graph_path:
                pytest.skip("this server does not report graph_path")
            conn.graph(first)
            conn.commit()
            conn.execute(f'set local graph_path = "{second_graph}"')
            conn.refresh_labels()
            assert conn.label_table.graph == second_graph
            conn.commit()
            assert conn.label_table.graph is None
            assert conn.execute("show graph_path").fetchone()[0] == first
            conn.rollback()

    def test_a_block_ended_behind_the_cursor_is_seen_by_what_describes(
        self, dsn: str, agens, second_graph: str
    ) -> None:  # type: ignore[no-untyped-def]
        """psycopg ends a ``transaction()`` block with a statement no cursor of the connection
        runs. Where the server reports the path, a describing method sees the move back by
        itself and lists the first graph's labels, not the second's."""
        agens.execute("create vlabel person")
        first = agens.label_table.graph
        with agensgraph.Connection.connect(dsn) as conn:
            if not conn._agens_reports_graph_path:
                pytest.skip("this server does not report graph_path")
            conn.graph(first)
            conn.commit()
            with conn.transaction():
                conn.execute(f'set graph_path = "{second_graph}"')
                conn.refresh_labels()
                assert conn.label_table.graph == second_graph
                raise psycopg.Rollback
            names = {label.name for label in conn.labels()}
            assert "person" in names
            assert "account" not in names
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
            (vertex,) = conn.execute_query("match (n:account) return n", binary_=True).records[
                0
            ]
            assert vertex.label == "account"


@pytest.mark.server
class TestATableNamingNoGraphIsAskedAboutRatherThanRefused:
    """A statement mentioning the setting in its data drops the table, and must cost no more.

    The mention is counted whatever it is inside, which is the safe direction to be wrong in.
    What made it expensive was that dropping the table dropped the graph's *name* with it, and
    a reader that wanted the name was refused rather than told to ask: measured, a plain read
    whose data held the words left the next `indexes()` raising.
    """

    def test_a_read_whose_data_mentions_the_setting_costs_a_reading(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel doc")
        agens.refresh_labels()
        before = agens.label_table.graph
        agens.execute("match (n:doc) where n.note = 'graph_path' return n")
        assert agens.label_table.graph is None, "the table is dropped, as it should be"
        assert [i.name for i in agens.indexes("doc")] == []
        assert agens.describe().graph == before

    def test_and_the_reading_answers_with_where_the_session_is_now(
        self, agens, second_graph: str
    ) -> None:  # type: ignore[no-untyped-def]
        """Which is why it is read rather than remembered: the name held before it is stale.

        A table that kept its name across a real move would answer for the graph the session
        has left, and every reader of it would describe the wrong graph.
        """
        agens.execute("create vlabel doc")
        agens.refresh_labels()
        agens.execute(f"select set_config('graph_path', '{second_graph}', false)")
        assert agens.describe().graph == second_graph

    def test_a_session_reading_no_graph_is_still_refused(self, agens) -> None:  # type: ignore[no-untyped-def]
        """The two states arrive as the same missing name, and only one of them is recoverable."""
        agens.execute("create vlabel doc")
        agens.execute("reset graph_path")
        with pytest.raises(ValueError, match="no graph is selected"):
            agens.indexes("doc")

    def test_naming_a_graph_never_reads_the_setting(self, agens) -> None:  # type: ignore[no-untyped-def]
        """A caller that said which graph is answered without a statement being sent for it."""
        agens.execute("create vlabel doc")
        graph = agens.label_table.graph
        agens.execute("reset graph_path")
        assert [i.name for i in agens.indexes("doc", graph=graph)] == []


@pytest.mark.server
class TestWhyTheMentionIsNotBlankedOutFirst:
    """Blanking the literals before searching would be a false negative where it costs most."""

    def test_a_dollar_quoted_body_can_move_the_session(self, agens, second_graph: str) -> None:  # type: ignore[no-untyped-def]
        """And a dollar-quoted body is blanked whole, so the blanked text mentions nothing."""
        statement = (
            f"do $$ begin perform set_config('graph_path', '{second_graph}', false); end $$"
        )
        assert "graph_path" not in without_literals(statement)
        agens.execute(statement)
        (now,) = agens.execute("select current_setting('graph_path')").fetchone()
        assert now == second_graph, "the session moved, so a reader of the text has to say so"
        assert changes_graph_path(statement), "and reading it as written does"

    def test_the_table_follows_it(self, agens, second_graph: str) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel doc")
        agens.refresh_labels()
        agens.execute(
            f"do $$ begin perform set_config('graph_path', '{second_graph}', false); end $$"
        )
        assert agens.describe().graph == second_graph
