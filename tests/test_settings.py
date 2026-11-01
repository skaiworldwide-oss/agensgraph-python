"""Settings for one statement, in the same round trip as it.

``SET LOCAL`` before a search is a round trip of its own, and in autocommit it does nothing: there
is no transaction for it to be local to, and the server only warns. These pin that the settings
reach the statement they were given with, end with it in autocommit, last as ``SET LOCAL`` does
inside a transaction, and that ``vector_search_options`` warns where it can do nothing.
"""

from __future__ import annotations

import warnings

import pytest
import pytest_asyncio

import agensgraph
from agensgraph.settings import check_settings, settings_statement

TIMEOUT = "select current_setting('statement_timeout')"


class TestCheckingThem:
    def test_each_kind_of_value_is_written_as_the_server_reads_it(self) -> None:
        assert check_settings(
            {"enable_hashjoin": False, "jit": True, "work_mem": 4096, "seq_page_cost": 1.5}
        ) == [
            ("enable_hashjoin", "off"),
            ("jit", "on"),
            ("work_mem", "4096"),
            ("seq_page_cost", "1.5"),
        ]
        assert check_settings({"application_name": "x"}) == [("application_name", "x")]

    def test_the_search_settings_are_held_to_their_list(self) -> None:
        """The server takes an unknown ``hnsw.`` name in silence on PostgreSQL 14."""
        with pytest.raises(ValueError, match="not a vector search option"):
            check_settings({"hnsw.ef_serach": 100})
        with pytest.raises(ValueError, match="whole number"):
            check_settings({"hnsw.ef_search": True})
        assert check_settings({"HNSW.ef_search": 100}) == [("HNSW.ef_search", "100")]

    def test_graph_path_is_refused_with_what_to_do_instead(self) -> None:
        with pytest.raises(ValueError, match=r"graph\(\)"):
            check_settings({"graph_path": "g"})
        with pytest.raises(ValueError, match=r"graph\(\)"):
            check_settings({"GRAPH_PATH": "g"})

    @pytest.mark.parametrize("name", ["", "1abc", "a;b", "a b", "a.", "set_config('x'"])
    def test_a_name_the_server_could_not_hold_is_refused(self, name: str) -> None:
        with pytest.raises(ValueError, match="not the name of a setting"):
            check_settings({name: 1})

    def test_a_value_of_any_other_kind_is_refused(self) -> None:
        with pytest.raises(TypeError, match="takes a string, a number or a boolean"):
            check_settings({"work_mem": [1]})
        with pytest.raises(ValueError, match="NUL"):
            check_settings({"application_name": "a\x00b"})

    def test_they_go_in_one_statement_whose_text_depends_only_on_how_many(self) -> None:
        statement, values = settings_statement([("a.b", "1"), ("c", "on")])
        assert statement == "select set_config(%s, %s, true), set_config(%s, %s, true)"
        assert values == ("a.b", "1", "c", "on")
        assert settings_statement([("x", "2"), ("y", "off")])[0] == statement
        assert "false" in settings_statement([("a", "1")], local=False)[0]

    def test_no_settings_is_not_a_statement(self) -> None:
        with pytest.raises(ValueError, match="no settings"):
            settings_statement([])


@pytest.mark.server
class TestSendingThemWithAStatement:
    def test_the_statement_sees_them(self, agens) -> None:  # type: ignore[no-untyped-def]
        result = agens.execute_query(TIMEOUT, settings_={"statement_timeout": 4321})
        assert result.records == [("4321ms",)]

    def test_and_in_autocommit_they_end_with_it(self, agens) -> None:  # type: ignore[no-untyped-def]
        before = agens.execute_query(TIMEOUT).records
        agens.execute_query(TIMEOUT, settings_={"statement_timeout": 4321})
        assert agens.execute_query(TIMEOUT).records == before

    def test_a_cypher_statement_sees_them_too(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel doc")
        agens.execute("create (:doc {n: 1})")
        result = agens.execute_query(
            "match (n:doc) return n.n as n, (select current_setting('enable_hashjoin')) as hj",
            settings_={"enable_hashjoin": False},
        )
        assert result.records == [(1, "off")]
        assert result.keys == ["n", "hj"]

    def test_inside_a_transaction_they_last_until_it_ends(self, agens) -> None:  # type: ignore[no-untyped-def]
        """As ``SET LOCAL`` would: there is no ending them sooner without a round trip."""
        before = agens.execute_query(TIMEOUT).records
        with agens.transaction():
            agens.execute_query(TIMEOUT, settings_={"statement_timeout": 4321})
            assert agens.execute_query(TIMEOUT).records == [("4321ms",)]
        assert agens.execute_query(TIMEOUT).records == before

    def test_inside_a_pipeline_of_the_caller_s_they_still_reach_the_statement(
        self, agens
    ) -> None:  # type: ignore[no-untyped-def]
        with agens.pipeline():
            agens.execute("select 1")
            result = agens.execute_query(TIMEOUT, settings_={"statement_timeout": 4321})
        assert result.records == [("4321ms",)]

    def test_inside_a_read_only_block_they_are_not_mistaken_for_a_write(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel doc")
        # The test role is a superuser; the boundary under test is the statement check.
        with agens.read_only_transaction(allow_server_programs=True):
            result = agens.execute_query(
                "match (n:doc) return count(n)", settings_={"enable_hashjoin": False}
            )
        assert result.records == [(0,)]

    def test_both_go_through_the_reporting_cursor(self, agens) -> None:  # type: ignore[no-untyped-def]
        sent: list[str] = []

        def record(record) -> None:  # type: ignore[no-untyped-def]
            sent.append(record.statement)

        agensgraph.add_query_logger(record)
        try:
            agens.execute_query(TIMEOUT, settings_={"statement_timeout": 4321, "jit": False})
        finally:
            agensgraph.remove_query_logger(record)
        assert sent == ["select set_config(%s, %s, true), set_config(%s, %s, true)", TIMEOUT]

    def test_a_setting_the_server_does_not_know_raises_its_error(self, agens) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(agensgraph.errors.Error) as caught:
            agens.execute_query("select 1", settings_={"no_such_setting": 1})
        assert caught.value.sqlstate == "42704"
        assert agens.execute_query("select 1").records == [(1,)]

    def test_a_failing_statement_raises_its_own_error(self, agens) -> None:  # type: ignore[no-untyped-def]
        """It arrives when the pipeline ends rather than from the cursor, and still reads the same."""
        with pytest.raises(agensgraph.errors.Error) as caught:
            agens.execute_query("select 1/0", settings_={"statement_timeout": 4321})
        assert caught.value.sqlstate == "22012"
        assert agens.execute_query(TIMEOUT).records != [("4321ms",)]

    def test_a_refused_setting_costs_no_round_trip(self, agens) -> None:  # type: ignore[no-untyped-def]
        sent: list[str] = []

        def record(record) -> None:  # type: ignore[no-untyped-def]
            sent.append(record.statement)

        agensgraph.add_query_logger(record)
        try:
            with pytest.raises(ValueError, match=r"graph\(\)"):
                agens.execute_query("select 1", settings_={"graph_path": "elsewhere"})
        finally:
            agensgraph.remove_query_logger(record)
        assert sent == []

    def test_no_settings_is_the_ordinary_route(self, agens) -> None:  # type: ignore[no-untyped-def]
        assert agens.execute_query("select 1", settings_={}).records == [(1,)]


@pytest.mark.server
class TestTuningAVectorSearchByItself:
    """``vector_search_options`` defaults to settings local to the transaction."""

    EF = "select current_setting('hnsw.ef_search', true)"

    def test_in_autocommit_it_warns_rather_than_doing_nothing_in_silence(self, agens) -> None:  # type: ignore[no-untyped-def]
        with pytest.warns(RuntimeWarning, match="does nothing in autocommit"):
            agens.vector_search_options({"hnsw.ef_search": 77})

    def test_a_misspelt_option_is_still_refused_there(self, agens) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="not a vector search option"):
            agens.vector_search_options({"hnsw.ef_serach": 77})

    def test_in_a_transaction_every_option_applies(self, agens) -> None:  # type: ignore[no-untyped-def]
        with agens.transaction():
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                agens.vector_search_options(
                    {"hnsw.ef_search": 77, "hnsw.iterative_scan": "relaxed_order"}
                )
            assert agens.execute(self.EF).fetchone() == ("77",)
            assert agens.execute(
                "select current_setting('hnsw.iterative_scan', true)"
            ).fetchone() == ("relaxed_order",)

    def test_in_a_pipeline_of_the_caller_s_it_applies_without_a_warning(self, agens) -> None:  # type: ignore[no-untyped-def]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with agens.pipeline():
                agens.vector_search_options({"hnsw.ef_search": 77})
                cursor = agens.execute(self.EF)
        assert cursor.fetchone() == ("77",)

    def test_for_the_session_it_applies_in_autocommit(self, agens) -> None:  # type: ignore[no-untyped-def]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            agens.vector_search_options({"hnsw.ef_search": 55}, local=False)
        assert agens.execute(self.EF).fetchone() == ("55",)


class TestTheAwaitingInterface:
    @pytest_asyncio.fixture
    async def conn(self, dsn: str):  # type: ignore[no-untyped-def]
        connection = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
        async with connection:
            yield connection

    @pytest.mark.server
    @pytest.mark.asyncio
    async def test_the_statement_sees_them_and_they_end_with_it(self, conn) -> None:  # type: ignore[no-untyped-def]
        result = await conn.execute_query(TIMEOUT, settings_={"statement_timeout": 4321})
        assert result.records == [("4321ms",)]
        assert (await conn.execute_query(TIMEOUT)).records != [("4321ms",)]

    @pytest.mark.server
    @pytest.mark.asyncio
    async def test_a_search_option_in_autocommit_warns(self, conn) -> None:  # type: ignore[no-untyped-def]
        with pytest.warns(RuntimeWarning, match="does nothing in autocommit"):
            await conn.vector_search_options({"hnsw.ef_search": 77})
