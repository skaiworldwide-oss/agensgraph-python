"""Subscribing to what the server announces, and announcing.

The statements are built without a server, since a channel is an identifier in ``LISTEN`` and has
to be quoted rather than bound. The live half asserts that an announcement made on one connection
reaches another, and that reading them two ways at once is refused rather than left to chance.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

import agensgraph
from agensgraph.notify import listen_statement, unlisten_statement


class TestTheStatements:
    def test_a_plain_channel(self) -> None:
        assert listen_statement("changes") == "listen changes"

    def test_a_channel_needing_quotes(self) -> None:
        assert listen_statement("My Chan") == 'listen "My Chan"'

    def test_a_channel_holding_a_quote(self) -> None:
        assert listen_statement('a"b') == 'listen "a""b"'

    def test_a_channel_that_would_otherwise_carry_a_statement(self) -> None:
        assert listen_statement('x"; drop graph g; --') == 'listen "x""; drop graph g; --"'

    def test_unlistening_from_one(self) -> None:
        assert unlisten_statement("changes") == "unlisten changes"

    def test_unlistening_from_all_of_them(self) -> None:
        assert unlisten_statement() == "unlisten *"

    @pytest.mark.parametrize("statement", [listen_statement, unlisten_statement])
    def test_a_channel_with_no_name(self, statement) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="a channel has a name"):
            statement("")


@pytest.mark.server
class TestAgainstAServer:
    def test_a_channel_is_subscribed_to_and_reported(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.listen("changes", "My Chan")
        assert set(agens.listening()) == {"changes", "My Chan"}

    def test_unlistening_from_one_leaves_the_others(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.listen("a", "b")
        agens.unlisten("a")
        assert agens.listening() == ["b"]

    def test_unlistening_from_all(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.listen("a", "b")
        agens.unlisten()
        assert agens.listening() == []

    def test_an_announcement_reaches_a_handler(self, agens, dsn: str) -> None:  # type: ignore[no-untyped-def]
        seen: list[agensgraph.Notify] = []
        agens.add_notify_handler(seen.append)
        agens.listen("changes")

        with agensgraph.Connection.connect(dsn, autocommit=True) as other:
            other.notify("changes", "a vertex was written")

        # A handler is called while the connection is processing results, so something has to be
        # asked of it before the announcement is noticed.
        for _ in range(200):
            agens.execute("select 1")
            if seen:
                break
        assert [(item.channel, item.payload) for item in seen] == [
            ("changes", "a vertex was written")
        ]

    def test_a_channel_needing_quotes_carries_announcements_too(self, agens, dsn: str) -> None:  # type: ignore[no-untyped-def]
        seen: list[agensgraph.Notify] = []
        agens.add_notify_handler(seen.append)
        agens.listen("My Chan")
        with agensgraph.Connection.connect(dsn, autocommit=True) as other:
            other.notify("My Chan", "x")
        for _ in range(200):
            agens.execute("select 1")
            if seen:
                break
        assert [item.channel for item in seen] == ["My Chan"]

    def test_an_announcement_reaches_the_iterator(self, agens, dsn: str) -> None:  # type: ignore[no-untyped-def]
        agens.listen("changes")
        with agensgraph.Connection.connect(dsn, autocommit=True) as other:
            other.notify("changes", "one")
        received = list(agens.notifications(timeout=5, stop_after=1))
        assert [(item.channel, item.payload) for item in received] == [("changes", "one")]

    def test_reading_both_ways_at_once_is_refused(self, agens) -> None:  # type: ignore[no-untyped-def]
        """psycopg warns and carries on, which loses announcements to whichever route looks first."""
        agens.add_notify_handler(lambda notice: None)
        agens.listen("changes")
        with pytest.raises(RuntimeError, match="already has a notify handler"):
            next(iter(agens.notifications(timeout=1)))

    def test_removing_the_handler_makes_the_iterator_usable_again(
        self, agens, dsn: str
    ) -> None:  # type: ignore[no-untyped-def]
        def handler(notice: agensgraph.Notify) -> None:
            return None

        agens.add_notify_handler(handler)
        agens.remove_notify_handler(handler)
        agens.listen("changes")
        with agensgraph.Connection.connect(dsn, autocommit=True) as other:
            other.notify("changes", "two")
        assert [item.payload for item in agens.notifications(timeout=5, stop_after=1)] == [
            "two"
        ]

    def test_a_graph_write_can_announce_itself(self, agens, dsn: str) -> None:  # type: ignore[no-untyped-def]
        """What this is for: a trigger on a label table announcing that the graph changed."""
        graph = agens.label_table.graph
        agens.execute("create vlabel doc")
        agens.execute(
            f'create function "{graph}".announce() returns trigger language plpgsql as '
            f"$$ begin perform pg_notify('graph_changed', 'doc'); return new; end $$"
        )
        agens.execute(
            f'create trigger announce_doc after insert on "{graph}".doc '
            f'for each row execute function "{graph}".announce()'
        )
        seen: list[agensgraph.Notify] = []
        agens.add_notify_handler(seen.append)
        agens.listen("graph_changed")
        agens.execute("create (:doc {a: 1})")
        for _ in range(200):
            agens.execute("select 1")
            if seen:
                break
        assert [(item.channel, item.payload) for item in seen] == [("graph_changed", "doc")]


@pytest.mark.server
class TestTheAwaitingInterface:
    @pytest_asyncio.fixture
    async def conn(self, dsn: str):  # type: ignore[no-untyped-def]
        connection = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
        async with connection:
            yield connection

    @pytest.mark.asyncio
    async def test_listening_and_reporting(self, conn) -> None:  # type: ignore[no-untyped-def]
        await conn.listen("changes")
        assert await conn.listening() == ["changes"]
        await conn.unlisten()
        assert await conn.listening() == []

    @pytest.mark.asyncio
    async def test_an_announcement_reaches_the_iterator(self, conn, dsn: str) -> None:  # type: ignore[no-untyped-def]
        await conn.listen("changes")
        other = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
        async with other:
            await other.notify("changes", "one")
        received = [item async for item in conn.notifications(timeout=5, stop_after=1)]
        assert [(item.channel, item.payload) for item in received] == [("changes", "one")]

    @pytest.mark.asyncio
    async def test_reading_both_ways_at_once_is_refused(self, conn) -> None:  # type: ignore[no-untyped-def]
        async def first() -> None:
            async for _ in conn.notifications(timeout=1):
                break

        conn.add_notify_handler(lambda notice: None)
        await conn.listen("changes")
        with pytest.raises(RuntimeError, match="already has a notify handler"):
            await first()
