"""Writing elements that may already be there.

A copy only creates, so reading a source in twice with ``load_vertices`` makes a second element for
everything the graph already holds. Every knowledge-graph package re-ingests, so what they all
reach for is a merge, and every one of them writes it as a statement per row.
"""

from __future__ import annotations

import contextlib
import threading

import psycopg
import pytest
import pytest_asyncio

import agensgraph

pytestmark = pytest.mark.server


@pytest.fixture
def docs(agens):  # type: ignore[no-untyped-def]
    """A label whose key is unique, which is what an upsert asks for."""
    agens.execute("create vlabel doc")
    agens.execute("create unique property index on doc (k)")
    agens.refresh_labels()
    return agens


def count(conn, statement: str = "match (n:doc) return count(*)") -> int:  # type: ignore[no-untyped-def]
    return int(conn.execute_query(statement).records[0][0])


class TestWhatItWrites:
    def test_the_first_pass_writes_everything(self, docs) -> None:  # type: ignore[no-untyped-def]
        assert docs.upsert_vertices("doc", "k", [{"k": i} for i in range(50)]) == (50, 0)
        assert count(docs) == 50

    def test_the_second_pass_writes_nothing(self, docs) -> None:  # type: ignore[no-untyped-def]
        rows = [{"k": i, "v": 0} for i in range(50)]
        docs.upsert_vertices("doc", "k", rows)
        assert docs.upsert_vertices("doc", "k", [{"k": i, "v": 1} for i in range(50)]) == (0, 0)
        assert count(docs, "match (n:doc) where n.v = 0 return count(*)") == 50

    def test_only_the_rows_that_are_new_are_written(self, docs) -> None:  # type: ignore[no-untyped-def]
        docs.upsert_vertices("doc", "k", [{"k": i} for i in range(50)])
        assert docs.upsert_vertices("doc", "k", [{"k": i} for i in range(25, 75)]) == (25, 0)
        assert count(docs) == 75

    def test_asking_for_an_update_writes_the_overlap_too(self, docs) -> None:  # type: ignore[no-untyped-def]
        docs.upsert_vertices("doc", "k", [{"k": i, "v": 0} for i in range(50)])
        written = docs.upsert_vertices(
            "doc", "k", [{"k": i, "v": 1} for i in range(25, 75)], on_existing="update"
        )
        assert written == (25, 25)
        assert count(docs, "match (n:doc) where n.v = 1 return count(*)") == 50
        assert count(docs, "match (n:doc) where n.v = 0 return count(*)") == 25

    def test_an_update_merges_rather_than_replaces(self, docs) -> None:  # type: ignore[no-untyped-def]
        """A property the caller did not mention keeps the value it had."""
        docs.upsert_vertices("doc", "k", [{"k": 1, "kept": "yes", "v": 0}])
        docs.upsert_vertices("doc", "k", [{"k": 1, "v": 9}], on_existing="update")
        assert docs.execute_query("match (n:doc) return n.kept, n.v").records[0] == ("yes", 9)

    def test_an_empty_list_writes_nothing(self, docs) -> None:  # type: ignore[no-untyped-def]
        assert docs.upsert_vertices("doc", "k", []) == (0, 0)


class TestWhatItRefuses:
    def test_a_key_with_no_uniqueness_behind_it(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel doc")
        agens.execute("create property index on doc (k)")
        agens.refresh_labels()
        with pytest.raises(ValueError, match="unique"):
            agens.upsert_vertices("doc", "k", [{"k": 1}])

    def test_unless_the_caller_accepts_the_hazard(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel doc")
        agens.refresh_labels()
        assert agens.upsert_vertices("doc", "k", [{"k": 1}], require_unique=False) == (1, 0)

    def test_a_uniqueness_constraint_counts_as_well_as_an_index(self, agens) -> None:  # type: ignore[no-untyped-def]
        """It is kept as an exclusion constraint, so reading indexes alone would not find it."""
        agens.execute("create vlabel doc")
        agens.execute("create constraint doc_k_unique on doc assert k is unique")
        agens.refresh_labels()
        assert agens.upsert_vertices("doc", "k", [{"k": 1}]) == (1, 0)

    def test_a_row_with_no_key(self, docs) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="identify it by"):
            docs.upsert_vertices("doc", "k", [{"other": 1}])

    def test_a_row_whose_key_is_null(self, docs) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="identifies nothing"):
            docs.upsert_vertices("doc", "k", [{"k": None}])

    def test_a_third_thing_to_do_with_an_existing_element(self, docs) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="skip"):
            docs.upsert_vertices("doc", "k", [{"k": 1}], on_existing="replace")


class TestTheHazardTheUniqueKeyIsFor:
    """Why the key must be unique, shown rather than asserted.

    Two writers merging on the same key both miss and both create. This is the measurement the
    refusal above is based on, so it is worth having on the record.
    """

    def test_concurrent_merges_duplicate_without_it(self, agens, dsn) -> None:  # type: ignore[no-untyped-def]
        keys, writers = 25, 8
        agens.execute("create vlabel t")
        agens.execute("create property index on t (k)")
        graph = agens.execute("select current_setting('graph_path')").fetchone()[0]
        ready = threading.Barrier(writers)

        def merge_them() -> None:
            with agensgraph.connect(dsn, autocommit=True) as conn:
                conn.graph(graph)
                ready.wait()
                for k in range(keys):
                    # A conflict is the expected path here, not a surprise: this is what two
                    # writers merging on one key do.
                    with contextlib.suppress(psycopg.Error):
                        conn.execute("merge (n:t {k: %s}) set n.seen = 1", (k,))

        threads = [threading.Thread(target=merge_them) for _ in range(writers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        made = count(agens, "match (n:t) return count(*)")
        distinct = count(agens, "match (n:t) return count(distinct n.k)")
        assert distinct == keys
        assert made >= distinct, "which is the point: more elements than keys"


class TestTheKeysAreAskedForByName:
    """The identities come from a lookup, not from reading the label.

    A property map is one column, so extracting one key from it reassembles the whole map. Reading
    the label to find out which keys exist therefore costs the maps of every element whatever the
    batch is, so a caller feeding a stream in batches would pay a full read for each of them.
    """

    def test_the_label_is_not_read_to_find_the_keys(self, docs) -> None:  # type: ignore[no-untyped-def]
        """Asserted by what the driver sends, since the cost is invisible from the result."""
        docs.upsert_vertices("doc", "k", [{"k": i} for i in range(20)])
        sent: list[str] = []

        def record(statement) -> None:  # type: ignore[no-untyped-def]
            sent.append(statement.statement)

        agensgraph.add_query_logger(record)
        try:
            docs.upsert_vertices("doc", "k", [{"k": 1}])
        finally:
            agensgraph.remove_query_logger(record)
        read_whole_label = [text for text in sent if "properties ->>" in text]
        assert not read_whole_label, f"the whole label was read: {read_whole_label}"
        assert any("unwind" in text and "id(n)" in text for text in sent)

    def test_a_key_stored_as_a_number_is_found_when_given_as_a_string(self, docs) -> None:  # type: ignore[no-untyped-def]
        """The two routes have to agree, and the full read compares text on both sides.

        This is the shape that breaks a naive lookup: load once from a source whose keys are
        numbers, again from one whose keys are strings, and every key misses. Since a miss means
        insert, every element would be written a second time -- and the unique index does not catch
        it, because the index holds the property as jsonb, where the number ``7`` and the string
        ``"7"`` are different values and so occupy different entries.
        """
        docs.upsert_vertices("doc", "k", [{"k": i} for i in range(30)])
        again = docs.upsert_vertices("doc", "k", [{"k": str(i)} for i in range(30)])
        assert again == (0, 0)
        assert count(docs) == 30

    def test_and_the_other_way_round(self, docs) -> None:  # type: ignore[no-untyped-def]
        docs.upsert_vertices("doc", "k", [{"k": str(i)} for i in range(30)])
        again = docs.upsert_vertices("doc", "k", [{"k": i} for i in range(30)])
        assert again == (0, 0)
        assert count(docs) == 30

    def test_a_label_holding_both_spellings_is_still_ambiguous(self, docs) -> None:  # type: ignore[no-untyped-def]
        """Two elements answer to one text key, so neither is the one to attach anything to."""
        docs.execute("create (:doc {k: 7})")
        docs.execute("create (:doc {k: '7'})")
        with pytest.raises(ValueError, match="does not identify"):
            docs.upsert_vertices("doc", "k", [{"k": 7}])

    def test_a_key_that_is_not_there_is_written(self, docs) -> None:  # type: ignore[no-untyped-def]
        docs.upsert_vertices("doc", "k", [{"k": "aa"}])
        assert docs.upsert_vertices("doc", "k", [{"k": "aa"}, {"k": "bb"}]) == (1, 0)
        assert count(docs) == 2

    def test_a_label_with_nothing_to_look_a_key_up_by_reads_the_label(self, agens) -> None:  # type: ignore[no-untyped-def]
        """A lookup with no index would read the label once per key, which is worse."""
        agens.execute("create vlabel plain")
        agens.refresh_labels()
        agens.upsert_vertices("plain", "k", [{"k": 1}], require_unique=False)
        sent: list[str] = []

        def record(statement) -> None:  # type: ignore[no-untyped-def]
            sent.append(statement.statement)

        agensgraph.add_query_logger(record)
        try:
            agens.upsert_vertices("plain", "k", [{"k": 1}], require_unique=False)
        finally:
            agensgraph.remove_query_logger(record)
        assert any("properties ->>" in text for text in sent)


class TestItIsWorthDoing:
    def test_it_beats_a_merge_per_row(self, docs, agens) -> None:  # type: ignore[no-untyped-def]
        """Not a benchmark -- a floor, so a regression that makes it slower than the alternative
        fails rather than merely disappoints."""
        import time

        rows = [{"k": i, "v": 1} for i in range(2000)]
        docs.upsert_vertices("doc", "k", rows[:1000])
        started = time.monotonic()
        docs.upsert_vertices("doc", "k", rows, on_existing="update")
        upserting = time.monotonic() - started

        docs.execute("match (n:doc) delete n")
        docs.upsert_vertices("doc", "k", rows[:1000])
        started = time.monotonic()
        for row in rows:
            docs.execute("merge (n:doc {k: %s}) set n.v = %s", (row["k"], row["v"]))
        one_at_a_time = time.monotonic() - started
        assert upserting < one_at_a_time

    def test_the_cost_does_not_follow_the_label(self, docs) -> None:  # type: ignore[no-untyped-def]
        """Not a benchmark -- a floor. Reading the label to find the keys costs the same whatever
        the batch is, so ten small batches would cost ten reads of it. Asking by name does not."""
        import time

        docs.upsert_vertices("doc", "k", [{"k": i, "pad": "x" * 400} for i in range(4000)])
        one = [{"k": 1}]
        started = time.monotonic()
        for _ in range(10):
            docs.upsert_vertices("doc", "k", one)
        ten_small = time.monotonic() - started

        started = time.monotonic()
        docs.identity_map("doc", "k")
        one_full_read = time.monotonic() - started
        assert ten_small < one_full_read * 10, (
            "ten batches should not cost ten reads of the whole label"
        )


class TestTheAwaitingInterface:
    @pytest_asyncio.fixture
    async def conn(self, dsn: str):  # type: ignore[no-untyped-def]
        name = "upsert_async"
        connection = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
        async with connection:
            await connection.execute(f'drop graph if exists "{name}" cascade')
            await connection.execute(f'create graph "{name}"')
            await connection.graph(name)
            await connection.execute("create vlabel doc")
            await connection.execute("create unique property index on doc (k)")
            await connection.refresh_labels()
            try:
                yield connection
            finally:
                await connection.execute("reset graph_path")
                await connection.execute(f'drop graph "{name}" cascade')

    @pytest.mark.asyncio
    async def test_it_writes_and_then_does_not(self, conn) -> None:  # type: ignore[no-untyped-def]
        rows = [{"k": i, "v": 0} for i in range(20)]
        assert await conn.upsert_vertices("doc", "k", rows) == (20, 0)
        assert await conn.upsert_vertices("doc", "k", rows) == (0, 0)
        written = await conn.upsert_vertices(
            "doc", "k", [{"k": i, "v": 1} for i in range(20)], on_existing="update"
        )
        assert written == (0, 20)
        result = await conn.execute_query("match (n:doc) where n.v = 1 return count(*)")
        assert result.records[0][0] == 20


@pytest.fixture
def joined(agens):  # type: ignore[no-untyped-def]
    """A graph with elements to join and an edge label, and the identities to join them by."""
    agens.execute("create vlabel p")
    agens.execute("create elabel links")
    agens.refresh_labels()
    agens.load_vertices("p", [{"n": i} for i in range(60)])
    ids = [
        row[0] for row in agens.execute("match (n:p) return id(n) order by id(n)").fetchall()
    ]
    return agens, ids


def edges_of(conn) -> int:  # type: ignore[no-untyped-def]
    (found,) = conn.execute("match ()-[r:links]->() return count(*)").fetchone()
    return int(found)


def keyed(conn) -> None:  # type: ignore[no-untyped-def]
    """The index that keeps one edge per pair, which a property index cannot express."""
    graph = conn.label_table.graph
    conn.execute(f'create unique index links_pair on "{graph}".links (start, "end")')


class TestUpsertingEdges:
    """An edge is identified by the pair it joins, so that is what it is written by."""

    def test_the_second_run_writes_nothing(self, joined) -> None:  # type: ignore[no-untyped-def]
        """Which is the whole point: a copy alone makes a second edge for every one already there."""
        conn, ids = joined
        keyed(conn)
        pairs = [(ids[i], ids[i + 1], {"w": i}) for i in range(25)]
        assert conn.upsert_edges("links", pairs) == (25, 0)
        assert conn.upsert_edges("links", pairs) == (0, 0)
        assert edges_of(conn) == 25

    def test_a_copy_alone_would_have_doubled_them(self, joined) -> None:  # type: ignore[no-untyped-def]
        conn, ids = joined
        pairs = [(ids[i], ids[i + 1], None) for i in range(25)]
        conn.load_edges("links", pairs)
        conn.load_edges("links", pairs)
        assert edges_of(conn) == 50

    def test_updating_keeps_what_it_was_not_told_about(self, joined) -> None:  # type: ignore[no-untyped-def]
        conn, ids = joined
        keyed(conn)
        conn.upsert_edges("links", [(ids[0], ids[1], {"w": 1})])
        conn.execute("match ()-[e:links]->() set e.kept = 'yes'")
        counts = conn.upsert_edges("links", [(ids[0], ids[1], {"w": 99})], on_existing="update")
        assert counts == (0, 1)
        (weight, kept) = conn.execute("match ()-[e:links]->() return e.w, e.kept").fetchone()
        assert (weight, kept) == (99, "yes")

    def test_a_pair_given_twice_in_one_call_is_written_once(self, joined) -> None:  # type: ignore[no-untyped-def]
        """The copy reads nothing back, so a repeat left in would be the duplicate this prevents."""
        conn, ids = joined
        keyed(conn)
        counts = conn.upsert_edges(
            "links",
            [(ids[0], ids[1], None), (ids[0], ids[1], None), (ids[2], ids[3], None)],
        )
        assert counts.inserted == 2
        assert edges_of(conn) == 2

    @pytest.mark.parametrize("on_existing", ["replace", "merge", ""])
    def test_there_is_no_third_thing_to_do(self, joined, on_existing: str) -> None:  # type: ignore[no-untyped-def]
        conn, ids = joined
        with pytest.raises(ValueError, match="no third thing"):
            conn.upsert_edges("links", [(ids[0], ids[1], None)], on_existing=on_existing)


class TestTheHazardTheEndpointIndexIsFor:
    def test_a_label_with_nothing_keeping_one_edge_per_pair_is_refused(self, joined) -> None:  # type: ignore[no-untyped-def]
        conn, ids = joined
        with pytest.raises(ValueError, match="edge per pair of endpoints"):
            conn.upsert_edges("links", [(ids[0], ids[1], None)])

    def test_the_refusal_names_the_statement_that_fixes_it(self, joined) -> None:  # type: ignore[no-untyped-def]
        conn, ids = joined
        with pytest.raises(ValueError) as caught:
            conn.upsert_edges("links", [(ids[0], ids[1], None)])
        assert "create unique index" in str(caught.value)
        assert '(start, "end")' in str(caught.value)

    def test_the_hazard_can_be_accepted(self, joined) -> None:  # type: ignore[no-untyped-def]
        conn, ids = joined
        counts = conn.upsert_edges("links", [(ids[0], ids[1], None)], require_unique=False)
        assert counts.inserted == 1

    def test_eight_writers_on_one_set_of_pairs_leave_one_edge_each(self, joined, dsn) -> None:  # type: ignore[no-untyped-def]
        """Measured without the index: twenty-seven edges for twenty-five pairs, and no failure."""
        conn, ids = joined
        keyed(conn)
        graph = conn.label_table.graph
        pairs = [(ids[i], ids[i + 1], {"w": i}) for i in range(25)]
        failures: list[str] = []

        def writer() -> None:
            other = agensgraph.Connection.connect(dsn, autocommit=True)
            with other:
                other.graph(graph)
                with contextlib.suppress(psycopg.Error):
                    other.upsert_edges("links", pairs)

        threads = [threading.Thread(target=writer) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert edges_of(conn) == 25, f"one edge per pair, failures seen: {failures}"


class TestWhichReadItChooses:
    """Asking about the pairs costs two identities each; reading the label costs none."""

    def test_the_two_shapes_agree(self, joined) -> None:  # type: ignore[no-untyped-def]
        conn, ids = joined
        keyed(conn)
        pairs = [(ids[i], ids[i + 1], None) for i in range(25)]
        conn.upsert_edges("links", pairs)
        graph = conn.label_table.graph
        asked = conn._edges_present("links", pairs, graph=graph, estimate=-1.0)
        whole = conn._edges_present("links", pairs, graph=graph, estimate=25.0)
        assert asked == whole
        assert len(asked) == 25

    def test_a_label_nobody_has_analysed_reports_no_size(self, joined) -> None:  # type: ignore[no-untyped-def]
        conn, _ids = joined
        keyed(conn)
        _, estimate = conn._edge_label_facts("links", graph=conn.label_table.graph)
        assert estimate == -1.0, "which is read as not knowing, and asks about the pairs"

    def test_and_reports_one_once_it_has(self, joined) -> None:  # type: ignore[no-untyped-def]
        conn, ids = joined
        keyed(conn)
        conn.upsert_edges("links", [(ids[i], ids[i + 1], None) for i in range(25)])
        graph = conn.label_table.graph
        conn.execute(f'analyze "{graph}".links')
        keyed_now, estimate = conn._edge_label_facts("links", graph=graph)
        assert keyed_now is True
        assert estimate == 25.0

    def test_a_label_that_is_not_there_says_so(self, joined) -> None:  # type: ignore[no-untyped-def]
        conn, _ = joined
        with pytest.raises(ValueError, match="not a label"):
            conn._edge_label_facts("nosuchlabel", graph=conn.label_table.graph)
