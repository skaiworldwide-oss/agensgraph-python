"""Reading and writing the graph types through psycopg's own interfaces.

Nothing here needs a server: a loader is handed the bytes a server would have sent, which
is what makes it possible to test the renderings the server produces rarely as thoroughly
as the ones it produces constantly.
"""

from __future__ import annotations

import pytest
from psycopg import pq
from psycopg.abc import PyFormat
from psycopg.adapt import AdaptersMap, Transformer

from agensgraph import Edge, GraphId, Path, Vertex
from agensgraph._protocol.labels import LabelCache
from agensgraph.adapters import OIDS, graph_adapters, register_binary, register_text, type_info
from agensgraph.errors import ProgrammingError, StaleLabelCache

from . import wire

TEXT = pq.Format.TEXT
BINARY = pq.Format.BINARY


@pytest.fixture
def labels() -> LabelCache:
    cache = LabelCache()
    cache.load("g", [(1, "ag_vertex"), (2, "ag_edge"), (3, "person"), (4, "knows")])
    return cache


def load(adapters: AdaptersMap, oid: int, data: bytes, fmt: pq.Format) -> object:
    """Push one value through the loader psycopg would have chosen for it."""
    tx = Transformer(_Context(adapters))
    return tx.get_loader(oid, fmt).load(data)


def dump(adapters: AdaptersMap, obj: object, fmt: PyFormat) -> tuple[bytes | None, int]:
    """Push one value through the dumper psycopg would have chosen for it.

    A dumper is chosen by the format a parameter is written in, which is a different
    enumeration from the one a result column arrives in.
    """
    tx = Transformer(_Context(adapters))
    dumper = tx.get_dumper(obj, fmt)
    return dumper.dump(obj), dumper.oid


class _Context:
    """What psycopg's transformer asks of a context, and no more."""

    def __init__(self, adapters: AdaptersMap) -> None:
        self.adapters = adapters
        self.connection = None


@pytest.fixture
def text_adapters() -> AdaptersMap:
    return graph_adapters()


@pytest.fixture
def binary_adapters(labels: LabelCache) -> AdaptersMap:
    adapters = graph_adapters()
    register_binary(_Context(adapters), labels)
    return adapters


class TestOids:
    def test_every_graph_type_is_listed(self) -> None:
        assert set(OIDS) == {
            "graphid",
            "_graphid",
            "vertex",
            "_vertex",
            "edge",
            "_edge",
            "graphpath",
            "_graphpath",
            "rowid",
        }

    def test_an_array_oid_sits_one_below_its_element(self) -> None:
        for name in ("graphid", "vertex", "edge", "graphpath"):
            assert OIDS[f"_{name}"] == OIDS[name] - 1

    def test_a_type_info_carries_both_oids(self) -> None:
        info = type_info("vertex")
        assert (info.oid, info.array_oid) == (OIDS["vertex"], OIDS["_vertex"])

    def test_a_type_that_is_not_a_graph_type_is_refused(self) -> None:
        with pytest.raises(KeyError):
            type_info("jsonb")


class TestTextLoaders:
    def test_a_graph_id(self, text_adapters: AdaptersMap) -> None:
        assert load(text_adapters, OIDS["graphid"], b"3.1", TEXT) == GraphId(3, 1)

    def test_a_vertex(self, text_adapters: AdaptersMap) -> None:
        v = load(text_adapters, OIDS["vertex"], b'person[3.1]{"name": "a"}', TEXT)
        assert isinstance(v, Vertex)
        assert (v.id, v.label, v.properties) == (GraphId(3, 1), "person", {"name": "a"})

    def test_an_edge(self, text_adapters: AdaptersMap) -> None:
        e = load(text_adapters, OIDS["edge"], b"knows[4.1][3.1,3.2]{}", TEXT)
        assert isinstance(e, Edge)
        assert (e.id, e.start, e.end) == (GraphId(4, 1), GraphId(3, 1), GraphId(3, 2))

    def test_a_path(self, text_adapters: AdaptersMap) -> None:
        p = load(
            text_adapters,
            OIDS["graphpath"],
            b"[person[3.1]{},knows[4.1][3.1,3.2]{},person[3.2]{}]",
            TEXT,
        )
        assert isinstance(p, Path)
        assert p.length == 1
        assert len(p) == 3

    def test_an_empty_path(self, text_adapters: AdaptersMap) -> None:
        """A legal value the earlier driver could not read at all."""
        p = load(text_adapters, OIDS["graphpath"], b"[]", TEXT)
        assert isinstance(p, Path)
        assert len(p) == 0
        assert p.length == 0

    def test_a_vertex_array(self, text_adapters: AdaptersMap) -> None:
        got = load(text_adapters, OIDS["_vertex"], b"[person[3.1]{},person[3.2]{}]", TEXT)
        assert [v.id for v in got] == [GraphId(3, 1), GraphId(3, 2)]

    def test_a_vertex_array_with_a_hole_in_it(self, text_adapters: AdaptersMap) -> None:
        got = load(text_adapters, OIDS["_vertex"], b"[person[3.1]{},NULL]", TEXT)
        assert got[1] is None

    def test_an_edge_array(self, text_adapters: AdaptersMap) -> None:
        got = load(text_adapters, OIDS["_edge"], b"[knows[4.1][3.1,3.2]{}]", TEXT)
        assert [e.id for e in got] == [GraphId(4, 1)]

    def test_a_label_holding_the_formats_own_characters(
        self, text_adapters: AdaptersMap
    ) -> None:
        """The server writes a label name with no escaping, so this is producible."""
        v = load(text_adapters, OIDS["vertex"], b'a,b}c{[3.1]{"k": "},{"}', TEXT)
        assert v.label == "a,b}c{"
        assert v.properties == {"k": "},{"}

    def test_a_memoryview_is_read_as_readily_as_bytes(self, text_adapters: AdaptersMap) -> None:
        """psycopg hands over one or the other depending on how it read the result."""
        payload = memoryview(b'person[3.1]{"name": "a"}')
        v = load(text_adapters, OIDS["vertex"], payload, TEXT)
        assert v.properties == {"name": "a"}


class TestBinaryLoaders:
    def test_a_graph_id(self, text_adapters: AdaptersMap) -> None:
        """Registered alongside the text loader, because it needs no label name."""
        assert load(text_adapters, OIDS["graphid"], wire.graphid(3, 1), BINARY) == GraphId(3, 1)

    def test_a_graph_id_above_the_signed_range(self, text_adapters: AdaptersMap) -> None:
        """A label id at or above 32768 reads negative if the value is taken as signed."""
        got = load(text_adapters, OIDS["graphid"], wire.graphid(65535, 1), BINARY)
        assert got == GraphId(65535, 1)

    def test_a_vertex(self, binary_adapters: AdaptersMap) -> None:
        v = load(binary_adapters, OIDS["vertex"], wire.vertex(3, 1, b'{"name": "a"}'), BINARY)
        assert (v.id, v.label, v.properties) == (GraphId(3, 1), "person", {"name": "a"})

    def test_an_edge(self, binary_adapters: AdaptersMap) -> None:
        e = load(binary_adapters, OIDS["edge"], wire.edge(4, 1, (3, 1), (3, 2), b"{}"), BINARY)
        assert (e.id, e.label, e.start, e.end) == (
            GraphId(4, 1),
            "knows",
            GraphId(3, 1),
            GraphId(3, 2),
        )

    def test_a_path(self, binary_adapters: AdaptersMap) -> None:
        payload = wire.path(
            [wire.vertex(3, 1, b"{}"), wire.vertex(3, 2, b"{}")],
            [wire.edge(4, 1, (3, 1), (3, 2), b"{}")],
        )
        p = load(binary_adapters, OIDS["graphpath"], payload, BINARY)
        assert p.length == 1
        assert [v.label for v in p.vertices] == ["person", "person"]

    def test_an_empty_path(self, binary_adapters: AdaptersMap) -> None:
        p = load(binary_adapters, OIDS["graphpath"], wire.path([], []), BINARY)
        assert len(p) == 0

    def test_a_vertex_array(self, binary_adapters: AdaptersMap) -> None:
        payload = wire.array(OIDS["vertex"], [wire.vertex(3, 1, b"{}"), None])
        got = load(binary_adapters, OIDS["_vertex"], payload, BINARY)
        assert got[0].id == GraphId(3, 1)
        assert got[1] is None

    def test_an_edge_array(self, binary_adapters: AdaptersMap) -> None:
        payload = wire.array(OIDS["edge"], [wire.edge(4, 1, (3, 1), (3, 2), b"{}")])
        got = load(binary_adapters, OIDS["_edge"], payload, BINARY)
        assert got[0].label == "knows"

    def test_a_label_the_cache_has_not_heard_of_says_so(
        self, binary_adapters: AdaptersMap
    ) -> None:
        """A label created since the cache was filled, which reconnecting would not fix."""
        with pytest.raises(StaleLabelCache) as caught:
            load(binary_adapters, OIDS["vertex"], wire.vertex(99, 1, b"{}"), BINARY)
        assert caught.value.labid == 99
        assert caught.value.graph == "g"

    def test_the_tuple_id_column_is_read_and_dropped(
        self, binary_adapters: AdaptersMap
    ) -> None:
        """It is present in this rendering and absent from the text one."""
        v = load(binary_adapters, OIDS["vertex"], wire.vertex(3, 1, b'{"a": 1}'), BINARY)
        assert v.properties == {"a": 1}


class TestBothRenderingsAgree:
    """The same value by two independent routes, which is what makes each checkable."""

    def test_a_vertex(self, text_adapters: AdaptersMap, binary_adapters: AdaptersMap) -> None:
        from_text = load(
            text_adapters, OIDS["vertex"], b'person[3.1]{"name": "a", "n": 1}', TEXT
        )
        from_binary = load(
            binary_adapters, OIDS["vertex"], wire.vertex(3, 1, b'{"name": "a", "n": 1}'), BINARY
        )
        assert from_text == from_binary
        assert from_text.label == from_binary.label
        assert from_text.properties == from_binary.properties

    def test_an_edge(self, text_adapters: AdaptersMap, binary_adapters: AdaptersMap) -> None:
        from_text = load(text_adapters, OIDS["edge"], b'knows[4.1][3.1,3.2]{"w": 2}', TEXT)
        from_binary = load(
            binary_adapters, OIDS["edge"], wire.edge(4, 1, (3, 1), (3, 2), b'{"w": 2}'), BINARY
        )
        assert from_text == from_binary
        assert (from_text.start, from_text.end) == (from_binary.start, from_binary.end)
        assert from_text.properties == from_binary.properties

    def test_a_path(self, text_adapters: AdaptersMap, binary_adapters: AdaptersMap) -> None:
        from_text = load(
            text_adapters,
            OIDS["graphpath"],
            b"[person[3.1]{},knows[4.1][3.1,3.2]{},person[3.2]{}]",
            TEXT,
        )
        from_binary = load(
            binary_adapters,
            OIDS["graphpath"],
            wire.path(
                [wire.vertex(3, 1, b"{}"), wire.vertex(3, 2, b"{}")],
                [wire.edge(4, 1, (3, 1), (3, 2), b"{}")],
            ),
            BINARY,
        )
        assert from_text == from_binary


class TestDumpers:
    def test_a_graph_id_as_text(self, text_adapters: AdaptersMap) -> None:
        assert dump(text_adapters, GraphId(3, 1), PyFormat.TEXT) == (b"3.1", OIDS["graphid"])

    def test_a_graph_id_as_bytes(self, text_adapters: AdaptersMap) -> None:
        assert dump(text_adapters, GraphId(3, 1), PyFormat.BINARY) == (
            wire.graphid(3, 1),
            OIDS["graphid"],
        )

    def test_a_graph_id_survives_the_round_trip(self, text_adapters: AdaptersMap) -> None:
        for gid in (GraphId(1, 1), GraphId(3, 7), GraphId(65535, 281474976710655)):
            for out, back in ((PyFormat.TEXT, TEXT), (PyFormat.BINARY, BINARY)):
                payload, oid = dump(text_adapters, gid, out)
                assert load(text_adapters, oid, payload, back) == gid

    def test_a_vertex_is_not_sent(self, text_adapters: AdaptersMap) -> None:
        """Nothing binds a vertex: a property map binds as jsonb, and identity as a graph id.

        Refusing it is better than quietly sending the property map, which would drop the
        label and the identity without saying so.
        """
        with pytest.raises(ProgrammingError, match=r"(?i)vertex"):
            dump(text_adapters, Vertex(GraphId(3, 1), "person", {}), PyFormat.TEXT)


class TestRegistration:
    def test_psycopgs_own_map_is_left_alone(self) -> None:
        """A plain PostgreSQL connection in the same process must be unaffected."""
        from psycopg import postgres

        before = postgres.adapters.get_loader(OIDS["vertex"], TEXT)
        graph_adapters()
        assert postgres.adapters.get_loader(OIDS["vertex"], TEXT) is before

    def test_registering_on_a_map_and_on_a_context_do_the_same_thing(self) -> None:
        from psycopg import postgres

        on_map = AdaptersMap(postgres.adapters)
        register_text(on_map)
        on_context = AdaptersMap(postgres.adapters)
        register_text(_Context(on_context))
        assert on_map.get_loader(OIDS["vertex"], TEXT) is on_context.get_loader(
            OIDS["vertex"], TEXT
        )

    def test_registering_twice_is_harmless(self) -> None:
        adapters = graph_adapters()
        register_text(adapters)
        assert load(adapters, OIDS["graphid"], b"3.1", TEXT) == GraphId(3, 1)
