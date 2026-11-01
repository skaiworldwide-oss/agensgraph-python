"""Where a label's vectors are kept: in the vertices' properties, or in a table of their own.

The properties are the default and need 2.17. A table works on every release
and is chosen by a value, ``storage="table"``. These pin that each design does what the other
does -- ensure, write, read, delete, search -- that the choice is made in one place, and that a
table whose key a ``DROP ... CASCADE`` took is refused rather than used.
"""

from __future__ import annotations

import dataclasses
import random

import psycopg
import pytest
import pytest_asyncio
from psycopg.rows import dict_row, tuple_row

import agensgraph
from agensgraph import (
    HNSW,
    Capabilities,
    Distance,
    InProperties,
    InTable,
    IVFFlat,
    SparseVector,
    Storage,
    StorageState,
    VectorHit,
    VectorStorage,
)
from agensgraph._vector_plans import (
    CopyIn,
    Execute,
    Found,
    PropertyPlan,
    TablePlan,
    plan_for,
    refusal,
)
from agensgraph.errors import CapabilityError, StaleVectorStorage, VectorStorageMismatch
from agensgraph.vector import TABLE_HINT, nearest, vector_index

D = 4
WRITER = "agens_vector_writer"
WRITER_PASSWORD = "agens-writer-only"
"""So the test connects as the role whether or not the server trusts local connections."""


def storage(**overrides: object) -> VectorStorage:
    return VectorStorage("g", "doc", **{"dimensions": D, **overrides})  # type: ignore[arg-type]


class TestMakingAStorage:
    def test_the_default_is_the_properties(self) -> None:
        made = storage()
        assert made.storage == InProperties()
        assert made.kind() is Storage.PROPERTIES

    def test_a_table_is_chosen_by_name_and_named_for_its_label(self) -> None:
        made = storage(storage="table")
        assert made.storage == InTable(schema="g_vectors", name="doc_embedding")
        assert made.kind() is Storage.TABLE

    def test_a_table_named_by_the_caller_keeps_its_names(self) -> None:
        made = storage(storage=InTable(schema="g_store", name="chunk_vec"))
        assert made.storage == InTable(schema="g_store", name="chunk_vec")
        assert storage(storage=InTable(schema="s")).storage == InTable(
            schema="s", name="doc_embedding"
        )

    @pytest.mark.parametrize(
        ("given", "plan"),
        [
            ("properties", PropertyPlan),
            (Storage.PROPERTIES, PropertyPlan),
            (InProperties(), PropertyPlan),
            (InProperties(promoted=True), PropertyPlan),
            ("table", TablePlan),
            (Storage.TABLE, TablePlan),
            (InTable(), TablePlan),
            (InTable(schema="s", name="t"), TablePlan),
        ],
    )
    def test_every_form_of_the_choice_reaches_its_one_plan(
        self, given: object, plan: type
    ) -> None:
        assert type(plan_for(storage(storage=given))) is plan

    def test_the_forms_that_mean_the_same_storage_are_equal_and_share_a_plan(self) -> None:
        assert storage(storage="table") == storage(storage=InTable())
        assert plan_for(storage(storage="table")) is plan_for(storage(storage=InTable()))

    def test_an_unknown_design_is_refused_with_the_known_ones(self) -> None:
        with pytest.raises(ValueError, match="properties"):
            storage(storage="tabel")
        with pytest.raises(TypeError, match="InTable"):
            storage(storage=3)

    def test_it_cannot_be_changed_after_it_is_made(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            storage().dimensions = 8  # type: ignore[misc]

    @pytest.mark.parametrize("field", ["graph", "label", "property"])
    def test_a_name_the_server_would_cut_is_refused(self, field: str) -> None:
        names = {"graph": "g", "label": "doc", "property": "embedding", field: "x" * 64}
        with pytest.raises(ValueError, match="63 bytes"):
            VectorStorage(
                names["graph"], names["label"], dimensions=D, property=names["property"]
            )

    def test_and_so_is_a_table_name_derived_too_long(self) -> None:
        with pytest.raises(ValueError, match="table name"):
            VectorStorage("g", "l" * 40, dimensions=D, property="p" * 30, storage="table")

    @pytest.mark.parametrize("dimensions", [0, -1, True, 1.5, 16_001])
    def test_a_dimension_the_type_cannot_hold_is_refused(self, dimensions: object) -> None:
        with pytest.raises(ValueError, match="dimension"):
            storage(dimensions=dimensions, index=None)

    def test_an_unknown_type_is_refused(self) -> None:
        with pytest.raises(ValueError, match="type"):
            storage(type="bit")

    def test_a_distance_for_bits_is_refused(self) -> None:
        with pytest.raises(ValueError, match="bit strings"):
            storage(distance=Distance.HAMMING)

    def test_a_distance_may_be_given_as_its_operator(self) -> None:
        assert storage(distance="<->").distance is Distance.L2

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"type": "sparsevec", "index": IVFFlat()}, "no ivfflat index on sparsevec"),
            ({"distance": Distance.L1, "index": IVFFlat()}, "cannot order by L1"),
            ({"dimensions": 2_001}, "at most 2000"),
            ({"type": "halfvec", "dimensions": 4_001}, "at most 4000"),
        ],
    )
    def test_an_index_pgvector_cannot_build_is_refused(
        self, overrides: dict[str, object], message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            storage(**overrides)

    def test_without_an_index_any_stored_dimension_is_allowed(self) -> None:
        assert storage(dimensions=2_001, index=None).index is None
        assert storage(type="halfvec", dimensions=4_000).dimensions == 4_000


class TestThePlans:
    def test_what_each_design_needs_from_the_server(self) -> None:
        assert plan_for(storage()).requires == "cypher_typmod_cast"
        assert (
            plan_for(storage(storage=InProperties(promoted=True))).requires
            == "property_promotion"
        )
        assert plan_for(storage(storage="table")).requires is None

    def test_the_properties_are_searched_by_the_expression_the_index_holds(self) -> None:
        """Ordered by the expression, not its alias: 2.17 sorts an alias with a sequential scan."""
        text = plan_for(storage()).nearest([1.0] * D, 3).text
        assert "order by n.embedding::vector(4) <=> %s::vector(4) limit %s" in text
        assert "order by distance" not in text

    def test_a_promoted_property_is_searched_without_a_cast(self) -> None:
        text = plan_for(storage(storage=InProperties(promoted=True))).nearest([1.0] * D, 3).text
        assert "order by n.embedding <=> %s::vector(4)" in text

    def test_a_table_is_searched_in_sql_joined_to_its_label(self) -> None:
        text = plan_for(storage(storage="table")).nearest([1.0] * D, 3).text
        assert text == (
            "select v.id, l.properties, v.embedding <=> %s::vector(4) as distance "
            "from g_vectors.doc_embedding v join g.doc l on l.id = v.id "
            "order by distance limit %s"
        )

    @pytest.mark.parametrize("design", ["properties", "table"])
    @pytest.mark.parametrize("k", [0, -1, 2.0, "3", True, None])
    def test_a_search_asks_for_a_whole_number_of_hits(self, design: str, k: object) -> None:
        """Refused while the statement is built, before anything is sent."""
        with pytest.raises(ValueError, match="k is a whole number of at least one"):
            plan_for(storage(storage=design)).nearest([1.0] * D, k)  # type: ignore[arg-type]

    def test_a_table_is_keyed_to_its_label_and_cascades(self) -> None:
        plan = plan_for(storage(storage="table"))
        (schema, table) = plan.create(Found(StorageState.ABSENT, label_exists=True))
        assert schema.text == "create schema if not exists g_vectors"
        assert "references g.doc (id) on delete cascade" in table.text
        assert "with (toast_tuple_target = 8160)" in table.text

    def test_a_row_too_wide_for_its_page_is_not_asked_to_stay_there(self) -> None:
        plan = plan_for(storage(dimensions=4_000, type="halfvec", storage="table"))
        (_, table) = plan.create(Found(StorageState.ABSENT, label_exists=True))
        assert "toast_tuple_target" not in table.text

    def test_a_missing_label_is_made_first_and_needs_the_graph(self) -> None:
        plan = plan_for(storage(storage="table"))
        statements = plan.create(Found(StorageState.ABSENT, label_exists=False))
        assert statements[0] == Execute("create vlabel if not exists doc", on_graph=True)
        assert not any(s.on_graph for s in statements[1:])

    @pytest.mark.parametrize(
        ("overrides", "operator_class"),
        [
            ({}, "vector_cosine_ops"),
            ({"distance": Distance.L2}, "vector_l2_ops"),
            ({"type": "halfvec", "distance": Distance.INNER_PRODUCT}, "halfvec_ip_ops"),
            ({"type": "sparsevec", "distance": Distance.L1}, "sparsevec_l1_ops"),
        ],
    )
    def test_the_index_serves_the_type_and_distance(
        self, overrides: dict[str, object], operator_class: str
    ) -> None:
        for design in ("properties", "table"):
            built = plan_for(storage(storage=design, **overrides)).index(concurrently=False)
            assert built is not None
            assert operator_class in built.text

    def test_index_settings_and_concurrency(self) -> None:
        built = plan_for(storage(storage="table", index=HNSW(m=8, ef_construction=32))).index(
            concurrently=True
        )
        assert built is not None
        assert built.text.startswith(
            "create index concurrently if not exists doc_embedding_hnsw"
        )
        assert built.text.endswith("with (m=8, ef_construction=32)")
        with pytest.raises(ValueError, match="no concurrent build"):
            plan_for(storage()).index(concurrently=True)

    def test_no_index_means_nothing_to_build(self) -> None:
        assert plan_for(storage(index=None)).index(concurrently=False) is None
        assert plan_for(storage(storage="table", index=None)).index(concurrently=False) is None

    def test_a_derived_index_name_is_kept_within_bounds_and_distinct(self) -> None:
        one = plan_for(VectorStorage("g", "a" * 60, dimensions=D)).index(concurrently=False)
        two = plan_for(VectorStorage("g", "a" * 59 + "b", dimensions=D)).index(
            concurrently=False
        )
        assert one is not None
        assert two is not None
        assert one.text != two.text

    def test_a_table_write_is_an_upsert_or_a_copy(self) -> None:
        plan = plan_for(storage(storage="table"))
        rows = [(agensgraph.GraphId(3, 1), [1.0, 2.0, 3.0, 4.0])]
        assert isinstance(plan.write(rows, upsert=True), Execute)
        copied = plan.write(rows, upsert=False)
        assert isinstance(copied, CopyIn)
        assert copied.types == ("graphid", "vector")

    def test_a_vector_of_another_dimension_is_refused_before_anything_is_sent(self) -> None:
        for design in ("properties", "table"):
            with pytest.raises(ValueError, match="2 dimensions"):
                plan_for(storage(storage=design)).write(
                    [(agensgraph.GraphId(3, 1), [1.0, 2.0])], upsert=True
                )

    def test_a_sparse_storage_takes_only_a_sparse_vector(self) -> None:
        plan = plan_for(storage(type="sparsevec", storage="table"))
        with pytest.raises(TypeError, match="from_dense"):
            plan.write([(agensgraph.GraphId(3, 1), [1.0, 0.0, 0.0, 0.0])], upsert=True)

    def test_every_properties_statement_runs_on_the_graph(self) -> None:
        plan = plan_for(storage())
        ids = [agensgraph.GraphId(3, 1)]
        made = [plan.read(ids), plan.delete(ids), plan.nearest([1.0] * D, 1), *plan.drop()]
        assert all(statement.on_graph for statement in made)

    def test_no_table_statement_but_making_the_label_needs_the_graph(self) -> None:
        plan = plan_for(storage(storage="table"))
        ids = [agensgraph.GraphId(3, 1)]
        made = [plan.read(ids), plan.delete(ids), plan.nearest([1.0] * D, 1), *plan.drop()]
        assert not any(statement.on_graph for statement in made)

    def test_each_state_that_cannot_be_used_has_its_refusal(self) -> None:
        assert isinstance(refusal(Found(StorageState.STALE, True, "x")), StaleVectorStorage)
        assert isinstance(
            refusal(Found(StorageState.MISMATCHED, True, "x")), VectorStorageMismatch
        )
        assert refusal(Found(StorageState.READY, True)) is None
        assert refusal(Found(StorageState.ABSENT, False)) is None


class TestTheCypherBuilders:
    """``vector_index()`` and ``nearest()``: a cast in Cypher, and a refusal that names the table."""

    def test_they_index_and_search_through_the_cast(self) -> None:
        assert vector_index("movie", "embedding", dimensions=4) == (
            "create property index on movie using hnsw "
            "((embedding::vector(4)) vector_cosine_ops)"
        )
        assert nearest("movie", "embedding", dimensions=4) == (
            "match (n:movie) return n order by n.embedding::vector(4) <=> %s::vector(4) limit 10"
        )

    def test_a_refusal_points_to_a_table(self) -> None:
        with pytest.raises(CapabilityError) as caught:
            vector_index("movie", "embedding", dimensions=4, capabilities=Capabilities("2.16"))
        assert TABLE_HINT in caught.value.__notes__


# -- against a server ---------------------------------------------------------------------------

rng = random.Random(5)


def unit(seed: int) -> list[float]:
    """A vector whose nearest neighbours are easy to state: mostly along one axis."""
    values = [0.01 * random.Random(seed).random() for _ in range(D)]
    values[seed % D] = 1.0
    return values


@pytest.fixture
def graph(agens):  # type: ignore[no-untyped-def]
    if not agens.has_vectors():
        pytest.skip("the vector extension is not created in this database")
    agens.execute("create vlabel doc")
    agens.ids = agens.load_vertices_with_ids("doc", [{"n": n} for n in range(8)])
    return agens


def table(agens, **overrides: object) -> VectorStorage:  # type: ignore[no-untyped-def]
    name = agens.label_table.graph
    return VectorStorage(name, "doc", **{"dimensions": D, "storage": "table", **overrides})  # type: ignore[arg-type]


def properties(agens, **overrides: object) -> VectorStorage:  # type: ignore[no-untyped-def]
    name = agens.label_table.graph
    return VectorStorage(name, "doc", **{"dimensions": D, **overrides})  # type: ignore[arg-type]


@pytest.fixture
def stored(graph):  # type: ignore[no-untyped-def]
    """The table design, made and filled: vertex n holds unit(n)."""
    made = table(graph)
    try:
        assert graph.ensure_vectors(made) is StorageState.READY
        graph.write_vectors(made, [(gid, unit(n)) for n, gid in enumerate(graph.ids)])
        yield graph, made
    finally:
        graph.drop_vectors(made)
        graph.execute(f"drop schema if exists {made.storage.schema} cascade")


@pytest.mark.server
class TestTheTableDesign:
    """On every release this driver supports."""

    def test_ensuring_twice_finds_it_ready(self, stored) -> None:  # type: ignore[no-untyped-def]
        conn, made = stored
        assert conn.ensure_vectors(made) is StorageState.READY
        assert conn.vector_storage_state(made) is StorageState.READY

    def test_what_is_written_reads_back(self, stored) -> None:  # type: ignore[no-untyped-def]
        conn, made = stored
        read = conn.vectors_of(made, conn.ids[:3])
        assert {gid: v.tolist() for gid, v in read.items()} == {
            gid: pytest.approx(unit(n)) for n, gid in enumerate(conn.ids[:3])
        }

    def test_an_unchanged_rewrite_changes_nothing_and_a_change_is_counted(self, stored) -> None:  # type: ignore[no-untyped-def]
        conn, made = stored
        rows = [(gid, unit(n)) for n, gid in enumerate(conn.ids)]
        assert conn.write_vectors(made, rows) == 0
        assert conn.write_vectors(made, [(conn.ids[0], unit(1))]) == 1

    def test_the_bulk_route_copies_and_refuses_a_row_already_there(self, graph) -> None:  # type: ignore[no-untyped-def]
        made = table(graph)
        graph.ensure_vectors(made, index=False)
        try:
            assert (
                graph.write_vectors(
                    made, [(gid, unit(n)) for n, gid in enumerate(graph.ids)], upsert=False
                )
                == 8
            )
            graph.build_vector_index(made)
            with pytest.raises(psycopg.errors.UniqueViolation):
                graph.write_vectors(made, [(graph.ids[0], unit(0))], upsert=False)
        finally:
            graph.execute(f"drop schema {made.storage.schema} cascade")

    def test_the_nearest_come_first_as_vertices(self, stored) -> None:  # type: ignore[no-untyped-def]
        conn, made = stored
        hits = conn.nearest(made, unit(2), k=3)
        assert all(isinstance(hit, VectorHit) for hit in hits)
        assert hits[0].vertex.id == conn.ids[2]
        assert hits[0].vertex.label == "doc"
        assert hits[0].vertex.properties == {"n": 2}
        assert [hit.distance for hit in hits] == sorted(hit.distance for hit in hits)
        assert len(conn.nearest(made, unit(2), k=5, settings_={"hnsw.ef_search": 50})) == 5

    def test_the_search_uses_the_index(self, stored) -> None:  # type: ignore[no-untyped-def]
        conn, made = stored
        statement = plan_for(made).nearest(unit(1), 3)
        with conn.transaction():
            conn.execute("set local enable_seqscan = off")
            plan = "\n".join(
                row[0] for row in conn.execute("explain " + statement.text, statement.params)
            )
        assert "Index Scan using doc_embedding_hnsw" in plan

    def test_deleting_a_vector_keeps_the_vertex(self, stored) -> None:  # type: ignore[no-untyped-def]
        conn, made = stored
        assert conn.delete_vectors(made, conn.ids[:2]) == 2
        assert conn.vectors_of(made, conn.ids[:2]) == {}
        assert conn.execute_query("match (n:doc) return count(n)").records == [(8,)]

    def test_deleting_the_vertex_takes_its_vector(self, stored) -> None:  # type: ignore[no-untyped-def]
        conn, made = stored
        conn.execute("match (n:doc) where n.n = 0 detach delete n")
        assert conn.ids[0] not in conn.vectors_of(made, conn.ids)
        assert len(conn.vectors_of(made, conn.ids)) == 7

    def test_a_table_outliving_its_label_is_refused(self, stored) -> None:  # type: ignore[no-untyped-def]
        """``DROP VLABEL ... CASCADE`` drops the key and keeps the rows; a new label reuses ids."""
        conn, made = stored
        conn.execute("drop vlabel doc cascade")
        conn.execute("create vlabel doc")
        assert conn.vector_storage_state(made) is StorageState.STALE
        with pytest.raises(StaleVectorStorage, match="Truncate the table"):
            conn.ensure_vectors(made)
        conn.drop_vectors(made)
        assert conn.ensure_vectors(made) is StorageState.READY

    def test_another_dimension_where_one_is_kept_is_refused(self, stored) -> None:  # type: ignore[no-untyped-def]
        conn, _ = stored
        with pytest.raises(VectorStorageMismatch, match="vector\\(8\\)"):
            conn.ensure_vectors(table(conn, dimensions=8))

    def test_an_index_built_for_another_distance_is_refused(self, stored) -> None:  # type: ignore[no-untyped-def]
        """It keeps the name and serves none of this storage's searches."""
        conn, _ = stored
        other = table(conn, distance="<->")
        assert conn.vector_storage_state(other) is StorageState.MISMATCHED
        with pytest.raises(VectorStorageMismatch, match="vector_l2_ops"):
            conn.ensure_vectors(other)

    def test_an_index_left_for_later_is_reported_missing_until_built(self, graph) -> None:  # type: ignore[no-untyped-def]
        made = table(graph)
        try:
            assert graph.ensure_vectors(made, index=False) is StorageState.ABSENT
            assert graph.vector_storage_state(made) is StorageState.ABSENT
            graph.build_vector_index(made)
            assert graph.vector_storage_state(made) is StorageState.READY
            graph.build_vector_index(made)  # there already, so nothing is asked of the server
        finally:
            graph.execute(f"drop schema if exists {made.storage.schema} cascade")

    def test_a_role_that_only_reads_and_writes_can_ensure_what_is_there(
        self, stored, dsn
    ) -> None:  # type: ignore[no-untyped-def]
        """Making a schema, a table or an index is checked against the role before ``IF NOT
        EXISTS`` is, so a storage that is all there must be found rather than made again."""
        conn, made = stored
        schema, name = made.storage.schema, made.storage.name
        graph = conn.label_table.graph
        conn.execute(f'drop role if exists "{WRITER}"')
        try:
            conn.execute(f"create role \"{WRITER}\" login password '{WRITER_PASSWORD}'")
        except psycopg.Error:
            pytest.skip("this role cannot create another")
        try:
            for statement in (
                f'grant usage on schema "{graph}", "{schema}" to "{WRITER}"',
                f'grant select on all tables in schema "{graph}" to "{WRITER}"',
                f'grant select, insert, update, delete on "{schema}"."{name}" to "{WRITER}"',
            ):
                conn.execute(statement)
            theirs = psycopg.conninfo.make_conninfo(dsn, user=WRITER, password=WRITER_PASSWORD)
            with agensgraph.connect(theirs, autocommit=True) as writer:
                writer.graph(graph)
                assert writer.ensure_vectors(made) is StorageState.READY
                assert writer.write_vectors(made, [(conn.ids[0], unit(3))]) == 1
        finally:
            conn.execute(f'drop owned by "{WRITER}" cascade')
            conn.execute(f'drop role "{WRITER}"')

    def test_hits_are_read_whatever_the_row_factory(self, stored) -> None:  # type: ignore[no-untyped-def]
        conn, made = stored
        conn.row_factory = dict_row
        try:
            assert conn.nearest(made, unit(2), k=1)[0].vertex.id == conn.ids[2]
        finally:
            conn.row_factory = tuple_row

    def test_a_missing_label_is_made_on_its_graph_and_refused_from_another(
        self, graph, dsn
    ) -> None:  # type: ignore[no-untyped-def]
        made = VectorStorage(graph.label_table.graph, "fresh", dimensions=D, storage="table")
        try:
            with (
                agensgraph.connect(dsn, autocommit=True) as elsewhere,
                pytest.raises(agensgraph.errors.ConfigurationError, match="graph"),
            ):
                elsewhere.ensure_vectors(made)
            assert graph.ensure_vectors(made) is StorageState.READY
        finally:
            graph.execute(f"drop schema if exists {made.storage.schema} cascade")

    def test_but_reads_and_writes_need_no_graph(self, stored, dsn) -> None:  # type: ignore[no-untyped-def]
        conn, made = stored
        with agensgraph.connect(dsn, autocommit=True) as elsewhere:
            assert elsewhere.write_vectors(made, [(conn.ids[0], unit(5))]) == 1
            assert elsewhere.nearest(made, unit(5), k=1)[0].vertex.id in (
                conn.ids[0],
                conn.ids[5],
            )

    @pytest.mark.parametrize("kind", ["halfvec", "sparsevec"])
    def test_the_other_types(self, graph, kind: str) -> None:  # type: ignore[no-untyped-def]
        made = table(graph, type=kind)
        value = (lambda n: SparseVector({n % D: 1.0}, D)) if kind == "sparsevec" else unit
        try:
            graph.ensure_vectors(made)
            graph.write_vectors(made, [(gid, value(n)) for n, gid in enumerate(graph.ids)])
            assert graph.nearest(made, value(3), k=1)[0].vertex.id in graph.ids
            assert len(graph.vectors_of(made, graph.ids)) == 8
        finally:
            graph.execute(f"drop schema if exists {made.storage.schema} cascade")

    def test_numpy_rows(self, graph) -> None:  # type: ignore[no-untyped-def]
        np = pytest.importorskip("numpy")
        made = table(graph)
        try:
            graph.ensure_vectors(made)
            values = np.eye(D, dtype="float32")[np.arange(8) % D]
            assert graph.write_vectors(made, zip(graph.ids, values, strict=True)) == 8
        finally:
            graph.execute(f"drop schema if exists {made.storage.schema} cascade")


def needs_properties(conn) -> None:  # type: ignore[no-untyped-def]
    if not conn.capabilities.has("cypher_typmod_cast"):
        pytest.skip("vectors in properties need 2.17")


@pytest.mark.server
class TestThePropertiesDesign:
    """The default: 2.17 and later, and refused below with the way out named."""

    def test_below_2_17_every_use_is_refused_and_points_to_a_table(self, graph) -> None:  # type: ignore[no-untyped-def]
        if graph.capabilities.has("cypher_typmod_cast"):
            pytest.skip("this server keeps vectors in properties")
        made = properties(graph)
        calls = [
            lambda: graph.ensure_vectors(made),
            lambda: graph.write_vectors(made, [(graph.ids[0], unit(0))]),
            lambda: graph.vectors_of(made, graph.ids),
            lambda: graph.delete_vectors(made, graph.ids),
            lambda: graph.nearest(made, unit(0)),
            lambda: graph.build_vector_index(made),
            lambda: graph.drop_vectors(made),
        ]
        for call in calls:
            with pytest.raises(CapabilityError) as caught:
                call()
            assert TABLE_HINT in caught.value.__notes__

    def test_the_whole_lifecycle(self, graph) -> None:  # type: ignore[no-untyped-def]
        needs_properties(graph)
        made = properties(graph)
        assert graph.ensure_vectors(made) is StorageState.READY
        assert (
            graph.write_vectors(made, [(gid, unit(n)) for n, gid in enumerate(graph.ids)]) == 8
        )
        (stored,) = graph.execute_query("match (n:doc {n: 1}) return n.embedding").records[0]
        assert stored == pytest.approx(unit(1))
        assert graph.vectors_of(made, graph.ids[:1])[graph.ids[0]].tolist() == pytest.approx(
            unit(0)
        )
        hits = graph.nearest(made, unit(2), k=3)
        assert hits[0].vertex.id == graph.ids[2]
        assert [hit.distance for hit in hits] == sorted(hit.distance for hit in hits)
        assert graph.delete_vectors(made, graph.ids[:1]) == 1
        assert graph.ids[0] not in graph.vectors_of(made, graph.ids)
        graph.drop_vectors(made)
        assert graph.vector_storage_state(made) is StorageState.ABSENT

    def test_the_search_uses_the_property_index(self, graph) -> None:  # type: ignore[no-untyped-def]
        needs_properties(graph)
        made = properties(graph)
        graph.ensure_vectors(made)
        graph.write_vectors(made, [(gid, unit(n)) for n, gid in enumerate(graph.ids)])
        statement = plan_for(made).nearest(unit(1), 3)
        with graph.transaction():
            graph.execute("set local enable_seqscan = off")
            plan = "\n".join(
                row[0] for row in graph.execute("explain " + statement.text, statement.params)
            )
        assert "Index Scan using doc_embedding_hnsw" in plan

    def test_a_promoted_property_on_a_new_label(self, graph) -> None:  # type: ignore[no-untyped-def]
        if not graph.capabilities.has("property_promotion"):
            pytest.skip("a property of its own needs 2.18")
        made = VectorStorage(
            graph.label_table.graph, "emb", dimensions=D, storage=InProperties(promoted=True)
        )
        assert graph.ensure_vectors(made) is StorageState.READY
        ids = graph.load_vertices_with_ids("emb", [{"k": n} for n in range(4)])
        graph.write_vectors(made, [(gid, unit(n)) for n, gid in enumerate(ids)])
        assert graph.nearest(made, unit(3), k=1)[0].vertex.id == ids[3]

    def test_a_promoted_property_on_a_label_without_its_column_is_refused(self, graph) -> None:  # type: ignore[no-untyped-def]
        if not graph.capabilities.has("property_promotion"):
            pytest.skip("a property of its own needs 2.18")
        made = properties(graph, storage=InProperties(promoted=True))
        with pytest.raises(VectorStorageMismatch, match="no column"):
            graph.ensure_vectors(made)

    def test_it_runs_only_on_its_own_graph(self, graph, dsn) -> None:  # type: ignore[no-untyped-def]
        needs_properties(graph)
        made = properties(graph)
        with (
            agensgraph.connect(dsn, autocommit=True) as elsewhere,
            pytest.raises(agensgraph.errors.ConfigurationError, match="graph"),
        ):
            elsewhere.nearest(made, unit(0))

    def test_a_session_put_on_its_graph_by_a_setting_is_on_it(self, graph, dsn) -> None:  # type: ignore[no-untyped-def]
        """Selected by ``SET`` or by ``options`` rather than by graph(), it is on the graph all
        the same, though the driver has not read its labels."""
        needs_properties(graph)
        made = properties(graph)
        graph.ensure_vectors(made)
        graph.write_vectors(made, [(gid, unit(n)) for n, gid in enumerate(graph.ids)])
        name = graph.label_table.graph
        with agensgraph.connect(dsn, autocommit=True) as by_hand:
            by_hand.execute(f'set graph_path = "{name}"')
            assert by_hand.nearest(made, unit(4), k=1)[0].vertex.id == graph.ids[4]
        with agensgraph.connect(
            dsn, autocommit=True, options=f"-c graph_path={name}"
        ) as by_option:
            assert by_option.nearest(made, unit(5), k=1)[0].vertex.id == graph.ids[5]

    def test_hits_are_read_whatever_the_row_factory(self, graph) -> None:  # type: ignore[no-untyped-def]
        needs_properties(graph)
        made = properties(graph)
        graph.ensure_vectors(made)
        graph.write_vectors(made, [(gid, unit(n)) for n, gid in enumerate(graph.ids)])
        graph.row_factory = dict_row
        try:
            assert graph.nearest(made, unit(6), k=1)[0].vertex.id == graph.ids[6]
        finally:
            graph.row_factory = tuple_row


@pytest.mark.server
class TestBothDesignsAgree:
    def test_the_same_vectors_give_the_same_neighbours(self, graph) -> None:  # type: ignore[no-untyped-def]
        """Exact search in both, so the comparison is of the designs and not of two indexes."""
        needs_properties(graph)
        rows = [(gid, [rng.uniform(-1, 1) for _ in range(D)]) for gid in graph.ids]
        both = [properties(graph, index=None), table(graph, index=None)]
        try:
            for made in both:
                graph.ensure_vectors(made)
                graph.write_vectors(made, rows)
            query = [rng.uniform(-1, 1) for _ in range(D)]
            found = [graph.nearest(made, query, k=8) for made in both]
            assert [hit.vertex.id for hit in found[0]] == [hit.vertex.id for hit in found[1]]
            assert [hit.distance for hit in found[0]] == pytest.approx(
                [hit.distance for hit in found[1]], rel=1e-5
            )
        finally:
            graph.execute(f"drop schema if exists {both[1].storage.schema} cascade")


class TestTheAwaitingInterface:
    @pytest_asyncio.fixture
    async def conn(self, dsn: str):  # type: ignore[no-untyped-def]
        name = "vector_storage_async"
        connection = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
        async with connection:
            if not await connection.has_vectors():
                pytest.skip("the vector extension is not created in this database")
            await connection.execute(f'drop graph if exists "{name}" cascade')
            await connection.execute(f"drop schema if exists {name}_vectors cascade")
            await connection.execute(f'create graph "{name}"')
            await connection.graph(name)
            try:
                yield connection
            finally:
                await connection.execute(f"drop schema if exists {name}_vectors cascade")
                await connection.execute("reset graph_path")
                await connection.execute(f'drop graph "{name}" cascade')

    @pytest.mark.server
    @pytest.mark.asyncio
    async def test_a_table_round_trip(self, conn) -> None:  # type: ignore[no-untyped-def]
        made = VectorStorage("vector_storage_async", "doc", dimensions=D, storage="table")
        assert await conn.ensure_vectors(made) is StorageState.READY
        ids = await conn.load_vertices_with_ids("doc", [{"n": n} for n in range(4)])
        assert (
            await conn.write_vectors(made, [(gid, unit(n)) for n, gid in enumerate(ids)]) == 4
        )
        hits = await conn.nearest(made, unit(1), k=2)
        assert hits[0].vertex.id == ids[1]
        assert set(await conn.vectors_of(made, ids)) == set(ids)
        assert await conn.delete_vectors(made, ids[:1]) == 1
