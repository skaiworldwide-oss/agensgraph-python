"""Taking the indexes and constraints that exist to the ones somebody asked for.

The diffing is a pure function over what the catalogs reported and what was wanted, so most of
this needs no server: it is a list in, a list of statements out. The live half asserts the two
things a pure test cannot -- that the statements are ones the server accepts, and that a second
run has nothing left to do.

The server's rules that the tests turn on:

* A definition is printed with every default left out. ``ASC`` never appears, ``NULLS LAST`` only
  alongside ``DESC``, an operator class only when it is not the default.
* A predicate is stored normalised: ``a > 0`` prints as ``(a) > cypher_to_jsonb(0)``.
* A uniqueness assertion is over an expression. ``ASSERT lower(name) IS UNIQUE`` is accepted,
  ``ASSERT (a, b) IS UNIQUE`` is not.
* An unnamed constraint is named ``<label>_unique_constraint`` or ``<label>_properties_check``, then
  the same with a counter.
"""

from __future__ import annotations

import psycopg
import pytest
import pytest_asyncio

import agensgraph
from agensgraph.cypher import quote_identifier
from agensgraph.introspect import (
    Check,
    Constraint,
    DesiredIndex,
    Index,
    IndexElement,
    Unique,
    constraint_name,
    create_constraint_statement,
    create_index_statement,
    index_elements,
    index_is_partial,
    index_method,
    index_properties,
    parse_index_element,
    reconcile_constraints,
    reconcile_indexes,
)


def an_index(label: str, columns: str, *, name: str = "i", unique: bool = False) -> Index:
    """An index as the catalog reports one, definition included, since that is what is read."""
    keyword = "UNIQUE PROPERTY INDEX" if unique else "PROPERTY INDEX"
    return Index(
        label,
        name,
        unique,
        f"CREATE {keyword} {name} ON {label} USING btree ({columns})",
    )


class TestReadingTheColumnsOffADefinition:
    """The definition the server printed is the only description of an index both sides share."""

    def test_one_property(self) -> None:
        assert index_properties(an_index("doc", "name").definition) == ("name",)

    def test_several_in_order(self) -> None:
        assert index_properties(an_index("doc", "a, b, c").definition) == ("a", "b", "c")

    def test_a_unique_one_reads_the_same(self) -> None:
        assert index_properties(an_index("doc", "sku", unique=True).definition) == ("sku",)

    def test_a_quoted_name_is_unquoted(self) -> None:
        assert index_properties(an_index("doc", '"order"').definition) == ("order",)

    def test_an_expression_is_not_a_list_of_properties(self) -> None:
        """The shape a vector index takes. It has to be distinguishable, not merely parsed."""
        definition = an_index("doc", "((properties ->> 'v'))").definition
        assert index_properties(definition) is None

    def test_an_expression_holding_a_comma_is_not_split_into_properties(self) -> None:
        definition = an_index("doc", "(coalesce(a, b))").definition
        assert index_properties(definition) is None

    def test_a_mix_of_the_two_is_not_a_list_of_properties(self) -> None:
        definition = an_index("doc", "a, ((properties ->> 'v'))").definition
        assert index_properties(definition) is None

    def test_something_that_is_not_a_definition(self) -> None:
        assert index_properties("CREATE PROPERTY INDEX i ON doc") is None
        assert index_properties("") is None

    def test_an_unclosed_definition(self) -> None:
        assert index_properties("CREATE PROPERTY INDEX i ON doc USING btree (a, b") is None

    def test_a_nested_property_path_is_not_a_plain_property(self) -> None:
        assert index_properties(an_index("doc", "a.b.c").definition) is None

    def test_a_sort_order_is_read_rather_than_refused(self) -> None:
        """The names are still the names; how they are keyed is read separately."""
        assert index_properties(an_index("doc", "a DESC, b").definition) == ("a", "b")
        assert index_properties(an_index("doc", "a, b NULLS FIRST").definition) == ("a", "b")

    def test_a_partial_index_is_not_a_plain_index_over_its_properties(self) -> None:
        """It covers them conditionally, so reading it as a plain index would report a desired
        index as already there when the only one present applies to some of the rows."""
        definition = (
            "CREATE PROPERTY INDEX i ON doc USING btree (c) WHERE (c) > cypher_to_jsonb(0)"
        )
        assert index_properties(definition) is None

    def test_extra_included_columns_likewise(self) -> None:
        definition = "CREATE PROPERTY INDEX i ON doc USING btree (d) INCLUDE (e)"
        assert index_properties(definition) is None

    def test_a_property_named_where_is_still_a_property(self) -> None:
        """The partial-index check reads the tail after the key list, not the key list itself."""
        assert index_properties(an_index("doc", "where").definition) == ("where",)
        assert index_properties(an_index("doc", "include").definition) == ("include",)


class TestDiffingIndexes:
    def test_nothing_there_means_make_all_of_them(self) -> None:
        statements = reconcile_indexes([DesiredIndex("doc", ("name",))], [])
        assert statements == ["create property index on doc (name)"]

    def test_already_there_means_nothing_to_do(self) -> None:
        """The assertion that makes running this twice free rather than a second round of work."""
        assert (
            reconcile_indexes([DesiredIndex("doc", ("name",))], [an_index("doc", "name")]) == []
        )

    def test_matched_on_properties_not_on_the_name_it_happens_to_have(self) -> None:
        existing = an_index("doc", "name", name="whatever_somebody_called_it")
        assert reconcile_indexes([DesiredIndex("doc", ("name",))], [existing]) == []

    def test_property_order_is_part_of_what_an_index_is(self) -> None:
        """``(a, b)`` and ``(b, a)`` are different indexes, and a diff that said otherwise would
        quietly refuse to make the second one."""
        statements = reconcile_indexes(
            [DesiredIndex("doc", ("b", "a"))], [an_index("doc", "a, b")]
        )
        assert statements == ["create property index on doc (b, a)"]

    def test_the_same_properties_on_another_label_is_another_index(self) -> None:
        statements = reconcile_indexes(
            [DesiredIndex("note", ("name",))], [an_index("doc", "name")]
        )
        assert statements == ["create property index on note (name)"]

    def test_becoming_unique_is_a_drop_and_a_remake(self) -> None:
        """Uniqueness is not something an index can be altered into, so it is done the long way."""
        existing = an_index("doc", "sku", name="doc_sku_idx")
        statements = reconcile_indexes([DesiredIndex("doc", ("sku",), unique=True)], [existing])
        assert statements == [
            "drop property index doc_sku_idx",
            "create unique property index on doc (sku)",
        ]

    def test_ceasing_to_be_unique_is_too(self) -> None:
        existing = an_index("doc", "sku", name="u", unique=True)
        statements = reconcile_indexes([DesiredIndex("doc", ("sku",))], [existing])
        assert statements == ["drop property index u", "create property index on doc (sku)"]

    def test_an_extra_one_is_left_alone_unless_asked_about(self) -> None:
        assert reconcile_indexes([], [an_index("doc", "name")]) == []

    def test_and_dropped_when_asked(self) -> None:
        existing = an_index("doc", "name", name="doc_name_idx")
        assert reconcile_indexes([], [existing], drop_extra=True) == [
            "drop property index doc_name_idx"
        ]

    def test_a_partial_index_does_not_satisfy_a_desired_plain_one(self) -> None:
        partial = Index(
            "doc",
            "doc_c_partial",
            False,
            "CREATE PROPERTY INDEX doc_c_partial ON doc USING btree (c) "
            "WHERE (c) > cypher_to_jsonb(0)",
        )
        assert reconcile_indexes([DesiredIndex("doc", ("c",))], [partial]) == [
            "create property index on doc (c)"
        ]
        assert reconcile_indexes([], [partial], drop_extra=True) == []

    @pytest.mark.parametrize(
        "definition",
        [
            # A vector index over a property still in the map, as the catalog prints it.
            "CREATE PROPERTY INDEX doc_v ON doc USING hnsw "
            "(((v)::vector(4)) vector_cosine_ops)",
            # One made as plain SQL over the jsonb arrow operator, likewise an expression.
            "CREATE PROPERTY INDEX doc_v ON doc USING btree ((properties ->> 'v'))",
        ],
    )
    def test_an_index_over_an_expression_survives_drop_extra(self, definition: str) -> None:
        """Nothing a desired index can say describes these, so they are neither matched nor
        dropped -- otherwise reconciling a list of property names would take out a vector index."""
        existing = Index("doc", "doc_v", False, definition)
        assert index_elements(definition) is None
        assert reconcile_indexes([], [existing], drop_extra=True) == []

    def test_a_vector_index_over_a_column_is_describable_and_so_is_dropped(self) -> None:
        """It keys a plain property with an operator class, which a desired index can say, so
        ``drop_extra`` reads the list it was given as the whole of what should exist."""
        definition = "CREATE PROPERTY INDEX doc_v ON doc USING hnsw (v vector_l2_ops)"
        existing = Index("doc", "doc_v", False, definition)
        assert index_elements(definition) == (IndexElement("v", "vector_l2_ops", False, False),)
        assert reconcile_indexes([], [existing], drop_extra=True) == [
            "drop property index doc_v"
        ]
        wanted = [DesiredIndex("doc", (IndexElement("v", "vector_l2_ops"),), method="hnsw")]
        assert reconcile_indexes(wanted, [existing], drop_extra=True) == []

    def test_a_name_is_used_when_one_is_given(self) -> None:
        statements = reconcile_indexes([DesiredIndex("doc", ("name",), name="my_idx")], [])
        assert statements == ["create property index my_idx on doc (name)"]

    def test_a_label_or_property_needing_quotes_gets_them(self) -> None:
        statement = create_index_statement(DesiredIndex("a b", ("order",)))
        assert statement == 'create property index on "a b" ("order")'

    def test_an_index_over_no_properties_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one property"):
            create_index_statement(DesiredIndex("doc", ()))


class TestDiffingConstraints:
    def test_a_uniqueness_assertion_is_named_after_its_property(self) -> None:
        assert constraint_name(Unique("doc", "sku")) == "doc_sku_unique"

    def test_a_given_name_wins(self) -> None:
        assert constraint_name(Unique("doc", "sku", name="chosen")) == "chosen"

    def test_a_long_derived_name_is_truncated_as_the_server_would(self) -> None:
        name = constraint_name(Unique("d" * 40, "p" * 40))
        assert len(name) == 63

    def test_nothing_there_means_make_it(self) -> None:
        statements = reconcile_constraints([Unique("doc", "sku")], [])
        assert statements == ["create constraint doc_sku_unique on doc assert sku is unique"]

    def test_a_check_is_written_as_given(self) -> None:
        statements = reconcile_constraints([Check("doc", "age > 0", "doc_age_ok")], [])
        assert statements == ["create constraint doc_age_ok on doc assert age > 0"]

    def test_already_there_means_nothing_to_do(self) -> None:
        existing = Constraint("doc", "doc_sku_unique", True, "ASSERT (sku) IS UNIQUE")
        assert reconcile_constraints([Unique("doc", "sku")], [existing]) == []

    def test_a_check_already_there_is_not_rebuilt_for_being_printed_differently(self) -> None:
        """The server prints ``age > 0`` as ``ASSERT ((age) > cypher_to_jsonb(0))``. Comparing the
        expression as written against that would call every check different and remake it on
        every run, which is the failure this matches by name to avoid."""
        existing = Constraint("doc", "doc_age_ok", False, "ASSERT ((age) > cypher_to_jsonb(0))")
        assert reconcile_constraints([Check("doc", "age > 0", "doc_age_ok")], [existing]) == []

    def test_a_name_reused_for_the_other_kind_is_a_drop_and_a_remake(self) -> None:
        existing = Constraint("doc", "doc_thing", False, "ASSERT ((age) > cypher_to_jsonb(0))")
        statements = reconcile_constraints([Unique("doc", "x", name="doc_thing")], [existing])
        assert statements == [
            "drop constraint doc_thing on doc",
            "create constraint doc_thing on doc assert x is unique",
        ]

    def test_two_asked_for_under_one_name_is_refused_rather_than_one_winning(self) -> None:
        with pytest.raises(ValueError, match="same name"):
            reconcile_constraints(
                [Check("doc", "age > 0", "same"), Check("doc", "age < 200", "same")], []
            )

    def test_the_same_name_on_two_labels_is_two_constraints(self) -> None:
        statements = reconcile_constraints(
            [Check("doc", "a", "n"), Check("note", "a", "n")],
            [Constraint("doc", "n", False, "ASSERT (a)")],
        )
        assert statements == ["create constraint n on note assert a"]

    def test_an_extra_one_is_left_alone_unless_asked_about(self) -> None:
        existing = Constraint("doc", "doc_sku_unique", True, "ASSERT (sku) IS UNIQUE")
        assert reconcile_constraints([], [existing]) == []

    def test_and_dropped_when_asked(self) -> None:
        existing = Constraint("doc", "doc_sku_unique", True, "ASSERT (sku) IS UNIQUE")
        assert reconcile_constraints([], [existing], drop_extra=True) == [
            "drop constraint doc_sku_unique on doc"
        ]

    def test_a_label_needing_quotes_gets_them(self) -> None:
        statement = create_constraint_statement(Unique("a b", "order", name="n"))
        assert statement == 'create constraint n on "a b" assert "order" is unique'


@pytest.mark.server
class TestAgainstAServer:
    """That the statements are accepted, and that a second run has nothing left to do."""

    @pytest.fixture
    def graph(self, agens):  # type: ignore[no-untyped-def]
        agens.execute("create vlabel doc")
        return agens

    def test_indexes_are_made_and_then_already_there(self, graph) -> None:  # type: ignore[no-untyped-def]
        desired = [
            DesiredIndex("doc", ("name",)),
            DesiredIndex("doc", ("sku",), unique=True),
            DesiredIndex("doc", ("a", "b")),
        ]
        first = graph.ensure_indexes(desired)
        assert len(first) == 3
        assert graph.ensure_indexes(desired) == [], "the second run found work to do"
        made = {index.name: index for index in graph.indexes("doc")}
        assert len(made) == 3
        assert sum(index.unique for index in made.values()) == 1

    def test_constraints_are_made_and_then_already_there(self, graph) -> None:  # type: ignore[no-untyped-def]
        desired = [Unique("doc", "sku"), Check("doc", "age > 0", "doc_age_ok")]
        assert len(graph.ensure_constraints(desired)) == 2
        assert graph.ensure_constraints(desired) == [], "the second run found work to do"
        names = {constraint.name for constraint in graph.constraints("doc")}
        assert {"doc_sku_unique", "doc_age_ok"} <= names

    def test_a_uniqueness_assertion_is_enforced_once_made(self, graph) -> None:  # type: ignore[no-untyped-def]
        """Reconciling is only worth anything if what it made actually constrains."""
        graph.ensure_constraints([Unique("doc", "sku")])
        graph.execute("create (:doc {sku: 'one'})")
        with pytest.raises(agensgraph.errors.Error):
            graph.execute("create (:doc {sku: 'one'})")

    def test_changing_uniqueness_is_carried_out_not_merely_planned(self, graph) -> None:  # type: ignore[no-untyped-def]
        graph.ensure_indexes([DesiredIndex("doc", ("sku",))])
        assert [index.unique for index in graph.indexes("doc")] == [False]
        graph.ensure_indexes([DesiredIndex("doc", ("sku",), unique=True)])
        assert [index.unique for index in graph.indexes("doc")] == [True]

    def test_a_dry_run_reports_without_doing(self, graph) -> None:  # type: ignore[no-untyped-def]
        desired = [DesiredIndex("doc", ("name",))]
        assert graph.ensure_indexes(desired, dry_run=True) == [
            "create property index on doc (name)"
        ]
        assert graph.indexes("doc") == []

    def test_drop_extra_removes_what_was_not_asked_for(self, graph) -> None:  # type: ignore[no-untyped-def]
        graph.execute("create property index on doc (stale)")
        assert graph.ensure_indexes([DesiredIndex("doc", ("name",))], drop_extra=True)
        assert [index_properties(i.definition) for i in graph.indexes("doc")] == [("name",)]

    def test_drop_extra_leaves_a_label_the_caller_did_not_name(self, graph) -> None:  # type: ignore[no-untyped-def]
        """What is read here is the whole graph's, and a caller declaring one label is not
        saying every other label should have nothing."""
        graph.execute("create vlabel other")
        graph.execute("create unique property index other_key on other (k)")
        graph.execute("create property index on doc (stale)")

        planned = graph.ensure_indexes(
            [DesiredIndex("doc", ("name",))], drop_extra=True, dry_run=True
        )
        assert not any("other_key" in statement for statement in planned)
        assert any("stale" in statement for statement in planned)

        graph.ensure_indexes([DesiredIndex("doc", ("name",))], drop_extra=True)
        assert [i.name for i in graph.indexes("other")] == ["other_key"]
        assert [index_properties(i.definition) for i in graph.indexes("doc")] == [("name",)]

    def test_a_real_partial_index_does_not_satisfy_a_desired_plain_one(self, graph) -> None:  # type: ignore[no-untyped-def]
        """Against the server, so the rendering of the WHERE clause is the real one."""
        graph.execute("create property index on doc (c) where c > 0")
        (existing,) = graph.indexes("doc")
        assert "WHERE" in existing.definition.upper()
        assert index_properties(existing.definition) is None
        assert graph.ensure_indexes([DesiredIndex("doc", ("c",))]) == [
            "create property index on doc (c)"
        ]
        assert graph.ensure_indexes([DesiredIndex("doc", ("c",))]) == []

    def test_an_index_no_declaration_could_describe_is_left_alone(self, graph) -> None:  # type: ignore[no-untyped-def]
        """A nested path and an expression cannot be written as a DesiredIndex, so neither may be
        dropped while reconciling a list that could not have mentioned them."""
        graph.execute("create property index on doc (a.b.c)")
        graph.execute("create property index on doc ((f + g))")
        before = {index.name for index in graph.indexes("doc")}
        graph.ensure_indexes([DesiredIndex("doc", ("name",))], drop_extra=True)
        assert before <= {index.name for index in graph.indexes("doc")}

    def test_an_expression_index_survives_a_real_drop_extra(self, graph) -> None:  # type: ignore[no-untyped-def]
        """The live form of the case that would hurt. An index made as plain SQL over the property
        map is reported as a property index over an expression, and must not be dropped for not
        being in a list of property names."""
        # Plain SQL names the table, not the label, so it has to be schema-qualified -- the graph
        # schema is not on the search path the way a selected graph_path is for Cypher.
        schema = graph.label_table.graph
        graph.execute(f"create index doc_expr_idx on \"{schema}\".doc ((properties ->> 'v'))")
        graph.ensure_indexes([DesiredIndex("doc", ("name",))], drop_extra=True)
        assert "doc_expr_idx" in {index.name for index in graph.indexes("doc")}

    def test_a_property_name_needing_quotes_round_trips(self, graph) -> None:  # type: ignore[no-untyped-def]
        desired = [DesiredIndex("doc", ("order",))]
        assert graph.ensure_indexes(desired)
        assert graph.ensure_indexes(desired) == [], "a quoted name did not match itself"

    def test_naming_a_graph_other_than_the_selected_one(self, graph, dsn: str) -> None:  # type: ignore[no-untyped-def]
        graph.execute("drop graph if exists reconcile_other cascade")
        graph.execute("create graph reconcile_other")
        try:
            graph.execute("create vlabel other_doc")  # in the selected graph, not the other one
            statements = graph.ensure_indexes(
                [DesiredIndex("doc", ("name",))], graph="reconcile_other", dry_run=True
            )
            assert statements == ["create property index on doc (name)"]
        finally:
            graph.execute("drop graph reconcile_other cascade")


@pytest.mark.server
class TestTheAwaitingInterface:
    @pytest_asyncio.fixture
    async def conn(self, dsn: str):  # type: ignore[no-untyped-def]
        name = "reconcile_async"
        connection = await agensgraph.AsyncConnection.connect(dsn, autocommit=True)
        async with connection:
            await connection.execute(f'drop graph if exists "{name}" cascade')
            await connection.execute(f'create graph "{name}"')
            await connection.graph(name)
            await connection.execute("create vlabel doc")
            try:
                yield connection
            finally:
                await connection.execute("reset graph_path")
                await connection.execute(f'drop graph "{name}" cascade')

    @pytest.mark.asyncio
    async def test_indexes(self, conn) -> None:  # type: ignore[no-untyped-def]
        desired = [DesiredIndex("doc", ("name",)), DesiredIndex("doc", ("sku",), unique=True)]
        assert len(await conn.ensure_indexes(desired)) == 2
        assert await conn.ensure_indexes(desired) == []

    @pytest.mark.asyncio
    async def test_constraints(self, conn) -> None:  # type: ignore[no-untyped-def]
        desired = [Unique("doc", "sku"), Check("doc", "age > 0", "doc_age_ok")]
        assert len(await conn.ensure_constraints(desired)) == 2
        assert await conn.ensure_constraints(desired) == []


class TestReadingHowAnIndexKeysItsProperties:
    """The server prints these canonically, omitting every default, so they round-trip."""

    @pytest.mark.parametrize(
        ("printed", "expected"),
        [
            # Ascending with nulls last is the default, so neither is printed.
            ("a", IndexElement("a", None, False, False)),
            ("a DESC", IndexElement("a", None, True, True)),
            ("a NULLS FIRST", IndexElement("a", None, False, True)),
            ("a DESC NULLS LAST", IndexElement("a", None, True, False)),
            ("m jsonb_path_ops", IndexElement("m", "jsonb_path_ops", False, False)),
            ("m jsonb_path_ops DESC", IndexElement("m", "jsonb_path_ops", True, True)),
            ('"odd name"', IndexElement("odd name", None, False, False)),
        ],
    )
    def test_an_element(self, printed: str, expected: IndexElement) -> None:
        assert parse_index_element(printed) == expected

    @pytest.mark.parametrize(
        "printed", ["", "((a) + (b))", "a.b.c", 'a COLLATE "C"', "a b c d"]
    )
    def test_something_that_is_not_a_plain_element(self, printed: str) -> None:
        assert parse_index_element(printed) is None

    def test_the_defaults_an_element_leaves_implicit_are_resolved_before_comparing(
        self,
    ) -> None:
        """Ascending puts nulls last and descending puts them first, so an element that says so
        explicitly has to equal one that leaves it out -- otherwise the two would never match."""
        assert IndexElement("a").resolved() == IndexElement("a", nulls_first=False).resolved()
        assert (
            IndexElement("a", descending=True).resolved()
            == IndexElement("a", descending=True, nulls_first=True).resolved()
        )

    def test_the_method_is_read(self) -> None:
        assert index_method("CREATE PROPERTY INDEX i ON doc USING gin (m)") == "gin"
        assert index_method("CREATE PROPERTY INDEX i ON doc USING btree (a)") == "btree"
        assert index_method("nonsense") is None

    def test_a_predicate_is_noticed_and_only_after_the_key_list(self) -> None:
        assert index_is_partial("CREATE PROPERTY INDEX i ON doc USING btree (a) WHERE (a) > 0")
        assert not index_is_partial("CREATE PROPERTY INDEX i ON doc USING btree (a)")
        assert not index_is_partial(an_index("doc", "where").definition)

    def test_a_multi_element_key_list(self) -> None:
        definition = "CREATE PROPERTY INDEX i ON doc USING btree (a, b DESC, c NULLS FIRST)"
        assert index_elements(definition) == (
            IndexElement("a", None, False, False),
            IndexElement("b", None, True, True),
            IndexElement("c", None, False, True),
        )


class TestDiffingTheWiderIndexes:
    def test_the_method_is_part_of_what_an_index_is(self) -> None:
        """A gin index over a property is not the btree index somebody asked for."""
        gin = Index("doc", "g", False, "CREATE PROPERTY INDEX g ON doc USING gin (m)")
        assert reconcile_indexes([DesiredIndex("doc", ("m",))], [gin]) == [
            "create property index on doc (m)"
        ]
        assert reconcile_indexes([DesiredIndex("doc", ("m",), method="gin")], [gin]) == []

    def test_a_sort_order_is_part_of_it_too(self) -> None:
        plain = an_index("doc", "a")
        descending = (DesiredIndex("doc", (IndexElement("a", descending=True),)),)
        assert reconcile_indexes(descending, [plain]) == [
            "create property index on doc (a desc)"
        ]
        assert reconcile_indexes(descending, [an_index("doc", "a DESC")]) == []

    def test_an_operator_class_is_part_of_it(self) -> None:
        actual = [Index("doc", "g", False, "CREATE PROPERTY INDEX g ON doc USING gin (m)")]
        wanted = (DesiredIndex("doc", (IndexElement("m", "jsonb_path_ops"),), method="gin"),)
        assert reconcile_indexes(wanted, actual) == [
            "create property index on doc using gin (m jsonb_path_ops)"
        ]

    def test_a_partial_index_is_matched_by_name(self) -> None:
        wanted = [DesiredIndex("doc", ("a",), name="doc_hot", where="a > 0")]
        assert reconcile_indexes(wanted, []) == [
            "create property index doc_hot on doc (a) where a > 0"
        ]
        existing = Index(
            "doc",
            "doc_hot",
            False,
            "CREATE PROPERTY INDEX doc_hot ON doc USING btree (a) WHERE (a) > cypher_to_jsonb(0)",
        )
        assert reconcile_indexes(wanted, [existing]) == []

    def test_a_partial_index_without_a_name_is_refused(self) -> None:
        with pytest.raises(ValueError, match="needs a name"):
            reconcile_indexes([DesiredIndex("doc", ("a",), where="a > 0")], [])

    def test_a_partial_index_is_not_dropped_as_an_extra_one(self) -> None:
        """It is not describable as an unconditional index, so it is not one to drop."""
        existing = Index(
            "doc",
            "doc_hot",
            False,
            "CREATE PROPERTY INDEX doc_hot ON doc USING btree (a) WHERE (a) > cypher_to_jsonb(0)",
        )
        assert reconcile_indexes([], [existing], drop_extra=True) == []


@pytest.mark.server
class TestTheWiderFormsAgainstAServer:
    """That each form is accepted, and that a second run of the same declaration does nothing."""

    @pytest.fixture
    def graph(self, agens):  # type: ignore[no-untyped-def]
        agens.execute("create vlabel doc")
        return agens

    @pytest.mark.parametrize(
        "wanted",
        [
            DesiredIndex("doc", ("a",)),
            DesiredIndex("doc", ("a", "b")),
            DesiredIndex("doc", ("a",), unique=True),
            DesiredIndex("doc", (IndexElement("a", descending=True),)),
            DesiredIndex("doc", (IndexElement("a", nulls_first=True),)),
            DesiredIndex("doc", (IndexElement("a", descending=True, nulls_first=False),)),
            DesiredIndex("doc", ("a", IndexElement("b", descending=True)), name="mixed"),
            DesiredIndex("doc", (IndexElement("m", "jsonb_path_ops"),), method="gin"),
            DesiredIndex("doc", ("h",), method="hash"),
            DesiredIndex("doc", ("a",), name="doc_hot", where="a > 0"),
            DesiredIndex("doc", ("a",), name="doc_hot2", where="a > 0 and b < 10"),
        ],
    )
    def test_it_is_accepted_and_then_already_there(self, graph, wanted) -> None:  # type: ignore[no-untyped-def]
        applied = graph.ensure_indexes([wanted])
        assert len(applied) == 1, applied
        assert graph.ensure_indexes([wanted]) == [], (
            f"a second run repeated the work for {wanted!r}"
        )

    def test_naming_an_operator_class_that_is_already_the_default_is_reported(
        self, graph
    ) -> None:  # type: ignore[no-untyped-def]
        """The server omits a default operator class when it prints a definition, so asking for one
        by name can never be recognised afterwards. Without the check that follows a run, every run
        would drop and remake the index; with it, the first run says so."""
        wanted = [DesiredIndex("doc", (IndexElement("m", "jsonb_ops"),), method="gin")]
        with pytest.raises(RuntimeError, match="still do not match"):
            graph.ensure_indexes(wanted)

    def test_a_gin_index_is_not_confused_with_a_btree_one_over_the_same_property(
        self, graph
    ) -> None:  # type: ignore[no-untyped-def]
        both = [DesiredIndex("doc", ("m",)), DesiredIndex("doc", ("m",), method="gin")]
        assert len(graph.ensure_indexes(both)) == 2
        assert graph.ensure_indexes(both) == []
        assert len(graph.indexes("doc")) == 2

    def test_changing_the_predicate_of_a_partial_index_is_not_noticed(self, graph) -> None:  # type: ignore[no-untyped-def]
        """Asserted because it is a limitation rather than a bug, and one a caller has to know:
        the predicate is stored normalised, so it cannot be compared against the one written."""
        graph.ensure_indexes([DesiredIndex("doc", ("a",), name="doc_hot", where="a > 0")])
        assert (
            graph.ensure_indexes([DesiredIndex("doc", ("a",), name="doc_hot", where="a > 5")])
            == []
        )


class TestAnIndexDeclarationQuotesEveryNameInIt:
    """A method and an operator class are identifiers, and sat beside a quoted label unquoted."""

    HOSTILE = "x) ; drop graph y cascade --"

    def test_the_method_is_quoted(self) -> None:
        (statement,) = reconcile_indexes(
            [DesiredIndex("doc", ("name",), method=self.HOSTILE)], []
        )
        assert f'using "{self.HOSTILE}"' in statement

    def test_the_operator_class_is_quoted(self) -> None:
        element = IndexElement("name", operator_class=self.HOSTILE)
        (statement,) = reconcile_indexes([DesiredIndex("doc", (element,))], [])
        assert f'"{self.HOSTILE}"' in statement

    def test_an_ordinary_declaration_is_unchanged(self) -> None:
        (statement,) = reconcile_indexes([DesiredIndex("doc", ("name",))], [])
        assert statement == "create property index on doc (name)"
        (statement,) = reconcile_indexes([DesiredIndex("doc", ("tags",), method="gin")], [])
        assert statement == "create property index on doc using gin (tags)"


class TestDeclaringLabels:
    """A write to a label that is not there makes one, which is DDL inside the write."""

    def test_it_asks_for_what_is_missing(self) -> None:
        from agensgraph import DesiredLabel
        from agensgraph.introspect import Label, reconcile_labels

        actual = [Label(3, "doc", "v", None)]
        assert reconcile_labels([DesiredLabel("doc")], actual) == []
        assert reconcile_labels([DesiredLabel("KNOWS", "e")], actual) == [
            'create elabel if not exists "KNOWS"'
        ]

    def test_a_name_needing_quoting_gets_it(self) -> None:
        from agensgraph import DesiredLabel
        from agensgraph.introspect import reconcile_labels

        assert reconcile_labels([DesiredLabel("WORKS AT", "e")], []) == [
            'create elabel if not exists "WORKS AT"'
        ]

    def test_the_same_name_under_the_other_kind_is_refused(self) -> None:
        """The server refuses it too, and the message is clearer from here."""
        from agensgraph import DesiredLabel
        from agensgraph.introspect import Label, reconcile_labels

        with pytest.raises(
            ValueError, match="as a vertex label and the graph has it as an edge"
        ):
            reconcile_labels([DesiredLabel("KNOWS", "v")], [Label(4, "KNOWS", "e", None)])


@pytest.mark.server
class TestDeclaringLabelsAgainstTheServer:
    def test_it_converges_and_creates_both_kinds(self, agens) -> None:  # type: ignore[no-untyped-def]
        from agensgraph import DesiredLabel

        want = [
            DesiredLabel("Memory"),
            DesiredLabel("KNOWS", "e"),
            DesiredLabel("WORKS AT", "e"),
        ]
        assert len(agens.ensure_labels(want)) == 3
        assert agens.ensure_labels(want) == [], "the second run found work to do"
        kinds = {label.name: label.kind for label in agens.labels()}
        assert kinds["Memory"] == "v"
        assert kinds["KNOWS"] == kinds["WORKS AT"] == "e"

    def test_a_declared_edge_label_takes_concurrent_writers(self, agens, dsn: str) -> None:  # type: ignore[no-untyped-def]
        """Undeclared, eight writers merging one edge report 42P07 from each other's label."""
        import threading

        from agensgraph import DesiredLabel

        graph = agens.label_table.graph
        agens.execute("create vlabel m")
        agens.execute("create (:m {k: 'x'})")
        agens.execute("create (:m {k: 'y'})")
        agens.ensure_labels([DesiredLabel("LINKS", "e")])

        failures: list[str | None] = []

        def writer() -> None:
            conn = agensgraph.connect(dsn, autocommit=False)
            conn.graph(graph)
            try:
                conn.execute("""match (a:m {k:'x'}), (b:m {k:'y'}) merge (a)-[:"LINKS"]->(b)""")
                conn.commit()
            except Exception as exc:
                failures.append(getattr(exc, "sqlstate", None))
                conn.rollback()
            finally:
                conn.close()

        threads = [threading.Thread(target=writer) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert "42P07" not in failures, "the label was declared, so no writer had to make it"


class TestWritingDdlWithNoConnection:
    """A caller with no connection cannot reconcile, so it writes the statements out instead."""

    def test_the_builders_are_reachable_from_the_top_level(self) -> None:
        assert agensgraph.create_index_statement is create_index_statement
        assert agensgraph.create_constraint_statement is create_constraint_statement

    def test_if_not_exists_makes_an_index_script_re_runnable(self) -> None:
        statement = create_index_statement(
            DesiredIndex("Memory", ("name",), unique=True, name="memory_name"),
            if_not_exists=True,
        )
        assert statement == (
            'create unique property index if not exists memory_name on "Memory" (name)'
        )

    def test_and_needs_a_name_because_the_server_takes_one_there(self) -> None:
        with pytest.raises(ValueError, match="needs a name"):
            create_index_statement(DesiredIndex("Memory", ("name",)), if_not_exists=True)

    @pytest.mark.parametrize(
        ("label", "prop"),
        [
            ("X {}) DETACH DELETE n //", "k"),
            ("Zap; CREATE TABLE t(i int); --", "k"),
            ("Memory", "a) ; drop table t; --"),
            ("iPhone", "Name"),
        ],
    )
    def test_a_hostile_name_becomes_one_quoted_identifier(self, label: str, prop: str) -> None:
        """The statements go to somebody else to run, so the quoting has to hold here."""
        statement = create_index_statement(DesiredIndex(label, (prop,)))
        assert statement.count(quote_identifier(label)) == 1
        assert quote_identifier(prop) in statement


@pytest.mark.server
class TestTheGeneratedDdlAgainstTheServer:
    def test_the_index_script_can_be_run_twice(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel doc")
        statement = create_index_statement(
            DesiredIndex("doc", ("name",), name="doc_name_once"), if_not_exists=True
        )
        agens.execute(statement)
        agens.execute(statement)
        assert [i.name for i in agens.indexes("doc")] == ["doc_name_once"]

    def test_the_unnamed_statement_is_what_is_not_re_runnable(self, agens) -> None:  # type: ignore[no-untyped-def]
        """Which is what the name is for: the server skips on it and picks one when not given."""
        agens.execute("create vlabel doc")
        statement = create_index_statement(DesiredIndex("doc", ("name",)))
        for _ in range(3):
            agens.execute(statement)
        assert sorted(i.name for i in agens.indexes("doc")) == [
            "doc_name_idx",
            "doc_name_idx1",
            "doc_name_idx2",
        ]

    def test_asking_for_it_unnamed_is_a_syntax_error(self, agens) -> None:  # type: ignore[no-untyped-def]
        """Which is why the builder refuses to write one rather than leaving it to the server."""
        agens.execute("create vlabel doc")
        with pytest.raises(agensgraph.errors.Error) as caught:
            agens.execute("create property index if not exists on doc (name)")
        assert caught.value.sqlstate == "42601"

    def test_a_constraint_script_cannot_be_run_twice(self, agens) -> None:  # type: ignore[no-untyped-def]
        """Which is why the docstring says so: the grammar has no `if not exists` for one."""
        agens.execute("create vlabel doc")
        statement = create_constraint_statement(Unique("doc", "sku"))
        agens.execute(statement)
        with pytest.raises(agensgraph.errors.Error) as caught:
            agens.execute(statement)
        assert caught.value.sqlstate == "42P07"


class TestAUniqueIndexOnAnEdgesEndpoints:
    """The endpoints are columns, and a property index keys on the property of that name.

    So asking for one over ``start`` and ``end`` builds it over ``properties.'start'`` and
    ``properties.'end'``, which an edge carrying no such property leaves NULL -- and no two NULLs
    conflict, so a unique index over them refuses nothing. Measured: the index is
    accepted, ``indexes()`` reports it unique, a duplicate edge is still taken, and eight concurrent
    merges of one triple leave two edges. The server prints it as ``(start, "end")``, which is what
    the columns would print as, so nothing downstream can tell the two apart.
    """

    @pytest.mark.parametrize(
        "properties",
        [("start", "end"), ("start",), ("end",), ("END",), ("start", "weight")],
    )
    def test_asking_for_one_is_refused(self, properties: tuple[str, ...]) -> None:
        with pytest.raises(ValueError, match="would guarantee nothing"):
            create_index_statement(DesiredIndex("links", properties, unique=True, name="x"))

    def test_the_refusal_says_how_to_get_the_guarantee(self) -> None:
        with pytest.raises(ValueError) as caught:
            create_index_statement(DesiredIndex("links", ("start", "end"), unique=True))
        message = str(caught.value)
        assert "create unique index" in message, "the form that does refuse a duplicate"
        assert '(start, "end")' in message

    @pytest.mark.parametrize("properties", [("start", "end"), ("start",)])
    def test_one_that_is_not_unique_promises_nothing_and_is_left_alone(
        self, properties: tuple[str, ...]
    ) -> None:
        """Only the guarantee is refused. A useless index is not a wrong answer."""
        assert "property index" in create_index_statement(DesiredIndex("links", properties))

    def test_an_ordinary_unique_index_is_untouched(self) -> None:
        assert (
            create_index_statement(DesiredIndex("doc", ("sku",), unique=True))
            == "create unique property index on doc (sku)"
        )


@pytest.mark.server
class TestWhatDoesGuaranteeAnEdgesEndpoints:
    def test_the_reconciler_refuses_it_rather_than_running_it(self, agens) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create elabel links")
        agens.refresh_labels()
        with pytest.raises(ValueError, match="would guarantee nothing"):
            agens.ensure_indexes([DesiredIndex("links", ("start", "end"), unique=True)])

    def test_a_plain_unique_index_on_the_columns_does_refuse_a_duplicate(self, agens) -> None:  # type: ignore[no-untyped-def]
        """Which is what the refusal points at, so it is asserted rather than only recommended."""
        graph = agens.label_table.graph
        agens.execute("create vlabel p")
        agens.execute("create elabel links")
        agens.refresh_labels()
        agens.execute("create (:p {n: 1}), (:p {n: 2})")
        agens.execute(f'create unique index links_pair on "{graph}".links (start, "end")')
        agens.execute("match (a:p {n:1}), (b:p {n:2}) create (a)-[:links]->(b)")
        with pytest.raises(psycopg.Error) as caught:
            agens.execute("match (a:p {n:1}), (b:p {n:2}) create (a)-[:links]->(b)")
        assert caught.value.sqlstate == "23505"
        (count,) = agens.execute("match ()-[r:links]->() return count(*)").fetchone()
        assert count == 1
