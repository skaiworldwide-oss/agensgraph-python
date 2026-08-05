"""What is in a database, read from the catalogs.

There is no ``\\d`` for a graph. psql knows nothing about ``ag_graph`` or ``ag_label``, so the
only way to find out what labels exist, what is indexed and what is constrained is to read the
catalogs directly -- which makes this the introspection layer rather than a convenience over
one.

The statements live here and the methods that run them are on the connection, so that both
interfaces get them from one copy.

Two things are easy to get wrong and are got right here. A uniqueness constraint is an
*exclusion* constraint, and ``ag_property_indexes`` filters exclusion constraints out -- its
own definition carries ``AND x.indisexclusion = false`` -- so a driver that read indexes and
constraints from that view alone would report a graph as having no uniqueness constraints when
it has them. And ``ag_get_graphconstraintdef`` raises on a constraint that is not a graph
constraint: every label carries not-null constraints and a primary key, and asking about one of
those reports an invalid constraint type rather than returning nothing. Both are filtered.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from .cypher import quote_identifier

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "CONSTRAINTS_QUERY",
    "DECLARED_PROPERTIES_QUERY",
    "GRAPHS_QUERY",
    "INDEXES_QUERY",
    "LABELS_QUERY",
    "Check",
    "Constraint",
    "DeclaredProperty",
    "DesiredIndex",
    "Graph",
    "Index",
    "Label",
    "Unique",
    "constraint_name",
    "element_count_query",
    "index_properties",
    "reconcile_constraints",
    "reconcile_indexes",
]

MAX_IDENTIFIER = 63
"""How long a name the server will keep. A longer one is truncated rather than refused."""

GRAPHS_QUERY = """
select g.graphname::text,
       n.nspname::text,
       (select count(*) from pg_catalog.ag_label l where l.graphid = g.oid)::bigint
from pg_catalog.ag_graph g
join pg_catalog.pg_namespace n on n.oid = g.nspid
order by g.graphname
"""

# The parent is read from the inheritance the server sets up between label tables, which is
# what a label inheriting another one is. A label with no parent is one the graph came with.
LABELS_QUERY = """
select l.labid,
       l.labname::text,
       l.labkind,
       (select p.relname::text
          from pg_catalog.pg_inherits i
          join pg_catalog.pg_class p on p.oid = i.inhparent
         where i.inhrelid = l.relid
         limit 1)
from pg_catalog.ag_label l
join pg_catalog.ag_graph g on g.oid = l.graphid
where g.graphname = %s
order by l.labid
"""

# Only a property with a column of its own is listed, because only that is declared anywhere.
# A property living in the JSON map is not declared at all, so nothing can be read about it.
#
# The label is cast in both places it appears. Left uncast, a null in a comparison against
# nothing else typed gives the server nothing to infer from, and it says so rather than
# guessing -- which is the same refusal a bare parameter gets as an argument to concat.
DECLARED_PROPERTIES_QUERY = """
select l.labname::text,
       p.propname::text,
       format_type(a.atttypid, a.atttypmod),
       not a.attnotnull
from pg_catalog.ag_label_property p
join pg_catalog.ag_label l on l.oid = p.laboid
join pg_catalog.ag_graph g on g.oid = l.graphid
join pg_catalog.pg_attribute a on a.attrelid = l.relid and a.attnum = p.attnum
where g.graphname = %s and (%s::text is null or l.labname::text = %s::text)
order by l.labname, p.propname
"""

INDEXES_QUERY = """
select labelname::text, indexname::text, "unique", indexdef
from pg_catalog.ag_property_indexes
where graphname = %s and (%s::text is null or labelname::text = %s::text)
order by labelname, indexname
"""

# Restricted to the two types a graph constraint can be. Asking about a not-null constraint or
# a primary key -- which every label has -- reports an invalid constraint type instead of
# nothing, so those are not asked about. 'x' is a uniqueness assertion and 'c' a check.
CONSTRAINTS_QUERY = """
select l.labname::text,
       c.conname::text,
       c.contype = 'x',
       pg_catalog.ag_get_graphconstraintdef(c.oid)
from pg_catalog.pg_constraint c
join pg_catalog.ag_label l on l.relid = c.conrelid
join pg_catalog.ag_graph g on g.oid = l.graphid
where g.graphname = %s
  and c.contype in ('x', 'c')
  and (%s::text is null or l.labname::text = %s::text)
order by l.labname, c.conname
"""


def element_count_query(graph: str, *, edges: bool = False) -> str:
    """Count elements per label without reading a single property.

    The label id is part of every element's identity, so grouping on it needs nothing from the
    row but its id -- no property map decoded, no wider column read. The graph's name is quoted
    into the statement because a schema cannot be bound as a parameter.
    """
    table = "ag_edge" if edges else "ag_vertex"
    return (
        f"select graphid_labid(id) as labid, count(*)::bigint "
        f"from {quote_identifier(graph)}.{table} group by 1"
    )


class Graph(NamedTuple):
    """A graph, and how many labels it holds."""

    name: str
    schema: str
    labels: int


class Label(NamedTuple):
    """A label, its kind, and the label it inherits."""

    id: int
    name: str
    kind: str
    """``'v'`` for a vertex label, ``'e'`` for an edge label."""
    parent: str | None
    """The label this one inherits, or ``None`` for the two a graph is created with."""

    @property
    def is_vertex(self) -> bool:
        return self.kind == "v"

    @property
    def is_edge(self) -> bool:
        return self.kind == "e"

    @property
    def is_builtin(self) -> bool:
        """Whether the server made this label rather than a user.

        The two a graph comes with have ids 1 and 2, and every user label inherits one of them.
        """
        return self.parent is None


class DeclaredProperty(NamedTuple):
    """A property given a column of its own, and the type of that column.

    Only a promoted property appears. Before 2.18 nothing can be promoted, so this is always
    empty there -- which is not the same as a label having no properties, and does not read
    like it.
    """

    label: str
    name: str
    type: str
    nullable: bool


class Index(NamedTuple):
    """A property index."""

    label: str
    name: str
    unique: bool
    definition: str


class Constraint(NamedTuple):
    """A constraint on a label's properties."""

    label: str
    name: str
    unique: bool
    """Whether this is a uniqueness assertion, which the server keeps as an exclusion."""
    definition: str


class DesiredIndex(NamedTuple):
    """A plain btree property index somebody wants to exist.

    Matched against what is there by the properties it covers and whether it is unique, read off
    the definition the server printed. Not by name: the server derives one from the columns, but
    truncates it at the identifier limit and appends a counter on a collision.

    A property index can be a good deal more than this -- over a nested path or an expression, with
    a sort order, an operator class, an access method, ``INCLUDE`` columns, or a ``WHERE`` making it
    partial. None of that is described here, and reconciliation leaves any such index alone rather
    than treating it as one of these. Write those as DDL, or as :func:`agensgraph.vector.vector_index`
    for a vector.
    """

    label: str
    properties: tuple[str, ...]
    unique: bool = False
    name: str | None = None
    """A name to give it. The server picks one from the properties if this is ``None``."""


class Unique(NamedTuple):
    """An assertion that a property holds a different value on every element of a label.

    The server asserts over an *expression*, so ``ASSERT lower(name) IS UNIQUE`` is accepted while
    ``ASSERT (a, b) IS UNIQUE`` is not -- a parenthesised list is not an expression. Only a plain
    property name is described here; an assertion over anything else is written as DDL.
    """

    label: str
    property: str
    name: str | None = None
    """A name to give it. Derived from the label and property if this is ``None``."""


class Check(NamedTuple):
    """A condition every element of a label has to satisfy.

    The name is required. An unnamed check is named ``<label>_properties_check``, then ``check1``
    and ``check2`` as more are added, which identifies no condition in particular.

    The expression is Cypher, written into the statement as given rather than parsed or quoted.
    """

    label: str
    expression: str
    name: str


def _bare_property(token: str) -> str | None:
    """The property a column of an index definition reads, if it reads one plainly.

    ``None`` for an expression, which no desired property index matches and which
    :func:`reconcile_indexes` therefore never drops.
    """
    token = token.strip()
    if not token:
        return None
    if token.startswith('"') and token.endswith('"') and len(token) > 2:
        inner = token[1:-1]
        # A doubled quote stands for one quote. A lone one means the name was split across
        # tokens by the comma scan, so this is not a name that can be read plainly.
        return None if '"' in inner.replace('""', "") else inner.replace('""', '"')
    if token[0].isdigit():
        return None
    return token if all(char == "_" or char.isalnum() for char in token) else None


def index_properties(definition: str) -> tuple[str, ...] | None:
    """The properties a plain index covers, read off the definition the server printed.

    ``None`` for anything a desired index cannot describe, which reconciliation therefore leaves
    alone. The grammar allows a great deal more than a list of property names: a nested path
    (``a.b.c``), an expression (``((a) + (b))``), a sort or nulls order per element, an operator
    class, an access method, ``INCLUDE``, and a ``WHERE`` making the index partial. All of those
    read as ``None`` here.
    """
    start = definition.find("(", definition.find(" USING "))
    if start < 0:
        return None
    depth, parts, current, end = 0, [], "", 0
    for offset, char in enumerate(definition[start:]):
        if char == "(":
            depth += 1
            if depth == 1:
                continue
        elif char == ")":
            depth -= 1
            if depth == 0:
                parts.append(current)
                end = start + offset + 1
                break
        if depth == 1 and char == ",":
            parts.append(current)
            current = ""
            continue
        current += char
    else:
        return None
    # A partial index, or one carrying extra columns, covers its properties conditionally or holds
    # more than it is keyed on. Neither is the index a bare list of property names asks for, and
    # reading one as though it were would report a desired index as already present.
    rest = definition[end:].upper()
    if " WHERE " in f" {rest} " or "INCLUDE" in rest:
        return None
    properties = [_bare_property(part) for part in parts]
    if not properties or any(name is None for name in properties):
        return None
    return tuple(name for name in properties if name is not None)


def constraint_name(desired: Unique | Check) -> str:
    """The name a constraint is given, and matched by."""
    if desired.name is not None:
        return desired.name[:MAX_IDENTIFIER]
    if isinstance(desired, Check):  # unreachable: Check requires a name
        raise ValueError("a check constraint needs a name")
    return f"{desired.label}_{desired.property}_unique"[:MAX_IDENTIFIER]


def create_index_statement(desired: DesiredIndex) -> str:
    if not desired.properties:
        raise ValueError(f"an index covers at least one property, got none for {desired.label}")
    unique = "unique " if desired.unique else ""
    name = f"{quote_identifier(desired.name)} " if desired.name else ""
    columns = ", ".join(quote_identifier(prop) for prop in desired.properties)
    return (
        f"create {unique}property index {name}on {quote_identifier(desired.label)} ({columns})"
    )


def drop_index_statement(name: str) -> str:
    return f"drop property index {quote_identifier(name)}"


def create_constraint_statement(desired: Unique | Check) -> str:
    name = quote_identifier(constraint_name(desired))
    label = quote_identifier(desired.label)
    if isinstance(desired, Unique):
        assertion = f"{quote_identifier(desired.property)} is unique"
    else:
        assertion = desired.expression
    return f"create constraint {name} on {label} assert {assertion}"


def drop_constraint_statement(label: str, name: str) -> str:
    return f"drop constraint {quote_identifier(name)} on {quote_identifier(label)}"


def reconcile_indexes(
    desired: Sequence[DesiredIndex], actual: Sequence[Index], *, drop_extra: bool = False
) -> list[str]:
    """The statements that take the indexes that exist to the ones asked for.

    Empty when they already agree. An index whose properties are asked for but whose uniqueness
    differs is dropped and remade; uniqueness cannot be altered into an index.
    """
    have: dict[tuple[str, tuple[str, ...]], Index] = {}
    for index in actual:
        properties = index_properties(index.definition)
        if properties is not None:
            have[index.label, properties] = index

    statements: list[str] = []
    wanted: set[tuple[str, tuple[str, ...]]] = set()
    for want in desired:
        key = (want.label, tuple(want.properties))
        wanted.add(key)
        existing = have.get(key)
        if existing is None:
            statements.append(create_index_statement(want))
        elif existing.unique != want.unique:
            statements.append(drop_index_statement(existing.name))
            statements.append(create_index_statement(want))
    if drop_extra:
        for key, index in have.items():
            if key not in wanted:
                statements.append(drop_index_statement(index.name))
    return statements


def reconcile_constraints(
    desired: Sequence[Unique | Check],
    actual: Sequence[Constraint],
    *,
    drop_extra: bool = False,
) -> list[str]:
    """The statements that take the constraints that exist to the ones asked for.

    Matched by name. A definition comes back normalised -- ``age > 0`` is printed as
    ``ASSERT ((age) > cypher_to_jsonb(0))`` -- so an expression as written never equals one as
    printed.
    """
    have = {(constraint.label, constraint.name): constraint for constraint in actual}
    statements: list[str] = []
    wanted: set[tuple[str, str]] = set()
    for want in desired:
        key = (want.label, constraint_name(want))
        if key in wanted:
            raise ValueError(f"two constraints asked for the same name {key[1]!r} on {key[0]}")
        wanted.add(key)
        existing = have.get(key)
        if existing is None:
            statements.append(create_constraint_statement(want))
        elif existing.unique != isinstance(want, Unique):
            statements.append(drop_constraint_statement(*key))
            statements.append(create_constraint_statement(want))
    if drop_extra:
        for label, name in have:
            if (label, name) not in wanted:
                statements.append(drop_constraint_statement(label, name))
    return statements
