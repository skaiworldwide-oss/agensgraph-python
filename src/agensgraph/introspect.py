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

from typing import NamedTuple

from .cypher import quote_identifier

__all__ = [
    "CONSTRAINTS_QUERY",
    "DECLARED_PROPERTIES_QUERY",
    "GRAPHS_QUERY",
    "INDEXES_QUERY",
    "LABELS_QUERY",
    "Constraint",
    "DeclaredProperty",
    "Graph",
    "Index",
    "Label",
    "element_count_query",
]

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
