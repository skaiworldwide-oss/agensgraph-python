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

from typing import TYPE_CHECKING, NamedTuple, TypeVar

from .cypher import quote_identifier

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Protocol

    class _HasLabel(Protocol):
        """Anything the reconcilers key by label: an index, a constraint, or one asked for."""

        @property
        def label(self) -> str: ...


_Labelled = TypeVar("_Labelled", bound="_HasLabel")
"""What a narrowing hands back, which is whatever kind it was given."""

__all__ = [
    "CONSTRAINTS_QUERY",
    "DECLARED_PROPERTIES_QUERY",
    "GRAPHS_QUERY",
    "INDEXES_QUERY",
    "LABELS_QUERY",
    "META_FLAG_QUERY",
    "META_VALID_QUERY",
    "PROMOTION_CATALOG_QUERY",
    "SERVER_PROGRAM_QUERY",
    "TRIPLES_QUERY",
    "Check",
    "Constraint",
    "DeclaredProperty",
    "DesiredIndex",
    "Graph",
    "GraphDescription",
    "Index",
    "IndexElement",
    "Label",
    "PropertyShape",
    "Triple",
    "Unique",
    "constraint_name",
    "describe_kind",
    "element_count_query",
    "for_labels",
    "index_elements",
    "index_is_partial",
    "index_method",
    "index_properties",
    "parse_index_element",
    "property_sample_query",
    "reconcile_constraints",
    "reconcile_indexes",
]

MAX_IDENTIFIER = 63
"""How long a name the server will keep. A longer one is truncated rather than refused."""

GRAPHS_QUERY = """
select g.graphname::text,
       n.nspname::text,
       count(l.oid)::bigint
from pg_catalog.ag_graph g
join pg_catalog.pg_namespace n on n.oid = g.nspid
left join pg_catalog.ag_label l on l.graphid = g.oid
group by g.graphname, n.nspname
order by g.graphname
"""
"""Every graph and how many labels it has.

Counted by a grouped join rather than a subquery per graph, which read the whole of ``ag_label``
once for each of them: 9 buffers against 141, and 0.294 milliseconds against 1.299.
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
_DECLARED_PROPERTIES = """
select l.labname::text,
       p.propname::text,
       format_type(a.atttypid, a.atttypmod),
       not a.attnotnull
from pg_catalog.ag_label_property p
join pg_catalog.ag_label l on l.oid = p.laboid
join pg_catalog.ag_graph g on g.oid = l.graphid
join pg_catalog.pg_attribute a on a.attrelid = l.relid and a.attnum = p.attnum
where g.graphname = %s{label}
order by l.labname, p.propname
"""

PROMOTION_CATALOG_QUERY = """
select to_regclass('pg_catalog.ag_label_property') is not null
"""
"""Whether this server can store a property in a column of its own.

Asked of the catalog rather than worked out from the version, because the version cannot answer
it: the 2.18 release branch and main both report ``2.18-devel`` and only one of them has the
catalog. A server that cannot promote a property has nowhere to record one, so the presence of the
catalog is the feature.
"""

SERVER_PROGRAM_QUERY = """
select rolsuper or pg_has_role(current_user, 'pg_execute_server_program', 'member')
from pg_roles where rolname = current_user
"""
"""Whether this role could run a command on the server's host.

``COPY ... TO PROGRAM`` does that, and a read-only transaction does not stop it: it moves rows out
of the database rather than into it, so it is not a write for the server to refuse. Measured inside
a read-only transaction, every other write was refused with ``25006`` and this one ran.

Asked of the role rather than looked for in the statement, because looking for it in the statement
does not work: a second statement after a semicolon, or a leading comment, gets a copy past any
reading of the text, and all three were demonstrated. Whether the copy can run at all is a
privilege, and a privilege is a thing the server can be asked about once.

Asked as ``member`` rather than ``usage``, because the question is what the role can reach and not
what it holds right now. A membership granted ``WITH INHERIT FALSE`` carries no privilege until
``SET ROLE`` names it, so ``usage`` answers no -- and ``SET ROLE`` moves no rows, so a read-only
transaction has no reason to refuse it. Measured: a role that ``usage`` called safe opened a
read-only transaction, named the membership, and ran a command on the host.
"""

DECLARED_PROPERTIES_QUERY = _DECLARED_PROPERTIES.format(label="")
DECLARED_PROPERTIES_FOR_LABEL = _DECLARED_PROPERTIES.format(
    label="\n  and l.labname::text = %s"
)

_INDEXES = """
select l.labname::text,
       i.relname::text,
       x.indisunique,
       pg_catalog.ag_get_propindexdef(i.oid)
from pg_catalog.ag_graph g
join pg_catalog.ag_label l on l.graphid = g.oid
join pg_catalog.pg_index x on x.indrelid = l.relid
join pg_catalog.pg_class i on i.oid = x.indexrelid
where g.graphname = %s{label}
  and x.indisexclusion = false
  and i.relkind = 'i'
  and (x.indexprs is not null
       or exists (select 1
                    from pg_catalog.pg_attribute a
                   where a.attrelid = l.relid
                     and a.attnum = any(x.indkey::smallint[])
                     and not a.attisdropped
                     and a.attgenerated <> ''))
order by l.labname, i.relname
"""
"""Every property index of a graph, read from the catalogs rather than through the view.

``ag_property_indexes`` answers the same question, and reading it costs what its other five
columns cost -- among them ``pg_size_pretty(pg_table_size(...))``, which the planner may not drop
however few columns are selected, because a volatile expression is not prunable. Reading the
catalogs starts from the graph instead of filtering every index in the database at the top:
93 buffers against 1,574, and 0.99 milliseconds against 28.25.

The conditions are the view's own, kept so the two answer alike: not an exclusion constraint's
index -- that is a uniqueness assertion and belongs to constraints() -- and either an expression
index or one over a promoted column.
"""

INDEXES_QUERY = _INDEXES.format(label="")
INDEXES_FOR_LABEL = _INDEXES.format(label="\n  and l.labname::text = %s")

# Restricted to the two types a graph constraint can be. Asking about a not-null constraint or
# a primary key -- which every label has -- reports an invalid constraint type instead of
# nothing, so those are not asked about. 'x' is a uniqueness assertion and 'c' a check.
_CONSTRAINTS = """
select l.labname::text,
       c.conname::text,
       c.contype = 'x',
       pg_catalog.ag_get_graphconstraintdef(c.oid)
from pg_catalog.pg_constraint c
join pg_catalog.ag_label l on l.relid = c.conrelid
join pg_catalog.ag_graph g on g.oid = l.graphid
where g.graphname = %s
  and c.contype in ('x', 'c'){label}
order by l.labname, c.conname
"""

CONSTRAINTS_QUERY = _CONSTRAINTS.format(label="")
CONSTRAINTS_FOR_LABEL = _CONSTRAINTS.format(label="\n  and l.labname::text = %s")

ASKING_FOR_ONE_LABEL = """Why each of the three above is two statement texts and not one.

One text holding ``(%s::text is null or labname = %s::text)`` has to plan for both.

Once psycopg prepares it -- which it does on the sixth execution -- the generic plan is the one
both get, and that plan estimates one row of ``ag_label`` against the eighty-seven to a hundred it
reads, abandons the index and nested-loops instead, for six and a half to eleven times the buffers.
Two texts each plan for what they are.
"""


def element_count_query(graph: str, *, edges: bool = False) -> str:
    """Count elements per label, grouping on the label id inside every element's identity.

    Nothing decodes a property map: the group key is a function of the id column alone. Whether
    the *heap* is read anyway is the planner's to decide and often yes -- against a narrow label
    it prefers a sequential scan to an index-only one, correctly, because there are fewer heap
    pages than index pages. And an edge can never avoid it: the engine creates an edge's id index
    as BRIN (``parse_utilcmd.c``, ``edge_id_idx->accessMethod = "brin"``), and BRIN carries no
    tuple pointers, so there is no index-only scan for it to choose.

    The graph's name is quoted into the statement because a schema cannot be bound as a parameter.
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


class IndexElement(NamedTuple):
    """One property an index is keyed on, with how it is keyed.

    The server omits every default when it prints a definition: ``ASC`` never appears, ``NULLS
    LAST`` only with ``DESC``, ``NULLS FIRST`` only without it, and an operator class only when it
    is not the default for the type. Name an operator class only when it differs from the default.
    """

    property: str
    operator_class: str | None = None
    descending: bool = False
    nulls_first: bool | None = None
    """``None`` for whichever way round the sort order implies: last ascending, first descending."""

    def nulls_come_first(self) -> bool:
        """Where nulls go, with the default resolved."""
        return self.descending if self.nulls_first is None else self.nulls_first

    def resolved(self) -> IndexElement:
        """The same element with nothing left implicit."""
        return self._replace(nulls_first=self.nulls_come_first())

    def rendered(self) -> str:
        """How this element is written into a statement."""
        parts = [quote_identifier(self.property)]
        if self.operator_class is not None:
            parts.append(quote_identifier(self.operator_class))
        if self.descending:
            parts.append("desc")
        if self.nulls_first is not None and self.nulls_first != self.descending:
            parts.append("nulls first" if self.nulls_first else "nulls last")
        return " ".join(parts)


class DesiredIndex(NamedTuple):
    """A property index somebody wants to exist.

    Matched by the access method and the elements it is keyed on, read off the definition the
    server printed. A difference in uniqueness is a drop and a remake.

    ``properties`` takes plain names, or :class:`IndexElement` for an operator class or a sort
    order. The two can be mixed.

    ``where`` makes the index partial and requires a ``name``, which it is then matched by. **A
    change to the predicate of a partial index is not noticed**; change its name to have it rebuilt.

    An index over a nested path or an expression, or one with ``INCLUDE`` columns, is not described
    here, and is neither matched nor dropped. Write those as DDL, or
    :func:`agensgraph.vector.vector_index` for a vector.
    """

    label: str
    properties: tuple[str | IndexElement, ...]
    unique: bool = False
    name: str | None = None
    """A name to give it. The server picks one from the properties if this is ``None``."""
    method: str = "btree"
    """The access method. ``btree`` unless said otherwise, as it is for the server."""
    where: str | None = None
    """A predicate making the index partial. Requires ``name``; see above."""

    @property
    def elements(self) -> tuple[IndexElement, ...]:
        """The elements, with a plain name read as an element with everything defaulted."""
        return tuple(
            IndexElement(item) if isinstance(item, str) else item for item in self.properties
        )

    def key(self) -> tuple[str, str, tuple[IndexElement, ...]]:
        """What this index is matched on."""
        return (
            self.label,
            self.method.lower(),
            tuple(element.resolved() for element in self.elements),
        )


class Unique(NamedTuple):
    """An assertion that a property holds a different value on every element of a label.

    The assertion is over an expression: ``ASSERT lower(name) IS UNIQUE`` is accepted,
    ``ASSERT (a, b) IS UNIQUE`` is not. Only a plain property name is described here.
    """

    label: str
    property: str
    name: str | None = None
    """A name to give it. Derived from the label and property if this is ``None``."""


class Check(NamedTuple):
    """A condition every element of a label has to satisfy.

    The name is required. An unnamed check is named ``<label>_properties_check``, then the same
    with a counter.

    The expression is Cypher, written into the statement as given.
    """

    label: str
    expression: str
    name: str


def _bare_property(token: str) -> str | None:
    """The property a column of an index definition reads, if it reads one plainly."""
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


def _split_top_level(text: str, separator: str) -> list[str]:
    """Split on a separator that is not inside parentheses or quotes."""
    depth, quoted, parts, current = 0, False, [], ""
    for char in text:
        if char == '"':
            quoted = not quoted
        elif not quoted:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == separator and depth == 0:
                parts.append(current)
                current = ""
                continue
        current += char
    parts.append(current)
    return parts


def _split_words(text: str) -> list[str]:
    """Split on whitespace that is not inside a quoted name or parentheses.

    A property whose name holds a space is printed quoted.
    """
    depth, quoted, words, current = 0, False, [], ""
    for char in text:
        if char == '"':
            quoted = not quoted
        elif not quoted:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char.isspace() and depth == 0:
                if current:
                    words.append(current)
                current = ""
                continue
        current += char
    if current:
        words.append(current)
    return words


def _key_list(definition: str) -> tuple[str, int] | None:
    """The text between the parentheses an index is keyed on, and where it ends."""
    using = definition.find(" USING ")
    start = definition.find("(", using)
    if using < 0 or start < 0:
        return None
    depth = 0
    for offset, char in enumerate(definition[start:]):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return definition[start + 1 : start + offset], start + offset + 1
    return None


def index_method(definition: str) -> str | None:
    """The access method, which the server always prints."""
    using = definition.find(" USING ")
    if using < 0:
        return None
    rest = definition[using + len(" USING ") :].lstrip()
    name = rest.split("(", 1)[0].split()[0] if rest else ""
    return name.lower() or None


def index_is_partial(definition: str) -> bool:
    """Whether a predicate limits which elements the index covers."""
    found = _key_list(definition)
    if found is None:
        return False
    return " WHERE " in f" {definition[found[1] :].upper()} "


def _has_included_columns(definition: str) -> bool:
    found = _key_list(definition)
    return found is not None and "INCLUDE" in definition[found[1] :].upper()


def parse_index_element(token: str) -> IndexElement | None:
    """One element of a printed key list, or ``None`` if it is not a plain property.

    Read from the right: the nulls order, then the sort order, then an operator class. What is
    left has to be a bare property name.
    """
    words = _split_words(token)
    if not words:
        return None
    nulls_first: bool | None = None
    if (
        len(words) >= 2
        and words[-2].upper() == "NULLS"
        and words[-1].upper() in ("FIRST", "LAST")
    ):
        nulls_first = words[-1].upper() == "FIRST"
        words = words[:-2]
    descending = bool(words) and words[-1].upper() == "DESC"
    if descending or (words and words[-1].upper() == "ASC"):
        words = words[:-1]
    if not words:
        return None
    operator_class: str | None = None
    if len(words) > 1:
        # A collation lands here too, and is not described.
        if words[-2].upper() == "COLLATE":
            return None
        operator_class = words[-1]
        words = words[:-1]
    if len(words) != 1:
        return None
    name = _bare_property(words[0])
    if name is None:
        return None
    return IndexElement(name, operator_class, descending, nulls_first).resolved()


def index_elements(definition: str) -> tuple[IndexElement, ...] | None:
    """The elements a plain index is keyed on, read off the definition the server printed.

    ``None`` for anything a desired index cannot describe: a nested path, an expression, a
    collation, ``INCLUDE`` columns, or a predicate.
    """
    found = _key_list(definition)
    if found is None or index_is_partial(definition) or _has_included_columns(definition):
        return None
    elements = [parse_index_element(token) for token in _split_top_level(found[0], ",")]
    if not elements or any(element is None for element in elements):
        return None
    return tuple(element for element in elements if element is not None)


def index_properties(definition: str) -> tuple[str, ...] | None:
    """The property names a plain index is keyed on, ignoring how it keys them."""
    elements = index_elements(definition)
    return None if elements is None else tuple(element.property for element in elements)


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
    if desired.where is not None and not desired.name:
        raise ValueError(
            f"a partial index on {desired.label} needs a name, because its predicate is stored "
            f"normalised and so cannot be compared against the one written here"
        )
    unique = "unique " if desired.unique else ""
    name = f"{quote_identifier(desired.name)} " if desired.name else ""
    # An access method is named by an identifier, and the server takes it quoted.
    method = (
        ""
        if desired.method.lower() == "btree"
        else f"using {quote_identifier(desired.method)} "
    )
    keys = ", ".join(element.rendered() for element in desired.elements)
    where = f" where {desired.where}" if desired.where is not None else ""
    return (
        f"create {unique}property index {name}on {quote_identifier(desired.label)} "
        f"{method}({keys}){where}"
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


def for_labels(
    actual: Sequence[_Labelled], desired: Sequence[_HasLabel], scoped: bool
) -> Sequence[_Labelled]:
    """``actual`` narrowed to the labels ``desired`` names, when the caller is dropping.

    The reconcilers read the list they are given as the whole of what should exist, which is what
    lets an empty one mean "none of these". That only holds while both lists describe the same
    thing, and a caller declaring one label's indexes is handed the whole graph's -- so the graph's
    are narrowed to what the caller was talking about before the two are compared. Without the
    narrowing a list naming one label drops the indexes of every label it did not name.

    Nothing is narrowed when the caller is not dropping, since then the extra ones are only read
    past.
    """
    if not scoped:
        return actual
    spoken_for = {want.label for want in desired}
    return [have for have in actual if have.label in spoken_for]


def reconcile_indexes(
    desired: Sequence[DesiredIndex], actual: Sequence[Index], *, drop_extra: bool = False
) -> list[str]:
    """The statements that take the indexes that exist to the ones asked for.

    Empty when they already agree. An index whose elements are asked for but whose uniqueness
    differs is dropped and remade. A partial index is matched by name. Anything not describable as
    a :class:`DesiredIndex` is neither matched nor dropped.
    """
    by_key: dict[tuple[str, str, tuple[IndexElement, ...]], Index] = {}
    by_name: dict[tuple[str, str], Index] = {}
    for index in actual:
        by_name[index.label, index.name] = index
        elements = index_elements(index.definition)
        method = index_method(index.definition)
        if elements is not None and method is not None:
            by_key[index.label, method, elements] = index

    statements: list[str] = []
    matched: set[str] = set()
    for want in desired:
        if want.where is not None:
            existing = by_name.get((want.label, constraint_name_of_index(want)))
            if existing is None:
                statements.append(create_index_statement(want))
            else:
                matched.add(existing.name)
                if existing.unique != want.unique:
                    statements.append(drop_index_statement(existing.name))
                    statements.append(create_index_statement(want))
            continue
        found = by_key.get(want.key())
        if found is None:
            statements.append(create_index_statement(want))
            continue
        matched.add(found.name)
        if found.unique != want.unique:
            statements.append(drop_index_statement(found.name))
            statements.append(create_index_statement(want))
    if drop_extra:
        for index in by_key.values():
            if index.name not in matched:
                statements.append(drop_index_statement(index.name))
    return statements


def constraint_name_of_index(desired: DesiredIndex) -> str:
    """The name a partial index is matched by, which it has to have been given."""
    if not desired.name:
        raise ValueError(f"a partial index on {desired.label} needs a name")
    return desired.name[:MAX_IDENTIFIER]


def reconcile_constraints(
    desired: Sequence[Unique | Check],
    actual: Sequence[Constraint],
    *,
    drop_extra: bool = False,
) -> list[str]:
    """The statements that take the constraints that exist to the ones asked for.

    Matched by name. A definition comes back normalised: ``age > 0`` prints as
    ``ASSERT ((age) > cypher_to_jsonb(0))``.
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


TRIPLES_QUERY = """
select start::text, edge::text, "end"::text, edgecount
from pg_catalog.ag_graphmeta_view
where graphname = %s
order by edgecount desc, start, edge, "end"
"""
"""Which labels an edge label joins, and how many edges of it there are.

Read from the catalog the server keeps for the planner, so it costs an indexed read of a small
table rather than a pass over every edge. The alternative every integration writes is
``MATCH (a)-[r]->(b)`` with a ``DISTINCT`` over the labels, which reads the whole graph to return a
handful of rows.

The catalog is only as current as the last gather, and ``auto_gather_graphmeta`` is off by default,
so an empty answer means nobody has gathered rather than that the graph has no edges.
:data:`META_VALID_QUERY` is how to tell the two apart, and how to tell a current answer from a
stale one.
"""

META_VALID_QUERY = """
select g.graphmeta_valid
from pg_catalog.ag_graph g
where g.graphname = %s
"""

META_FLAG_QUERY = """
select count(*) > 0
from pg_catalog.pg_attribute
where attrelid = 'pg_catalog.ag_graph'::regclass and attname = 'graphmeta_valid'
"""
"""Whether this server records that the triple catalog is current.

The flag arrived with 2.18. Asked of the catalog rather than of the version, for the same reason
:data:`PROMOTION_CATALOG_QUERY` is: a development build may report a version whose columns it does
not have. Where there is no flag, whether a gather has happened cannot be established, and the
question has to be answered from what the catalog holds instead.
"""
"""Whether the triple catalog still describes this graph.

``regather_graphmeta()`` sets the flag, and a transaction that wrote to the graph clears it at
commit (``pgstat_xact.c``, "ag_graph.graphmeta_valid flag must be cleared at commit"). A read
leaves it alone. So this distinguishes the three states that matter: never gathered, gathered and
current, gathered and since written to.
"""

GATHER_META = "select pg_catalog.regather_graphmeta()"
"""Fill in the catalog :data:`TRIPLES_QUERY` reads.

A write, and not a cheap one on a large graph, which is why nothing calls it without being asked.
"""


def property_sample_query(graph: str, label: str) -> str:
    """What keys a label's elements carry, and what kind of value each holds.

    Over a bounded sample rather than the whole label: the shape of a label is not a thing that
    needs every row to establish, and reading every row of a label full of embeddings to find out
    that one of its keys holds an array is the difference between milliseconds and minutes.

    ``jsonb_typeof`` is built in. Three of the surveyed packages install a plpgsql function that
    calls it and narrows ``number`` to integer or float; that narrowing is done here instead, so
    nothing is created in the caller's database.
    """
    table = f"{quote_identifier(graph)}.{quote_identifier(label)}"
    return (
        f"select key, jsonb_typeof(value) as kind, "
        f"count(*) filter (where jsonb_typeof(value) = 'number' "
        f"and value::text like '%%.%%') as fractional, count(*) as seen "
        f"from (select properties from {table} limit %s) sampled, "
        f"jsonb_each(sampled.properties) "
        f"group by key, kind order by key, kind"
    )


class Triple(NamedTuple):
    """An edge label, and the labels it was found joining.

    The count is not called ``count`` because a named tuple is a tuple, and a field of that name
    would take the place of the method every tuple has.
    """

    start: str
    edge: str
    end: str
    edge_count: int


class PropertyShape(NamedTuple):
    """A property of a label, and what was found in it.

    ``declared`` says the property has a column of its own, which makes ``kind`` exact. Otherwise
    it was read from a sample and describes what that sample held.
    """

    name: str
    kind: str
    declared: bool


class GraphDescription(NamedTuple):
    """What is in a graph, in the shape a prompt or a schema browser wants.

    ``meta_gathered`` is false when the catalog behind :attr:`triples` has never been filled, which
    is its state on a new server. The triples are empty in that case rather than wrong, and asking
    for them again with ``refresh`` is what fills it.
    """

    graph: str
    labels: tuple[Label, ...]
    properties: dict[str, tuple[PropertyShape, ...]]
    triples: tuple[Triple, ...]
    counts: dict[str, int]
    meta_gathered: bool


def describe_kind(kind: str, fractional: int) -> str:
    """The name for a JSON type, with a number split by whether any was not whole.

    A caller writing a prompt wants to know an integer from a float, and JSON does not distinguish
    them: both are ``number``.
    """
    if kind != "number":
        return kind
    return "float" if fractional else "integer"
