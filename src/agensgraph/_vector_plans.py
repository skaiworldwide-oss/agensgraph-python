"""How each vector storage design is carried out, as statements. Nothing here waits.

One class per design, both answering :class:`Plan`, and one registry from the design's type to its
class -- so the connection, which runs what a plan builds, never asks which design it holds. Every
statement is built from the storage once, when its plan is made, and plans are kept per storage.

Every write and delete answers one row holding how many rows it changed, in both designs, so the
connection reads the count the same way whichever it ran.
"""

from __future__ import annotations

import functools
import hashlib
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol

from psycopg.types.json import Jsonb

from .cypher import quote_identifier
from .errors import StaleVectorStorage, VectorStorageMismatch
from .types import Vertex
from .vector import Distance, SparseVector, Vector, _with_options, vector_index
from .vector_storage import (
    MAX_NAME_BYTES,
    InProperties,
    InTable,
    StorageState,
    VectorHit,
    VectorStorage,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from ._protocol.graphid import GraphId

__all__ = ["CopyIn", "Execute", "Found", "Plan", "plan_for", "refusal"]


class Execute(NamedTuple):
    """A statement and its parameters."""

    text: str
    params: tuple[Any, ...] = ()
    on_graph: bool = False
    """Whether it names the label without its graph, and so runs on the session's graph."""


class CopyIn(NamedTuple):
    """A binary ``COPY``, its column types and its rows.

    Not named ``Copy``: the blocking connection is generated from the awaiting one, and psycopg's
    ``AsyncCopy`` becomes ``Copy`` there.
    """

    text: str
    types: tuple[str, ...]
    rows: list[tuple[Any, ...]]


class Found(NamedTuple):
    """What the probe found, and why, for the error a state other than ready raises.

    ``table_exists`` is the table design's table; the properties need none. ``indexed`` is the
    design's index, there and over the right shape. A storage asking for an index is ready only
    once it has one.
    """

    state: StorageState
    label_exists: bool
    detail: str = ""
    table_exists: bool = False
    indexed: bool = False


class Plan(Protocol):
    """Everything a design does, as statements and the decoding of their rows."""

    requires: str | None
    """The capability the design needs, or ``None`` for one every release has."""

    def probe(self) -> Execute: ...
    def found(self, row: Sequence[Any]) -> Found: ...
    def create(self, found: Found) -> list[Execute]: ...
    def index(self, *, concurrently: bool) -> Execute | None: ...
    def write(
        self, rows: Sequence[tuple[GraphId, object]], *, upsert: bool
    ) -> Execute | CopyIn: ...
    def read(self, ids: Sequence[GraphId]) -> Execute: ...
    def vectors(
        self, rows: Iterable[Sequence[Any]]
    ) -> dict[GraphId, Vector | SparseVector]: ...
    def delete(self, ids: Sequence[GraphId]) -> Execute: ...
    def nearest(self, vector: object, k: int) -> Execute: ...
    def hits(self, rows: Iterable[Sequence[Any]]) -> list[VectorHit]: ...
    def drop(self) -> list[Execute]: ...


def _bounded(name: str) -> str:
    """A derived identifier, shortened with a hash of the whole where it would not fit.

    The server cuts a longer one, and two cut names can collide; the hash keeps them apart and
    keeps the name the same every time it is derived.
    """
    if len(name.encode()) <= MAX_NAME_BYTES:
        return name
    digest = hashlib.blake2s(name.encode(), digest_size=4).hexdigest()
    head = name.encode()[: MAX_NAME_BYTES - len(digest) - 1].decode(errors="ignore")
    return f"{head}_{digest}"


def _operator_class(storage: VectorStorage) -> str:
    return Distance(storage.distance).operator_class_for(storage.type)


def _index_found(name: str, definition: str | None, signature: str) -> tuple[bool, str]:
    """Whether the design's index is there, and what is wrong with the one there if anything.

    The name says which index is the design's; its definition has to hold the *signature*: what
    it orders by, followed by the operator class it orders with. An index left from a storage that
    measured by another distance keeps the name and serves none of this storage's searches, which
    would otherwise pass for ready and scan every row. Quotes are set aside, since the server
    quotes a name only where its own grammar needs it to.
    """
    if definition is None:
        return False, ""
    if signature not in definition.replace('"', ""):
        return False, (
            f"the index {name!r} is not over {signature.strip('()')}: {definition}; drop it, or "
            f"use the distance it was built for"
        )
    return True, ""


def _value(value: object, storage: VectorStorage) -> Vector | SparseVector:
    """A value as the storage's type, refused here if its dimension is not the storage's."""
    if storage.type == "sparsevec":
        if not isinstance(value, SparseVector):
            raise TypeError(
                f"a sparsevec storage takes a SparseVector, got {type(value).__name__}; "
                f"SparseVector.from_dense() makes one"
            )
        size = value.dimensions
        made: Vector | SparseVector = value
    else:
        made = value if isinstance(value, Vector) else Vector(value)  # type: ignore[arg-type]
        size = len(made)
    if size != storage.dimensions:
        raise ValueError(
            f"a vector of {size} dimensions, for a storage of {storage.dimensions}"
        )
    return made


def _limit(k: object) -> int:
    """How many hits a search asks for, refused here unless it is a whole number of at least one.

    A count read from a request or a configuration file is often a float or a string, which the
    server would take as a malformed ``LIMIT`` or not at all.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"k is a whole number of at least one, got {k!r}")
    return k


def _as_read(value: object) -> Vector | SparseVector:
    """A vector read back, as one whether it arrived typed or as the property's JSON."""
    if isinstance(value, Vector | SparseVector):
        return value
    if isinstance(value, str):
        return SparseVector.from_text(value)
    return Vector(value)  # type: ignore[arg-type]


def _json_value(value: Vector | SparseVector) -> object:
    """How a vector is written into a property map: its numbers, or a sparse one's text."""
    return value.to_text() if isinstance(value, SparseVector) else value.values.tolist()


class PropertyPlan(Plan):
    """The vector as a property, searched through a Cypher cast. Every statement is Cypher, or DDL
    that names the label alone, so all of them run on the session's graph."""

    def __init__(self, storage: VectorStorage) -> None:
        design = storage.design()
        assert isinstance(design, InProperties)
        self._storage = storage
        self.requires = "property_promotion" if design.promoted else "cypher_typmod_cast"
        label = quote_identifier(storage.label)
        name = quote_identifier(storage.property)
        typed = f"{storage.type}({storage.dimensions})"
        # A column of its own carries its type; a value in the map is cast to it, dimension and
        # all, or the index built over the cast is never matched. The query vector is bound twice
        # rather than once by name: one parameter used in both places plans as a sequential scan
        # on 2.17, measured, where the ORDER BY's own parameter is matched to the index.
        searched = f"n.{name}" if design.promoted else f"n.{name}::{typed}"
        measure = f"{searched} {storage.distance} %s::{typed}"
        method = storage.index.method if storage.index is not None else "exact"
        self._index_name = _bounded(f"{storage.label}_{storage.property}_{method}")
        self._typed = typed
        opclass = _operator_class(storage)
        self._index_signature = (
            f"({storage.property} {opclass})" if design.promoted else f"::{typed}) {opclass}"
        )
        self._label_statement = (
            f"create vlabel if not exists {label} ({name} {typed} generated)"
            if design.promoted
            else f"create vlabel if not exists {label}"
        )
        self._promoted = design.promoted
        self._write = (
            f"unwind %s as r match (n:{label}) where id(n) = (r.id)::graphid "
            f"set n.{name} = r.v return count(n)"
        )
        self._read = f"unwind %s as i match (n:{label}) where id(n) = (i)::graphid return id(n), n.{name}"
        self._delete = (
            f"unwind %s as i match (n:{label}) where id(n) = (i)::graphid and n.{name} is not null "
            f"remove n.{name} return count(n)"
        )
        # Ordered by the expression rather than by its alias: 2.17 sorts an alias with a
        # sequential scan, where the expression is matched to the index on every release.
        self._nearest = (
            f"match (n:{label}) return n, {measure} as distance order by {measure} limit %s"
        )

    def probe(self) -> Execute:
        return Execute(
            _PROPERTY_PROBE,
            (
                self._storage.graph,
                self._storage.label,
                self._storage.property,
                self._index_name,
            ),
        )

    def found(self, row: Sequence[Any]) -> Found:
        relid, column_type, index_definition = row
        if relid is None:
            return Found(StorageState.ABSENT, label_exists=False)
        if self._promoted and column_type != self._typed:
            return Found(
                StorageState.MISMATCHED,
                label_exists=True,
                detail=f"the label has {column_type or 'no column'} for {self._storage.property!r}, "
                f"not {self._typed}; a label is given its column when it is made",
            )
        indexed, wrong = _index_found(self._index_name, index_definition, self._index_signature)
        if wrong:
            return Found(StorageState.MISMATCHED, label_exists=True, detail=wrong)
        if not indexed and self._storage.index is not None:
            return Found(StorageState.ABSENT, label_exists=True)
        return Found(StorageState.READY, label_exists=True, indexed=indexed)

    def create(self, found: Found) -> list[Execute]:
        return [] if found.label_exists else [Execute(self._label_statement, on_graph=True)]

    def index(self, *, concurrently: bool) -> Execute | None:
        storage = self._storage
        if storage.index is None:
            return None
        if concurrently:
            raise ValueError("a property index is built in place; there is no concurrent build")
        return Execute(
            vector_index(
                storage.label,
                storage.property,
                dimensions=None if self._promoted else storage.dimensions,
                type=storage.type,
                method=storage.index.method,
                operator_class=_operator_class(storage),
                name=self._index_name,
                options=storage.index.options(),
                if_not_exists=True,
            ),
            on_graph=True,
        )

    def write(self, rows: Sequence[tuple[GraphId, object]], *, upsert: bool) -> Execute:
        # A property is set whether or not it was there, so an upsert and a plain write are one.
        payload = [
            {"id": str(gid), "v": _json_value(_value(v, self._storage))} for gid, v in rows
        ]
        return Execute(self._write, (Jsonb(payload),), on_graph=True)

    def read(self, ids: Sequence[GraphId]) -> Execute:
        return Execute(self._read, (Jsonb([str(gid) for gid in ids]),), on_graph=True)

    def vectors(self, rows: Iterable[Sequence[Any]]) -> dict[GraphId, Vector | SparseVector]:
        return {gid: _as_read(value) for gid, value in rows if value is not None}

    def delete(self, ids: Sequence[GraphId]) -> Execute:
        return Execute(self._delete, (Jsonb([str(gid) for gid in ids]),), on_graph=True)

    def nearest(self, vector: object, k: int) -> Execute:
        value = _value(vector, self._storage)
        return Execute(self._nearest, (value, value, _limit(k)), on_graph=True)

    def hits(self, rows: Iterable[Sequence[Any]]) -> list[VectorHit]:
        return [VectorHit(vertex, float(distance)) for vertex, distance in rows]

    def drop(self) -> list[Execute]:
        name = quote_identifier(self._index_name)
        return [Execute(f"drop property index if exists {name}", on_graph=True)]


_LABEL = """
with l as (
    select l.relid from pg_catalog.ag_label l
    join pg_catalog.ag_graph g on g.oid = l.graphid
    where g.graphname = %s and l.labname = %s and l.labkind = 'v'
)"""
"""The label a storage is for, by graph and name, as ``l``; both probes start with it."""


def _column_type(relation: str) -> str:
    """The declared type of a column of *relation*, its name the next parameter."""
    return f"""
       (select pg_catalog.format_type(a.atttypid, a.atttypmod) from pg_catalog.pg_attribute a
         where a.attrelid = {relation} and a.attname = %s and not a.attisdropped)"""


def _index_definition(relation: str) -> str:
    """The definition of an index on *relation*, its name the next parameter."""
    return f"""
       (select pg_catalog.pg_get_indexdef(i.indexrelid) from pg_catalog.pg_index i
         join pg_catalog.pg_class c on c.oid = i.indexrelid
         where i.indrelid = {relation} and c.relname = %s)"""


_PROPERTY_PROBE = f"""{_LABEL}
select (select relid from l),{_column_type("(select relid from l)")},{
    _index_definition("(select relid from l)")
}
"""
"""The label, a column of its own for the property, and the design's index, in one statement."""


_TABLE_PROBE = f"""{_LABEL}, t as (select pg_catalog.to_regclass(%s) as oid)
select (select relid from l),
       (select oid from t),
       (select count(*) from pg_catalog.pg_constraint k
         where k.conrelid = (select oid from t) and k.contype = 'f' and k.confdeltype = 'c'
           and k.confrelid = (select relid from l)),{_column_type("(select oid from t)")},{
    _index_definition("(select oid from t)")
}
"""
"""The label, the table, whether the table's key still cascades from that label, the column, and
the design's index."""

_INLINE_WIDTH: Mapping[str, int] = {"vector": 4, "halfvec": 2}
"""Bytes a number takes in each type that is stored packed; a sparse vector's size varies."""

_INLINE_LIMIT = 8_000
"""Above this a row cannot be kept on its page, whatever the table asks for."""


class TablePlan(Plan):
    """The vectors in a table keyed by the vertex. Every statement names its schema, so none of
    them depends on the session's graph -- except making the label, which cannot be qualified."""

    requires = None

    def __init__(self, storage: VectorStorage) -> None:
        design = storage.design()
        assert isinstance(design, InTable)
        assert design.schema is not None
        assert design.name is not None
        self._storage = storage
        table = self._qualified_name = (
            f"{quote_identifier(design.schema)}.{quote_identifier(design.name)}"
        )
        label = f"{quote_identifier(storage.graph)}.{quote_identifier(storage.label)}"
        column = quote_identifier(storage.property)
        typed = self._typed = f"{storage.type}({storage.dimensions})"
        self._index_signature = f"({storage.property} {_operator_class(storage)})"
        method = storage.index.method if storage.index is not None else "exact"
        self._index_name = _bounded(f"{design.name}_{method}")
        self._schema_statement = (
            f"create schema if not exists {quote_identifier(design.schema)}"
        )
        width = _INLINE_WIDTH.get(storage.type)
        inline = width is not None and width * storage.dimensions + 32 <= _INLINE_LIMIT
        # Kept on the row's own page where it fits, so a search reads one page per row rather
        # than following each vector into the toast table.
        self._table_statement = (
            f"create table if not exists {table} ("
            f"id graphid primary key references {label} (id) on delete cascade, "
            f"{column} {typed} not null)"
            + (" with (toast_tuple_target = 8160)" if inline else "")
        )
        self._upsert = (
            f"with written as (insert into {table} (id, {column}) "
            f"select * from unnest(%s::graphid[], %s::{typed}[]) "
            f"on conflict (id) do update set {column} = excluded.{column} "
            f"where {table}.{column} is distinct from excluded.{column} returning 1) "
            f"select count(*) from written"
        )
        self._copy = f"copy {table} (id, {column}) from stdin (format binary)"
        self._read = f"select id, {column} from {table} where id = any(%s::graphid[])"
        self._delete = (
            f"with gone as (delete from {table} where id = any(%s::graphid[]) returning 1) "
            f"select count(*) from gone"
        )
        self._nearest = (
            f"select v.id, l.properties, v.{column} {storage.distance} %s::{typed} as distance "
            f"from {table} v join {label} l on l.id = v.id order by distance limit %s"
        )
        self._label_statement = f"create vlabel if not exists {quote_identifier(storage.label)}"

    def probe(self) -> Execute:
        storage = self._storage
        return Execute(
            _TABLE_PROBE,
            (
                storage.graph,
                storage.label,
                self._qualified_name,
                storage.property,
                self._index_name,
            ),
        )

    def found(self, row: Sequence[Any]) -> Found:
        label, table, keyed, column_type, index_definition = row
        exists = label is not None
        if table is None:
            return Found(StorageState.ABSENT, label_exists=exists)
        if not keyed:
            return Found(
                StorageState.STALE,
                label_exists=exists,
                table_exists=True,
                detail=(
                    f"{self._qualified_name} is no longer keyed to {self._storage.label}: the "
                    f"label or its graph was dropped with CASCADE, which drops the key and keeps "
                    f"the rows, and their identities may now name other vertices. Truncate the "
                    f"table, or drop it with drop_vectors(), before using it again"
                ),
            )
        if column_type != self._typed:
            return Found(
                StorageState.MISMATCHED,
                label_exists=exists,
                table_exists=True,
                detail=f"{self._qualified_name} holds {column_type}, not {self._typed}",
            )
        indexed, wrong = _index_found(self._index_name, index_definition, self._index_signature)
        if wrong:
            return Found(
                StorageState.MISMATCHED, label_exists=exists, table_exists=True, detail=wrong
            )
        if not indexed and self._storage.index is not None:
            return Found(StorageState.ABSENT, label_exists=exists, table_exists=True)
        return Found(
            StorageState.READY, label_exists=exists, table_exists=True, indexed=indexed
        )

    def create(self, found: Found) -> list[Execute]:
        label = [] if found.label_exists else [Execute(self._label_statement, on_graph=True)]
        if found.table_exists:
            return label
        return [*label, Execute(self._schema_statement), Execute(self._table_statement)]

    def index(self, *, concurrently: bool) -> Execute | None:
        storage = self._storage
        if storage.index is None:
            return None
        how = "concurrently " if concurrently else ""
        return Execute(
            f"create index {how}if not exists {quote_identifier(self._index_name)} "
            f"on {self._qualified_name} using {storage.index.method} "
            f"({quote_identifier(storage.property)} {_operator_class(storage)})"
            f"{_with_options(storage.index.options())}"
        )

    def write(
        self, rows: Sequence[tuple[GraphId, object]], *, upsert: bool
    ) -> Execute | CopyIn:
        ids = [gid for gid, _ in rows]
        values = [_value(value, self._storage) for _, value in rows]
        if upsert:
            return Execute(self._upsert, (ids, values))
        return CopyIn(
            self._copy, ("graphid", self._storage.type), list(zip(ids, values, strict=True))
        )

    def read(self, ids: Sequence[GraphId]) -> Execute:
        return Execute(self._read, (list(ids),))

    def vectors(self, rows: Iterable[Sequence[Any]]) -> dict[GraphId, Vector | SparseVector]:
        return {gid: _as_read(value) for gid, value in rows}

    def delete(self, ids: Sequence[GraphId]) -> Execute:
        return Execute(self._delete, (list(ids),))

    def nearest(self, vector: object, k: int) -> Execute:
        return Execute(self._nearest, (_value(vector, self._storage), _limit(k)))

    def hits(self, rows: Iterable[Sequence[Any]]) -> list[VectorHit]:
        label = self._storage.label
        return [VectorHit(Vertex(gid, label, props), float(d)) for gid, props, d in rows]

    def drop(self) -> list[Execute]:
        return [Execute(f"drop table if exists {self._qualified_name}")]


_REFUSALS: Mapping[StorageState, type[Exception]] = {
    StorageState.STALE: StaleVectorStorage,
    StorageState.MISMATCHED: VectorStorageMismatch,
}


def refusal(found: Found) -> Exception | None:
    """The error a state that cannot be used raises, or ``None`` for one that can."""
    error = _REFUSALS.get(found.state)
    return error(found.detail) if error is not None else None


_PLANS: Mapping[type[InProperties | InTable], Callable[[VectorStorage], Plan]] = {
    InProperties: PropertyPlan,
    InTable: TablePlan,
}
"""The one place a design is matched to how it is carried out."""


@functools.lru_cache(maxsize=256)
def plan_for(storage: VectorStorage) -> Plan:
    """The plan for a storage, built once per distinct storage."""
    return _PLANS[type(storage.design())](storage)
