"""Handing a result to something that works in columns.

An escape hatch, not the way results are held. Arrow is worth reaching for when the consumer stays
columnar; taking a table apart into Python objects costs more than never having built one.

A graph element has no columnar form -- a vertex is an identity, a label and a map, which is three
columns and not one -- so a column holding one is refused, naming the column and what to project
instead. Everything a statement can return as a scalar goes through:

============================  ==========================
in a result                   in a column
============================  ==========================
``int`` ``float`` ``str``     itself
``bool`` ``bytes`` ``None``   itself
:class:`~agensgraph.GraphId`  its text form, ``3.1``
a property map                a map, if the backend can hold one
a :class:`~agensgraph.Vector` a list of numbers
a sparse vector               its text form
============================  ==========================

The backends are imported when asked for rather than at import time, so none of them is a
dependency of the driver and none is loaded by a program that does not use them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._protocol.graphid import GraphId
from .types import Edge, Path, Vertex
from .vector import SparseVector, Vector

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["columns", "to_arrow", "to_pandas", "to_polars"]

_REFUSED = (Vertex, Edge, Path)


def _scalar(value: Any) -> Any:
    """One value as a column can hold it."""
    if isinstance(value, _REFUSED):
        raise TypeError(
            f"a {type(value).__name__.lower()} has no columnar form: it is an identity, a label "
            f"and a property map rather than one value. Return the parts wanted instead, as in "
            f"'return id(n), label(n), n.name'."
        )
    if isinstance(value, GraphId):
        return str(value)
    if isinstance(value, Vector):
        return value.tolist()
    if isinstance(value, SparseVector):
        return value.to_text()
    return value


def columns(records: Sequence[Sequence[Any]], keys: Sequence[str]) -> dict[str, list[Any]]:
    """A result turned on its side, one list per column.

    What every backend below is built from, and useful on its own where none of them is wanted.
    """
    if not keys:
        return {}
    out: dict[str, list[Any]] = {name: [] for name in keys}
    for row in records:
        if len(row) != len(keys):
            raise ValueError(f"a row of {len(row)} against {len(keys)} column names")
        for name, value in zip(keys, row, strict=True):
            out[name].append(_scalar(value))
    return out


def to_arrow(records: Sequence[Sequence[Any]], keys: Sequence[str]) -> Any:
    """An Arrow table. Needs ``pyarrow``."""
    import pyarrow

    return pyarrow.table(columns(records, keys))


def to_pandas(records: Sequence[Sequence[Any]], keys: Sequence[str]) -> Any:
    """A pandas frame. Needs ``pandas``."""
    import pandas

    return pandas.DataFrame(columns(records, keys), columns=list(keys))


def to_polars(records: Sequence[Sequence[Any]], keys: Sequence[str]) -> Any:
    """A polars frame. Needs ``polars``."""
    import polars

    return polars.DataFrame(columns(records, keys))
