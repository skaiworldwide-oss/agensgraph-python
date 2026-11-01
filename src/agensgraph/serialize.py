"""Turning what a graph query returned into JSON.

None of the values a graph query produces is JSON-serializable, and the obvious way to make one so
is worse than the failure: encoding a vertex with msgspec emits its *private* fields, so
``_id`` arrives as a nested ``{"labid": 3, "locid": 1}`` alongside ``_raw`` and ``_props``. That
publishes names the driver reserves the right to change, in a shape nobody would choose.

So the shape is written down here instead, and it is the shape the wire uses. An identity is the
text the server prints, ``"3.1"``, which reads back through ``%s::graphid`` and keeps ``labid`` and
``locid`` recoverable from the value it parses to. The structured form was considered and dropped:
it invites a reader to build an identity out of two numbers, and one built that way has never been
anywhere near the server.

Two ways in. :func:`to_builtins` gives dicts and lists, and :func:`json_default` hands the same
thing to ``json.dumps``. :func:`to_json` is the one to reach for, because it is the only one that
can leave a property map alone: it embeds the bytes the server sent through :class:`msgspec.Raw`,
so a map nobody read is copied to the output rather than decoded and built again.

There is deliberately no encoder hook to register. msgspec calls one only for a type it does not
already know, and a vertex is a :class:`msgspec.Struct`, so it would encode by its own fields and
the hook would never run -- which is how the private names leak in the first place. The conversion
therefore happens before the encoder sees anything.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING, Any

import msgspec

from ._core import Result
from ._protocol.graphid import GraphId
from .types import Edge, Path, Vertex
from .vector import SparseVector, Vector

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["json_default", "to_builtins", "to_json"]


_SCALARS = (str, int, float, bool, type(None), Decimal)
"""What JSON, or the encoder, takes as it is."""


def _rows(result: Result, convert: Callable[[Any], Any]) -> list[Any]:
    """A result as one object per row, keyed by the columns it returned."""
    keys = result.keys
    return [
        {key: convert(value) for key, value in row.items()}
        if isinstance(row, dict)
        else dict(zip(keys, (convert(value) for value in row), strict=False))
        for row in result.records
    ]


def _tuple(row: tuple[Any, ...], convert: Callable[[Any], Any]) -> Any:
    """A tuple as an object where it names its fields, as a named-tuple row does, else a list."""
    fields = getattr(row, "_asdict", None)
    if fields is None:
        return [convert(each) for each in row]
    return {key: convert(value) for key, value in fields().items()}


def to_builtins(obj: Any) -> Any:
    """A graph value -- or a row, a list of rows, or a whole :class:`~agensgraph.Result` -- as
    dicts, lists, strings and numbers.

    A result becomes one object per row, keyed by the columns it returned. A tuple that names its
    fields becomes an object too. Reads the property map, since a dict is what is being asked for;
    :func:`to_json` leaves an unread one alone.
    """
    kind = type(obj)
    if kind is Result:
        return _rows(obj, to_builtins)
    if kind is list:
        return [to_builtins(each) for each in obj]
    if isinstance(obj, tuple):
        return _tuple(obj, to_builtins)
    if kind is dict:
        return {key: to_builtins(value) for key, value in obj.items()}
    if isinstance(obj, _SCALARS):
        return obj
    if type(obj) is GraphId:
        return str(obj)
    if type(obj) is Vertex:
        return {
            "id": str(obj.id),
            "label": obj.label,
            "properties": obj.properties,
        }
    if type(obj) is Edge:
        return {
            "id": str(obj.id),
            "label": obj.label,
            "start": str(obj.start),
            "end": str(obj.end),
            "properties": obj.properties,
        }
    if type(obj) is Path:
        return {
            "vertices": [to_builtins(each) for each in obj.vertices],
            "edges": [to_builtins(each) for each in obj.edges],
        }
    if type(obj) is Vector:
        return obj.tolist()
    if type(obj) is SparseVector:
        return {
            "dimensions": obj.dimensions,
            "indices": list(obj.indices),
            "values": list(obj.values),
        }
    raise TypeError(f"cannot describe {type(obj).__name__} as JSON")


def json_default(obj: Any) -> Any:
    """For ``json.dumps(value, default=agensgraph.json_default)``.

    The standard library builds its output from Python objects alone, so a property map is decoded
    on the way past whether or not the caller wanted it. :func:`enc_hook` is the way to keep the
    bytes.
    """
    return to_builtins(obj)


def _prepared(obj: Any) -> Any:
    """The same shape as :func:`to_builtins`, with an unread property map left as its bytes.

    Walks lists, tuples and dicts, so a whole result goes through in one call and anything that is
    not a graph value is passed on untouched for the encoder to deal with. A :class:`Result` and a
    tuple that names its fields are walked too: left to the encoder, a tuple subclass is encoded
    as it stands, vertices and all, which is how their private fields would get out.
    """
    kind = type(obj)
    if kind is Result:
        return _rows(obj, _prepared)
    if kind is list:
        return [_prepared(each) for each in obj]
    if isinstance(obj, tuple):
        return _tuple(obj, _prepared)
    if kind is dict:
        return {key: _prepared(value) for key, value in obj.items()}
    if kind is Vertex:
        return {
            "id": str(obj.id),
            "label": obj.label,
            "properties": msgspec.Raw(obj.properties_json()),
        }
    if kind is Edge:
        return {
            "id": str(obj.id),
            "label": obj.label,
            "start": str(obj.start),
            "end": str(obj.end),
            "properties": msgspec.Raw(obj.properties_json()),
        }
    if kind is Path:
        return {
            "vertices": [_prepared(each) for each in obj.vertices],
            "edges": [_prepared(each) for each in obj.edges],
        }
    if kind is GraphId or kind is Vector or kind is SparseVector:
        return to_builtins(obj)
    return obj


_ENCODER = msgspec.json.Encoder()


def to_json(obj: Any) -> bytes:
    """A graph value as JSON bytes, without decoding a property map nobody read.

    What to hand a model, or anything else that wants JSON rather than Python. Takes a single
    value or a whole result: a list of rows goes through in one call.

    Kept apart from the encoder the driver writes property maps with, so that what this accepts on
    the way out cannot widen what that accepts on the way in.
    """
    return _ENCODER.encode(_prepared(obj))
