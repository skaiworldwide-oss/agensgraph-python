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

from typing import Any

import msgspec

from ._protocol.graphid import GraphId
from .types import Edge, Path, Vertex
from .vector import SparseVector, Vector

__all__ = ["json_default", "to_builtins", "to_json"]


def to_builtins(obj: Any) -> Any:
    """A graph value as dicts, lists and strings.

    Reads the property map, since a dict is what is being asked for. Where that is not wanted,
    :func:`enc_hook` leaves an unread map alone.
    """
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
    not a graph value is passed on untouched for the encoder to deal with.
    """
    kind = type(obj)
    if kind is list or kind is tuple:
        return [_prepared(each) for each in obj]
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
