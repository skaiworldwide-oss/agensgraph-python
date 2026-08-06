"""Structural equality for the differential oracles, without the traps of ``==``.

Shared by the two of them: one decodes a corpus by both routes with no server, the other
asks a live server for the same rows in both renderings. Plain ``==`` would pass on
disagreement and fail on agreement in five ways -- ``nan != nan``, ``-0.0 == 0.0``, dict
equality ignoring order, ``Decimal('1.5') == 1.5``, and a differing type comparing equal --
so the type check runs first and each of the rest is answered on its own.
"""

from __future__ import annotations

import math

from agensgraph.types import Edge, Path, Vertex

__all__ = ["same", "same_element"]


def same(a: object, b: object) -> bool:
    """Structural equality that does not paper over the traps of ``==``."""
    if type(a) is not type(b):
        return False
    if isinstance(a, float):
        assert isinstance(b, float)
        if math.isnan(a) and math.isnan(b):
            return True
        if a == 0.0 and b == 0.0:
            # -0.0 == 0.0 is true, so plain equality would hide a lost sign.
            return math.copysign(1.0, a) == math.copysign(1.0, b)
        return a == b
    if isinstance(a, dict):
        assert isinstance(b, dict)
        if list(a.keys()) != list(b.keys()):
            return False
        return all(same(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        assert isinstance(b, list | tuple)
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)


def same_element(a: object, b: object) -> bool:
    if type(a) is not type(b):
        return False
    if isinstance(a, Vertex):
        assert isinstance(b, Vertex)
        return a.id == b.id and a.label == b.label and same(a.properties, b.properties)
    if isinstance(a, Edge):
        assert isinstance(b, Edge)
        return (
            a.id == b.id
            and a.label == b.label
            and a.start == b.start
            and a.end == b.end
            and same(a.properties, b.properties)
        )
    if isinstance(a, Path):
        assert isinstance(b, Path)
        return len(a) == len(b) and all(same_element(x, y) for x, y in zip(a, b, strict=True))
    raise TypeError(f"not a graph value: {type(a)}")
