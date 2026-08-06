"""How a number in a property map is read.

jsonb keeps a number as an arbitrary-precision decimal. Python's float is not one, so the two do not
line up, and where they part company is worth knowing:

* An **integer of any size** is read exactly, however long. ``1e400`` included -- the server stores
  that as an exact four-hundred-and-one-digit integer, not an infinity.
* A **non-integer** is read as a float, so it keeps about seventeen significant digits. A property
  written as ``3.141592653589793238462643383279`` reads back as ``3.141592653589793``, and one written
  as ``1e-400`` reads back as ``0.0``.
* **A negative zero is already gone** before the driver sees it: the server stores ``-0.0`` as ``0.0``.

:func:`read_numbers_exactly` reads every non-integer as a :class:`~decimal.Decimal` instead, which
keeps whatever the server holds.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import msgspec

__all__ = ["decode_json", "encode_json", "read_numbers_exactly", "reading_numbers_exactly"]

_AS_FLOAT = msgspec.json.Decoder().decode
_AS_DECIMAL = msgspec.json.Decoder(float_hook=Decimal).decode

_decode: Any = _AS_FLOAT

_encode = msgspec.json.Encoder(decimal_format="number").encode


def decode_json(data: bytes) -> Any:
    """Read a property map, at whichever precision is currently asked for."""
    return _decode(data)


def encode_json(value: Any) -> bytes:
    """Write a value back as JSON.

    A decimal is written as a number rather than as a string, so a map read with
    :func:`read_numbers_exactly` and written again holds the numbers it held.
    """
    return _encode(value)


def read_numbers_exactly(enabled: bool = True) -> None:
    """Read a non-integer as a :class:`~decimal.Decimal` rather than a float.

    Process-wide, and meant to be called once at startup: a property map is decoded from a value that
    holds no connection, so there is nothing narrower to attach the choice to. Once at startup is
    also the only time it is unambiguous, because a map on the text path is decoded when it is first
    read rather than when its row arrived -- so a row read before this was called and read after it
    holds decimals, and one whose map was already touched holds floats. Two rows of one result can
    disagree that way.

    What it costs depends on what the map holds, because only a non-integer takes the slower path.
    Measured on maps of forty keys: a map of integers is unchanged, a mixed one costs twice as
    much, one of non-integers 4.4 times, and a single embedding of 1536 floats 5.8 times. An
    integer is read exactly either way, however long.
    """
    global _decode
    _decode = _AS_DECIMAL if enabled else _AS_FLOAT


def reading_numbers_exactly() -> bool:
    """Whether non-integers are currently read as decimals."""
    return _decode is _AS_DECIMAL
