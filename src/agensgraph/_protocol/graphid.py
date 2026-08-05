"""Encoding and decoding of the ``graphid`` type.

A graphid is a 64-bit unsigned value that packs a label id into the top 16 bits and
a per-label serial id into the low 48. On the wire it is eight bytes, big-endian,
with no version byte and no length prefix of its own. As text it is the two parts
written in decimal and joined by a dot.

Because the label id occupies the high bits, graphids sort by ``(labid, locid)`` and
any label occupies a contiguous range of the value space.
"""

from __future__ import annotations

import struct

import msgspec

__all__ = [
    "LABID_MAX",
    "LOCID_MAX",
    "GraphId",
    "pack",
    "parse_text",
    "unpack",
]

LABID_MAX = 65535
"""Largest label id the server will allocate."""

LOCID_MAX = (1 << 48) - 1
"""Largest serial id available within one label."""

_LOCID_MASK = LOCID_MAX
_LABID_SHIFT = 48

_unpack_u64 = struct.Struct(">Q").unpack
_pack_u64 = struct.Struct(">Q").pack


class GraphId(msgspec.Struct, frozen=True, gc=False, order=True):
    """The identity of a vertex or an edge.

    It holds the two parts, which is what both renderings hand over: the text form writes
    them separately, and the binary form is split to resolve the label. Ordering follows the
    parts in the order they are declared, which is the order the packed value sorts in.

    Frozen, so it can be a dict key or a set member, and untracked by the collector,
    which is what a result holding one per element rests on.
    """

    labid: int
    """The label this identity belongs to."""

    locid: int
    """The serial id within the label."""

    def __post_init__(self) -> None:
        if not 0 <= self.labid <= LABID_MAX:
            raise ValueError(f"labid out of range: {self.labid}")
        if not 0 <= self.locid <= LOCID_MAX:
            raise ValueError(f"locid out of range: {self.locid}")

    @classmethod
    def from_packed(cls, packed: int) -> GraphId:
        """Build from the packed 64-bit value, as it appears on the wire."""
        return cls(packed >> _LABID_SHIFT, packed & _LOCID_MASK)

    @property
    def packed(self) -> int:
        """The single 64-bit value, unsigned."""
        return (self.labid << _LABID_SHIFT) | self.locid

    def __str__(self) -> str:
        return f"{self.labid}.{self.locid}"


def unpack(data: bytes) -> GraphId:
    """Decode the eight wire bytes.

    The value is unsigned, so a label id at or above 32768 would read as a negative
    number under a signed interpretation.
    """
    if len(data) != 8:
        raise ValueError(f"graphid must be 8 bytes, got {len(data)}")
    return GraphId.from_packed(_unpack_u64(data)[0])


def pack(gid: GraphId) -> bytes:
    """Encode to the eight wire bytes."""
    return _pack_u64(gid.packed)


def parse_text(text: bytes | str) -> GraphId:
    """Decode the ``labid.locid`` text form.

    The whole string must be consumed. Trailing characters are an error rather than
    something to discard, so a truncated or corrupted value cannot pass as a valid
    identity.
    """
    if isinstance(text, str):
        text = text.encode()
    dot = text.find(b".")
    if dot <= 0 or dot == len(text) - 1:
        raise ValueError(f"invalid graphid: {text!r}")
    head, tail = text[:dot], text[dot + 1 :]
    if not (head.isdigit() and tail.isdigit()):
        raise ValueError(f"invalid graphid: {text!r}")
    return GraphId(int(head), int(tail))
