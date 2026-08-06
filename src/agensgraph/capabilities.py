"""What the connected server can do.

The server reports its own version in the startup packet, as the ``agversion`` run-time
parameter, so this costs nothing: by the time a connection is usable the answer has
already arrived and no query is needed to ask for it.

Two version numbers are in play and they move independently. ``agversion`` governs the
graph features and is what everything here reads. ``server_version`` is the PostgreSQL
release underneath, which differs between graph releases for reasons of its own -- so a
graph feature is never gated on it.

Reading the graph types, decoding either wire format, counting what a write changed, and
resolving an unknown commit all work on every supported version, so nothing in this
module gates them.
"""

from __future__ import annotations

import re
from typing import Protocol

from .errors import CapabilityError

__all__ = ["MINIMUM_VERSION", "Capabilities", "parse_version"]

MINIMUM_VERSION = (2, 16)
"""The oldest server this driver reads. Older ones lack catalogs it depends on."""

_PROPERTY_PROMOTION = (2, 18)
_GQL_CLAUSES = (2, 18)
_ELEMENT_ORDERING = (2, 18)
_ENDPOINT_ELISION = (2, 18)

# A release build reports '2.18' and a development build '2.18-devel', so only the two
# leading numbers are read and whatever follows them is kept for reporting.
_VERSION = re.compile(r"(\d+)\.(\d+)")


def parse_version(text: str) -> tuple[int, int]:
    """Read the major and minor numbers out of a reported version."""
    match = _VERSION.match(text.strip())
    if match is None:
        raise CapabilityError(
            f"cannot read an AgensGraph version from {text!r}. The server reports one in "
            f"`agversion`, as `2.18` or `2.18-devel`"
        )
    return int(match.group(1)), int(match.group(2))


class _Reports(Protocol):
    """Anything that can be asked what the server said at startup."""

    @property
    def info(self) -> _Startup: ...


class _Startup(Protocol):
    def parameter_status(self, param_name: str) -> str | None: ...


class Capabilities:
    """The features of one server.

    Each question can be asked two ways. Plainly, it answers yes or no, which is what a
    caller choosing between two ways of doing something wants. With ``check=True`` it
    raises instead of answering no, naming the feature and the version that would carry
    it, which is what a caller with only one way of doing something wants -- the refusal
    then says what is missing rather than letting the server fail on a syntax it has never
    heard of.
    """

    __slots__ = ("_reported", "_version")

    _version: tuple[int, int]
    _reported: str

    def __init__(self, reported: str) -> None:
        version = parse_version(reported)
        if version < MINIMUM_VERSION:
            required = ".".join(str(part) for part in MINIMUM_VERSION)
            raise CapabilityError(
                f"this driver needs AgensGraph {required} or later; this server is {reported}"
            )
        self._version = version
        self._reported = reported

    @classmethod
    def of(cls, conn: _Reports) -> Capabilities:
        """Read the version a connection was told at startup.

        A server that reports no ``agversion`` at all is a PostgreSQL without the graph
        extensions, which is refused here rather than at the first graph query.
        """
        reported = conn.info.parameter_status("agversion")
        if reported is None:
            raise CapabilityError(
                "this server does not report agversion, so it is not an AgensGraph server"
            )
        return cls(reported)

    @property
    def version(self) -> tuple[int, int]:
        """The major and minor numbers, for comparing."""
        return self._version

    @property
    def reported(self) -> str:
        """The version exactly as the server gave it, development suffix and all."""
        return self._reported

    def _at_least(self, version: tuple[int, int], feature: str, *, check: bool) -> bool:
        if self._version >= version:
            return True
        if check:
            raise CapabilityError.for_feature(
                feature,
                required=".".join(str(part) for part in version),
                found=self._reported,
            )
        return False

    def has_property_promotion(self, *, check: bool = False) -> bool:
        """Whether a property can be stored in a column of its own.

        Where it can, reading one returns that column's own type rather than a JSON value,
        so a property read is not always JSON. Where it cannot, it always is. Either way
        the decoder follows the type the result declares, so this is a question about what
        can be declared rather than about how anything is read.
        """
        return self._at_least(
            _PROPERTY_PROMOTION, "storing a property in its own column", check=check
        )

    def has_gql_clauses(self, *, check: bool = False) -> bool:
        """Whether the GQL clauses are understood: LET, NEXT, FINISH, FILTER, FOR and CALL.

        This also decides how much of a query can be wrapped for reading in chunks, since
        the wrap accepts only what the server will read.
        """
        return self._at_least(_GQL_CLAUSES, "the GQL clauses", check=check)

    def has_element_ordering(self, *, check: bool = False) -> bool:
        """Whether a vertex or an edge can be sorted on directly.

        Sorting on a property works everywhere; sorting on the element itself orders by
        identity and needs the operators for it.
        """
        return self._at_least(_ELEMENT_ORDERING, "ordering by a vertex or an edge", check=check)

    def has_endpoint_elision(self, *, check: bool = False) -> bool:
        """Whether an unread pattern endpoint is left unfetched.

        Visible only in a plan, which is the one place a caller would go looking for it.
        """
        return self._at_least(
            _ENDPOINT_ELISION, "leaving an unread endpoint unfetched", check=check
        )

    def __repr__(self) -> str:
        return f"Capabilities({self._reported!r})"


VECTOR_VERSION_QUERY = """
select extversion from pg_catalog.pg_extension where extname = 'vector'
"""
"""What version of pgvector is created here, or nothing.

Worth having as a version rather than a yes or no, because pgvector gates features on its own
version the way the server does: iterative index scans arrived in 0.8.0, and half precision and
sparse vectors in 0.7.0. A caller that has to know cannot read it off a boolean.
"""

VECTOR_AVAILABLE_QUERY = """
select count(*) > 0 from pg_catalog.pg_type t where t.typname = 'vector'
"""
"""Whether vectors can be read on this connection.

Not a version question, which is why it is not one of the gates above: the extension is created
per database, so the same server answers differently for two of them. And its type has no fixed
oid, so the only way to ask is by name.
"""
