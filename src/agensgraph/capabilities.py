"""What the connected server can do.

The server reports its own version in the startup packet, as the ``agversion`` run-time
parameter, so this costs nothing: by the time a connection is usable the answer has
already arrived and no query is needed to ask for it.

Two version numbers are in play and they move independently. ``agversion`` governs the
graph features and is what everything here reads. ``server_version`` is the PostgreSQL
release underneath, which differs between graph releases for reasons of its own -- so a
graph feature is never gated on it.

Every behaviour that differs between releases is one row of :data:`FEATURES`: the first
release that has it, and for a known bug the first release that fixed it. That table is the
only place in the driver, and in the packages built on it, where a version is compared. A
caller asks for a feature by name and chooses between its own implementations once, rather
than comparing versions where it works.

Reading the graph types, decoding either wire format, counting what a write changed, and
resolving an unknown commit all work on every supported version, so nothing in this
module gates them.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from .errors import CapabilityError

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "FEATURES",
    "MINIMUM_VERSION",
    "Capabilities",
    "Feature",
    "parse_release",
    "parse_version",
]

MINIMUM_VERSION = (2, 14)
"""The oldest server line this driver is built and tested against.

2.14 runs on PostgreSQL 14. 2.13, on PostgreSQL 13, is untested, so it is not a release this
driver is claimed to work on.
"""

_RELEASE_WIDTH = 4


def _pad(release: tuple[int, ...]) -> tuple[int, ...]:
    return release + (0,) * (_RELEASE_WIDTH - len(release))


@dataclass(frozen=True)
class Feature:
    """One behaviour that differs between releases.

    ``since`` is the first release that has it. For a known bug, ``until`` is the first release
    that no longer has it, so the feature is present on ``since <= release < until``. Releases
    are compared to the patch: ``(2, 18, 6)`` is 2.18.6.

    A ``safety`` feature is one the driver keeps its own guard for where it is missing, so a
    development build or a release candidate, which may predate the change, is not taken to have
    it: wrongly answering no costs a refusal the driver did not need, wrongly answering yes costs
    the guard.
    """

    since: tuple[int, ...]
    description: str
    until: tuple[int, ...] | None = None
    safety: bool = False

    def holds_for(self, release: tuple[int, ...]) -> bool:
        """Whether a release, padded to four numbers, has this behaviour."""
        if release < _pad(self.since):
            return False
        return self.until is None or release < _pad(self.until)


FEATURES: Mapping[str, Feature] = MappingProxyType(
    {
        # Safety. Where these are false, the driver refuses rather than relies on the server.
        "cypher_write_enforced": Feature(
            (2, 16), "Cypher writes checked against READ ONLY and table privileges", safety=True
        ),
        "detach_checks_edge_grants": Feature(
            (2, 17),
            "DETACH DELETE checked against DELETE on each edge label it removes from",
            safety=True,
        ),
        "unlabelled_read_checks_label_grants": Feature(
            (2, 18, 6),
            "an unlabelled read checked against each label's grants and row security",
            safety=True,
        ),
        # Statement forms.
        "order_by_unprojected": Feature(
            (2, 17), "ORDER BY a value the RETURN does not project"
        ),
        "in_parameter_list": Feature((2, 17), "x IN a list parameter"),
        "cypher_typmod_cast": Feature((2, 17), "a ::vector(N) cast in Cypher"),
        "cypher_vector_cast": Feature((2, 15), "a ::vector cast in Cypher"),
        "cypher_sql_functions": Feature(
            (2, 15), "start_id, end_id, range, split and jsonb_build_object called from Cypher"
        ),
        "range_in_unwind": Feature((2, 18), "UNWIND range(a, b)"),
        "missing_label_read_empty": Feature(
            (2, 15), "a read of a missing label returning no rows"
        ),
        "missing_label_write_noop": Feature(
            (2, 16), "a write to a missing label changing nothing"
        ),
        "merge_creates_label": Feature((2, 15), "MERGE creating its label"),
        "collect_drops_nulls": Feature(
            (2, 15), "collect() dropping nulls and giving [] for no rows"
        ),
        "fast_vertex_delete": Feature(
            (2, 17), "deleting a vertex without scanning every edge table in the graph"
        ),
        "element_comparison": Feature((2, 18), "comparing a vertex or an edge with < or >"),
        "element_array_negative_index": Feature(
            (2, 18), "a negative index into nodes(p) or relationships(p)"
        ),
        "list_filter_drops_json_null": Feature(
            (2, 18), "[x IN l WHERE x IS NOT NULL] dropping JSON nulls"
        ),
        "size_of_list": Feature((2, 18), "size() of a list"),
        "aggregate_slice_in_with": Feature(
            (2, 18), "slicing an aggregate in the WITH that computes it"
        ),
        "gql_clauses": Feature((2, 18), "the GQL surface"),
        "property_promotion": Feature((2, 18), "storing a property in its own column"),
        "endpoint_elision": Feature((2, 18), "leaving an unread endpoint unfetched"),
        "text_cast_sort_in_with": Feature((2, 18, 6), "sorting on a text cast inside a WITH"),
        # Server state the driver reads.
        "graph_dml_setting": Feature((2, 17), "the enable_graph_dml setting"),
        "graphmeta_notices": Feature((2, 17), "regather_graphmeta() reporting what it changed"),
        "graphmeta_catalog": Feature((2, 18, 4), "the graphmeta_valid flag"),
        "graph_path_reported": Feature(
            (2, 18, 6), "graph_path reported as a run-time parameter"
        ),
        "write_completes_under_limit": Feature(
            (2, 18, 6), "a write under LIMIT changing every matched row"
        ),
        "write_counters_fresh": Feature(
            (2, 18, 6), "write counters reporting this statement's changes"
        ),
        # Known bugs: present on [since, until).
        "jsonb_minmax_wrong": Feature(
            (2, 15), "min() and max() over jsonb comparing as text", until=(2, 18)
        ),
        "vle_unnamed_wrong": Feature(
            (2, 14), "an unnamed variable-length edge returning wrong paths", until=(2, 16)
        ),
    }
)
"""Every behaviour that differs between supported releases, by name."""

# The two leading numbers are the AgensGraph line and the PostgreSQL major under it; any more
# name the patch and its build. Whatever follows them -- '-rc1', '-devel', or a shape no release
# has worn -- marks a build that is not a plain release.
_VERSION = re.compile(r"(\d+)\.(\d+)((?:\.\d+)*)(.*)", re.DOTALL)


def parse_release(text: str) -> tuple[tuple[int, ...], bool]:
    """Read a reported version as its numbers and whether it is a pre-release.

    ``2.18.6.1`` is ``((2, 18, 6, 1), False)``; ``2.18.6.0-rc1`` is ``((2, 18, 6, 0), True)``;
    ``2.18-devel`` is ``((2, 18), True)``. Text after the numbers that no release has reported is
    read the same way rather than refused: the connection goes ahead, and a build that may not be
    a plain release is held to what the oldest release of its line does.
    """
    match = _VERSION.match(text.strip())
    if match is None:
        raise CapabilityError(
            f"cannot read an AgensGraph version from {text!r}. The server reports one in "
            "`agversion`, as `2.18.6.1`, `2.18.6.0-rc1`, `2.18-devel`, `2.17.0`, `2.16` "
            "or `2.14.1`"
        )
    numbers: tuple[int, ...] = (int(match.group(1)), int(match.group(2)))
    numbers += tuple(int(part) for part in match.group(3).split(".") if part)
    return numbers, bool(match.group(4).strip())


def parse_version(text: str) -> tuple[int, int]:
    """Read the AgensGraph line and the PostgreSQL major out of a reported version."""
    numbers, _ = parse_release(text)
    return numbers[0], numbers[1]


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

    A release candidate or a development build is judged as the first release of its line:
    ``2.18.6.0-rc1`` and ``2.18-devel`` both answer as 2.18.0. Such a build may lack a fix its
    number suggests, and answering as the older release makes every gate err on the side that
    refuses, or that takes the slower path which is correct everywhere.
    """

    __slots__ = ("_effective", "_prerelease", "_release", "_reported")

    _release: tuple[int, ...]
    _effective: tuple[int, ...]
    _prerelease: bool
    _reported: str

    def __init__(self, reported: str) -> None:
        release, prerelease = parse_release(reported)
        if release[:2] < MINIMUM_VERSION:
            required = ".".join(str(part) for part in MINIMUM_VERSION)
            raise CapabilityError(
                f"this driver needs AgensGraph {required} or later; this server is {reported}"
            )
        self._release = release
        self._prerelease = prerelease
        self._effective = _pad(release[:2] if prerelease else release)
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
        """The AgensGraph line and the PostgreSQL major under it, for comparing lines.

        A release, a candidate for it and a development build of one line all compare alike;
        :attr:`release` has the patch and :attr:`reported` the version entire.
        """
        return self._release[0], self._release[1]

    @property
    def release(self) -> tuple[int, ...]:
        """Every number the server reported, patch and build included: ``(2, 18, 6, 1)``."""
        return self._release

    @property
    def is_prerelease(self) -> bool:
        """Whether the server reported a release candidate or a development build."""
        return self._prerelease

    @property
    def reported(self) -> str:
        """The version exactly as the server gave it, development suffix and all."""
        return self._reported

    def has(self, name: str, *, check: bool = False) -> bool:
        """Whether this server has the named behaviour from :data:`FEATURES`.

        For a known bug the answer is whether the bug is present. ``check=True`` raises rather
        than answering no, naming the release that would carry the behaviour. A safety feature
        is never taken on a development build or a release candidate.
        """
        try:
            feature = FEATURES[name]
        except KeyError:
            raise ValueError(
                f"no such feature {name!r}; see agensgraph.capabilities.FEATURES"
            ) from None
        if feature.holds_for(self._effective) and not (feature.safety and self._prerelease):
            return True
        if check:
            raise CapabilityError.for_feature(
                feature.description,
                required=".".join(str(part) for part in feature.since),
                found=self._reported,
            )
        return False

    def has_property_promotion(self, *, check: bool = False) -> bool:
        """Whether a property can be stored in a column of its own.

        Where it can, reading one returns that column's own type rather than a JSON value,
        so a property read is not always JSON. Where it cannot, it always is. Either way
        the decoder follows the type the result declares, so this is a question about what
        can be declared rather than about how anything is read.

        This reads the version and nothing else, which is what makes it free -- and it is
        therefore a claim about the release rather than about the build in front of it. Two
        servers reporting ``2.18-devel`` were found to differ, one carrying the catalog a
        promoted property is recorded in and one not, so a development build may report a
        version whose features it does not have. Where the answer has to be right rather than
        free, ask the catalog: :meth:`~agensgraph.Connection.can_promote_properties`.
        """
        return self.has("property_promotion", check=check)

    def has_gql_clauses(self, *, check: bool = False) -> bool:
        """Whether the GQL surface is understood.

        Confirmed against the grammar and a server, and confirmed absent from 2.17's:
        ``INSERT`` (the same grammar arm as ``CREATE``, so it *writes*), ``LET``, ``NEXT``,
        ``FILTER``, ``FINISH``, ``FOR ... IN ... WITH OFFSET AS``, ``CALL { }`` and
        ``CALL func() YIELD``, ``OPTIONAL CALL``, ``OFFSET`` as a synonym for ``SKIP``,
        ``RETURN ALL`` and ``WITH ALL``, ``IS [NOT] UNKNOWN``, ``NULLIF``, ``XOR``,
        ``CURRENT_GRAPH``, and the subquery value expressions ``EXISTS``, ``COUNT``,
        ``COLLECT``, ``ARRAY`` and ``VALUE``.

        It does **not** decide what can be wrapped for reading in chunks. Three of these --
        ``FILTER``, ``NEXT`` and ``CALL func() YIELD`` -- the grammar keeps for the top of a
        statement, so a server that has them still refuses them inside the wrap, and the wrap's
        own boundary moves between versions in the other direction as well: 2.18 refuses a
        ``LIMIT`` that is not last where 2.17 takes it. What a subquery holds is the server's to
        say, and it says so with a syntax error naming the word.
        """
        return self.has("gql_clauses", check=check)

    def has_element_comparison(self, *, check: bool = False) -> bool:
        """Whether a vertex or an edge can be compared with ``<`` or ``>``.

        Before 2.18 such a comparison silently matches nothing. Sorting is a different
        question: ``ORDER BY`` a vertex or an edge the ``RETURN`` projects sorts by identity on
        every supported release, and ordering by one it does not project needs
        ``has("order_by_unprojected")``.
        """
        return self.has("element_comparison", check=check)

    def has_element_ordering(self, *, check: bool = False) -> bool:
        """Deprecated alias of :meth:`has_element_comparison`.

        The name promised ``ORDER BY`` a vertex or an edge, which works on every supported
        release; what 2.18 adds is comparing them.
        """
        warnings.warn(
            "has_element_ordering() is deprecated; use has_element_comparison(), or "
            "has('order_by_unprojected') for ORDER BY a value the RETURN does not project",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.has_element_comparison(check=check)

    def has_endpoint_elision(self, *, check: bool = False) -> bool:
        """Whether an unread pattern endpoint is left unfetched.

        Visible only in a plan, which is the one place a caller would go looking for it.
        """
        return self.has("endpoint_elision", check=check)

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
