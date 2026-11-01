"""Gating on the server's version."""

from __future__ import annotations

from typing import ClassVar

import psycopg
import pytest

from agensgraph.capabilities import (
    FEATURES,
    MINIMUM_VERSION,
    Capabilities,
    parse_release,
    parse_version,
)
from agensgraph.cypher import writable_counters
from agensgraph.errors import CapabilityError

GATED = [
    "has_property_promotion",
    "has_gql_clauses",
    "has_element_comparison",
    "has_endpoint_elision",
]


# Read off each release's own configure.ac or agversion, not imagined here. The 2.14.1 image
# reports 2.14.1 though its tag's configure.ac says 2.14.0.
SHIPPED = [
    ("2.14.1", (2, 14)),
    ("2.15.0", (2, 15)),
    ("2.16", (2, 16)),
    ("2.17.0", (2, 17)),
    ("2.18.4.0", (2, 18)),
    ("2.18.6.0-rc1", (2, 18)),
    ("2.18.6.1", (2, 18)),
    ("2.18-devel", (2, 18)),
]


@pytest.mark.parametrize(("reported", "expected"), SHIPPED)
def test_every_version_a_release_has_reported_is_read(
    reported: str, expected: tuple[int, int]
) -> None:
    """Three numbers before 2.18 and four from it, and servers on both are in the field."""
    assert parse_version(reported) == expected


# Shapes no release has worn, kept apart from SHIPPED so that list stays a record of fact.
@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        ("2.17", (2, 17)),
        ("2.18", (2, 18)),
        ("2.18.1", (2, 18)),
        ("2.18.6.0", (2, 18)),
        ("2.18.6.0-rc2", (2, 18)),
        ("2.18.6.1-rc10", (2, 18)),
        ("2.19-beta2", (2, 19)),
        ("2.19.0.0", (2, 19)),
        ("3.0", (3, 0)),
        ("  2.18-devel  ", (2, 18)),
        ("10.4", (10, 4)),
    ],
)
def test_a_reported_version_is_read_leniently(reported: str, expected: tuple[int, int]) -> None:
    """Whatever follows the two leading numbers is the build, and none of it is read."""
    assert parse_version(reported) == expected


@pytest.mark.parametrize("reported", ["", "devel", "2", "v2.18", "two.eighteen", "-2.18"])
def test_something_that_is_not_a_version_is_refused(reported: str) -> None:
    """As a driver error, so a caller catching this driver's failures catches it."""
    with pytest.raises(CapabilityError, match="version"):
        parse_version(reported)


@pytest.mark.parametrize("reported", ["2.13", "2.13.0", "2.12", "2.0", "1.9", "2.13-devel"])
def test_a_server_below_the_minimum_is_refused_at_once(reported: str) -> None:
    """Rather than at whichever later query first wants a catalog it does not have."""
    with pytest.raises(CapabilityError, match=r"2\.14"):
        Capabilities(reported)


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        ("2.14.1", ((2, 14, 1), False)),
        ("2.16", ((2, 16), False)),
        ("2.18.6.1", ((2, 18, 6, 1), False)),
        ("2.18.6.0-rc1", ((2, 18, 6, 0), True)),
        ("2.18-devel", ((2, 18), True)),
        ("  2.17.0  ", ((2, 17, 0), False)),
        # Shapes no release has worn: read rather than refused, and held to the oldest release of
        # their line.
        ("2.18.6.0rc1", ((2, 18, 6, 0), True)),
        ("2.17.0 (build 123)", ((2, 17, 0), True)),
        ("2.18.6.1.2", ((2, 18, 6, 1, 2), False)),
    ],
)
def test_a_release_is_read_to_the_patch(
    reported: str, expected: tuple[tuple[int, ...], bool]
) -> None:
    assert parse_release(reported) == expected


# What each shipped release has, as checked against that release's own server.
# Every feature in FEATURES appears in every row, so a new row must be placed on every release.
_ON_EVERY_SUPPORTED = set()
_SINCE_2_15 = {
    "cypher_vector_cast",
    "cypher_sql_functions",
    "missing_label_read_empty",
    "merge_creates_label",
    "collect_drops_nulls",
}
_SINCE_2_16 = _SINCE_2_15 | {"cypher_write_enforced", "missing_label_write_noop"}
_SINCE_2_17 = _SINCE_2_16 | {
    "detach_checks_edge_grants",
    "order_by_unprojected",
    "in_parameter_list",
    "cypher_typmod_cast",
    "fast_vertex_delete",
    "graph_dml_setting",
    "graphmeta_notices",
}
_SINCE_2_18 = _SINCE_2_17 | {
    "range_in_unwind",
    "element_comparison",
    "element_array_negative_index",
    "list_filter_drops_json_null",
    "size_of_list",
    "aggregate_slice_in_with",
    "gql_clauses",
    "property_promotion",
    "endpoint_elision",
}
_SINCE_2_18_4 = _SINCE_2_18 | {"graphmeta_catalog"}
_SINCE_2_18_6 = _SINCE_2_18_4 | {
    "unlabelled_read_checks_label_grants",
    "text_cast_sort_in_with",
    "graph_path_reported",
    "write_completes_under_limit",
    "write_counters_fresh",
}
_SAFETY = {name for name, feature in FEATURES.items() if feature.safety}
EXPECTED = {
    "2.14.1": _ON_EVERY_SUPPORTED | {"vle_unnamed_wrong"},
    "2.15.0": _SINCE_2_15 | {"vle_unnamed_wrong", "jsonb_minmax_wrong"},
    "2.16": _SINCE_2_16 | {"jsonb_minmax_wrong"},
    "2.17.0": _SINCE_2_17 | {"jsonb_minmax_wrong"},
    "2.18.4.0": _SINCE_2_18_4,
    "2.18.6.1": _SINCE_2_18_6,
    # Pre-releases answer as the first release of their line, and claim no safety feature: the
    # build may predate the change, and the driver's own guard is what answering no keeps.
    "2.18-devel": _SINCE_2_18 - _SAFETY,
    "2.18.6.0-rc1": _SINCE_2_18 - _SAFETY,
    "2.16.0-devel": (_SINCE_2_16 | {"jsonb_minmax_wrong"}) - _SAFETY,
}


def test_the_safety_features_are_the_ones_the_driver_guards_for() -> None:
    guarded = {"cypher_write_enforced", "detach_checks_edge_grants"}
    assert guarded | {"unlabelled_read_checks_label_grants"} == _SAFETY


@pytest.mark.parametrize("reported", sorted(EXPECTED))
def test_every_feature_on_every_release(reported: str) -> None:
    caps = Capabilities(reported)
    assert {name for name in FEATURES if caps.has(name)} == EXPECTED[reported]


def test_an_unknown_feature_is_a_mistake_not_a_no() -> None:
    with pytest.raises(ValueError, match="no such feature"):
        Capabilities("2.18").has("time_travel")


def test_a_checked_patch_gate_names_the_patch() -> None:
    with pytest.raises(CapabilityError) as caught:
        Capabilities("2.18.4.0").has("graph_path_reported", check=True)
    assert caught.value.required == "2.18.6"
    assert caught.value.found == "2.18.4.0"


def test_a_release_candidate_does_not_claim_its_patch() -> None:
    """It may lack a fix its number suggests, so it answers as the line's first release."""
    caps = Capabilities("2.18.6.0-rc1")
    assert caps.is_prerelease
    assert caps.release == (2, 18, 6, 0)
    assert not caps.has("unlabelled_read_checks_label_grants")


@pytest.mark.parametrize(
    ("reported", "dimensions", "allowed"),
    [
        ("2.16", 4, False),
        ("2.17.0", 4, True),
        ("2.17.0", None, False),
        ("2.18.6.1", None, True),
    ],
)
def test_the_cypher_vector_builders_refuse_a_server_that_cannot_carry_them(
    reported: str, dimensions: int | None, allowed: bool
) -> None:
    """A dimension in a Cypher cast needs 2.17, and a column of its own 2.18."""
    from agensgraph.vector import nearest, vector_index

    caps = Capabilities(reported)
    for build in (vector_index, nearest):
        if allowed:
            assert build("movie", "embedding", dimensions=dimensions, capabilities=caps)
        else:
            with pytest.raises(CapabilityError):
                build("movie", "embedding", dimensions=dimensions, capabilities=caps)


def test_the_deprecated_ordering_alias_answers_and_warns() -> None:
    with pytest.warns(DeprecationWarning, match="has_element_comparison"):
        assert Capabilities("2.18").has_element_ordering() is True
    with pytest.warns(DeprecationWarning, match="has_element_comparison"):
        assert Capabilities("2.17").has_element_ordering() is False


def test_the_minimum_is_itself_accepted() -> None:
    reported = ".".join(str(part) for part in MINIMUM_VERSION)
    assert Capabilities(reported).version == MINIMUM_VERSION


@pytest.mark.parametrize("reported", ["2.14.1", "2.15.0", "2.16", "2.17", "2.17.0", "2.17.3"])
@pytest.mark.parametrize("feature", GATED)
def test_the_older_servers_carry_none_of_the_gated_features(
    reported: str, feature: str
) -> None:
    caps = Capabilities(reported)
    assert getattr(caps, feature)() is False


@pytest.mark.parametrize(
    "reported", ["2.18", "2.18-devel", "2.18.4.0", "2.18.6.0", "2.18.6.0-rc1", "2.19", "3.0"]
)
@pytest.mark.parametrize("feature", GATED)
def test_the_newer_servers_carry_all_of_them(reported: str, feature: str) -> None:
    caps = Capabilities(reported)
    assert getattr(caps, feature)() is True
    assert getattr(caps, feature)(check=True) is True


@pytest.mark.parametrize("feature", GATED)
def test_a_checked_question_says_what_is_missing_and_what_would_carry_it(feature: str) -> None:
    caps = Capabilities("2.17")
    with pytest.raises(CapabilityError) as caught:
        getattr(caps, feature)(check=True)
    assert caught.value.required == "2.18"
    assert caught.value.found == "2.17"
    assert caught.value.feature
    assert "2.18" in str(caught.value)
    assert "2.17" in str(caught.value)


@pytest.mark.parametrize("reported", ["2.18-devel", "2.18.6.0", "2.18.6.0-rc1", "2.18.6.0-rc2"])
def test_the_version_reported_is_kept_exactly_as_given(reported: str) -> None:
    """Which build it was is part of what a person needs to see in a refusal."""
    caps = Capabilities(reported)
    assert caps.reported == reported
    assert caps.version == (2, 18)
    assert reported in repr(caps)


def test_capabilities_carry_no_dictionary() -> None:
    """One is built per connection, so it holds its few fields and nothing else."""
    with pytest.raises(AttributeError):
        Capabilities("2.18").anything = 1  # type: ignore[attr-defined]


@pytest.mark.server
class TestWhatTheGqlGateCovers:
    """The list in the docstring, asserted against the server rather than trusted.

    Both ways round, because a gate is a boundary: the server it says yes for takes every one of
    these, and the server it says no for refuses every one. Asserting only the first would let a
    gate that had gone quietly false pass as a skip.
    """

    GQL_ONLY: ClassVar[list[str]] = [
        "insert (:cap_probe {a: 1})",
        "let x = 1 return x",
        "match (n:cap_probe) return n.a as a next return a",
        "match (n:cap_probe) filter n.a is not null return n",
        "match (n:cap_probe) finish",
        "match (n:cap_probe) return all n.a",
        "match (n:cap_probe) with all n return n",
        "match (n:cap_probe) return n.a offset 0",
        "return 1 is unknown",
        "return 1 is not unknown",
        "return nullif(1, 1)",
        "call { match (n:cap_probe) return n as v } return v",
        "optional call { match (n:cap_probe) return n as v } return v",
        "match (n:cap_probe) call jsonb_each(n.bag) yield key return key",
        "for x in [1, 2] with offset as i return x, i",
        "return true xor false",
        "return current_graph",
        "match (n:cap_probe) where exists { match (q:cap_probe) return q } return n",
        "match (n:cap_probe) return count { match (q:cap_probe) return q }",
        "match (n:cap_probe) return collect { match (q:cap_probe) return q.a }",
        "match (n:cap_probe) return array { match (q:cap_probe) return q.a }",
        "match (n:cap_probe) return value { match (q:cap_probe) return q.a limit 1 }",
    ]

    @pytest.mark.parametrize("statement", GQL_ONLY)
    def test_the_gate_agrees_with_the_server_about_the_statement(
        self, agens, statement: str
    ) -> None:  # type: ignore[no-untyped-def]
        agens.execute("create vlabel cap_probe")
        agens.refresh_labels()
        # EXPLAIN parses and plans without running, which is all that is being asserted.
        if agens.capabilities.has_gql_clauses():
            agens.execute(f"explain {statement}")
            return
        with pytest.raises(psycopg.Error) as caught:
            agens.execute(f"explain {statement}")
        assert caught.value.sqlstate in {"42601", "42703"}, (
            "the gate says this server has no GQL surface, so the word is not in its grammar"
        )

    def test_insert_is_a_write_the_gate_does_not_make_safe(self, agens) -> None:  # type: ignore[no-untyped-def]
        """It shares its grammar arm with CREATE, so anything reading statements must know it."""
        assert writable_counters("insert (:cap_probe {a: 1})") == writable_counters(
            "create (:cap_probe {a: 1})"
        )
