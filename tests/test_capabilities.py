"""Gating on the server's version."""

from __future__ import annotations

from typing import ClassVar

import psycopg
import pytest

from agensgraph.capabilities import MINIMUM_VERSION, Capabilities, parse_version
from agensgraph.cypher import writable_counters
from agensgraph.errors import CapabilityError

GATED = [
    "has_property_promotion",
    "has_gql_clauses",
    "has_element_ordering",
    "has_endpoint_elision",
]


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        ("2.16", (2, 16)),
        ("2.17", (2, 17)),
        ("2.18", (2, 18)),
        ("2.18-devel", (2, 18)),
        ("2.18.1", (2, 18)),
        ("2.19-beta2", (2, 19)),
        ("3.0", (3, 0)),
        ("  2.18-devel  ", (2, 18)),
        ("10.4", (10, 4)),
    ],
)
def test_a_reported_version_is_read_leniently(reported: str, expected: tuple[int, int]) -> None:
    """A development build appends a suffix and a release does not, so both must be read."""
    assert parse_version(reported) == expected


@pytest.mark.parametrize("reported", ["", "devel", "2", "v2.18", "two.eighteen", "-2.18"])
def test_something_that_is_not_a_version_is_refused(reported: str) -> None:
    """As a driver error, so a caller catching this driver's failures catches it."""
    with pytest.raises(CapabilityError, match="version"):
        parse_version(reported)


@pytest.mark.parametrize("reported", ["2.16", "2.15", "2.0", "1.9", "2.16-devel"])
def test_a_server_below_the_minimum_is_refused_at_once(reported: str) -> None:
    """Rather than at whichever later query first wants a catalog it does not have."""
    with pytest.raises(CapabilityError, match=r"2\.17"):
        Capabilities(reported)


def test_the_minimum_is_itself_accepted() -> None:
    reported = ".".join(str(part) for part in MINIMUM_VERSION)
    assert Capabilities(reported).version == MINIMUM_VERSION


@pytest.mark.parametrize("reported", ["2.17", "2.17.3"])
@pytest.mark.parametrize("feature", GATED)
def test_the_older_servers_carry_none_of_the_gated_features(
    reported: str, feature: str
) -> None:
    caps = Capabilities(reported)
    assert getattr(caps, feature)() is False


@pytest.mark.parametrize("reported", ["2.18", "2.18-devel", "2.19", "3.0"])
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


def test_the_version_reported_is_kept_exactly_as_given() -> None:
    """A development suffix is part of what a person needs to see in a refusal."""
    caps = Capabilities("2.18-devel")
    assert caps.reported == "2.18-devel"
    assert caps.version == (2, 18)
    assert "2.18-devel" in repr(caps)


def test_capabilities_carry_no_dictionary() -> None:
    """One is built per connection, so it holds two fields and nothing else."""
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
