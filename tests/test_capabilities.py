"""Gating on the server's version."""

from __future__ import annotations

import pytest

from agensgraph.capabilities import MINIMUM_VERSION, Capabilities, parse_version
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
    with pytest.raises(ValueError, match="version"):
        parse_version(reported)


@pytest.mark.parametrize("reported", ["2.15", "2.0", "1.9", "2.15-devel"])
def test_a_server_below_the_minimum_is_refused_at_once(reported: str) -> None:
    """Rather than at whichever later query first wants a catalog it does not have."""
    with pytest.raises(CapabilityError, match=r"2\.16"):
        Capabilities(reported)


def test_the_minimum_is_itself_accepted() -> None:
    reported = ".".join(str(part) for part in MINIMUM_VERSION)
    assert Capabilities(reported).version == MINIMUM_VERSION


@pytest.mark.parametrize("reported", ["2.16", "2.17"])
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
    caps = Capabilities("2.16")
    with pytest.raises(CapabilityError) as caught:
        getattr(caps, feature)(check=True)
    assert caught.value.required == "2.18"
    assert caught.value.found == "2.16"
    assert caught.value.feature
    assert "2.18" in str(caught.value)
    assert "2.16" in str(caught.value)


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
