"""Settings for one statement, sent in the same round trip as it.

A vector search wants ``hnsw.ef_search`` raised and an iterative scan switched on, a join wants
hash joins off, a slow read wants its own ``statement_timeout``. ``SET LOCAL`` before the statement
costs a round trip of its own, and in autocommit it does nothing at all: there is no transaction for
it to be local to, and the server only warns.

``set_config(name, value, true)`` is the function form of ``SET LOCAL``, so both the name and the
value are parameters and nothing is quoted into the text. Every setting goes in one ``select``, and
the connection sends it in a pipeline with the statement it is for. In autocommit the two share the
pipeline's implicit transaction, so the settings end with that statement. Inside a transaction they
last until it ends, exactly as ``SET LOCAL`` would.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .vector import SEARCH_OPTIONS, check_search_option

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = ["REFUSED_SETTINGS", "check_settings", "settings_statement"]

REFUSED_SETTINGS: Mapping[str, str] = {
    "graph_path": (
        "graph_path chooses the graph, and changing it under the driver leaves the label table "
        "describing the graph the session has left. Select it with graph() on the connection, or "
        "graph= on a pool."
    ),
}
"""Settings refused here, each with the reason and what to do instead."""

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_$]*(?:\.[A-Za-z_][A-Za-z0-9_$]*)*")
"""What the server takes as a setting's name: identifiers, joined by dots for an extension's."""

_SEARCH_PREFIXES = tuple({name.split(".")[0] + "." for name in SEARCH_OPTIONS})


def check_settings(settings: Mapping[str, object]) -> list[tuple[str, str]]:
    """Each setting, checked, as the name and the text ``set_config`` takes.

    A boolean is written ``on`` or ``off``, a number as its digits and a string as it is. The
    pgvector settings are held to :data:`~agensgraph.vector.SEARCH_OPTIONS`, because the server
    takes an unknown one of those silently on PostgreSQL 14. Any other name the server checks
    itself, and refuses when it does not know it.
    """
    checked = []
    for name, value in settings.items():
        # The type says a name is a string, and a mapping built at run time can hold another key.
        if not isinstance(name, str) or not _NAME.fullmatch(name):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(f"not the name of a setting: {name!r}")
        refused = REFUSED_SETTINGS.get(name.lower())
        if refused is not None:
            raise ValueError(refused)
        if name.lower().startswith(_SEARCH_PREFIXES):
            checked.append((name, check_search_option(name.lower(), value)))
        elif isinstance(value, bool):
            checked.append((name, "on" if value else "off"))
        elif isinstance(value, int | float | str):
            if isinstance(value, str) and "\x00" in value:
                raise ValueError(f"{name} cannot hold a NUL character")
            checked.append((name, str(value)))
        else:
            raise TypeError(
                f"{name} takes a string, a number or a boolean, got {type(value).__name__}"
            )
    return checked


def settings_statement(
    checked: Iterable[tuple[str, str]], *, local: bool = True
) -> tuple[str, tuple[str, ...]]:
    """One statement making every setting, and its parameters.

    The text depends only on how many settings there are, so a statement made with the same number
    of them is the same statement, and the server's prepared copy of it is reused.
    """
    pairs = list(checked)
    if not pairs:
        raise ValueError("no settings to make")
    scope = "true" if local else "false"
    calls = ", ".join(f"set_config(%s, %s, {scope})" for _ in pairs)
    return f"select {calls}", tuple(part for pair in pairs for part in pair)
