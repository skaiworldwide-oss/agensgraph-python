"""Resolving a label id to a label name.

The binary rendering of a graph value carries the label id inside the graphid but not
the label name, because the server resolves the name at render time from the session
graph path. A binary reader therefore has to keep its own map, which it fills from
``ag_label`` and drops whenever the graph path changes.

Every graph starts with two labels the server creates itself, ``ag_vertex`` with id 1
and ``ag_edge`` with id 2, so the first label a user creates gets id 3. Id 0 is never
stored; the planner uses it as a stand-in for "unlabelled".
"""

from __future__ import annotations

__all__ = ["AG_EDGE_LABID", "AG_VERTEX_LABID", "CURRENT_GRAPH_QUERY", "LabelCache"]

AG_VERTEX_LABID = 1
AG_EDGE_LABID = 2

CURRENT_GRAPH_QUERY = "select current_setting('graph_path', true)"
"""Which graph the session is reading, reported as an empty string when it is reading none.

The setting is not one the server reports of its own accord, so a session moved by anything
other than :meth:`~agensgraph.Connection.graph` is only found by asking.
"""

_QUERY = """
select l.labid, l.labname
from pg_catalog.ag_label l
join pg_catalog.ag_graph g on g.oid = l.graphid
where g.graphname = %s
"""


class LabelCache:
    """A label id to label name map for one graph, on one connection.

    Lookups are a plain dict hit. Interning the names is deliberately not done: a warm
    dict lookup is cheaper than interning, and interning costs more than it saves here.

    The names and the graph they came from are one field, replaced in a single assignment.
    Held apart they are two, and a reader between the two assignments would see names that
    belong to one graph beside the name of another -- which is how the caller that reloads
    the table asks the server for the labels of a graph the session has left, and stores the
    answer as though it described the graph it is on. Reading the two together costs about
    twelve nanoseconds a lookup, against six thousand a vertex for the rendering that needs
    it.
    """

    __slots__ = ("_table",)

    def __init__(self) -> None:
        self._table: tuple[str | None, dict[int, str]] = (None, {})

    @property
    def graph(self) -> str | None:
        """The graph these names came from."""
        return self._table[0]

    @property
    def query(self) -> str:
        """The statement that fills the cache, taking the graph name as its parameter."""
        return _QUERY

    def load(self, graph: str, rows: list[tuple[int, str]]) -> None:
        """Replace the contents with the labels of one graph."""
        self._table = (graph, dict(rows))

    def invalidate(self) -> None:
        """Forget everything, so the next lookup has to reload."""
        self._table = (None, {})

    def get(self, labid: int) -> str | None:
        """Look up a name, or ``None`` if this id is not known."""
        return self._table[1].get(labid)

    def name(self, labid: int) -> str:
        """Look up a name, raising if this id is not known."""
        table = self._table
        try:
            return table[1][labid]
        except KeyError:
            raise KeyError(
                f"label id {labid} is not in the cache for graph {table[0]!r}"
            ) from None

    def __contains__(self, labid: object) -> bool:
        return labid in self._table[1]

    def __len__(self) -> int:
        return len(self._table[1])

    def __repr__(self) -> str:
        graph, names = self._table
        return f"LabelCache(graph={graph!r}, {len(names)} labels)"
