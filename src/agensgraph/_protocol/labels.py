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
    """

    __slots__ = ("_graph", "_names")

    def __init__(self) -> None:
        self._names: dict[int, str] = {}
        self._graph: str | None = None

    @property
    def graph(self) -> str | None:
        """The graph these names came from."""
        return self._graph

    @property
    def query(self) -> str:
        """The statement that fills the cache, taking the graph name as its parameter."""
        return _QUERY

    def load(self, graph: str, rows: list[tuple[int, str]]) -> None:
        """Replace the contents with the labels of one graph."""
        self._names = dict(rows)
        self._graph = graph

    def invalidate(self) -> None:
        """Forget everything, so the next lookup has to reload."""
        self._names = {}
        self._graph = None

    def get(self, labid: int) -> str | None:
        """Look up a name, or ``None`` if this id is not known."""
        return self._names.get(labid)

    def name(self, labid: int) -> str:
        """Look up a name, raising if this id is not known."""
        try:
            return self._names[labid]
        except KeyError:
            raise KeyError(
                f"label id {labid} is not in the cache for graph {self._graph!r}"
            ) from None

    def __contains__(self, labid: object) -> bool:
        return labid in self._names

    def __len__(self) -> int:
        return len(self._names)

    def __repr__(self) -> str:
        return f"LabelCache(graph={self._graph!r}, {len(self._names)} labels)"
